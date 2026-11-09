import argparse
import os
from types import SimpleNamespace

import numpy as np

from two_leds_sim import FOCAL, HEIGHT, WIDTH, EventCamera, drone_noise, load_path, save_raw

# Skip the start of each flight: every LED pixel fires at once when it first turns on
SETTLE_TIME = 0.02  # [s]

# width and height are the LED's size in the image. Each example's tensor has the
# shape [bins, 2, crop_size, crop_size].
LABEL_COLUMNS = (
    "example,flight,led,start_us,center_x,center_y,width,height,"
    "crop_x,crop_y,crop_size,bins,bin_us,freq_hz,label,distance_m"
)

# One row for each cell of a tensor that holds a 1, because almost all cells are 0.
# Polarity 0 is darker and 1 is brighter. y and x are relative to the crop corner.
TENSOR_COLUMNS = "example,bin,polarity,y,x"


def led_box(state, led, led_distance, led_radius):
    """True centre (x, y) and diameter [px] of one LED, as the simulator draws it."""
    x, y, z, yaw = state
    half = (-1 if led == 0 else 1) * led_distance / 2
    led_z = z + half * np.sin(np.radians(yaw))
    u = FOCAL * (x + half * np.cos(np.radians(yaw))) / led_z + WIDTH / 2
    v = FOCAL * y / led_z + HEIGHT / 2
    return np.array([u, v]), 2 * FOCAL * led_radius / led_z


def tensor_rows(events_us, start_us, n_bins, bin_us, corner, size):
    """Return the filled cells of one example's tensor as rows of bin, polarity, y, x."""
    t_us, x, y, p = events_us
    bins = (t_us - start_us) // bin_us  # whole numbers, so this matches the .raw file
    cols = x.astype(np.int64) - corner[0]
    rows = y.astype(np.int64) - corner[1]
    inside = (bins >= 0) & (bins < n_bins) & (cols >= 0) & (cols < size) & (rows >= 0) & (rows < size)

    tensor = np.zeros((n_bins, 2, size, size), dtype=bool)
    tensor[bins[inside], p[inside].astype(np.int64), rows[inside], cols[inside]] = True
    return np.column_stack(np.nonzero(tensor))


def make_flight(flight, first_example, labels_file, tensors_file, args, rng):
    """Simulate one flight, add its examples to both CSV files, and return how many."""
    freq = float(rng.choice(args.freqs))  # both LEDs share one frequency per flight
    settings = SimpleNamespace(
        freq1=freq,
        freq2=freq,
        led_distance=args.led_distance,
        led_radius=args.led_radius,
        camera_noise=args.camera_noise,
        led_trail=args.led_trail,
    )

    csv_path = rng.choice(args.csv)
    times, path = load_path(csv_path, args.duration)
    states = path + drone_noise(times, args.drone_noise, rng)
    camera = EventCamera(times, states, settings, rng)

    bin_us = round(args.bin * 1e6)
    n_bins = round(args.window / args.bin)
    window_us = bin_us * n_bins
    first_us = round((times[0] + SETTLE_TIME) * 1e6)
    last_us = round(times[-1] * 1e6) - window_us
    starts_us = np.arange(first_us, last_us, round(args.stride * 1e6))

    example = first_example
    pieces = []  # every event of the flight, for the .raw file
    now = times[0]

    # The camera must see the whole flight, so its pixels keep the right state.
    # Events between the windows are made, and only the .raw file keeps them.
    for start_us in starts_us:
        start = start_us / 1e6
        end = (start_us + window_us) / 1e6
        before = camera.events(now, start)
        during = camera.events(start, end)
        now = end
        if args.save_raw:
            pieces += [before, during]

        # A late event from just before the window can land inside it
        t, x, y, p = (np.concatenate(pair) for pair in zip(before[:4], during[:4]))
        events_us = ((t * 1e6).astype(np.int64), x, y, p)

        middle = (start + end) / 2
        state = np.array([np.interp(middle, times, states[:, i]) for i in range(4)])
        for led in (0, 1):
            center, diameter = led_box(state, led, args.led_distance, args.led_radius)
            if not (0 <= center[0] < WIDTH and 0 <= center[1] < HEIGHT):
                continue  # LED out of view

            # Shift the crop at random, so the LED is not always in the middle
            shift = rng.integers(-args.jitter, args.jitter + 1, 2)
            corner = np.floor(center).astype(np.int64) + shift - args.crop // 2

            labels_file.write(
                f"{example},{flight},{led},{start_us},{center[0]:.3f},{center[1]:.3f},"
                f"{diameter:.3f},{diameter:.3f},{corner[0]},{corner[1]},{args.crop},{n_bins},"
                f"{bin_us},{freq:g},{args.freqs.index(freq)},{state[2]:.3f}\n"
            )
            rows = tensor_rows(events_us, start_us, n_bins, bin_us, corner, args.crop)
            table = np.column_stack([np.full(len(rows), example), rows])
            np.savetxt(tensors_file, table, fmt="%d", delimiter=",")
            example += 1

    if args.save_raw:
        pieces.append(camera.events(now, times[-1]))
        save_raw(os.path.join(args.out, f"flight_{flight:03d}.raw"), pieces)

    print(f"flight {flight}: {example - first_example} examples, {freq:g} Hz")
    return example - first_example


def main():
    parser = argparse.ArgumentParser(
        description="Make SNN training data from the two LED simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", nargs="+", help="drone paths, one is picked for each flight")
    parser.add_argument("--flights", type=int, default=10, help="number of flights")
    parser.add_argument("--freqs", type=float, nargs="+", default=[500, 1000, 2000, 3000, 5000],
                        help="the frequency classes [Hz]")
    parser.add_argument("--window", type=float, default=0.01, help="length of each example [s]")
    parser.add_argument("--bin", type=float, default=0.00005, help="length of each time bin [s]")
    parser.add_argument("--stride", type=float, default=0.1, help="time between examples [s]")
    parser.add_argument("--crop", type=int, default=48, help="crop width and height [px]")
    parser.add_argument("--jitter", type=int, default=3, help="largest random crop shift [px]")
    parser.add_argument("--led-distance", type=float, default=0.25, help="between the LEDs [m]")
    parser.add_argument("--led-radius", type=float, default=0.02, help="of each LED [m]")
    parser.add_argument("--drone-noise", type=float, default=1.0, help="0 = follows the CSV exactly")
    parser.add_argument("--camera-noise", type=float, default=1.0, help="0 = perfect camera")
    parser.add_argument("--led-trail", type=float, default=0.015,
                        help="how fast a pixel's darker events fade after the LED leaves [s], 0 = none")
    parser.add_argument("--duration", type=float, help="use only this much of each path [s]")
    parser.add_argument("--save-raw", action="store_true", help="also save each flight as a .raw file")
    parser.add_argument("--seed", type=int, default=0, help="same seed, same dataset")
    parser.add_argument("--out", default="dataset", help="folder for the output files")
    args = parser.parse_args()

    if args.bin >= 0.5 / max(args.freqs):
        parser.error("--bin must be under half the period of the fastest frequency")
    if args.window < 1 / min(args.freqs):
        parser.error("--window must hold at least one period of the slowest frequency")

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.out, "labels.csv"), "w") as labels_file, \
            open(os.path.join(args.out, "tensors.csv"), "w") as tensors_file:
        labels_file.write(LABEL_COLUMNS + "\n")
        tensors_file.write(TENSOR_COLUMNS + "\n")
        n_examples = 0
        for flight in range(args.flights):
            n_examples += make_flight(flight, n_examples, labels_file, tensors_file, args, rng)

    print(f"{n_examples} examples saved in {args.out}/labels.csv and {args.out}/tensors.csv")


if __name__ == "__main__":
    main()
