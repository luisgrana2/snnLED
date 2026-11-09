import argparse
import itertools
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

# Camera
WIDTH, HEIGHT = 1280, 720  # EVK4 HD image size [px]
FOCAL = 1700.0             # focal length [px], from our 41.4 x 23.6 deg view

# Drone noise, scaled by --drone-noise
MOTION_NOISE = 0.08        # std of the velocity change per noise step [m/s]
YAW_NOISE = 2.0            # std of the yaw speed change per noise step [deg/s]
NOISE_DT = 0.03            # time between noise steps [s]
DRAG = 0.98                # stops the noise from speeding up forever
PULL_BACK = 0.98           # keeps the drone near the CSV path

# Event camera
MOTION_DT = 0.001          # longest time between simulation steps [s]
LED_BRIGHTNESS = 0.45      # LED log brightness when on (the background is 0)
THRESHOLD = 0.3            # log brightness change that fires an event
TRAIL_EVENTS = 2.0         # darker events each pixel fires after the LED leaves it
TRAIL_STEP = 0.002         # time between updates of the trail [s]

# Event camera noise, scaled by --camera-noise
THRESHOLD_NOISE = 0.06     # std of the threshold from pixel to pixel
LATENCY_STD = 2e-6         # std of the delay before an event fires [s]
BACKGROUND_RATE = 1.0      # random noise events per pixel per second
HOT_PIXELS = 100           # pixels that keep firing on their own
HOT_PIXEL_RATE = 500.0     # events per second from each hot pixel
EVENT_LOSS = 0.1           # fraction of events the camera loses

# Window
FRAME_MS = 33              # time between video frames [ms]
MAX_FRAMES_BEHIND = 3      # work per video frame, at most, when running late
NOISE_ON_SCREEN = 0.0005   # longest time a noise event stays on screen [s]
BACKGROUND = "#1a1a19"
BRIGHTER = "#e34948"
DARKER = "#3987e5"
TEXT = "#e8e7e3"
EDGE = "#383835"
EVENT_COLORS = np.array([to_rgba(DARKER), to_rgba(BRIGHTER)])  # index with p


def load_path(csv_path, duration):
    """Read the CSV and resample it every MOTION_DT seconds."""
    csv = pd.read_csv(csv_path)
    end = csv["t"].iloc[-1]
    if duration is not None:
        end = min(end, csv["t"].iloc[0] + duration)
    times = np.arange(csv["t"].iloc[0], end, MOTION_DT)
    path = np.column_stack(
        [np.interp(times, csv["t"], csv[axis]) for axis in ("x", "y", "z", "yaw")]
    )
    return times, path


def drone_noise(times, scale, rng):
    """Smooth random offset for x, y, z [m] and yaw [deg], one row per time."""
    noise_times = np.arange(times[0], times[-1] + NOISE_DT, NOISE_DT)
    offsets = np.zeros((len(noise_times), 4))
    offset = np.zeros(4)
    velocity = np.zeros(4)
    stds = scale * np.array([MOTION_NOISE, MOTION_NOISE, MOTION_NOISE, YAW_NOISE])

    for i in range(len(noise_times)):
        velocity = DRAG * velocity + rng.normal(0, stds)
        offset = PULL_BACK * offset + velocity * NOISE_DT
        offsets[i] = offset

    return np.column_stack(
        [np.interp(times, noise_times, offsets[:, axis]) for axis in range(4)]
    )


def blink_times(freq, start, end):
    """Times in [start, end) when an LED turns on or off. 0 Hz means always on."""
    if freq == 0:
        return np.array([])
    half_period = 0.5 / freq
    first = np.ceil(start / half_period)
    last = np.ceil(end / half_period)
    return np.arange(first, last) * half_period


def is_on(freq, times):
    """True where the LED is on. It starts each period on."""
    if freq == 0:
        return np.ones(len(times), dtype=bool)
    return np.floor(2 * freq * times).astype(np.int64) % 2 == 0


class EventCamera:
    """Makes the events one piece of time after another."""

    def __init__(self, times, states, args, rng):
        self.times = times
        self.states = states
        self.args = args
        self.rng = rng
        self.noise = args.camera_noise

        # Log brightness each pixel last fired at, and its own threshold
        # (Gaussian noise: real pixels never match exactly)
        self.memory = np.zeros((HEIGHT, WIDTH))
        thresholds = rng.normal(THRESHOLD, THRESHOLD_NOISE * self.noise, (HEIGHT, WIDTH))
        self.thresholds = np.maximum(thresholds, 0.05)

        self.hot_x = rng.integers(0, WIDTH, HOT_PIXELS)
        self.hot_y = rng.integers(0, HEIGHT, HOT_PIXELS)

        self.grid_x = np.arange(WIDTH)
        self.grid_y = np.arange(HEIGHT)[:, None]
        self.previous_boxes = [None, None]

        # When each LED last lit each pixel, for its trail
        self.last_lit = np.full((2, HEIGHT, WIDTH), -np.inf)

    def led_image(self, led, times):
        """Image position (u, v) and radius [px] of one LED at the given times."""
        x, y, z, yaw = (np.interp(times, self.times, self.states[:, i]) for i in range(4))
        half = (-1 if led == 0 else 1) * self.args.led_distance / 2
        led_z = z + half * np.sin(np.radians(yaw))  # this LED's own distance
        us = FOCAL * (x + half * np.cos(np.radians(yaw))) / led_z + WIDTH / 2
        vs = FOCAL * y / led_z + HEIGHT / 2
        return us, vs, FOCAL * self.args.led_radius / led_z

    def trail_events(self, led, start, end):
        """Darker events from the pixels an LED has just left, as lists of arrays.

        A real pixel does not go dark at once after a bright light passes: it keeps
        firing darker events for a while, less and less often. The LED's own pixels
        are left out, so the trail does not hide the blinking.
        """
        trail_time = self.args.led_trail
        out_t, out_x, out_y, out_p = [], [], [], []
        if trail_time <= 0:
            return out_t, out_x, out_y, out_p

        n_steps = max(1, int(np.ceil((end - start) / TRAIL_STEP)))
        edges = np.linspace(start, end, n_steps + 1)
        for first, last in zip(edges[:-1], edges[1:]):
            middle = (first + last) / 2

            # A box around where the LED has been lately, ending with where it is now
            recent = np.linspace(middle - 4 * trail_time, middle, 9)
            us, vs, radii = self.led_image(led, recent)
            x0 = max(int(us.min() - radii.max() - 1), 0)
            x1 = min(int(us.max() + radii.max() + 2), WIDTH)
            y0 = max(int(vs.min() - radii.max() - 1), 0)
            y1 = min(int(vs.max() + radii.max() + 2), HEIGHT)
            if x0 >= x1 or y0 >= y1:
                continue  # LED out of view

            dx = self.grid_x[x0:x1] - us[-1]
            dy = self.grid_y[y0:y1] - vs[-1]
            outside_led = dx * dx + dy * dy > radii[-1] ** 2
            age = middle - self.last_lit[led, y0:y1, x0:x1]

            # The firing rate fades with time since the LED left. Over a whole
            # fade, a pixel fires TRAIL_EVENTS on average.
            expected = np.zeros_like(age)
            active = outside_led & (age > 0) & np.isfinite(age)
            expected[active] = (
                TRAIL_EVENTS / trail_time * np.exp(-age[active] / trail_time) * (last - first)
            )

            counts = self.rng.poisson(expected)
            rows, cols = np.nonzero(counts)
            if len(rows) == 0:
                continue
            repeats = counts[rows, cols]
            out_x.append(np.repeat(cols + x0, repeats))
            out_y.append(np.repeat(rows + y0, repeats))
            out_p.append(np.zeros(repeats.sum(), dtype=bool))
            out_t.append(self.rng.uniform(first, last, repeats.sum()))

        return out_t, out_x, out_y, out_p

    def led_events(self, led, start, end):
        """Events of one LED between two times, as lists of arrays.

        Each LED only looks at the pixels around itself, which keeps this fast.
        So the LEDs must not touch in the image.
        """
        freq = self.args.freq1 if led == 0 else self.args.freq2

        # Step at every blink, plus often enough to follow the drone
        motion = self.times[np.searchsorted(self.times, start):np.searchsorted(self.times, end)]
        steps = np.unique(np.concatenate([[start], motion, blink_times(freq, start, end)]))

        # Check the LED halfway through each step, away from the exact blink time
        on = is_on(freq, (steps + np.append(steps[1:], end)) / 2)

        us, vs, radii = self.led_image(led, steps)

        out_t, out_x, out_y, out_p = [], [], [], []
        previous = self.previous_boxes[led]

        for i, t in enumerate(steps):
            u, v, radius = us[i], vs[i], radii[i]

            # The box covers the last box too, so pixels the LED left can fade
            box = (int(u - radius - 1), int(u + radius + 2), int(v - radius - 1), int(v + radius + 2))
            x0, x1, y0, y1 = box
            if previous is not None:
                x0, x1 = min(x0, previous[0]), max(x1, previous[1])
                y0, y1 = min(y0, previous[2]), max(y1, previous[3])
            previous = box
            x0, x1 = max(x0, 0), min(x1, WIDTH)
            y0, y1 = max(y0, 0), min(y1, HEIGHT)
            if x0 >= x1 or y0 >= y1:
                continue  # LED out of view

            # Brightness of the box: the LED disk if it is on, otherwise dark
            if on[i]:
                dx = self.grid_x[x0:x1] - u
                dy = self.grid_y[y0:y1] - v
                lit = dx * dx + dy * dy <= radius * radius
                brightness = LED_BRIGHTNESS * lit
                self.last_lit[led, y0:y1, x0:x1][lit] = t
            else:
                brightness = 0.0

            # A pixel fires one event for each threshold its brightness has crossed
            memory = self.memory[y0:y1, x0:x1]
            thresholds = self.thresholds[y0:y1, x0:x1]
            change = brightness - memory
            counts = (np.abs(change) / thresholds).astype(int)
            rows, cols = np.nonzero(counts)
            if len(rows) == 0:
                continue
            memory += np.sign(change) * counts * thresholds

            repeats = counts[rows, cols]
            out_x.append(np.repeat(cols + x0, repeats))
            out_y.append(np.repeat(rows + y0, repeats))
            out_p.append(np.repeat(change[rows, cols] > 0, repeats))
            delays = np.abs(self.rng.normal(0, LATENCY_STD * self.noise, repeats.sum()))
            out_t.append(t + delays)

        self.previous_boxes[led] = previous
        return out_t, out_x, out_y, out_p

    def events(self, start, end):
        """All the events between two times, as arrays t [s], x, y, p."""
        rng = self.rng
        all_t, all_x, all_y, all_p = [], [], [], []
        for led in (0, 1):
            for make_events in (self.led_events, self.trail_events):
                led_t, led_x, led_y, led_p = make_events(led, start, end)
                all_t += led_t
                all_x += led_x
                all_y += led_y
                all_p += led_p
        n_led = sum(len(column) for column in all_t)

        # Random noise events, anywhere in the image at any time
        n_noise = rng.poisson(self.noise * BACKGROUND_RATE * WIDTH * HEIGHT * (end - start))
        all_t.append(rng.uniform(start, end, n_noise))
        all_x.append(rng.integers(0, WIDTH, n_noise))
        all_y.append(rng.integers(0, HEIGHT, n_noise))
        all_p.append(rng.integers(0, 2, n_noise).astype(bool))

        # Hot pixels: a few fixed pixels that fire far more often than the rest
        n_hot = rng.poisson(self.noise * HOT_PIXEL_RATE * HOT_PIXELS * (end - start))
        which = rng.integers(0, HOT_PIXELS, n_hot)
        all_t.append(rng.uniform(start, end, n_hot))
        all_x.append(self.hot_x[which])
        all_y.append(self.hot_y[which])
        all_p.append(rng.integers(0, 2, n_hot).astype(bool))

        t = np.concatenate(all_t)
        x = np.concatenate(all_x)
        y = np.concatenate(all_y)
        p = np.concatenate(all_p)

        # Only the window uses this: a real camera cannot tell noise from an LED
        from_led = np.zeros(len(t), dtype=bool)
        from_led[:n_led] = True

        # The camera loses some events
        kept = rng.random(len(t)) >= min(EVENT_LOSS * self.noise, 0.9)
        return t[kept], x[kept], y[kept], p[kept], from_led[kept]


def save_raw(path, pieces):
    """Write the events to a .raw file in EVT3, the format of the EVK4 HD.

    After a text header, EVT3 is a list of 16-bit words. The top 4 bits say what
    a word holds. A word only sends what changed, so each event needs from one
    word (a new x) up to four (new time high, time low, y and x).
    """
    t, x, y, p, _ = (np.concatenate(column) for column in zip(*pieces))
    us = (t * 1e6).astype(np.int64)
    order = np.lexsort((x, y, us))  # by time, then by row, as a camera sends them
    us, x, y, p = us[order], x[order].astype(np.uint16), y[order].astype(np.uint16), p[order]

    # Which values change at each event. The first event sends everything.
    high = us >> 12
    low = us & 0xFFF
    new_high = np.diff(high, prepend=-1) != 0
    new_low = new_high | (np.diff(low, prepend=-1) != 0)
    new_y = np.diff(y.astype(np.int64), prepend=-1) != 0

    # Where each event's first word goes
    sizes = new_high.astype(np.int64) + new_low + new_y + 1
    position = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    words = np.zeros(sizes.sum(), dtype=np.uint16)

    words[position[new_high]] = 0x8000 | (high[new_high] & 0xFFF)  # time, bits 23-12
    position = position + new_high
    words[position[new_low]] = 0x6000 | low[new_low]               # time, bits 11-0
    position = position + new_low
    words[position[new_y]] = y[new_y]                              # y (top bits 0)
    position = position + new_y
    words[position] = 0x2000 | (p.astype(np.uint16) << 11) | x     # polarity and x

    header = (
        "% camera_integrator_name Prophesee\n"
        f"% date {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        "% evt 3.0\n"
        f"% format EVT3;height={HEIGHT};width={WIDTH}\n"
        "% generation 4.2\n"
        f"% geometry {WIDTH}x{HEIGHT}\n"
        "% integrator_name Prophesee\n"
        "% plugin_integrator_name Prophesee\n"
        "% plugin_name hal_plugin_prophesee\n"
        "% sensor_generation 4.2\n"
        "% sensor_name IMX636\n"
        "% serial_number 00000000\n"
        "% system_ID 49\n"
        "% end\n"
    )
    with open(path, "wb") as file:
        file.write(header.encode())
        words.astype("<u2").tofile(file)


def play(camera, times, states, args):
    """Make the events and draw them as the video plays. Return what was made."""
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # image y points down
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=BRIGHTER, label="brighter"),
            Line2D([], [], marker="o", ls="", color=DARKER, label="darker"),
        ],
        loc="upper right", facecolor=BACKGROUND, labelcolor=TEXT, edgecolor=EDGE,
    )
    dots = ax.scatter([], [], s=5, linewidths=0)  # noise events

    # LED events: each pixel shows its newest event, which fades over the trail
    # time. So an LED shows the colour of its last blink, and a moving LED leaves
    # a fading trail of the darker events it fires as it leaves each pixel.
    last_time = np.full((HEIGHT, WIDTH), -np.inf)
    last_polarity = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    picture = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
    trail_image = ax.imshow(picture, interpolation="nearest", extent=(0, WIDTH, HEIGHT, 0))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    event_rgb = (EVENT_COLORS[:, :3] * 255).astype(np.uint8)

    # Noise stays only briefly, or it fills the screen. The trail must last at
    # least half a period of the slowest LED, so that LED stays visible.
    half_periods = [0.5 / f for f in (args.freq1, args.freq2) if f > 0]
    window = min([NOISE_ON_SCREEN] + half_periods)
    trail = max([args.trail] + half_periods)

    first_time, final_time = times[0], times[-1]
    longest_piece = MAX_FRAMES_BEHIND * args.speed * FRAME_MS / 1000
    saved = []
    # Live state: flight time reached, events on screen, and when playing began
    live = dict(now=first_time, recent=None, started=None, finished=False)

    def update(_frame):
        if live["finished"]:
            return dots,
        if live["started"] is None:
            live["started"] = time.perf_counter()

        # Catch up with the clock, but never freeze the window to do it
        target = first_time + (time.perf_counter() - live["started"]) * args.speed
        end = min(target, live["now"] + longest_piece, final_time)
        if end <= live["now"]:
            return dots,

        piece = camera.events(live["now"], end)
        live["now"] = end
        if args.out:
            saved.append(piece)
        t, x, y, p, from_led = piece

        # LED events: in time order, so each pixel keeps its newest event
        led = np.flatnonzero(from_led)
        led = led[np.argsort(t[led])]
        last_time[y[led], x[led]] = t[led]
        last_polarity[y[led], x[led]] = p[led]
        fade = np.clip(1 - (end - last_time) / trail, 0, 1)
        picture[..., :3] = event_rgb[last_polarity]
        picture[..., 3] = (fade * 255).astype(np.uint8)
        trail_image.set_data(picture)

        # Noise events: keep only the ones that belong on screen
        noise = ~from_led
        piece = (t[noise], x[noise], y[noise], p[noise])
        if live["recent"] is not None:
            piece = tuple(np.concatenate(pair) for pair in zip(live["recent"], piece))
        keep = (piece[0] > end - window) & (piece[0] <= end)
        live["recent"] = tuple(column[keep] for column in piece)
        _, noise_x, noise_y, noise_p = live["recent"]
        dots.set_offsets(np.column_stack([noise_x, noise_y]))
        dots.set_facecolors(EVENT_COLORS[noise_p.astype(int)])

        z = np.interp(end, times, states[:, 2])
        ax.set_title(f"t = {end:.6f} s, distance = {z:.1f} m", color=TEXT)

        if end >= final_time:
            live["finished"] = True
            played = time.perf_counter() - live["started"]
            print(f"{final_time - first_time:.3f} s of flight played in {played:.1f} s")
        return dots,

    animation = FuncAnimation(  # noqa: F841 (must stay alive while the window is open)
        fig, update, frames=itertools.count(), interval=FRAME_MS, cache_frame_data=False
    )
    plt.show()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Two blinking LEDs on a drone, seen live by an event camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv", nargs="?", default="trajectory.csv", help="drone path")
    parser.add_argument("--freq1", type=float, default=1000, help="first LED [Hz], 0 = steady")
    parser.add_argument("--freq2", type=float, default=2500, help="second LED [Hz], 0 = steady")
    parser.add_argument("--led-distance", type=float, default=0.25, help="between the LEDs [m]")
    parser.add_argument("--led-radius", type=float, default=0.02, help="of each LED [m]")
    parser.add_argument("--drone-noise", type=float, default=1.0, help="0 = follows the CSV exactly")
    parser.add_argument("--camera-noise", type=float, default=1.0, help="0 = perfect camera")
    parser.add_argument("--led-trail", type=float, default=0.015,
                        help="how fast a pixel's darker events fade after the LED leaves [s], 0 = none")
    parser.add_argument("--speed", type=float, default=1.0, help="video speed, 0.01 = 100 times slower")
    parser.add_argument("--trail", type=float, default=0.3, help="how long LED events stay on screen [s]")
    parser.add_argument("--duration", type=float, help="play only this long [s]")
    parser.add_argument("--out", help="save the events to this .raw file when the window closes")
    args = parser.parse_args()

    rng = np.random.default_rng()
    times, path = load_path(args.csv, args.duration)
    states = path + drone_noise(times, args.drone_noise, rng)

    camera = EventCamera(times, states, args, rng)
    saved = play(camera, times, states, args)

    if args.out and saved:
        print(f"saving {args.out} ...")
        save_raw(args.out, saved)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
