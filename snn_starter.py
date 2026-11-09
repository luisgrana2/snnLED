from types import SimpleNamespace

import numpy as np

from two_leds_sim import FOCAL, HEIGHT, WIDTH, EventCamera, drone_noise, load_path

# ---------------------------------------------------------------------------
# Simulation settings: change these to make different flights
# ---------------------------------------------------------------------------
CSV_PATH = "trajectory.csv"  # drone path: t, x, y, z, yaw
DURATION = 2.0               # seconds of flight to simulate (None = whole CSV)
FREQ1 = 1000                 # first LED blink frequency [Hz], 0 = steady
FREQ2 = 2500                 # second LED blink frequency [Hz]
LED_DISTANCE = 0.25          # distance between the LEDs [m]
LED_RADIUS = 0.02            # radius of each LED [m]
DRONE_NOISE = 1.0            # 0 = drone follows the CSV exactly
CAMERA_NOISE = 1.0           # 0 = perfect camera
LED_TRAIL = 0.015            # how long pixels keep firing after the LED leaves [s]
SEED = 0                     # same seed, same flight


def run_simulation():
    """Simulate one flight. Return the events and the true drone state over time."""
    rng = np.random.default_rng(SEED)
    times, path = load_path(CSV_PATH, DURATION)
    states = path + drone_noise(times, DRONE_NOISE, rng)  # true x, y, z [m], yaw [deg]

    settings = SimpleNamespace(
        freq1=FREQ1, freq2=FREQ2, led_distance=LED_DISTANCE, led_radius=LED_RADIUS,
        camera_noise=CAMERA_NOISE, led_trail=LED_TRAIL,
    )
    camera = EventCamera(times, states, settings, rng)

    # Make the events in small pieces of time, as the camera would send them
    pieces = []
    for start in np.arange(times[0], times[-1], 0.05):
        t, x, y, p, _ = camera.events(start, min(start + 0.05, times[-1]))
        pieces.append((t, x, y, p))
    t, x, y, p = (np.concatenate(column) for column in zip(*pieces))
    order = np.argsort(t)

    # t [s], x and y [px], p: 1 = brighter, 0 = darker
    events = {"t": t[order], "x": x[order], "y": y[order], "p": p[order].astype(np.uint8)}
    return events, times, states


def make_labels(times, states):
    """Step 1: the true LED centres in the image, to train and test against."""
    # PUT HERE: for each time, the (x, y) pixel of each LED.
    # Each row of states is x, y, z [m] and yaw [deg] of the drone at that time.
    # The camera is a pinhole: pixel = FOCAL * position / z + image centre.
    # Look at led_image() in two_leds_sim.py to see how the simulator does it.
    # Also useful: the LED size in pixels, 2 * FOCAL * LED_RADIUS / z.
    return None


def make_tensors(events, labels):
    """Step 2: turn events into tensors the SNN can read."""
    # PUT HERE: cut the events into short time windows (for example 10 ms),
    
    # and each window into time bins. A bin must be shorter than half a blink
    # period, so 2500 Hz needs bins under 200 us.

    # Use 2 channels: darker events (p = 0) and brighter events (p = 1).
    # Crop around each LED, or shrink the whole image: 1280 x 720 is too big.
    # Shape of one example: [time bins, 2, height, width].
    # Keep the matching labels next to each tensor.
    return None


def build_snn():
    """Step 3: the spiking network."""
    # PUT HERE: a small network with snntorch, for example:
    #   Conv2d -> MaxPool -> snn.Leaky -> Conv2d -> MaxPool -> snn.Leaky -> Linear
    # Start with frequency classes on crops, then try finding the LED position.
    # The snntorch tutorials (snntorch.readthedocs.io) show each piece.
    return None


def train(network, tensors):
    """Step 4: train the network."""
    # PUT HERE: feed each example one time bin at a time, collect the output
    # spikes, compute the loss and update the weights.
    # Reset the neurons before each example (snntorch.utils.reset).
    # Tip: first make sure it can learn a handful of examples by heart.
    return None


def evaluate(network, tensors):
    """Step 5: check how well it works."""
    # PUT HERE: test on flights the network never saw during training
    # (a new SEED or a new CSV). Report the position error in pixels, or the
    # share of frequencies it gets right. Compare with a simple method without
    # a network, for example an FFT of the event counts around each LED.
    return None


def main():
    events, times, states = run_simulation()
    print(f"{len(events['t'])} events in {times[-1] - times[0]:.2f} s, image {WIDTH} x {HEIGHT}")

    labels = make_labels(times, states)
    if labels is None:
        print("Next: write make_labels() (step 1)")
        return

    tensors = make_tensors(events, labels)
    if tensors is None:
        print("Next: write make_tensors() (step 2)")
        return

    network = build_snn()
    if network is None:
        print("Next: write build_snn() (step 3)")
        return

    if train(network, tensors) is None:
        print("Next: write train() (step 4)")
        return

    if evaluate(network, tensors) is None:
        print("Next: write evaluate() (step 5)")


if __name__ == "__main__":
    main()
