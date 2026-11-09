# Two LEDs on a drone, seen by an event camera

A drone carries two blinking LEDs. An event camera (EVK4 HD, 1280 x 720) watches it fly. Your task is to train a spiking neural network (SNN) that finds the LEDs in the events and tells how fast they blink.

## Files

| File | What it is |
|---|---|
| `two_leds_sim.py` | The simulator. Run it to watch a flight, and to save the events as a `.raw` file. |
| `trajectory.csv` | An example drone path. |
| `snn_starter.py` | Your starting point. It runs the simulator and has empty steps for you to fill in. |

## Install

You need Python 3.10 or newer. These steps were tested with Python 3.12 on Ubuntu.

The simulator window needs Tk:

```
sudo apt install python3-tk
```

Make a virtual environment and install the packages:

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib torch snntorch
```

Check that it works:

```
python3 two_leds_sim.py trajectory.csv
```

A window should open with the drone flying. Red dots are brighter events, blue dots are darker events.

## Get to know the simulator

Play with the settings first. `--help` lists them all:

```
python3 two_leds_sim.py --help
python3 two_leds_sim.py trajectory.csv --freq1 500 --freq2 3000 --camera-noise 3
python3 two_leds_sim.py trajectory.csv --speed 0.001     # slow motion: see each blink
python3 two_leds_sim.py trajectory.csv --out events.raw  # save the events when the window closes
```

Things to notice:

- An event camera does not take pictures. A pixel fires an event only when its brightness changes. A still, steady LED fires nothing; a blinking LED fires each time it turns on or off.
- When the drone moves away, the LEDs shrink and come closer together.
- A moving LED leaves a trail of darker events behind it, as in real flights.
- The noise is random events across the image, a few hot pixels, and lost events.

### The drone path

`trajectory.csv` has columns `t` [s], `x`, `y`, `z` [m] and `yaw` [deg]. The camera looks along `z`, so `z` is the distance to the drone; `x` points right and `y` points down in the image. Yaw turns the drone around the vertical axis. At 0 deg it faces the camera. Near 90 deg the two LEDs overlap and the simulator gets them wrong, so keep yaw well below that.

Write your own CSV files to make new flights. You will need several, so you can test on paths the network never saw.

## Your task

Open `snn_starter.py` and run it:

```
python3 snn_starter.py
```

It simulates a flight, then stops at the first empty step and tells you what to write next. Each step is marked `# PUT HERE` and has hints. Change the settings at the top of the file to make different flights.

1. **`make_labels()`**: the true pixel position of each LED over time. Work it out from the drone path and the camera model.
2. **`make_tensors()`**: turn events into tensors of shape `[time bins, 2, height, width]`, with one channel for each polarity.
3. **`build_snn()`**: a small spiking network with snntorch.
4. **`train()`**: train it.
5. **`evaluate()`**: test it on flights it never saw.

Work in this order, and show your supervisor each step before moving on.

### Suggested order of difficulty

1. Frequency only: give the network a crop around one LED whose position you know, and have it pick from about 5 frequencies under 5 kHz.
2. Position: find the LEDs in the whole image, shrunk to a smaller size.
3. Harder flights: more noise, wider yaw, greater distance, and higher frequencies last.

### Tips

- A time bin must be shorter than half a blink period. 2500 Hz needs bins under 200 us; 100 kHz needs bins under 5 us, which makes the data much larger.
- Before training on everything, check the network can learn a handful of examples by heart. If it cannot, there is a bug.
- Keep whole flights apart when you split train and test data. Otherwise the test score will look better than it really is.
- Compare the network with a simple method that uses no network, such as an FFT of the event counts around each LED. On clean data the simple method may well win; the network becomes interesting when it holds up under heavy noise.
- The snntorch tutorials cover most of what you need: https://snntorch.readthedocs.io

If you get stuck, ask. Don't spend days stuck on your own.
