import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

FRAME_MS = 33          # time between video frames [ms]
BLINKS_SHOWN = 2       # blinks to play from each example before moving to the next
FRAMES_PER_BLINK = 8   # video frames for each blink, when the bins allow it

BACKGROUND = "#1a1a19"
TEXT = "#e8e7e3"
EDGE = "#383835"
BRIGHTER = (0xe3, 0x49, 0x48)
DARKER = (0x39, 0x87, 0xe5)
COLORS = np.array([DARKER, BRIGHTER], dtype=np.uint8)  # index with polarity


def load_flight(folder, flight):
    """Read one flight's labels, and its tensor cells in a dictionary by example."""
    labels = pd.read_csv(os.path.join(folder, "labels.csv"))
    available = sorted(labels["flight"].unique().tolist())
    if flight not in available:
        raise SystemExit(f"no flight {flight} in {folder}, choose one of {available}")
    labels = labels[labels["flight"] == flight]

    cells = pd.read_csv(os.path.join(folder, "tensors.csv"))
    cells = cells[cells["example"].isin(labels["example"])]
    by_example = {example: group[["bin", "polarity", "y", "x"]].to_numpy()
                  for example, group in cells.groupby("example")}
    return labels, by_example


def make_frames(labels):
    """List the (window start, bin) of each video frame.

    The frames are the last blinks of each window, when the whole window's trail
    has built up.
    """
    first = labels.iloc[0]
    n_bins = int(first["bins"])
    period_bins = 1e6 / first["freq_hz"] / first["bin_us"]
    step = max(1, round(period_bins / FRAMES_PER_BLINK))
    first_bin = max(0, n_bins - round(BLINKS_SHOWN * period_bins))

    return [
        (start_us, shown_bin)
        for start_us in sorted(labels["start_us"].unique())
        for shown_bin in range(first_bin, n_bins, step)
    ]


def build_viewer(labels, by_example):
    """Make the window with one panel per LED, and return it with its update function."""
    frames = make_frames(labels)
    size = int(labels.iloc[0]["crop_size"])
    by_window = {(row.start_us, row.led): row for row in labels.itertuples()}

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), facecolor=BACKGROUND)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.1, wspace=0.06)
    title = fig.suptitle("", color=TEXT)
    fig.legend(
        handles=[
            Line2D([], [], marker="s", ls="", color=np.array(BRIGHTER) / 255, label="brighter"),
            Line2D([], [], marker="s", ls="", color=np.array(DARKER) / 255, label="darker"),
            Line2D([], [], color=TEXT, label="true centre and size"),
        ],
        loc="lower center", ncol=3, facecolor=BACKGROUND, labelcolor=TEXT, edgecolor=EDGE,
    )

    panels = []
    for led, ax in enumerate(axes):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(EDGE)
        ax.set_title(f"LED {led}", color=TEXT)
        image = ax.imshow(np.zeros((size, size, 3), dtype=np.uint8), interpolation="nearest")
        box = ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=TEXT, linewidth=1))
        center, = ax.plot([], [], marker="+", color=TEXT, markersize=10, linestyle="")
        panels.append((image, box, center))

    def update(frame):
        start_us, shown_bin = frames[frame]
        distance = None

        for led, (image, box, center) in enumerate(panels):
            picture = np.full((size, size, 3), 0x1a, dtype=np.uint8)
            row = by_window.get((start_us, led))
            if row is None:  # LED out of view in this window
                image.set_data(picture)
                box.set_visible(False)
                center.set_data([], [])
                continue

            # Cells are in bin order, so each pixel ends with its newest event. The
            # LED turns red and blue as it blinks, and pixels it left stay blue.
            cells = by_example.get(row.example)
            if cells is not None:
                so_far = cells[cells[:, 0] <= shown_bin]
                picture[so_far[:, 2], so_far[:, 3]] = COLORS[so_far[:, 1]]
            image.set_data(picture)

            x = row.center_x - row.crop_x
            y = row.center_y - row.crop_y
            center.set_data([x], [y])
            box.set_visible(True)
            box.set_bounds(x - row.width / 2, y - row.height / 2, row.width, row.height)
            distance = row.distance_m

        now_s = (start_us + shown_bin * labels.iloc[0]["bin_us"]) / 1e6
        reading = "" if distance is None else f", distance = {distance:.1f} m"
        title.set_text(
            f"flight {labels.iloc[0]['flight']:.0f}, {labels.iloc[0]['freq_hz']:g} Hz, "
            f"t = {now_s:.4f} s{reading}"
        )
        return ()

    return fig, update, len(frames)


def main():
    parser = argparse.ArgumentParser(
        description="Watch the LED crops of a dataset from make_dataset.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("folder", nargs="?", default="dataset", help="folder with the two CSV files")
    parser.add_argument("--flight", type=int, default=0, help="which flight to watch")
    args = parser.parse_args()

    labels, by_example = load_flight(args.folder, args.flight)
    fig, update, n_frames = build_viewer(labels, by_example)
    animation = FuncAnimation(  # noqa: F841 (must stay alive while the window is open)
        fig, update, frames=n_frames, interval=FRAME_MS, repeat=True
    )
    plt.show()


if __name__ == "__main__":
    main()
