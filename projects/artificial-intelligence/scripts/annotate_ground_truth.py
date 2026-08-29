"""Interactive ground-truth annotation tool for video line crossings.

The tool is intentionally independent of detectors, trackers, crossing logic,
and T06. It uses OpenCV only for video playback and display.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2

# Named constants for platform-dependent key codes returned by cv2.waitKeyEx
KEY_LEFT_CODES = frozenset({2424832, 65361})
KEY_RIGHT_CODES = frozenset({2555904, 65363})


def normalize_key(key: int) -> str | int:
    """Normalize a waitKeyEx key code for portable cross-platform matching.

    Maps platform-specific arrow codes to semantic names ('left', 'right')
    and lowercases ASCII characters so 'm'/'M', 'u'/'U', 'q'/'Q' behave identically.
    """
    if key in KEY_LEFT_CODES:
        return "left"
    if key in KEY_RIGHT_CODES:
        return "right"
    if 0 <= key < 256:
        return chr(key).lower()
    return key


CSV_FIELDNAMES = [
    "frame_idx",
    "timestamp_seconds",
    "class_name",
    "direction",
    "line_id",
    "video_name",
]


@dataclass(frozen=True)
class AnnotationEvent:
    """A manually annotated line-crossing event."""

    frame_idx: int
    timestamp_seconds: float
    class_name: str
    direction: str
    line_id: str
    video_name: str


def calculate_timestamp(frame_idx: int, fps: float) -> float:
    """Calculate elapsed time from a zero-based frame index."""
    if fps <= 0:
        return 0.0

    return round(frame_idx / fps, 6)


def save_events_to_csv(
    events: list[AnnotationEvent],
    output_path: str | Path,
) -> None:
    """Save annotation events using the official T13 CSV schema."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for event in events:
            writer.writerow(
                {
                    "frame_idx": event.frame_idx,
                    "timestamp_seconds": event.timestamp_seconds,
                    "class_name": event.class_name,
                    "direction": event.direction,
                    "line_id": event.line_id,
                    "video_name": event.video_name,
                }
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and document keyboard controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactive Ground Truth Annotation Tool. "
            "Keyboard controls: SPACE=pause/resume, "
            "RIGHT=next frame, LEFT=previous frame, "
            "M=mark crossing, U=undo, Q or ESC=save and quit."
        )
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the output Ground Truth CSV file.",
    )
    return parser.parse_args()


def ask_crossing_details() -> tuple[str, str, str] | None:
    """Read and validate class, direction, and line information."""
    class_name = input("class_name: ").strip()
    if not class_name:
        print("Annotation cancelled: class_name cannot be empty.")
        return None

    direction = input("direction [IN/OUT]: ").strip().upper()
    if direction not in {"IN", "OUT"}:
        print("Annotation cancelled: direction must be IN or OUT.")
        return None

    line_id = input("line_id: ").strip()
    if not line_id:
        print("Annotation cancelled: line_id cannot be empty.")
        return None

    return class_name, direction, line_id


def draw_overlay(
    frame,
    frame_idx: int,
    total_frames: int,
    playing: bool,
    event_count: int,
):
    """Draw current playback state on the displayed frame."""
    status = "PLAYING" if playing else "PAUSED"
    text_lines = [
        f"Frame: {frame_idx}/{max(total_frames - 1, 0)}",
        f"Status: {status}",
        f"Annotations: {event_count}",
        "SPACE pause/resume | LEFT/RIGHT step | M mark | U undo | Q quit",
    ]

    result = frame.copy()

    for index, text in enumerate(text_lines):
        cv2.putText(
            result,
            text,
            (15, 30 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return result


def run_annotation(
    video_path: Path,
    output_path: Path,
) -> None:
    """Run the interactive OpenCV annotation loop."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.name
    events: list[AnnotationEvent] = []

    current_idx = 0
    playing = True
    window_name = "Ground Truth Annotation"

    last_decoded_idx = -1

    try:
        while True:
            if current_idx != last_decoded_idx + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, current_idx)

            success, frame = capture.read()

            if not success:
                break

            last_decoded_idx = current_idx

            displayed = draw_overlay(
                frame=frame,
                frame_idx=current_idx,
                total_frames=total_frames,
                playing=playing,
                event_count=len(events),
            )
            cv2.imshow(window_name, displayed)

            delay = max(1, round(1000 / fps)) if playing and fps > 0 else 0
            key = cv2.waitKeyEx(delay)

            if key in (ord("q"), 27):
                break

            if key == ord(" "):
                playing = not playing
                continue

            if key == ord("m"):
                print(f"\nMarking crossing at frame {current_idx}")
                details = ask_crossing_details()

                if details is not None:
                    class_name, direction, line_id = details
                    events.append(
                        AnnotationEvent(
                            frame_idx=current_idx,
                            timestamp_seconds=calculate_timestamp(
                                current_idx,
                                fps,
                            ),
                            class_name=class_name,
                            direction=direction,
                            line_id=line_id,
                            video_name=video_name,
                        )
                    )
                    print("Annotation added.")

                continue

            if key == ord("u"):
                if events:
                    removed = events.pop()
                    print(f"Removed annotation at frame {removed.frame_idx}.")
                else:
                    print("No annotation to remove.")

                continue

            key = cv2.waitKeyEx(delay)
            if key == -1:
                if playing:
                    current_idx = min(max(total_frames - 1, 0), current_idx + 1)
                continue

            action = normalize_key(key)

            if action == "left":
                current_idx = max(0, current_idx - 1)
                playing = False
                continue

            if action == "right":
                current_idx = min(max(total_frames - 1, 0), current_idx + 1)
                playing = False
                continue

            if action == " ":
                playing = not playing
                continue

            if action == "m":
                ts = calculate_timestamp(current_idx, fps)
                events.append(AnnotationEvent(frame_idx=current_idx, timestamp=ts, label="anomaly"))
                continue

            if action == "u":
                if events:
                    events.pop()
                continue

            if action in ("q", "\x1b"):
                break

            if playing:
                if current_idx >= max(total_frames - 1, 0):
                    break

                current_idx += 1

    finally:
        capture.release()
        cv2.destroyAllWindows()
        save_events_to_csv(events, output_path)
        print(f"Saved {len(events)} annotation(s) to {output_path}")


def main() -> None:
    """Validate inputs and start the annotation tool."""
    args = parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video file does not exist: {args.video}")

    run_annotation(
        video_path=args.video,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
