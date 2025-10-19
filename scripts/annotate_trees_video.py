#!/usr/bin/env python3
import argparse
import os
import sys
import glob
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable but deepforest can still run on CPU
    torch = None  # type: ignore

try:
    from deepforest import main as deepforest_main
except ImportError as exc:
    print("Error: deepforest is not installed. Install dependencies first: pip install -r requirements.txt", file=sys.stderr)
    raise


def discover_videos(input_path: str) -> List[str]:
    """Return a list of .mp4 files from a file, directory, or glob."""
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".mp4") else []
    if any(ch in input_path for ch in ["*", "?", "["]):
        return sorted([p for p in glob.glob(input_path, recursive=True) if p.lower().endswith(".mp4")])
    if os.path.isdir(input_path):
        return sorted([
            os.path.join(root, f)
            for root, _dirs, files in os.walk(input_path)
            for f in files
            if f.lower().endswith(".mp4")
        ])
    return []


def select_device(prefer_gpu: bool = True) -> str:
    """Select computation device string: 'cuda', 'mps', or 'cpu'."""
    if not prefer_gpu:
        return "cpu"
    try:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_deepforest_model(model_path: Optional[str], device: str):
    """Load DeepForest model; use release weights if no model_path provided."""
    model = deepforest_main.deepforest()
    if model_path:
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model.load_model(model_path)
    else:
        model.use_release()
    # Try moving model to device (best-effort; DeepForest may expose .to on wrapper)
    try:
        if hasattr(model, "to"):
            model.to(device)
        elif hasattr(model, "model") and hasattr(model.model, "to"):
            model.model.to(device)
    except Exception:
        # Fallback silently to whatever DeepForest configured
        pass
    return model


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_scale(width: int, max_width: Optional[int]) -> float:
    if not max_width or max_width <= 0 or width <= max_width:
        return 1.0
    return max(1e-6, max_width / float(width))


def predict_boxes(model, frame_bgr: np.ndarray, score_threshold: float, resize_scale: float) -> pd.DataFrame:
    """Run DeepForest on an image, optionally resizing before inference.

    Returns a DataFrame with columns [xmin, ymin, xmax, ymax, label, score] in ORIGINAL frame coordinates.
    """
    original_h, original_w = frame_bgr.shape[:2]
    if resize_scale != 1.0:
        resized_w = int(round(original_w * resize_scale))
        resized_h = int(round(original_h * resize_scale))
        resized = cv2.resize(frame_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        image_for_model = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        scale_back = 1.0 / resize_scale
    else:
        image_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        scale_back = 1.0

    try:
        predictions = model.predict_image(image=image_for_model, return_plot=False)
    except TypeError:
        # Older DeepForest versions use image= or path=; ensure we pass correctly
        predictions = model.predict_image(image=image_for_model, return_plot=False)

    if predictions is None or len(predictions) == 0:
        # Empty DataFrame with the expected columns
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])

    # Filter by score
    if "score" in predictions.columns:
        predictions = predictions[predictions["score"] >= score_threshold]

    # Scale boxes back to original frame coordinates if resized
    predictions = predictions.copy()
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        predictions[col] = predictions[col].astype(float) * scale_back

    # Clip to image bounds
    predictions["xmin"] = predictions["xmin"].clip(0, original_w - 1)
    predictions["xmax"] = predictions["xmax"].clip(0, original_w - 1)
    predictions["ymin"] = predictions["ymin"].clip(0, original_h - 1)
    predictions["ymax"] = predictions["ymax"].clip(0, original_h - 1)
    return predictions


def draw_boxes(frame_bgr: np.ndarray, boxes_df: pd.DataFrame, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """Draw bounding boxes and labels on the frame."""
    if boxes_df is None or len(boxes_df) == 0:
        return frame_bgr
    result = frame_bgr
    for _idx, row in boxes_df.iterrows():
        x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
        score = float(row.get("score", 0.0))
        label = str(row.get("label", "Tree"))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}" if score else label
        # Text background for readability
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 4, y1), color, thickness=-1)
        cv2.putText(result, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def annotate_video(
    video_path: str,
    output_dir: str,
    model,
    device: str,
    score_threshold: float,
    max_width: Optional[int],
    output_fps: Optional[float],
    codec: str,
    write_csv: bool,
    process_every_n: int,
) -> Tuple[str, Optional[str]]:
    """Annotate a single video, returning (output_video_path, output_csv_path)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = float(output_fps or input_fps)
    fourcc = cv2.VideoWriter_fourcc(*codec)

    ensure_dir(output_dir)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{base}.annotated.mp4")
    writer = cv2.VideoWriter(out_video, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different --codec (e.g., avc1 or H264) or install codecs.")

    csv_rows: List[dict] = []
    frame_index = 0
    processed = 0
    resize_scale = compute_scale(width, max_width)

    # Simple progress indicator without tqdm dependency in runtime loop
    def log_progress():
        if total_frames > 0:
            pct = 100.0 * min(frame_index, total_frames) / float(total_frames)
            print(f"\rProcessing {base}: frame {frame_index}/{total_frames} ({pct:5.1f}%)", end="", flush=True)
        else:
            print(f"\rProcessing {base}: frame {frame_index}", end="", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Determine whether to process this frame
            should_process = (process_every_n <= 1) or (frame_index % process_every_n == 0)

            if should_process:
                boxes = predict_boxes(model, frame, score_threshold=score_threshold, resize_scale=resize_scale)
                processed += 1
            else:
                boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "label", "score"])  # no boxes drawn

            # Draw boxes and write frame
            annotated = draw_boxes(frame, boxes)
            writer.write(annotated)

            # Append CSV rows if requested
            if write_csv and boxes is not None and len(boxes) > 0:
                ts = frame_index / input_fps if input_fps else 0.0
                for _i, r in boxes.iterrows():
                    csv_rows.append({
                        "video": os.path.basename(video_path),
                        "frame": frame_index,
                        "timestamp_sec": ts,
                        "xmin": float(r["xmin"]),
                        "ymin": float(r["ymin"]),
                        "xmax": float(r["xmax"]),
                        "ymax": float(r["ymax"]),
                        "score": float(r.get("score", 0.0)),
                        "label": str(r.get("label", "Tree")),
                        "frame_width": width,
                        "frame_height": height,
                    })

            frame_index += 1
            if frame_index % 10 == 0:
                log_progress()
        # Final progress line
        log_progress()
        print()
    finally:
        cap.release()
        writer.release()

    out_csv = None
    if write_csv:
        out_csv = os.path.join(output_dir, f"{base}.detections.csv")
        pd.DataFrame(csv_rows).to_csv(out_csv, index=False)

    return out_video, out_csv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate .mp4 videos with tree bounding boxes using DeepForest")
    parser.add_argument("--input", "-i", default=".", help="Input file/dir/glob to search for .mp4 (default: current directory)")
    parser.add_argument("--output", "-o", default="./deepforest_output", help="Output directory for annotated videos and CSVs")
    parser.add_argument("--model", "-m", default=None, help="Optional path to a custom DeepForest .pth model. Uses release model by default")
    parser.add_argument("--threshold", "-t", type=float, default=0.2, help="Score threshold for detections (default: 0.2)")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize frames to this max width for faster inference (scale boxes back). 0 to disable")
    parser.add_argument("--fps", type=float, default=None, help="Optional output video FPS. Defaults to input FPS")
    parser.add_argument("--codec", type=str, default="mp4v", help="FourCC video codec (e.g., mp4v, avc1, H264). Default: mp4v")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU/MPS is available")
    parser.add_argument("--csv", action="store_true", help="Write per-video CSV with per-frame detections")
    parser.add_argument("--every-n", type=int, default=1, help="Process every Nth frame (draw nothing on skipped frames). Default: 1 (all frames)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    videos = discover_videos(args.input)
    if not videos:
        print("No .mp4 files found for input pattern/path.", file=sys.stderr)
        return 2

    device = select_device(prefer_gpu=not args.cpu)
    print(f"Using device: {device}")

    model = load_deepforest_model(args.model, device=device)

    ensure_dir(os.path.abspath(args.output))

    failures = []
    outputs = []
    for vp in videos:
        try:
            out_video, out_csv = annotate_video(
                video_path=vp,
                output_dir=args.output,
                model=model,
                device=device,
                score_threshold=args.threshold,
                max_width=args.max_width if args.max_width and args.max_width > 0 else None,
                output_fps=args.fps,
                codec=args.codec,
                write_csv=args.csv,
                process_every_n=max(1, int(args.every_n)),
            )
            outputs.append((out_video, out_csv))
            print(f"Annotated: {vp} -> {out_video}")
            if out_csv:
                print(f"Detections CSV: {out_csv}")
        except Exception as e:
            print(f"Failed to process {vp}: {e}", file=sys.stderr)
            failures.append((vp, str(e)))

    if failures:
        print("\nSome files failed:", file=sys.stderr)
        for f, err in failures:
            print(f" - {f}: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


