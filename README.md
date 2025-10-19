## DeepForest video annotator

Annotate `.mp4` videos in this repo with tree bounding boxes using the [DeepForest](https://deepforest.readthedocs.io/) model. Outputs annotated videos and optional per-frame detection CSVs.

### 1) Setup (macOS)

```bash
# From the repo root
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you want GPU acceleration and have an Apple Silicon Mac, recent PyTorch provides MPS support automatically. CPU works fine too.
- If `cv2.VideoWriter` fails to write MP4, try a different codec with `--codec avc1`. You may optionally install system codecs with Homebrew:
  ```bash
  brew install ffmpeg
  ```

### 2) Run

Process all `.mp4` files under the repo recursively, write annotated videos to `deepforest_output/`:

```bash
python scripts/annotate_trees_video.py --input . --output deepforest_output --csv
```

Common options:
- `--threshold 0.3` increase confidence threshold
- `--max-width 960` speed up by resizing frames before inference
- `--every-n 3` process every third frame (skips drawing on in-between frames)
- `--codec avc1` switch video codec if `mp4v` doesn’t work
- `--model /path/to/custom.pth` use a custom DeepForest checkpoint
- `--cpu` force CPU even if GPU/MPS available

Example with tuned settings:

```bash
python scripts/annotate_trees_video.py \
  --input "./**/*.mp4" \
  --output deepforest_output \
  --threshold 0.25 \
  --max-width 1280 \
  --every-n 1 \
  --csv
```

Outputs:
- Annotated MP4 video: `deepforest_output/<video_basename>.annotated.mp4`
- Detections CSV (if `--csv`): `deepforest_output/<video_basename>.detections.csv` with columns:
  - `video, frame, timestamp_sec, xmin, ymin, xmax, ymax, score, label, frame_width, frame_height`

### 3) Troubleshooting

- If you see “Failed to open VideoWriter”, try `--codec avc1` or `--codec H264` and ensure you have codecs (e.g., `brew install ffmpeg`).
- DeepForest performs best on aerial imagery. Ground-level videos may still work but results can vary.


