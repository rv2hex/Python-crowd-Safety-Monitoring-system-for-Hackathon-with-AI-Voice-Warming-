<div align="center">
  <img src="./banner.svg" alt="CROWD_SENTINEL banner" width="100%" />
</div>

# Crowd Safety Monitoring System

Hackathon group project — real-time crowd risk detection from a video feed
using OpenCV motion analysis. Classifies crowd behavior as LOW / MEDIUM / HIGH
risk and raises alerts on abnormal motion.

Group: Rishi Vishwakarma, Hong Sovannarith, Dinesh Shrestha.

## How it works (`Main.py`)

1. Frame differencing (grayscale → absdiff → blur → threshold → dilate),
   contour areas summed into a total motion score.
2. **Risk classifier** from motion area, motion density, and frame-to-frame
   motion delta:
   - `HIGH` — rapid motion spike in a dense crowd (possible panic/stampede)
   - `MEDIUM` — sustained high group movement
   - `LOW` — minor/ambient movement
3. **Temporal consistency** — risk must persist 5 consecutive frames before it
   counts (kills single-frame false positives).
4. **Alerts** (3 s cooldown): console line + append to `alerts.log` + PC-speaker
   beep on HIGH (Windows-only via `winsound`).

Live overlay shows the current risk level and FPS. Press `q` to stop.

## Run

```sh
pip install opencv-python numpy
python Main.py
```

## Config (top of `Main.py`)

| Setting | Default | Purpose |
|---|---|---|
| `VIDEO_SOURCE` | `"s4.mp4"` | Video file, or `0` for webcam |
| `MIN_CONTOUR_AREA` | `800` | Ignore smaller motion blobs |
| `ALERT_COOLDOWN` | `3` | Seconds between alerts |
| `LOG_FILE` | `"alerts.log"` | Alert log path |
| `LOW_MOTION` / `MEDIUM_MOTION` | `15000` / `40000` | Risk area thresholds |
| `HIGH_MOTION_SPIKE` / `HIGH_DENSITY` | `25000` / `0.10` | Panic-spike thresholds |
| `TEMPORAL_FRAMES` | `5` | Abnormal frames required |

Needs a `s4.mp4` (or similar crowd clip) next to the script, or point
`VIDEO_SOURCE` at a webcam. Tested flow is desktop Python + OpenCV window —
no web UI, no packaging.
