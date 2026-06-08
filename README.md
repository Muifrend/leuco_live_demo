# Leuco Live Demo

## Manifesto

Drowning is the leading cause of death for children ages 1-4 in the United States. Not cars. Not disease. Drowning. In CPSC-reported pool and spa fatalities involving children younger than 15, 74 percent happened in residential settings; for children younger than 5, it was 86 percent. The problem this demo is aimed at is not the public pool with lifeguards and commercial safety infrastructure. It is the backyard, the side yard, the private pool, and the few seconds when an adult looks away.

AI-assisted drowning detection exists. Lynxight, Poseidon, and other commercial systems have shown that computer vision can help aquatic safety teams. But those systems are built for hotels, aquatic centers, waterparks, and municipalities. They assume commercial budgets, managed infrastructure, and deployment environments that do not fit a family home.

Home safety has a different constraint: the video cannot leave the home. Parents will not accept a product that continuously streams video of their children swimming in their backyard to a cloud service. We would not either.

Leuco is an on-device home-pool drowning-risk proof of concept. The current demo runs on a Jetson, reads a local Amcrest RTSP camera, performs local pose or motion inference, gates risk to a calibrated pool polygon, tracks one swimmer, and scores temporal distress cues on the device. The live annotated stream is served locally over HTTP. Alerts are optional ntfy notifications. The model, tracker, risk engine, overlay, and alert decision all run locally.

No remote inference server. No AI subscription required. No continuous cloud video stream. The camera feed stays on the local network.

There are millions of residential pools in America. Most have no AI-assisted drowning-risk layer at all.

Imagine how many more lives could be saved.

This demo is not a replacement for adult supervision, barriers, pool covers, alarms, CPR training, swimming lessons, or safe water practices. It is a proof of concept for the architecture a home product needs: local video in, local inference, local decision, immediate alert.

Sources for the safety context: [CDC drowning facts](https://www.cdc.gov/drowning/data-research/facts/index.html) and the [CPSC 2025 pool/spa submersion report](https://www.cpsc.gov/s3fs-public/Pool-or-Spa-Submersion-Estimated-Nonfatal-Drowning-Injuries-and-Reported-Drownings-2025-Report.pdf).

## Current Demo

This repository contains a host-native Jetson Python demo for live, local drowning-risk detection. It is built to run against an Amcrest pool camera, but it also includes mock and video-file modes for repeatable development.

The current pipeline is:

1. Video source: mock scene, video file, or Amcrest RTSP.
2. Inference: mock, YOLO pose, or frame-difference motion box fallback.
3. Optional inference ROI crop for the high/wide camera view.
4. One-person tracking.
5. Pool gate: disabled, full-frame stub, or calibrated polygon.
6. Local risk scoring from upper-body activity, low forward progress, and lost-swimmer state.
7. Temporal confirmation over a 6-second AI window.
8. Optional ntfy alert.
9. Local HTTP dashboard with MJPEG stream and JSON status.

This phase intentionally does not finalize physical camera placement. Pool gating supports `disabled`, `full_frame_stub`, and `polygon`; the default polygon is calibrated from a `2960x1668` live-frame reference and can be overridden locally.

## Run

Mock smoke test:

```bash
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --alerts-disabled
```

Open the local demo page:

```text
http://<jetson-ip>:8080/
```

Useful mock scenarios:

```bash
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate disabled --mock-scenario empty
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --mock-scenario normal
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --mock-scenario alert
```

RTSP smoke mode before polygon/risk calibration:

```bash
.venv/bin/python -m leuco_live_demo --source rtsp --backend motion_box --pool-gate disabled
```

RTSP YOLO pose mode with the current tightened high/wide camera ROI and pool polygon gate:

```bash
.venv/bin/python -m leuco_live_demo --source rtsp --rtsp-backend gstreamer --gstreamer-pipeline auto --backend yolo_pose --pool-gate polygon --inference-roi 360,285,1080,670 --inference-roi-reference-size 1554x882
```

Use debug logs while tuning camera/model startup:

```bash
.venv/bin/python -m leuco_live_demo --source rtsp --rtsp-backend gstreamer --gstreamer-pipeline auto --backend yolo_pose --pool-gate polygon --inference-roi 360,285,1080,670 --inference-roi-reference-size 1554x882 --log-level DEBUG
```

If the camera main stream is H.265, force the H.265 GStreamer pipeline:

```bash
.venv/bin/python -m leuco_live_demo --source rtsp --rtsp-backend gstreamer --gstreamer-pipeline h265 --backend yolo_pose --pool-gate polygon --inference-roi 360,285,1080,670 --inference-roi-reference-size 1554x882 --log-level DEBUG
```

## Configuration

Configuration is read in this order, with later values taking precedence:

1. Built-in defaults.
2. `/mnt/ssd/projects/drowning_detection/.amcrest` for camera credentials.
3. Local `.env`.
4. Process environment.
5. CLI flags.

Useful values are documented in [.env.example](/mnt/ssd/projects/leuco_live_demo/.env.example).

Important settings:

| Setting | Purpose |
| --- | --- |
| `LEUCO_SOURCE` | `mock`, `video`, or `rtsp` |
| `LEUCO_INFERENCE_BACKEND` | `mock`, `yolo_pose`, or `motion_box` |
| `LEUCO_MODEL_PATH` | YOLO pose weights, default `/mnt/ssd/models/leuco/yolo11n-pose.pt` |
| `LEUCO_POOL_GATE` | `disabled`, `full_frame_stub`, or `polygon` |
| `LEUCO_POOL_POLYGON` | Pool-water polygon as `x,y;x,y;x,y` |
| `LEUCO_POOL_POLYGON_REFERENCE_SIZE` | Reference image size for polygon scaling |
| `LEUCO_INFERENCE_ROI` | Optional `x1,y1,x2,y2` crop used only for inference |
| `LEUCO_INFERENCE_ROI_REFERENCE_SIZE` | Reference image size for ROI scaling |
| `LEUCO_AI_FPS` | AI processing rate, default `8` |
| `LEUCO_DECISION_WINDOW_SECONDS` | Risk window duration, default `6` |
| `LEUCO_HIGH_RISK_FRAMES` | Rolling alert threshold, default `15` out of `48` AI frames |
| `LEUCO_POSE_RISK_FRAMES` | Pose-risk accumulation threshold |
| `LEUCO_LOST_SWIMMER_FRAMES` | Lost-swimmer candidate threshold |
| `LEUCO_ALERTS_ENABLED` | `0` for dry run, `1` to send ntfy notifications |
| `LEUCO_NTFY_URL` | ntfy endpoint |

`--gstreamer-pipeline auto` tries H.264 forced TCP, H.264 default transport, H.265 forced TCP, H.265 default transport, `decodebin` forced TCP, and `decodebin` default transport. `--rtsp-backend auto` then tries OpenCV direct RTSP as a final fallback. Use `--rtsp-backend ffmpeg` only when you explicitly want the FFmpeg backend.

`LEUCO_INFERENCE_ROI` crops only the frame passed to inference. The HTTP stream remains full-frame, and the overlay draws the effective ROI rectangle. If ROI points were picked from a screenshot, set `LEUCO_INFERENCE_ROI_REFERENCE_SIZE` to that screenshot's `WIDTHxHEIGHT` so the app scales the crop to the true RTSP frame size.

With `--pool-gate polygon`, the app tests the tracked person's lower-center bounding-box point against the scaled pool polygon. Risk stays inactive while the person is outside the polygon.

## Risk and Alerts

The demo does not treat splashing alone as drowning. The current risk engine looks for a person inside the pool mask showing high upper-body or arm activity while making little or no forward progress. It also has a lost-swimmer path after a reliable in-pool track disappears or loses enough pose signal.

An alert decision can happen through three paths:

1. At least `LEUCO_HIGH_RISK_FRAMES` of the last 48 AI frames are high-risk.
2. Pose-risk accumulation persists after `LEUCO_POSE_RISK_FRAMES`.
3. Lost-swimmer accumulation persists after `LEUCO_LOST_SWIMMER_FRAMES`.

Candidate paths notify after `LEUCO_DISTRESS_PERSIST_SECONDS`. A detected outside-pool person must remain outside for `LEUCO_CONFIRMED_EXIT_SECONDS` before the in-pool memory is cleared. Alerts are rate-limited by `LEUCO_ALERT_COOLDOWN_SECONDS`.

When alerts are disabled, the system logs a dry-run alert and exposes that state in the overlay/status endpoint without sending a network notification.

## HTTP Endpoints

| Endpoint | Purpose |
| --- | --- |
| `/` | Local dashboard page |
| `/stream.mjpg` | Annotated MJPEG stream |
| `/status` | Current runtime status as JSON |

The HTTP stream is for local demo viewing. It is not a cloud relay.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests use mock frames and do not send network notifications.
