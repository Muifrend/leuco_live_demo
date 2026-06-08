# Leuco Live Demo

Host-native Jetson demo app for the Leuco live drowning-risk proof of concept.

This phase intentionally excludes physical camera placement finalization. Pool
gating supports `disabled`, `full_frame_stub`, and `polygon`.

## Run

```bash
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --alerts-disabled
```

Open:

```text
http://<jetson-ip>:8080/
```

Useful mock scenarios:

```bash
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate disabled --mock-scenario empty
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --mock-scenario normal
.venv/bin/python -m leuco_live_demo --source mock --backend mock --pool-gate full_frame_stub --mock-scenario alert
```

RTSP smoke mode, before polygon/risk calibration:

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

`--gstreamer-pipeline auto` tries H.264 forced TCP, H.264 default transport,
H.265 forced TCP, H.265 default transport, `decodebin` forced TCP, and
`decodebin` default transport. `--rtsp-backend auto` then tries OpenCV direct
RTSP open as a last fallback. Use `--rtsp-backend ffmpeg` only when you
explicitly want the FFmpeg backend.

`LEUCO_INFERENCE_ROI` accepts `x1,y1,x2,y2` and crops only the frame passed
to inference. If those points were picked from a screenshot, set
`LEUCO_INFERENCE_ROI_REFERENCE_SIZE` to that screenshot's `WIDTHxHEIGHT` so the
app scales the crop to the true RTSP frame size. The HTTP stream remains
full-frame, and the overlay draws the effective ROI rectangle.

`LEUCO_POOL_POLYGON` accepts `x,y;x,y;x,y` and defaults to the live-frame
pool-water mask from a `2960x1668` reference image. With
`--pool-gate polygon`, the app tests the tracked person's lower-center
bounding-box point against the scaled polygon. Risk stays inactive when the
person is outside the polygon.

The alert gate has three paths. A notification is sent immediately when at
least `LEUCO_HIGH_RISK_FRAMES` of the last 48 AI frames are high-risk; the
default is 15. Pose-risk accumulation can also start a distress candidate after
`LEUCO_POSE_RISK_FRAMES` high-risk frames while the current frame is high-risk.
Lost-swimmer accumulation starts a candidate after `LEUCO_LOST_SWIMMER_FRAMES`
frames without a fresh person or YOLO pose after the swimmer was last confirmed
inside the pool and no exit was confirmed. Candidate paths notify after
`LEUCO_DISTRESS_PERSIST_SECONDS`. A detected outside-pool person must remain
outside for `LEUCO_CONFIRMED_EXIT_SECONDS` before the in-pool memory is cleared.

The RTSP source reads Amcrest credentials from
`/mnt/ssd/projects/drowning_detection/.amcrest`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests use mock frames and do not send network notifications.
