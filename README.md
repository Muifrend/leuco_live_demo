# Leuco Live Demo

Host-native Jetson demo app for the Leuco live drowning-risk proof of concept.

This phase intentionally excludes physical camera placement and real pool polygon
calibration. The only pool gate modes are `disabled` and `full_frame_stub`.

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

RTSP smoke mode, before polygon calibration:

```bash
.venv/bin/python -m leuco_live_demo --source rtsp --backend motion_box --pool-gate disabled
```

The RTSP source reads Amcrest credentials from
`/mnt/ssd/projects/drowning_detection/.amcrest`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests use mock frames and do not send network notifications.
