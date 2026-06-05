# Leuco Live Demo Implementation Plan

## Local Repository Boundary

This proof-of-concept demo should live in its own repository:

```text
/mnt/ssd/projects/leuco_live_demo
```

The existing Leuco control-plane repository remains separate:

```text
/mnt/ssd/projects/drowning_detection
```

For local camera access, the demo can read the existing Amcrest environment file from:

```text
/mnt/ssd/projects/drowning_detection/.amcrest
```

Important existing project references:

| Purpose | Path |
| --- | --- |
| Camera credentials | `/mnt/ssd/projects/drowning_detection/.amcrest` |
| Current site config | `/mnt/ssd/projects/drowning_detection/config/sites/2550_van_ness.yaml` |
| Camera bring-up notes | `/mnt/ssd/projects/drowning_detection/docs/camera_bringup.md` |
| Software state reference | `/mnt/ssd/projects/drowning_detection/software.md` |

The demo repository should not write generated media, model artifacts, or runtime output back into the control-plane repository. Large outputs should continue to use SSD-backed paths under `/mnt/ssd/datasets/leuco` and `/mnt/ssd/models/leuco` when needed.

## Goal

Build a live proof-of-concept drowning-risk demo using the current Leuco hardware stack.

The demo should show a live pool camera stream where the system detects one person, only activates risk classification when that person is inside a rough pool polygon mask, identifies active drowning-like cues, confirms risk over a 6-second window, and sends an ntfy notification after sustained high-risk behavior.

This plan is for the demo implementation only. It is not the final production drowning-detection design.

---

## Verified Current Context

The following items have now been verified or decided for this demo:

| Area | Current status |
| --- | --- |
| Demo repository | `/mnt/ssd/projects/leuco_live_demo` is the standalone demo repo |
| Control-plane references | Referenced camera, site, bring-up, and software docs exist in `/mnt/ssd/projects/drowning_detection` |
| Camera secret interface | `/mnt/ssd/projects/drowning_detection/.amcrest` exists with `AMCREST_USER`, `AMCREST_PASS`, and `AMCREST_IP` |
| Camera access | RTSP access is treated as working based on the current Leuco bring-up state and user confirmation |
| Runtime device | Jetson Orin Nano with JetPack 6.2.2 / L4T 36.5.0 |
| Native media stack | FFmpeg, GStreamer, OpenCV, and TensorRT are available on the Jetson per the software reference |
| Python environment | Demo `.venv` exists with `--system-site-packages` and reuses Jetson-native packages |
| Model environment | PyTorch, torchvision, Ultralytics, OpenCV import path, and CUDA availability are verified |
| ntfy | Environment config has been added and a test notification to `https://ntfy.sh/leuco` worked on 2026-06-05 |
| Demo output | Use HTTP for the live annotated demo view |
| HTTP port | Port `8080` works and is the locked demo port |

The camera still needs to be physically set up or positioned for the final demo view. The exact pool polygon should be calibrated after that camera view is available. Implementation can start before those are ready by using a mock or test-video source and a temporary full-frame or unset polygon mode.

Verified model/runtime package details:

| Package or capability | Verified state |
| --- | --- |
| `torch` | `2.8.0` from the JetPack 6.2 / CUDA 12.6 Jetson index |
| `torchvision` | `0.23.0` |
| `ultralytics` | `8.4.60`, installed without replacing Jetson-native core packages |
| `cv2` | `4.8.0` from `/usr/lib/python3.10/dist-packages/cv2/__init__.py` |
| CUDA | `torch.cuda.is_available() == True`, device Orin, CUDA 12.6 |

`pip check` may report that Ultralytics requires `opencv-python`. That warning is intentional for this demo environment because the `.venv` uses Jetson's native `cv2` instead of the PyPI OpenCV wheel. Do not install `opencv-python` just to clear that warning.

---

## Final Demo Stack

The implementation should follow this pipeline:

1. Amcrest RTSP live camera stream
2. Jetson host-native video ingest
3. Rough pool polygon mask
4. Person or pose detection
5. One-person tracking
6. Active drowning cue scoring
7. 6-second temporal confirmation
8. ntfy notification
9. HTTP live annotated stream for demonstration

The live demo should prioritize stability, clarity, and low latency over perfect model accuracy. The first implementation should be a host-native Python app on the Jetson using GStreamer/OpenCV for ingest and an HTTP MJPEG stream for viewing.

---

## Locked Demo Decisions

| Area               | Decision                                      |
| ------------------ | --------------------------------------------- |
| Camera input       | Use the existing Amcrest RTSP stream          |
| Camera frame rate  | Leave camera stream unchanged                 |
| Runtime            | Host-native Python on the Jetson              |
| Live output        | HTTP annotated stream                         |
| HTTP port          | 8080                                          |
| AI processing rate | Process approximately 8 AI frames per second  |
| Decision window    | 6 seconds                                     |
| Window size        | 48 processed AI frames                        |
| Alert threshold    | 34 high-risk frames out of 48                 |
| Current-frame rule | Current frame must also be high-risk          |
| Pool gating        | Use a rough polygon mask                      |
| People in demo     | One person                                    |
| Alert system       | ntfy                                          |
| ntfy demo endpoint | `https://ntfy.sh/leuco` via environment config |
| Alert cooldown     | 60 seconds                                    |
| Initial model path | Lightweight YOLO pose if stable               |
| Model fallback     | Person detection plus box/region motion       |
| DeepStream         | Not included in first demo implementation     |
| Docker             | Not used for first-line camera/demo runtime   |
| Cloud escalation   | Not included for this demo                    |
| Recording          | Outside the scope of this implementation plan |

---

## Core Detection Principle

The system should not treat “splashing” alone as drowning.

The most reliable demo concept is:

**A person inside the pool mask showing high upper-body or arm activity while making little or no forward progress.**

This is the core active-drowning-like pattern the demo should detect.

Normal swimming usually includes motion plus forward displacement. Active drowning-like behavior is better represented as repeated effort with poor travel across the pool.

---

## Pool Mask Gate

Risk classification should only begin after the detected person is inside the pool polygon.

The system should have three basic location states:

| State               | Meaning                                                         |
| ------------------- | --------------------------------------------------------------- |
| No person           | No person currently detected                                    |
| Person outside pool | Person detected, but outside pool polygon                       |
| Person inside pool  | Person detected inside pool polygon, risk classification active |

The rough polygon mask does not need to be perfect for the demo. It only needs to clearly cover the visible water area and exclude most of the surrounding deck.

The mask should be visible or outlined in the live overlay so the viewer can understand why classification is active or inactive.

The exact polygon points are not locked yet because they depend on the final mounted camera view. Once the camera is positioned, capture a current frame and store the image-space polygon points in demo-local configuration.

---

## One-Person Tracking

The demo should assume one person.

If multiple people appear, the system should still choose one primary tracked person, but the demo should avoid that case.

The tracked person should be maintained across frames so the system can measure:

* whether the person remains inside the pool
* how much the person moves across the pool
* how much the upper body or arms move
* whether high-risk behavior persists over time

The tracking does not need to be production-grade. It only needs to remain stable for the single-person staged demo.

---

## Active Drowning Cue Stack

The frame-level risk decision should be based on a small set of visual cues.

### Primary Cues

These are the most important cues:

| Cue                             | Meaning                                                                      |
| ------------------------------- | ---------------------------------------------------------------------------- |
| High upper-body or arm activity | Repeated movement around the shoulders, elbows, wrists, head, or upper torso |
| Low forward progress            | The person’s center position does not move meaningfully across the pool      |
| Person remains inside pool mask | The person is actually in the water area, not on the deck                    |

The strongest high-risk signal is:

**high upper-body activity plus low forward progress while inside the pool mask.**

### Supporting Cues

These can increase confidence:

| Cue                             | Meaning                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| Vertical posture                | Person appears upright rather than horizontal like a normal swimmer   |
| Head or torso bobbing           | Head or upper body moves irregularly near the water surface           |
| Repeated localized water motion | Motion is concentrated near the person without pool-crossing movement |

### Weak Cue

Splashing should be treated only as a weak supporting cue.

Splashing can happen during normal swimming, play, jumping, or turning. It should not trigger high-risk by itself.

---

## Frame-Level Binary Decision

Each processed AI frame should be classified as one of two states:

* Normal
* High-risk

A frame should become high-risk only when the person is inside the pool mask and the active-drowning cue detector finds a strong cue combination.

Recommended high-risk frame rule:

| Condition                                  | Required?                                                |
| ------------------------------------------ | -------------------------------------------------------- |
| Person detected                            | Yes                                                      |
| Person inside pool polygon                 | Yes                                                      |
| One tracked person selected                | Yes                                                      |
| High upper-body or arm activity            | Yes                                                      |
| Low forward progress                       | Yes                                                      |
| Vertical posture or head/torso instability | Preferred, but not required for demo if pose is unstable |

In plain English:

**A frame is high-risk when the person is inside the pool, moving their upper body or arms repeatedly, and not making meaningful forward progress.**

---

## Temporal Alert Decision

The system should not send an alert from a single high-risk frame.

The demo should use a 6-second rolling window.

Recommended final setting:

| Parameter                        |               Value |
| -------------------------------- | ------------------: |
| AI processing rate               | 8 frames per second |
| Decision window                  |           6 seconds |
| Total processed frames in window |           48 frames |
| Required high-risk frames        |           34 frames |
| Required percentage              |           About 70% |
| Cooldown after alert             |          60 seconds |

Alert rule:

**Send an ntfy alert when at least 34 of the last 48 processed frames are high-risk and the current frame is also high-risk.**

This keeps the demo responsive while avoiding one-frame false triggers.

---

## Live Overlay Requirements

The live stream should be annotated clearly enough that a viewer can understand the system behavior without looking at logs.

Recommended overlay fields:

| Overlay Field       | Example State      |
| ------------------- | ------------------ |
| Person detected     | Yes / No           |
| In pool             | Yes / No           |
| Risk classification | Inactive / Active  |
| Risk state          | Normal / High-risk |
| High-risk frames    | 34 / 48 required   |
| Decision window     | 6 seconds          |
| AI processing rate  | 8 fps              |
| Alert               | Pending / Sent     |

The overlay should also show:

* bounding box around the person
* rough pool polygon outline or translucent mask
* tracked person identifier if available

Keep the overlay simple. The demo should emphasize the decision pipeline, not raw debugging data.

The HTTP app should expose:

* `/` for a minimal browser view with the live annotated stream
* `/stream.mjpg` for the MJPEG video stream
* `/status` for machine-readable current demo state

---

## Demo Behavior Flow

The demo should follow this behavioral sequence:

### 1. Empty Pool or No Person

Expected behavior:

* No person detected
* Risk classification inactive
* No alert

### 2. Person Outside Pool

Expected behavior:

* Person box appears
* Pool mask says person is outside pool
* Risk classification remains inactive
* No alert

### 3. Person Enters Pool Mask

Expected behavior:

* Person remains detected
* In-pool state becomes active
* Risk classification begins
* Risk state starts as normal

### 4. Normal Pool Activity

Expected behavior:

* Person is inside pool
* Person may move or swim normally
* Forward progress prevents high-risk classification
* High-risk frame count stays low

### 5. Active Drowning-Like Motion

Expected behavior:

* Person remains inside pool mask
* Upper-body or arm activity increases
* Forward progress remains low
* Frames begin counting as high-risk
* High-risk frame counter rises across the 6-second window

### 6. Alert Trigger

Expected behavior:

* High-risk frame count reaches 34 of 48
* Current frame is also high-risk
* ntfy notification sends once
* Overlay changes alert status to sent
* Cooldown prevents repeated alert spam

---

## Recommended Model Choice

The preferred model path is pose-based if stable on the Jetson.

### Preferred

Use a lightweight YOLO pose model, starting with `yolo11n-pose.pt` if the Jetson Python environment supports it cleanly.

Reason:

* Person boxes alone can show location.
* Pose keypoints make it easier to estimate arm activity, torso orientation, and head/upper-body movement.
* Pose gives a better foundation for active drowning cue detection.

Implementation guidance:

* Keep inference behind a small backend interface.
* Use this repo's demo `.venv`.
* Preserve the `.venv` with `--system-site-packages` so it can reuse Jetson host modules such as OpenCV and TensorRT.
* Keep the Jetson-compatible NVIDIA/PyTorch installation path already configured for this demo.
* Do not allow generic upstream `torch` or `opencv-python` wheels to replace the Jetson-compatible native stack.
* Do not force heavy ML dependencies directly into the demo repo.
* Leave room to swap to ONNX or TensorRT later without rewriting the demo logic.

### Acceptable Demo Fallback

If `yolo11n-pose.pt` is unstable, unavailable, or too slow, try `yolov8n-pose.pt`. If pose remains unstable or dependency setup becomes the bottleneck, use person detection plus motion features around the person box.

The fallback should still use:

* person box
* pool mask
* tracked center position
* upper-body region motion
* low displacement over time

The demo should favor whichever option produces the most stable live result.

---

## Runtime Architecture

The first implementation should be a single host-native app in this demo repository.

Recommended shape:

| Component | Recommendation |
| --- | --- |
| Config loading | Read demo-local `.env` plus `/mnt/ssd/projects/drowning_detection/.amcrest` |
| RTSP ingest | GStreamer/OpenCV, using Jetson hardware decode where available |
| Display output | HTTP MJPEG stream and simple browser page |
| Inference | Pluggable backend, mock/test backend for smoke tests, YOLO pose for real detections |
| Tracking/risk logic | Demo-local Python modules |
| Alerts | HTTP POST to ntfy |
| Runtime artifacts | Keep logs/cache local or under `/mnt/ssd`; do not write output into the control-plane repo |

Do not use Docker or DeepStream for the first demo runtime unless host-native ingest or model execution becomes a blocker. The existing Leuco software reference explicitly favors host-native camera bring-up and keeps DeepStream deferred.

The implementation should support building and smoke-testing before the final camera view and pool polygon are available. In that pre-camera mode, use a mock/test source and keep risk classification inactive unless a temporary polygon and test detections are explicitly configured.

---

## Tracking Choice

Use a simple tracker that keeps the person identity stable across frames.

The purpose of tracking is to support:

* low forward progress measurement
* repeated cue confirmation
* temporal stability
* one-person continuity across the decision window

For the demo, tracking only needs to handle one visible person.

---

## Performance Plan

The camera stream should remain at its existing verified frame rate.

The AI system should sample frames for inference rather than forcing the camera to output fewer frames.

Recommended performance design:

| Layer          | Recommendation                             |
| -------------- | ------------------------------------------ |
| Camera stream  | Leave unchanged                            |
| Display stream | HTTP MJPEG, keep smooth if possible        |
| AI inference   | Sample around 8 frames per second          |
| Decision logic | Based only on processed AI frames          |
| Alert timing   | Based on 6 seconds of processed AI history |

If performance is unstable, reduce AI processing from 8 fps to 6 fps and adjust the 6-second window to 36 processed frames with a 26-frame high-risk threshold.

---

## Implementation Order

Build the system in this order:

### Step 1: Live Stream Ingest

Build the HTTP annotated output first, using a mock/test source if the final camera is not set up yet. After the camera is positioned, confirm the Jetson can read the Amcrest RTSP stream through the same app.

Success condition:

* Live video or a test source is visible at `http://<jetson-ip>:8080/` with minimal delay.
* The same app can later switch to the real RTSP source by configuration.

### Step 2: Person Detection

Add person detection and draw a bounding box around the person.

Success condition:

* The person is consistently boxed in the live pool scene.

### Step 3: Rough Pool Polygon Mask

Add a rough polygon mask over the visible pool water.

Success condition:

* The system can distinguish person outside pool from person inside pool.

### Step 4: Classification Activation Gate

Only allow risk classification when the tracked person is inside the pool polygon.

Success condition:

* Risk state remains inactive when the person is on the deck.
* Risk state becomes active when the person is inside the pool mask.

### Step 5: One-Person Tracking

Track the primary person across processed frames.

Success condition:

* The system can measure whether the person is moving across the pool or staying in roughly the same area.

### Step 6: Active Drowning Cue Detection

Add cue scoring based on:

* high upper-body or arm activity
* low forward progress
* optional vertical posture
* optional head or torso bobbing

Success condition:

* Normal swimming-like movement stays mostly normal.
* Staged active distress-like movement becomes high-risk.

### Step 7: Rolling 6-Second Decision Window

Add the rolling frame window.

Success condition:

* The system counts high-risk frames over the last 6 seconds.
* A single bad frame does not trigger an alert.

### Step 8: ntfy Alert

Send ntfy only after the window-level threshold is met.

The manual ntfy path has already been verified with a test notification to `https://ntfy.sh/leuco`. The implementation should still read the ntfy URL/topic from environment config rather than hardcoding alert details deep in the code.

Success condition:

* Alert sends after sustained high-risk behavior.
* Alert does not send while the person is outside the pool.
* Cooldown prevents repeated alerts.

### Step 9: Final Demo Validation

Run the full staged flow and verify the overlay clearly shows:

* person detection
* pool-mask gating
* risk state transition
* high-risk frame counter
* ntfy alert sent

---

## Remaining Setup Before Implementation

These items do not block starting implementation, but they are still needed before the full live pool demo can be validated:

| Item | Needed action |
| --- | --- |
| Final camera view | Set up or position the Amcrest camera for the demo scene |
| Pool polygon | Capture a current frame and define the rough water-area polygon |

No additional product-level decisions are currently blocking the demo plan.

---

## Acceptance Criteria

The implementation is demo-ready when all of these are true:

| Requirement                 | Pass Condition                                             |
| --------------------------- | ---------------------------------------------------------- |
| Live video works            | Viewer can see the pool feed in real time                  |
| Person detection works      | Person receives a stable bounding box                      |
| Pool mask works             | Risk classification only activates inside pool polygon     |
| One-person tracking works   | The person remains tracked across the staged sequence      |
| Active cue scoring works    | Staged active drowning-like motion becomes high-risk       |
| Temporal confirmation works | Alert waits for the 6-second threshold                     |
| ntfy works                  | Notification arrives after sustained high-risk behavior    |
| Cooldown works              | System does not spam repeated notifications                |
| Overlay is understandable   | Viewer can follow the system decision from the video alone |

---

## Final Demo Configuration

Use this as the final locked configuration:

| Setting                     |                                                   Value |
| --------------------------- | ------------------------------------------------------: |
| Runtime                     |                           Host-native Python on Jetson |
| Live output                 |                                   HTTP annotated stream |
| HTTP port                   |                                                    8080 |
| AI processing rate          |                                                   8 fps |
| Decision window             |                                               6 seconds |
| Processed frames per window |                                                      48 |
| High-risk alert threshold   |                                               34 frames |
| Required proportion         |                                               About 70% |
| Alert cooldown              |                                              60 seconds |
| People                      |                                              One person |
| Pool mask                   |                                           Rough polygon |
| Primary cue                 | High upper-body or arm motion plus low forward progress |
| Supporting cues             |                    Vertical posture, head/torso bobbing |
| Weak cue                    |                   Splashing only as supporting evidence |
| Alert channel               |                                                    ntfy |
| Demo ntfy topic             |                                                   leuco |
| Initial model preference    |                                      Lightweight YOLO pose |
| Initial build backend       |                                      Mock/test backend first |
| Initial deployment style    |                             Host-native, not Docker/DeepStream |
| Cloud escalation            |                                           None for demo |

---

## Summary

The final demo should prove this chain:

**Live camera feed → HTTP annotated view → person detected → person enters pool mask → active drowning-like cues detected → sustained risk confirmed over 6 seconds → ntfy alert sent.**

The most important technical idea is not simple splashing detection. The core signal should be:

**repeated upper-body or arm activity with little or no forward progress while inside the pool.**

That gives the demo the strongest alignment with active drowning behavior and with sequence-based drowning-detection research.
