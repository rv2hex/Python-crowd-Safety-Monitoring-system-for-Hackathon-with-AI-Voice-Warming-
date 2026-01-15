# This is the raw version of the code

import cv2
import numpy as np
import time

# ===== OPTIONAL (Windows only sound alert) =====
try:
    import winsound
    SOUND_ENABLED = True
except:
    SOUND_ENABLED = False

# ================= SETTINGS =================
VIDEO_SOURCE = "s4.mp4"          # 0 = webcam OR "crowd.mp4"
MIN_CONTOUR_AREA = 800
ALERT_COOLDOWN = 3        # seconds
LOG_FILE = "alerts.log"

# Risk thresholds (tunable)
LOW_MOTION = 15000
MEDIUM_MOTION = 40000
HIGH_MOTION_SPIKE = 25000
HIGH_DENSITY = 0.10       # normalized motion density
TEMPORAL_FRAMES = 5       # abnormal frames required
# ============================================

# Initialize video
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Video source not accessible")
    exit()

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prev_motion = 0
last_alert_time = 0
abnormal_frames = 0

# FPS calculation
start_time = time.time()
frame_count = 0
fps = 0

print("🟢 Crowd Safety Monitoring Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # ===== FPS update =====
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    # ===== Frame Processing =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    total_motion_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            total_motion_area += area

    # ===== Density Estimation =====
    frame_area = frame.shape[0] * frame.shape[1]
    density = total_motion_area / frame_area

    # ===== Motion Acceleration =====
    motion_delta = abs(total_motion_area - prev_motion)

    # ===== Risk Classification =====
    risk = "LOW"
    reason = "Normal crowd behavior"

    if motion_delta > HIGH_MOTION_SPIKE and density > HIGH_DENSITY:
        risk = "HIGH"
        reason = "Rapid motion change with dense crowd - possible panic or stampede"
    elif total_motion_area > MEDIUM_MOTION:
        risk = "MEDIUM"
        reason = "Sustained high group movement - potentially unsafe situation"
    elif total_motion_area > LOW_MOTION:
        risk = "LOW"
        reason = "Minor irregular crowd movement"

    # ===== Temporal Consistency =====
    if risk in ["HIGH", "MEDIUM"]:
        abnormal_frames += 1
    else:
        abnormal_frames = 0

    # ===== Alert Generation =====
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")

    if (
        abnormal_frames >= TEMPORAL_FRAMES and
        risk != "LOW" and
        current_time - last_alert_time > ALERT_COOLDOWN
    ):
        alert_msg = (
            f"[{timestamp}] RISK: {risk} | "
            f"MotionArea: {int(total_motion_area)} | "
            f"Density: {density:.2f} | "
            f"Reason: {reason}"
        )

        print(alert_msg)

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(alert_msg + "\n")

        if risk == "HIGH" and SOUND_ENABLED:
            winsound.Beep(1200, 700)

        last_alert_time = current_time
        abnormal_frames = 0

    # ===== Display =====
    color = (0, 255, 0)
    if risk == "MEDIUM":
        color = (0, 255, 255)
    elif risk == "HIGH":
        color = (0, 0, 255)

    cv2.putText(frame, f"RISK: {risk}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)

    cv2.imshow("Crowd Safety Monitor", frame)

    prev_gray = gray.copy()
    prev_motion = total_motion_area

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# ===== Cleanup =====
cap.release()
cv2.destroyAllWindows()

print("🔴 Monitoring Stopped")
print(f"📊 Average FPS: {fps:.2f}")
