import cv2
import mediapipe as mp
import numpy as np

# ============================================
# Initialize MediaPipe Pose
# ============================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================
# Start Webcam
# ============================================

cap = cv2.VideoCapture(0)

print("Press Q to Quit")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.resize(frame, (800, 600))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        h, w, c = frame.shape

        # Nose Detection
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        cv2.circle(frame, (nose_x, nose_y), 10, (0, 0, 255), -1)

        # Shoulder Width
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        ls_x, ls_y = int(left_shoulder.x * w), int(left_shoulder.y * h)
        rs_x, rs_y = int(right_shoulder.x * w), int(right_shoulder.y * h)

        shoulder_distance = int(
            np.sqrt((rs_x - ls_x)**2 + (rs_y - ls_y)**2)
        )

        cv2.line(frame, (ls_x, ls_y), (rs_x, rs_y), (255,255,0), 3)

        cv2.putText(
            frame,
            f'Shoulder Width: {shoulder_distance}px',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    cv2.putText(
        frame,
        "Improved Body Landmark Detection",
        (20, 580),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()