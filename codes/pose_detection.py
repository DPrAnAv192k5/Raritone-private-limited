import cv2
import mediapipe as mp
mp_pose=mp.solutions.pose
pose=mp_pose.Pose()
mp_draw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    if not success:
        break
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        cv2.putText(
            img,
            "Skeleton Tracking Active",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )
    cv2.imshow("Human Pose Detection",img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()