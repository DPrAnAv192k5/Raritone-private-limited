import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
offset_x = 0
offset_y = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = pose.process(rgb)
    h,w,_ = frame.shape
    cv2.putText(
        frame,
        "M1 -> M2 -> M3 -> M4",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ls = lm[11]
        rs = lm[12]
        lh = lm[23]
        rh = lm[24]
        lsx,lsy = int(ls.x*w),int(ls.y*h)
        rsx,rsy = int(rs.x*w),int(rs.y*h)
        lhx,lhy = int(lh.x*w),int(lh.y*h)
        rhx,rhy = int(rh.x*w),int(rh.y*h)
        shirt_w = abs(rsx-lsx)+120
        shirt_h = abs(lhy-lsy)+120
        x = min(lsx,rsx)-60+offset_x
        y = min(lsy,rsy)+offset_y
        x2 = x+shirt_w
        y2 = y+shirt_h
        cv2.rectangle(
            frame,
            (x,y),
            (x2,y2),
            (255,0,0),
            -1
        )
        cv2.rectangle(
            frame,
            (x-40,y+20),
            (x,y+80),
            (255,0,0),
            -1
        )
        cv2.rectangle(
            frame,
            (x2,y+20),
            (x2+40,y+80),
            (255,0,0),
            -1
        )
        cv2.circle(
            frame,
            (
                (x+x2)//2,
                y+20
            ),
            25,
            (0,0,0),
            -1
        )
        cv2.putText(
            frame,
            "SHIRT",
            (
                x+40,
                y+shirt_h//2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            3
        )
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    cv2.putText(
        frame,
        "WASD move shirt | ESC exit",
        (20,h-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )
    cv2.imshow(
        "Virtual Try-On Demo",
        frame
    )
    key=cv2.waitKey(1)
    if key==ord('a'):
        offset_x-=10
    elif key==ord('d'):
        offset_x+=10
    elif key==ord('w'):
        offset_y-=10
    elif key==ord('s'):
        offset_y+=10
    elif key==27:
        break
cap.release()
cv2.destroyAllWindows()