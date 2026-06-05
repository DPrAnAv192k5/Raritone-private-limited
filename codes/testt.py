import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
image_path = "data/test/image/000001_0.jpg"
image = cv2.imread(image_path)
# Check if image loaded
if image is None:
    print("Image not found!")
    exit()
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(rgb)
if results.pose_landmarks:
    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )
    print("Pose Detected Successfully")
else:
    print("Pose Detection Failed")
rgb_output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,10))
plt.imshow(rgb_output)
plt.title("Virtual Try-On Dataset Testing")
plt.axis("off")
plt.show()