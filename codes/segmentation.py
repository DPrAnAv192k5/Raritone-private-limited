import cv2
import mediapipe as mp
import numpy as np
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(
    model_selection=1
)
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    result = segmenter.process(rgb)
    mask = result.segmentation_mask
    condition = mask > 0.5
    bg = np.ones_like(frame) * 255
    segmented = np.where(
        condition[:, :, None],
        frame,
        bg
    )
    cv2.imshow(
        "Original Webcam",
        frame
    )
    cv2.imshow(
        "Segmentation Mask",
        mask
    )
    cv2.imshow(
        "Clothing Segmentation Prototype",
        segmented
    )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()