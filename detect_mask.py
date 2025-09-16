import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("models/mask_detector.h5")
IMG_SIZE = 224

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0][0]

    label = "Mask" if pred < 0.5 else "No Mask"
    color = (0,255,0) if label == "Mask" else (0,0,255)

    cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
