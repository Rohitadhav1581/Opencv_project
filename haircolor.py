import cv2
import numpy as np

def detect_skin_tone(face_img):
    # Convert face region to HSV color space
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

    # Sample region around center upper face (forehead area approx)
    h, w = hsv.shape[:2]
    sample_region = hsv[h//8:h//4, w//3:2*w//3]

    # Calculate mean HSV values in sample region
    mean_hue = np.mean(sample_region[:, :, 0])
    mean_saturation = np.mean(sample_region[:, :, 1])
    mean_value = np.mean(sample_region[:, :, 2])

    # Simple classification by brightness (value) and saturation could be refined
    if mean_value < 60:
        return "Dark"
    elif mean_value < 130:
        return "Medium"
    else:
        return "Light"

def detect_hair_color(frame, face_bbox):
    x, y, w, h = face_bbox
    # Sample region above face bounding box for hair
    hair_y_start = max(y - h // 2, 0)
    hair_region = frame[hair_y_start:y, x:x+w]

    if hair_region.size == 0:
        return "Unknown"

    # Convert hair region to HSV color space
    hsv = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)

    # Calculate mean HSV values
    mean_hue = np.mean(hsv[:, :, 0])
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])

    # Basic thresholding for hair color (can be improved with clustering)
    if (mean_value < 50 and mean_saturation > 50):
        return "Black"
    elif (mean_hue > 15 and mean_hue < 35):
        return "Brown"
    elif (mean_hue >= 0 and mean_hue <= 15):
        return "Red"
    elif (mean_saturation < 50 and mean_value > 120):
        return "Blonde"
    else:
        return "Other"

# Demo usage with webcam feed and OpenCV face detection:
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        skin_tone = detect_skin_tone(face_img)
        hair_color = detect_hair_color(frame, (x, y, w, h))

        label = f"Skin: {skin_tone}, Hair: {hair_color}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Hair and Skin Tone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
