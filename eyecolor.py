import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices (left and right iris/eye)
LEFT_EYE_IDX = [33, 133, 160, 158, 144, 153, 154, 155, 246]  # approximate left eye contour points
RIGHT_EYE_IDX = [362, 263, 387, 385, 373, 380, 381, 382, 466]  # approximate right eye contour points

def get_eye_region(image, landmarks, eye_indices):
    h, w, _ = image.shape
    points = []
    for idx in eye_indices:
        lm = landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))
    # Create mask for eye region
    mask = np.zeros((h, w), dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 255)
    # Extract eye region pixels
    eye_region = cv2.bitwise_and(image, image, mask=mask)
    return eye_region, points_array

def detect_eye_color(eye_region, points_array):
    # Crop bounding rect for eye
    x, y, w, h = cv2.boundingRect(points_array)
    cropped_eye = eye_region[y:y+h, x:x+w]
    if cropped_eye.size == 0:
        return "Unknown"

    # Convert to HSV
    hsv_eye = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2HSV)

    # Compute mean HSV ignoring black outside region
    eye_pixels = hsv_eye[np.where((hsv_eye != [0, 0, 0]).all(axis=2))]
    if eye_pixels.size == 0:
        return "Unknown"
    mean_hue = np.mean(eye_pixels[:,0])
    mean_sat = np.mean(eye_pixels[:,1])
    mean_val = np.mean(eye_pixels[:,2])

    # Simple color classification based on hue value (can tune)
    if mean_sat < 30 and mean_val > 180:
        return "Light Grey"
    if 0 <= mean_hue <= 20:
        return "Brown"
    elif 20 < mean_hue <= 35:
        return "Amber"
    elif 35 < mean_hue <= 85:
        return "Green"
    elif 85 < mean_hue <= 130:
        return "Blue"
    else:
        return "Other"

# Demo on webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Left eye
            left_eye_region, left_eye_points = get_eye_region(frame, face_landmarks, LEFT_EYE_IDX)
            left_eye_color = detect_eye_color(left_eye_region, left_eye_points)

            # Right eye
            right_eye_region, right_eye_points = get_eye_region(frame, face_landmarks, RIGHT_EYE_IDX)
            right_eye_color = detect_eye_color(right_eye_region, right_eye_points)

            label = f"Left Eye: {left_eye_color}, Right Eye: {right_eye_color}"

            # Put text near eyes
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Eye Color Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
