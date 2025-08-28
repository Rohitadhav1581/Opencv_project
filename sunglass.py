import cv2
import numpy as np
import mediapipe as mp

# Load sunglasses image with transparency
sunglasses = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)  # ensure 4 channels (RGBA)

mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Left and right eye keypoints for fitting sunglasses
LEFT_EYE = [33, 133]
RIGHT_EYE = [263, 362]

def overlay_sunglasses(frame, face_landmarks):
    h, w = frame.shape[:2]
    # Get eye coordinates
    left_eye_pts = []
    right_eye_pts = []
    for idx in LEFT_EYE:
        lm = face_landmarks.landmark[idx]
        left_eye_pts.append((int(lm.x * w), int(lm.y * h)))
    for idx in RIGHT_EYE:
        lm = face_landmarks.landmark[idx]
        right_eye_pts.append((int(lm.x * w), int(lm.y * h)))
    # Center between eyes
    x1 = np.mean([pt[0] for pt in left_eye_pts])
    y1 = np.mean([pt[1] for pt in left_eye_pts])
    x2 = np.mean([pt[0] for pt in right_eye_pts])
    y2 = np.mean([pt[1] for pt in right_eye_pts])
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # Estimate size between eyes for scaling sunglasses
    eye_width = int(abs(x2 - x1)) * 2
    s_h, s_w = sunglasses.shape[:2]
    scale = eye_width / s_w
    new_w = int(s_w * scale)
    new_h = int(s_h * scale)
    resized_sunglasses = cv2.resize(sunglasses, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Position for overlay: place sunglasses with center between eyes
    x_offset = center_x - new_w // 2
    y_offset = center_y - new_h // 2

    # Overlay sunglasses with transparency mask
    for c in range(0, 3):  # BGR
        frame_slice = frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
        sung_slice = resized_sunglasses[:,:,c]
        alpha = resized_sunglasses[:,:,3] / 255.0
        frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = frame_slice * (1-alpha) + sung_slice * alpha
    return frame

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, \
     mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Facial landmarks
        face_results = face_mesh.process(frame_rgb)
        # Body pose
        pose_results = pose.process(frame_rgb)

        # Draw body skeleton
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS)

        # Draw facemesh and overlay sunglasses
        if face_results.multi_face_landmarks:
            for fl in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    fl,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None)
                frame = overlay_sunglasses(frame, fl)

        cv2.imshow("Sunglasses Overlay, Facemesh, and Skeleton", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
