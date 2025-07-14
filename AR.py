import cv2
import mediapipe as mp
import trimesh
import numpy as np

# Load 3D mask model (mask.obj must be in filters/ directory)
mesh = trimesh.load('filters/mask.obj')
vertices = np.array(mesh.vertices)
faces = np.array(mesh.faces)

# Camera intrinsics (assuming fx = fy = width)
def get_camera_matrix(width, height):
    focal_length = width
    center = (width / 2, height / 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

# Landmark indices used for estimating pose
LANDMARK_IDS = [1, 33, 263, 61, 291, 152]  # Nose, Eyes, Mouth corners, Chin

# Dummy 3D model points corresponding to those landmark IDs (adjust if needed)
model_points = np.array([
    [0.0, 0.0, 0.0],      # Nose tip
    [-30.0, 0.0, -30.0],  # Right eye
    [30.0, 0.0, -30.0],   # Left eye
    [-20.0, -30.0, -30.0],# Mouth left
    [20.0, -30.0, -30.0], # Mouth right
    [0.0, -60.0, 0.0]     # Chin
], dtype=np.float64)

# MediaPipe face mesh and drawing tools
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

green_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

with mp_face.FaceMesh(refine_landmarks=True) as face_mesh, \
     mp_pose.Pose() as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face and pose detection
        face_results = face_mesh.process(rgb)
        pose_results = pose.process(rgb)

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Keep video unflipped

        # Draw face mesh and body skeleton in green
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face.FACEMESH_TESSELATION,
                landmark_drawing_spec=green_spec,
                connection_drawing_spec=green_spec
            )
            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face.FACEMESH_CONTOURS,
                landmark_drawing_spec=green_spec,
                connection_drawing_spec=green_spec
            )
            mp_drawing.draw_landmarks(
                frame, face_landmarks,
                mp_face.FACEMESH_IRISES,
                landmark_drawing_spec=green_spec,
                connection_drawing_spec=green_spec
            )

            # Get 2D image points for pose estimation
            image_points = []
            for idx in LANDMARK_IDS:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points.append([x, y])

            image_points = np.array(image_points, dtype=np.float64)
            camera_matrix = get_camera_matrix(w, h)
            dist_coeffs = np.zeros((4, 1))

            # Solve pose
            success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

            # Project mask vertices to image
            projected, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
            projected = projected.reshape(-1, 2).astype(np.int32)

            # Draw the 3D mask on face
            for face in faces:
                pts = projected[face]
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
                # Optionally: cv2.fillPoly(frame, [pts], color=(0, 255, 0)) for solid mask

        # Draw skeleton
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=green_spec,
                connection_drawing_spec=green_spec
            )

        cv2.imshow("AR Face Mesh + 3D Mask", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
