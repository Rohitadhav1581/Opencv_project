from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run DeepFace analysis on the frame for multiple faces
    results = DeepFace.analyze(frame, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

    # results is always a list when multiple faces detected
    for result in results:
        x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
        age = result['age']
        gender = result['gender']
        skin_tone = result['dominant_race']
        emotion = result['dominant_emotion']

        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Compose label
        label = f"Age: {age}, Gender: {gender}, Skin: {skin_tone}, Emotion: {emotion}"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('Multi-face Attribute Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
