import cv2

# Load pre-trained models for face, age and gender detection
face_proto = "deploy.prototxt.txt"
face_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

# Model mean values and list of age and gender classes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load networks
face_net = cv2.dnn.readNet(face_model, face_proto)
age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

def get_faces(net, frame, conf_threshold=0.7):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bboxes.append([x1, y1, x2, y2])
    return bboxes

cap = cv2.VideoCapture(0)

padding = 20

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = get_faces(face_net, frame)
    for bbox in faces:
        x1, y1, x2, y2 = bbox
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(frame.shape[1] - 1, x2 + padding)
        y2_p = min(frame.shape[0] - 1, y2 + padding)
        
        face = frame[y1_p:y2_p, x1_p:x2_p]

        # Gender prediction
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(frame, label, (x1_p, y1_p-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), (255, 0, 0), 2)

    cv2.imshow("Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
