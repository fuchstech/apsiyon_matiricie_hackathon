import os
import cv2
import numpy as np
import easyocr
import util

# Define constants
model_cfg_path = r"model\cfg\darknet-yolov3.cfg"
model_weights_path = r"weights\model.weights"
class_names_path = r"model\class.names"
input_video_path = r"files\plate.mp4"

# Load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

# Load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get frame width, height, and FPS from the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initialize OCR reader
reader = easyocr.Reader(['en'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    # Convert frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), True)

    # Get detections
    net.setInput(blob)
    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # Apply NMS
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # Process each detected license plate
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox
        license_plate = frame[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()
        frame = cv2.rectangle(frame,
                              (int(xc - (w / 2)), int(yc - (h / 2))),
                              (int(xc + (w / 2)), int(yc + (h / 2))),
                              (0, 255, 0),
                              2)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        output = reader.readtext(license_plate_thresh)

        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(text, text_score)

    # Write the processed frame to the output video
    #out.write(frame)

    # Display the frame (optional)
    cv2.imshow('Apsiyon|Matiricie Cam #2', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
#out.release()
cv2.destroyAllWindows()

