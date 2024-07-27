import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', device='cpu')

video_path = 'files\otopark-short.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video dosyası açılamıyor.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Video karesi okunamadı.")
        break

    height, width, _ = frame.shape

    start_point = (178, 150)
    end_point = (270, frame.shape[0])
    color = (0, 10, 250)
    thickness = 2

    cv2.line(frame, start_point, end_point, color, thickness)

    interval = 35
    areas = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    results = model(frame)
    car_number = 0
    for i, y in enumerate(range(start_point[1]+interval, end_point[1], interval)):
        cv2.line(frame, (start_point[0]-28, y), (end_point[0], y-40), (0, 0, 250), 2)
        areas.append((start_point[0]-28, y-interval, end_point[0], y))
        cv2.putText(frame, letters[i], (start_point[0] - 58, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        car_areas = {letter: 0 for letter in letters[:len(areas)]}

    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in ['car', 'truck', 'bus']:
            x1, y1, x2, y2 = map(int, box)
            car_number += 1

            for j, (ax1, ay1, ax2, ay2) in enumerate(areas):
                if ax1 < (x1 + x2) // 2 < ax2 and ay1 < (y1 + y2) // 2 < ay2:
                    car_areas[letters[j]] += 1

    cv2.putText(frame, f"Toplam Arac Sayisi: {car_number}", (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 2)

    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in ['car', 'truck', 'bus']:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for j, (ax1, ay1, ax2, ay2) in enumerate(areas):
        cv2.putText(frame, f"{letters[j]}: {car_areas[letters[j]]} arac", ((ax1 - 130, ay1 - 130)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 2)

    cv2.imshow('Apsiyon|Matiricie CAM #1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
