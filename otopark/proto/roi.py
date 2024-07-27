import cv2
import torch
import numpy as np
from time import sleep

# YOLOv5 modelini yükleyelim
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# Görüntüyü yükleyelim
for image_num in range(30,180,15):
    image_num = str(image_num)
    if len(image_num) == 2:
        image_num = '0' + image_num
    image_path = f'files/photo/ezgif-frame-{image_num}.jpg'  
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # ROI bölgesini belirleyelim (x, y, w, h)
    roi = (350, 100, 500, 200)  # Örnek ROI koordinatları
    x, y, w, h = roi

    # ROI bölgesini kırpalım
    roi_image = image[y:y+y,x:x+150]

    cv2.rectangle(image,roi[2:],roi[:2],(0,255,0),2)
    # YOLOv5 ile ROI bölgesinde nesne tespiti yapalım
    results = model(roi_image)
    results_all = model(image)
    car_number = 0
    for *box, conf, cls in results_all.xyxy[0]:
        if model.names[int(cls)] == 'car':
            car_number += 1
            
    # Sonuçları orijinal görüntüdeki ROI bölgesine göre çizelim
    for *box, conf, cls in results.xyxy[0]:
        label = f'{model.names[int(cls)]} {conf:.2f}'
        x1, y1, x2, y2 = map(int, box)
        x1 += x
        y1 += y
        x2 += x
        y2 += y
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Sonuç görüntüyü gösterelim
    cv2.putText(image,f"Toplam Arac Sayisi{str(car_number)}",(0,height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow('YOLO Object Detection', image)
    cv2.imshow('roi', roi_image)
    # Bekle ve pencereyi kapat
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

