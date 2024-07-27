import cv2
import torch
from time import sleep

# YOLOv5 modelini yükleyelim
model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')

# Görüntüyü yükleyelim
for image_num in range(20, 180, 5):
    image_num = str(image_num)
    if len(image_num) == 2:
        image_num = '0' + image_num
    image_path = f'files/photo/ezgif-frame-{image_num}.jpg'
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    start_point = (178, 150)
    end_point = (270, image.shape[0])
    color = (0, 10, 250)  # Çizgi rengi (kırmızı)
    thickness = 2  # Çizgi kalınlığı

    # Ana çizgiyi çizelim
    cv2.line(image, start_point, end_point, color, thickness)

    # 35 piksel aralıklarla yatay çizgiler çizelim ve alanları isimlendirelim
    interval = 35
    areas = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # YOLOv5 ile nesne tespiti yapalım
    results = model(image)
    car_number = 0
    for i, y in enumerate(range(start_point[1]+interval, end_point[1], interval)):
        # Yatay çizgileri çiz
        cv2.line(image, (start_point[0]-28, y), (end_point[0], y-40), (0, 0, 250), 2)
        # Park yerlerini alan olarak ekle
        areas.append((start_point[0]-28, y-interval, end_point[0], y))
        # Alanı isimlendir
        cv2.putText(image, letters[i], (start_point[0] - 58, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        car_areas = {letter: 0 for letter in letters[:len(areas)]}

    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in ['car', 'truck', 'bus']:
            x1, y1, x2, y2 = map(int, box)
            car_number += 1

            for j, (ax1, ay1, ax2, ay2) in enumerate(areas):
                # Araç dikdörtgeninin merkezinin alan dikdörtgeni içinde olup olmadığını kontrol edin
                if ax1 < (x1 + x2) // 2 < ax2 and ay1 < (y1 + y2) // 2 < ay2:
                    car_areas[letters[j]] += 1

    # Toplam araç sayısını yaz
    cv2.putText(image, f"Toplam Arac Sayisi: {car_number}", (0, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 2)

    # Sonuçları çiz ve araçların park yerlerini belirt
    for *box, conf, cls in results.xyxy[0]:
        if model.names[int(cls)] in ['car', 'truck', 'bus']:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Park yerlerindeki araç sayısını yaz
    for j, (ax1, ay1, ax2, ay2) in enumerate(areas):
        cv2.putText(image, f"{letters[j]}: {car_areas[letters[j]]} arac", (ax1 - 130, ay1 - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 200, 50), 2)

    # Sonuç görüntüyü gösterelim
    cv2.imshow('YOLO Object Detection', image)

    # Bekle ve pencereyi kapat
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
