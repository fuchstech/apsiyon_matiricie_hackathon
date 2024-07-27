import cv2
import numpy as np

# Renk aralığı tanımları
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
lower_red_alt = np.array([170, 50, 50])
upper_red_alt = np.array([180, 255, 255])
# Video akışını başlat
cap = cv2.VideoCapture(r'files/otopark.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk alanına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renkler için maske oluştur
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red_alt, upper_red_alt)
    mask = mask1 | mask2

    # Maskeyi uygula
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Masked Frame", result)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
