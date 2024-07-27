import cv2
import torch
import imutils
# YOLOv5 modelini yükleyelim
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


TrDict = {'csrt': cv2.legacy.TrackerCSRT_create,
         'kcf' : cv2.TrackerKCF_create,
         'boosting' : cv2.legacy.TrackerBoosting_create,
         'mil': cv2.TrackerMIL_create,
         'tld': cv2.legacy.TrackerTLD_create,
         'medianflow': cv2.legacy.TrackerMedianFlow_create,
         'mosse':cv2.legacy.TrackerMOSSE_create}

tracker = TrDict['csrt']()

v = cv2.VideoCapture(r'C:\Users\dest4\Desktop\apsiyon_hack\apsiyon_hack\files\otopark.mp4')
grid_size = 10
ret, frame = v.read()
frame = imutils.resize(frame,width=600)
cv2.imshow('Frame',frame)
bb = cv2.selectROI('Frame',frame)
tracker.init(frame,bb)

while True:
    ret, frame = v.read()
    if not ret:
        break
    frame = imutils.resize(frame,width=600)
    height, width, _ = frame.shape

    # Yatay çizgiler
    for y in range(-width, height, grid_size):
        start_point = (0, y)
        end_point = (width, y + int(width * np.tan(angle)))
        cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    # Dikey çizgiler
    for x in range(0, width, grid_size):
        start_point = (x, 0)
        end_point = (x - int(height * np.tan(angle)), height)
        cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

    (success,box) = tracker.update(frame)
    if success:
        (x,y,w,h) = [int(a) for a in box]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,255,0),2)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
v.release()
cv2.destroyAllWindows()