    start_point = (0, 30)
    end_point = (width, 30 + int(width * np.tan(angle)))
    cv2.line(frame, start_point, end_point, (255, 0, 0), 1)

# Dikey çizgiler
    start_point = (110, 50)
    end_point = (190, height)
    cv2.line(frame, start_point, end_point, (0, 0, 250), 2)