import cv2
import numpy as np

def detect_color(frame):
    # Chuyển đổi frame từ BGR sang HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Định nghĩa các ngưỡng màu cho từng màu sắc cơ bản
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Tạo các mask cho từng màu
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Tìm contour và vẽ bounding box cho các vùng màu
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours_red:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, 'Red', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for contour in contours_blue:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Blue', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for contour in contours_yellow:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, 'Yellow', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    return frame


# Khởi tạo camera
cap = cv2.VideoCapture(1)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện màu sắc và vẽ bounding box
    detected_frame = detect_color(frame)

    # Hiển thị frame với các vùng màu được phát hiện
    cv2.imshow('Color Detection', detected_frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'e'
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()
