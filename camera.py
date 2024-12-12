import cv2
from main import *
# 打开默认摄像头
model_path = 'runs/train/exp9/weights/last.pt'
conf_value = float(0.4)
rect_thickness = int(2)
text_thickness = int(2)
classes_input = ""
classes = classes_input.split() if classes_input else []

cap = cv2.VideoCapture(0)

model = YOLO(model_path)


while True:
    # 读取一帧视频
    ret, frame = cap.read()
    image = frame
    result_img, results = predict_and_detect(model, image, classes, conf_value, rect_thickness, text_thickness,model_path)
    if ret:
        # 显示视频帧
        cv2.imshow('Video Stream', result_img)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()