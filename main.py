import os

from ultralytics import YOLO
import cv2


def predict(chosen_model, img, classes=[], conf=0.5):
    """
    对输入图像使用指定的YOLO模型进行目标检测预测。

    参数:
    chosen_model (YOLO): 已经加载的YOLO模型对象。
    img: 输入的图像数据（可以是由cv2.imread读取后的图像等）。
    classes (list, 可选): 需要检测的目标类别列表，默认为空列表，表示检测所有类别。
    conf (float, 可选): 置信度阈值，默认为0.5。

    返回:
    results: 目标检测的结果。
    """
    try:
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf)
        else:
            results = chosen_model.predict(img, conf=conf)
        return results
    except Exception as e:
        print(f"目标检测预测过程出现错误: {e}")
        return None


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1, model_path="yolo11x.pt"):
    """
    对输入图像进行目标检测，并将检测结果可视化绘制在图像上。

    参数:
    chosen_model (YOLO): 已经加载的YOLO模型对象（也可通过传入model_path重新加载）。
    img: 输入的图像数据（可以是由cv2.imread读取后的图像等）。
    classes (list, 可选): 需要检测的目标类别列表，默认为空列表，表示检测所有类别。
    conf (float, 可选): 置信度阈值，默认为0.5。
    rectangle_thickness (int, 可选): 绘制目标矩形框的线条厚度，默认为2。
    text_thickness (int, 可选): 绘制目标类别文字的线条厚度，默认为1。
    model_path (str, 可选): YOLO模型文件路径，默认为"yolo11x.pt"。

    返回:
    img: 绘制好检测结果的图像。
    results: 目标检测的结果。
    """
    # 若传入的模型对象为空，则重新加载模型
    if chosen_model is None:
        chosen_model = YOLO(model_path)

    results = predict(chosen_model, img, classes, conf)
    if results is None:
        return None, None
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

def process_video(image_path, save_path, model, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    """
    处理视频文件，对视频中的每一帧图像进行目标检测并将检测结果可视化后保存为新的视频。

    参数:
    video_path (str): 输入的视频文件路径。
    save_video_path (str): 保存检测结果视频的文件路径。
    model (YOLO): 已经加载的YOLO模型对象。
    classes (list, 可选): 需要检测的目标类别列表，默认为空列表，表示检测所有类别。
    conf (float, 可选): 置信度阈值，默认为0.5。
    rectangle_thickness (int, 可选): 绘制目标矩形框的线条厚度，默认为2。
    text_thickness (int, 可选): 绘制目标类别文字的线条厚度，默认为1。

    返回:
    None
    """
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened():
        print(f"无法打开视频文件 {image_path}，请检查路径及视频文件是否正确！")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame, _ = predict_and_detect(model, frame, classes, conf, rectangle_thickness, text_thickness)
        if result_frame is None:
            continue
        out.write(result_frame)
        # cv2.imshow("Video", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # 确保保存的视频文件能被正确关闭和保存
    #os.sync()

if __name__ == "__main__":
    """
    # 获取用户输入的模型路径、图像路径、保存路径以及相关参数
    model_path = input("请输入YOLO模型文件路径（例如：yolo11x.pt）: ")
    image_path = input("请输入要检测的图像文件路径（例如：YourImagePath.png）: ")
    save_path = input("请输入检测结果图像的保存路径（例如：YourSavePath.png）: ")
    conf_value = float(input("请输入置信度阈值（例如：0.5）: "))
    rect_thickness = int(input("请输入矩形框厚度（例如：2）: "))
    text_thickness = int(input("请输入文字厚度（例如：1）: "))
    classes_input = input("请输入要检测的目标类别（多个类别用空格分隔，若检测所有类别则直接回车）: ")
    classes = classes_input.split() if classes_input else []
    """
    # 获取用户输入的模型路径、图像路径、保存路径以及相关参数
    model_path = 'runs/train/exp9/weights/last.pt'
    image_path = "F:\\yolov5\\yolo\\img\\"
    save_path = "F:\\yolov5\\yolo\\runs\\output_path\\"
    conf_value = float(0.4)
    rect_thickness = int(2)
    text_thickness = int(2)
    classes_input = ""
    classes = classes_input.split() if classes_input else []
    # 加载模型
    model = YOLO(model_path)

    # 检查图像文件夹和保存文件夹是否存在，不存在则创建
    if not os.path.exists(image_path):
        print(f"图像路径 {image_path} 不存在，请检查路径！")
        exit(1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 判断输入路径是文件夹还是文件
    if os.path.isdir(image_path):
        # 如果是文件夹，分别处理其中的图像文件和视频文件
        image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if
                       f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        video_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if
                       f.lower().endswith(('.mp4', '.avi', '.mov'))]

        for image_file in image_files:
            image = cv2.imread(image_file)
            if image is None:
                print(f"无法读取图像 {image_file}，请检查文件是否正确！")
                continue
            result_img, results = predict_and_detect(model, image, classes, conf_value, rect_thickness, text_thickness,
                                                     model_path)
            if result_img is None:
                print(f"目标检测及可视化过程出现问题，无法得到 {image_file} 的结果图像！")
                continue

            # 构建保存文件名（使用原图像文件名）
            save_file_name = os.path.join(save_path, os.path.basename(image_file))
            # 展示图像
            # cv2.imshow("Image", result_img)
            # 保存图像，添加错误处理
            if not cv2.imwrite(save_file_name, result_img):
                print(f"图像 {image_file} 保存失败，请检查保存路径的权限等相关问题！")
            cv2.waitKey(5)

        for video_file in video_files:
            # 构建保存的视频文件名，这里简单示例为在原文件名后添加'_result'后缀，可根据实际需求调整
            save_video_file_name = os.path.join(save_path, os.path.basename(video_file).split('.')[0] + '_result.' +
                                                os.path.basename(video_file).split('.')[1])
            process_video(video_file, save_video_file_name, model, classes, conf_value, rect_thickness, text_thickness)
    elif os.path.isfile(image_path):
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in ('.jpg', '.jpeg', '.png', '.bmp'):
            # 处理单张图像情况
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像 {image_path}，请检查文件是否正确！")
                exit(1)

            result_img, results = predict_and_detect(model, image, classes, conf_value, rect_thickness, text_thickness,
                                                     model_path)
            if result_img is None:
                print(f"目标检测及可视化过程出现问题，无法得到 {image_path} 的结果图像！")
                exit(1)

            # 构建保存文件名（使用原图像文件名）
            save_file_name = os.path.join(save_path, os.path.basename(image_path))
            # 展示图像
            # cv2.imshow("Image", result_img)
            # 保存图像，添加错误处理
            if not cv2.imwrite(save_file_name, result_img):
                print(f"图像 {image_path} 保存失败，请检查保存路径的权限等相关问题！")

            cv2.waitKey(5)
        elif file_extension in ('.mp4', '.avi', '.mov'):
            # 构建保存的视频文件名，这里简单示例为在原文件名后添加'_result'后缀，可根据实际需求调整
            save_video_file_name = os.path.join(save_path, os.path.basename(image_path).split('.')[0] + '_result.' +
                                                os.path.basename(image_path).split('.')[1])
            process_video(image_path, save_video_file_name, model, classes, conf_value, rect_thickness, text_thickness)
    else:
        print(f"输入的文件路径 {image_path} 不合法，请检查是否为正确的图像或视频文件路径！")

    cv2.destroyAllWindows()
    # 释放内存（这里释放可能占用较多内存的变量，可根据实际情况调整）
    del model
    cv2.waitKey(5)
    # 释放内存
    #del image
    #del result_img