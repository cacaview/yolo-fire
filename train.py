import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    # model.load('runs/train/exp6/weights/last.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model = YOLO(model='runs/train/exp9/weights/last.pt')
    model.train(data=r'dataset.yaml',
                imgsz=640,
                epochs=100,
                batch=8,
                workers=8,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )

"""
model参数：该参数填入模型配置文件的路径，改进的话建议不需要填预训练模型权重
data参数：该参数可以填入训练数据集配置文件的路径
imgsz参数：该参数代表输入图像的尺寸，指定为 640x640 像素
epochs参数：该参数代表训练的轮数
batch参数：该参数代表批处理大小，电脑显存越大，就设置越大，根据自己电脑性能设置
workers参数：该参数代表数据加载的工作线程数，出现显存爆了的话可以设置为0，默认是8
device参数：该参数代表用哪个显卡训练，留空表示自动选择可用的GPU或CPU
optimizer参数：该参数代表优化器类型
close_mosaic参数：该参数代表在多少个 epoch 后关闭 mosaic 数据增强
resume参数：该参数代表是否从上一次中断的训练状态继续训练。设置为False表示从头开始新的训练。如果设置为True，则会加载上一次训练的模型权重和优化器状态，继续训练。这在训练被中断或在已有模型的基础上进行进一步训练时非常有用。
project参数：该参数代表项目文件夹，用于保存训练结果
name参数：该参数代表命名保存的结果文件夹
single_cls参数：该参数代表是否将所有类别视为一个类别，设置为False表示保留原有类别
cache参数：该参数代表是否缓存数据，设置为False表示不缓存。
"""