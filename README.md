# yolov11火灾检测项目
这是一个用来检测图书馆火灾的项目,支持使用摄像头。

## get-started

安装依赖（建议使用miniconda，python版本3.11）：
`pytouch:`

```
# use the GPU:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

```
# use the CPU:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

`ultralytics:`

```
# use pip install the ultralytics package.
pip install ultralytics
```

```
# also,you can use the conda too.
conda install -c conda-forge ultralytics
```

`opencv:`

```
pip install opencv-python
```

## 如何使用？

`main.py`:主文件，用来直接检测指定目录下的图片和视频。

`camera.py`可以使用摄像头来进行实时检测

## TO-List

- [ ] 等待并对接学校图书馆的摄像头
- [ ] 优化模型，减少资源占用
- [ ] 将配置文件转移到json文件中，而不是在文件中
- [ ] 给软件写一个好看的皮囊