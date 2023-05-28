#项目名称
这是一个使用Faster-RCNN结构进行训练的目标检测项目，主干（backbone）使用的是`fasterrcnn_resnet50_fpn`。数据集使用的是VOC2012数据集。

##项目结构
```
.
├── VOCdevkit
│   └── VOC2012
├── caculate_mAP.py
├── fasterRCNN_predictor.py
├── fasterRCNN_train.py
├── loss_and_mAP.json
├── pictures
│   ├── aeroplane.png
│   ├── car.png
│   └── person.jpg
├── readme.md
├── results
│   └── my_fasterRCNN.pth
└── utils.py
```
`VOCdevkit`文件下是`VOC`数据集，该数据集已经划分好训练集与测试集且`pytorch`有直接解析该数据集的函数，该文件太大了无法上传至GitHub，使用时需自行下载 `VOC`数据集并放在根目录下。

`results`目录下为训练好的神经网络的模型参数。`loss_and_mAP.json`文件中保存了训练与测试集上的损失值与mAP，可以通过 `json.loads()`读取`loss_and_mAP.json`文件。

###代码文件包括以下内容：
+   `utils.py` 提供了一些必要的方法，包括`VOC`数据集中的类别字典，和图像预处理的方法函数。

+   `fasterRCNN_train.py` 提供了训练`Faster-RCNN`模型的代码，包括解析并封装`VOC`数据集，优化器的选择，fasterRCNN的构建与训练。训练完成后的网络参数会保存在`results`目录下，训练过程会保存在`loss_and_mAP.json`文件中。
    
+   `fasterRCNN_predict.py` 提供了测试`Faster-RCNN`模型的代码，会显示给定图片的检测结果，包括检测框（box）、类别标签（label）和得分（score）。

+   `caculate_mAP.py`提供了计算`mAP`的代码。
    
+   `visualize.py`提供了训练过程和网络参数可视化代码。

##使用方法
+   用户可以在`fasterRCNN_train.py`中训练自己的模型，如有必要，用户可以进行如下调整。
    +   `my_device`变量设置为自己能够使用的加速设备，可以是`cuda`、`cpu`、`mps`。
    +   `root_dir`变量设置为自己的`VOC`数据集路径。
    +   `batch_size`这里设置为16，用户可以根据自己的设备调整。
    +   优化器的学习率与权重衰减尽量自己调整，此处设置为初始学习率0.01，衰减权重为0.8。
    +   `num_epochs`变量保存的是训练的总轮数，此处设置为20，用户可以根据实际情况调整。
    +   为了不浪费大量时间计算`loss`和`mAP`，训练中只在数据集上随机采样并计算`loss`和`mAP`，用户可以调整计算`loss`和`mAP`的频率以及采样频率。

+   用户可以在`fasterRCNN_test.py`中测试训练好的模型。代码会输出用户给定图片的检测框、标签和得分，这些信息都会被标注在图片上。测试需要的图片保存在`pictures`目录下。
    
+   用户可以在`visualize.py`中可视化模型训练过程。

##已训练的模型

results目录下没有训练好的模型，需要自己训练，本人训练好的模型保存在百度云。
链接: https://pan.baidu.com/s/1akRg0PpzFu-AVvBL0s5cpg 提取码: 9c3d