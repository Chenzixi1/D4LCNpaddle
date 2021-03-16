# D4LCN: Learning Depth-Guided Convolutions for Monocular 3D Object Detection（CVPR2020)
The paddlepaddle version of D4LCN(CVPR2020) https://github.com/dingmyu/D4LCN
BAIDU AISTUDIO 项目链接 https://aistudio.baidu.com/aistudio/projectdetail/1420778

# 1. 论文简介
## 论文地址
https://arxiv.org/pdf/1912.04799v1

## 项目地址
https://github.com/dingmyu/D4LCN

## 简介

单目3D目标检测最大的挑战在于没法得到精确的深度信息，传统的二维卷积算法不适合这项任务，因为它不能捕获局部目标及其尺度信息，而这对三维目标检测至关重要。为了更好地表示三维结构，现有技术通常将二维图像估计的深度图转换为伪激光雷达表示，然后应用现有3D点云的物体检测算法。因此他们的结果在很大程度上取决于估计深度图的精度，从而导致性能不佳。在本文中，作者通过提出一种新的称为深度引导的局部卷积网络(LCN)，更改了二维全卷积Dynamic-Depthwise-Dilated LCN ，其中的filter及其感受野可以从基于图像的深度图中自动学习，使不同图像的不同像素具有不同的filter。D4LCN克服了传统二维卷积的局限性，缩小了图像表示与三维点云表示的差距。D4LCN相对于最先进的KITTI的相对改进是9.1%，单目3D检测的SOTA方法。

## 网络主要结构
![](https://ai-studio-static-online.cdn.bcebos.com/97eaef6b333843e1bf14993414b3bed99a71ac8f6ee84f5da734d4f868c407f5)


# 2.复现心得
先回顾下之前百度论文复现营老师指导的复现流程
## 整体流程
* 数据集获取
* 数据预处理
* 构建前向网络
* 构建反向传播
* 精度对齐，小数据集训练两轮

## 精度对齐
* 去除随机性：本项目中没有dropout项，只需对数据预处理中mirror的随机性置0
* 数据对齐，输入数据：本项目数据预处理部分涉及深度图，需要对其进行预处理
* 模型参数对齐
pytorch 和 paddle 网络参数输出并手动对齐
pytorch参数转化为paddle
将保存的模型载入并设置paddle模型初始化
* loss对齐
* 小规模实验

# 3.复现过程
## （1）数据处理
* 数据集采用[kitti 数据集](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
* dataloader 因paddlepaddle官方给的dataloader方式，无法载入字典easydict类型的文件（原代码需要，保存相机参数等），固参考之前论文复现营的方法，自己定义了dataloader类，详细可见我的博客[paddle复现pytorch踩坑(十）：dataloader读取](https://blog.csdn.net/qq_32097577/article/details/112385033)
## （2）模型的复现
这一步是项目的核心，主要分成两个部分：
* 前向传播

主要分为ResNet、ResNet、DeformConv2d、RPN这几个主要的网络，复习基本流程是这样：
1. 每个子网络单独复现
2. 先定义一个适合网络输入的全1的矩阵
3. 根据[paddleAPI对照表](https://editor.csdn.net/md/?articleId=112383360)依据forward顺序逐一修改
4. 打开pytorch文件，同步debug
5. 最后RPN的复现，需要同时设置train模型或eval模型，用于矫正

* 反向传播

反向传播过程复现类似，但需要在前向传播复现好的基础上，保存上一阶段跑的结果，输入反向层进行调试。
其中复现遇到的主要问题在损失函数的使用，具体在smooth_l1的用法不一致，输出的维度不同[paddle复现pytorch踩坑(八）：smooth_l1的用法](https://editor.csdn.net/md/?articleId=112385457)

## （3）精度对齐
做完前面的两个步骤，就完成任务的一半了，没错前面只是一半的工作量，后面才是重点

1. **初步对齐**：将全1矩阵输入网络前向传播反向传播后对比精度
这里遇到的问题主要是损失函数：pytorch中F.cross_entropy应使用 paddle中的fluid.layers.softmax_with_cross_entropy,其中有很多干扰的API，在复现过程维度上相同的，但精度就相去甚远，相关的分析在我的博客中有详细说明[paddle复现pytorch踩坑(七）：softmax_with_cross_entropy的用法](https://editor.csdn.net/md/?articleId=112385132)
2. **进阶对齐**：将原网络预训练模型输入并预测，对比预测结果
首先需要对pytorch模型进行转化，这一步比较简单，社区内的方法较多，可参考我的博客[paddle复现pytorch踩坑(十一）：转换pytorch预训练模型](https://editor.csdn.net/md/?articleId=112679438)
最后输入paddle模型加载pdparams文件成功并对比预测结果相同，则过关。
3. **高级对齐**：设置相同的config，训练两轮对比结果，如果对比的训练结果相差不远的话，就可以跑实验了。

## （4）其他问题
复现过程中的其他问题，可以参考我的博客，这个模型比较复杂，能踩过的API坑基本都踩过了

[paddle复现pytorch踩坑(四）：Tensor](https://editor.csdn.net/md/?articleId=112384360)

[paddle复现pytorch踩坑(五）：dygraph的一些用法示例](https://editor.csdn.net/md/?articleId=112384842)

[paddle复现pytorch踩坑(六）：多维度index下gather的用法](https://editor.csdn.net/md/?articleId=112384654)

# 4.运行步骤
## (1) 环境安装
```
paddlepaddle == 1.8.4
opnecv-python
easydict
```
## (2)数据集准备
[KITTI 3D 目标检测数据](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)

## (3)模型训练
```
python scripts/train.py
```

## （4）模型预测
模型结果输出为txt标签文件
```
unzip data/data68262/D4LCNpretrain.zip -d pretrain/
# 可视化输出预测文件
!python scripts/infer.py
```

## （5）模型评估
编译KITTI数据集评估工具
* 编译
```
cd scripts/eval/ && g++ -O3 -DNDEBUG -o evaluate_object evaluate_object.cpp && cd ../..
```
* 评估
```
# 评估
python scripts/test.py

# 或者直接调用evaluate_object 可执行文件
./scripts/eval/evaluate_object ./data/training/label_2/ ./pretrain/
```

## （6）3D框可视化
3D框可视化可借助代码
[kitti_object_vis](https://github.com/kuixu/kitti_object_vis)。
通过设置文件路径即可
Windows10、Linux下亲测有效

### 原始图像KITTI
![](https://ai-studio-static-online.cdn.bcebos.com/4831a9c9cd424338939b509a77bbcc50fd43103039fc48e397422d993717aae9)

### Ground Truth
![](https://ai-studio-static-online.cdn.bcebos.com/52ac796fa2f445a2b32d07459e4415ec7ffd8076ca09435c868429fa06a0fadc)

### 测试结果
![](https://ai-studio-static-online.cdn.bcebos.com/6cc5ae20f9c74db2a38e8fd0db8ce584eb2368ae89a94aeba5b6c45ec90b8ecf)

## VisualDL 可视化训练结果
![](https://ai-studio-static-online.cdn.bcebos.com/ce86b339a8cd407788195d1cf3906d532b6d1a31c536425f96f1b2763174b149)

# 5.总结与展望
这篇项目是利用深度图进行3D目标检测的工作，目前3D目标检测的工作主要借助雷达等传感器上，借助点云方法部署在自动驾驶等领域。而对于消费级的深度传感器，则缺乏有效的推理方案，希望这个复现工作可以帮助开发者在paddle框架基础上开展3D目标检测相关的工作。

# 附录：文件说明
* **scripts**
  
  * train.py 训练脚本
  * reader.py 数据集读取脚本
  * infer.py 预测脚本
  * test.py 评估脚本：会在train.py文件中调用
  * config --> depth_guided_config.py 权重配置文件
  * eval 文件夹：kitti官网给的评估cpp文件，需编译为可执行文件
* **models**
  
  * resnet.py backbone脚本
  * deform_conv_v2.py 膨胀卷积引导的backbone脚本
  * resnet_dilate.py 网络主体RPN脚本

* **lib**
  * cpu_nms.py 目标检测NMS后处理算法脚本
  * lr.py 学习率调整脚本
  * core.py 核心函数脚本
  * util.py 核心工具脚本
  * augmentations.py 数据预处理类脚本
  * rpn_util.py RPN文件辅助脚本
  * loss --> rpn_3d.py 损失函数脚本

