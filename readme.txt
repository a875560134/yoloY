修复了YOLOv5的一部分bug，并进行了代码清理和格式化
原始YOLOv5：39错误 1185警告 18275弱警告 37语法错误
YOLOY：23错误 492警告 1074弱警告 28语法错误

基于原始YOLOv5-7.0加入了大量的改进方法：
1更多激活函数 Hard_Swish leaky_relu linspace gelu lrelu Hard_Sigmoid relu Squareplus selu elu sigmoid softmax softplus softsign step_function tanh SiLU Mish FReLU AconC MetaAconC celu glu hardshrink hardtanh prelu rrelu softmin softsign tanhshrink
2更多数据增强 通道丢弃 像素丢弃 CLAHE HE 压缩 均值模糊 中值模糊 高斯模糊 棱镜模糊 运动模糊 超像素 下采样 ISO噪声 高斯噪声 乘性噪声 浮雕 锐化 USM 过曝 随机亮度对比度 随机色调 灰度化 褐色化 随机伽玛 RGB变换 色调分离 通道随机化 BCS变换 PCA变换 HSV变换 随机90度旋转
3更多卷积 conv crossconv robustconv robustconv2 dwconv conv6 pwconv convsig convsqu simconv gnconv repconv xbnconv ghostconv
4更多池化 simsppf spp aspp RFB sppcspc sppcspcgroup
5更多检测头 ASFF Decoupled IDetect IAuxDetect DetectX DetectYoloX IBin MT
6更多fpn bifpn affpn carafe panet elandpn deconvfpn
7更多loss函数 focalloss qfocalloss vfocalloss gfocalloss efocalloss giouloss diouloss ciouloss eiouloss siouloss aiouloss
8更多NMS Merge-NMS Soft-NMS CIoU_NMS DIoU_NMS GIoU_NMS EIoU_NMS SIoU_NMS Soft-SIoUNMS Soft-CIoUNMS Soft-DIoUNMS Soft-EIoUNMS Soft-GIoUNMS andNMS clusterNMS clusterdiouNMS clusterspmNMS clusterspmdistNMS clusterciouNMS clustereiouNMS
9更多注意力 GAM NAM SAM S2 SU SK CC CA ECA SE BOT PN CBAM ACMIX COT
10更多主干
BottleneckCSP BottleneckCSPA BottleneckCSPB BottleneckCSPC BottleneckCSP2 BottleneckCSPF BottleneckG BottleneckCSPL BottleneckCSPLG BottleneckCSPSE BottleneckCSPSEA BottleneckCSPSAM BottleneckCSPSAMA  BottleneckCSPSAMB  BottleneckCSPGC  BottleneckCSPDNL BottleneckCSP2SAM InvolutionBottleneck BottleneckCSPTR BottleneckCSP2TR
ResCSPA ResCSPB ResCSPC ResXCSPA ResXCSPB ResXCSPC RepResXCSPA RepResXCSPB RepResXCSPC RepRes RepResCSPA RepResCSPB RepResCSPC RepBottleneckCSPA RepBottleneckCSPB RepBottleneckCSPC 
C3TR C3Ghost C3HB C3RFEM C3STR C3x C3GC C3SPP C3C2 CTR3
GhostCSPA GhostCSPB GhostCSPC GhostSPPCSPC GhostBottleneck DownC
STCSPA STCSPB STCSPC SPPCSP ST2CSPA ST2CSPB ST2CSPC SPPCSPC SPPCSPTR
efficient involution mobilenet hornet spd addcon cneb convnext replknet frem seam horblock implicit shffule vovcsp HorBlock CNeB ConvNextBlock RepVGGBlockv6 VoVCSP  ImplicitA ImplicitM 
11更多评价指标 F1分数 map75 map95
