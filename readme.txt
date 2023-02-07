修复了YOLOv5的一部分bug，并进行了代码清理和格式化
原始YOLOv5：39错误 1185警告 18275弱警告 37语法错误
YOLOY：23错误 492警告 1074弱警告 28语法错误

基于原始YOLOv5-7.0加入了大量的改进方法：
1更多激活函数 Hard_Swish Leaky_Relu Linspace Gelu Lrelu Hard_Sigmoid Relu Squareplus Selu Elu Sigmoid Softmax Softplus Softsign Step_function Tanh SiLU Mish FReLU AconC MetaAconC Celu Glu Hardshrink Hardtanh Prelu Rrelu Softmin Softsign Tanhshrink
2更多数据增强 通道丢弃 像素丢弃 CLAHE HE 压缩 均值模糊 中值模糊 高斯模糊 棱镜模糊 运动模糊 超像素 下采样 ISO噪声 高斯噪声 乘性噪声 浮雕 锐化 USM 过曝 随机亮度对比度 随机色调 灰度化 褐色化 随机伽玛 RGB变换 色调分离 通道随机化 BCS变换 PCA变换 HSV变换 随机90度旋转
3更多卷积 conv CrossConv Conv6 PwConv Convsig Convsqu SimConv GnConv RepConv XbnConv GhostConv
4更多池化 SimSPPF SPP ASPP RFB SPPSCPC SPPSCPCGroup GhostSPPCSPC SPPFCSPC
5更多检测头 ASFF Decoupled IDetect IAuxDetect DetectX DetectYoloX IBin MT
6更多上采样 bifpn affpn carafe panet elandpn deconvfpn dwconvfpn
7更多loss函数 focalloss qfocalloss vfocalloss gfocalloss efocalloss giouloss diouloss ciouloss eiouloss siouloss alphagiouloss alphadiouloss alphaciouloss alphaeiouloss alphasiouloss
8更多NMS Merge-NMS Soft-NMS CIoU_NMS DIoU_NMS GIoU_NMS EIoU_NMS SIoU_NMS Soft-SIoUNMS Soft-CIoUNMS Soft-DIoUNMS Soft-EIoUNMS Soft-GIoUNMS andNMS clusterNMS clusterdiouNMS clusterspmNMS clusterspmdistNMS clusterciouNMS clustereiouNMS
9更多注意力 GAM NAM SAM S2 SU SK CC CA ECA SE BOT PN CBAM ACMIX COT
10更多主干 C3TR C3Ghost C3HB C3RFEM C3STR C3x C3GC C3SPP C3C2 CTR3 C2f
BottleneckCSP BottleneckCSPA BottleneckCSPB BottleneckCSPC BottleneckCSP2 BottleneckCSPF BottleneckG BottleneckCSPL BottleneckCSPLG BottleneckCSPSE BottleneckCSPSEA BottleneckCSPSAM BottleneckCSPSAMA  BottleneckCSPSAMB  BottleneckCSPGC  BottleneckCSPDNL BottleneckCSP2SAM InvolutionBottleneck BottleneckCSPTR BottleneckCSP2TR
ResCSPA ResCSPB ResCSPC ResXCSPA ResXCSPB ResXCSPC RepResXCSPA RepResXCSPB RepResXCSPC RepRes RepResCSPA RepResCSPB RepResCSPC RepBottleneckCSPA RepBottleneckCSPB RepBottleneckCSPC
GhostCSPA GhostCSPB GhostCSPC GhostBottleneck DownC
STCSPA STCSPB STCSPC SPPCSP ST2CSPA ST2CSPB ST2CSPC SPPCSPC SPPCSPTR
efficient involution mobilenet hornet spd addcon cneb convnext replknet frem seam horblock implicit shffule vovcsp HorBlock CNeB ConvNextBlock RepVGGBlockv6 VoVCSP ImplicitA ImplicitM 
11更多评价指标 F1分数 map75 map95
12独创改进方法 MIoUloss GhostSPPFCSPC YOLOv5NN主干

coco128,100epoch	layers	gflops	F1	mAP@0.5	mAP@0.5：0.95
N	213	4.5	0.818 	0.866 	0.619 
GIOUloss	213	4.5	0.814 	0.866 	0.617 
DIOUloss	213	4.5	0.817 	0.866 	0.619 
EIOUloss	213	4.5	0.826 	0.881 	0.643 
SIOUloss	213	4.5	0.821 	0.868 	0.625 
MIOUloss	213	4.5	0.802 	0.875 	0.649 
FOCALloss	213	4.5	0.710 	0.785 	0.559 
QFOCALloss	213	4.5	0.710 	0.788 	0.555 
GFOCALloss	213	4.5	0.759 	0.827 	0.596 
VFOCALloss	213	4.5	0.770 	0.836 	0.605 
HARDSWISH	213	4.5	0.794 	0.859 	0.603 
MISH	213	4.5	0.802 	0.860 	0.622 
LeakyReLU	213	4.5	0.755 	0.828 	0.559 
CLAHE	213	4.5	0.800 	0.847 	0.601 
USM	213	4.5	0.808 	0.866 	0.629 
SHARP	213	4.5	0.789 	0.854 	0.604 
HE	213	4.5	0.755 	0.848 	0.573 
SIGCONV	213	4.5	0.726 	0.793 	0.516 
SQUCONV	213	4.5	0.758 	0.828 	0.559 
XBNCONV	222	4.5	0.781 	0.847 	0.614 
Decoupled	267	45.2	0.274 	0.462 	0.227 
N6	280	4.6	0.775 	0.841 	0.610 
Idetect	278	4.6	0.803 	0.864 	0.615 
C3GC	317	4.7	0.808	0.868	0.623
C3HB	312	5.0 	0.191	0.31	0.106
C3FREM	224	3.1	0.0951	0.189	0.0551
C3TR	226	4.5	0.729	0.813	0.54
C3X	213	3.9	0.226	0.324	0.117
ASPP	213	6.1	0.804	0.859	0.586
BasicRFB	250	4.6	0.752	0.84	0.558
SPPCSPC	231	5.8	0.787	0.855	0.591
SPPFCSPC	228	5.8	0.803	0.869	0.592
CARAFEFPN	219	4.6	0.748	0.818	0.551
BIFPN	229	4.8	0.73	0.823	0.494
DECONVFPN	213	4.9	0.716	0.786	0.534
DWCONVFPN	213	6.1	0.742	0.801	0.557
