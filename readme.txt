�޸���YOLOv5��һ����bug�����ҽ����˴�������͸�ʽ��
ԭʼYOLOv5��39���� 1185���� 18275������ 37�﷨���� 905ƴд����
YOLOY��23���� 492���� 1074������ 28�﷨����

����ԭʼYOLOv5-6.2�����˴����ĸĽ�������
1���༤���30 Hard_Swish leaky_relu linspace gelu lrelu Hard_Sigmoid relu Squareplus selu elu sigmoid softmax softplus softsign step_function tanh SiLU Mish FReLU AconC MetaAconC celu glu hardshrink hardtanh prelu rrelu softmin softsign tanhshrink
2����������ǿ31 ͨ������ ���ض��� CLAHE HE ѹ�� ��ֵģ�� ��ֵģ�� ��˹ģ�� �⾵ģ�� �˶�ģ�� ������ �²��� ISO���� ��˹���� �������� ���� �� ���� ���� ������ȶԱȶ� ���ɫ�� �ҶȻ� ��ɫ�� ���٤�� RGB�任 ɫ������ ͨ������� BCS�任 PCA�任 HSV�任 ���90����ת
3������14 conv CrossConv DWConv Conv6 DepthWiseConv PointWiseConv ConvSig ConvSqu deconv simconv gnconv RepConv XBNConv ghostconv
4����ػ�6 simsppf spp aspp RFB sppcspc sppcspcgroup
5������ͷ8 ASFF Decoupled IDetect IAuxDetect DetectX DetectYoloX IBin MT
6����fpn6 bifpn affpn carafe panet elandpn
7����loss����19 focalloss Qfocalloss vfocalloss gfocalloss efocalloss giouloss diouloss ciouloss eiouloss siouloss aiouloss
8����NMS19 Merge-NMS Soft-NMS CIoU_NMS DIoU_NMS GIoU_NMS EIoU_NMS SIoU_NMS Soft-SIoUNMS Soft-CIoUNMS Soft-DIoUNMS Soft-EIoUNMS Soft-GIoUNMS andNMS clusterNMS clusterdiouNMS clusterspmNMS clusterspmdistNMS clusterciouNMS clustereiouNMS
9����ע����17 GAM NAM S2 SE SU SAM SK CC CBAM CA GAM ECA SE BOT3 RESCBAM BOT ACMIX
10��������83
BottleneckCSP BottleneckCSPA BottleneckCSPB BottleneckCSPC BottleneckCSP2 BottleneckCSPF BottleneckG BottleneckCSPL BottleneckCSPLG BottleneckCSPSE BottleneckCSPSEA BottleneckCSPSAM BottleneckCSPSAMA  BottleneckCSPSAMB  BottleneckCSPGC  BottleneckCSPDNL BottleneckCSP2SAM InvolutionBottleneck BottleneckCSPTR BottleneckCSP2TR
ResCSPA ResCSPB ResCSPC ResXCSPA ResXCSPB ResXCSPC RepResXCSPA RepResXCSPB RepResXCSPC RepRes RepResCSPA RepResCSPB RepResCSPC RepBottleneckCSPA RepBottleneckCSPB RepBottleneckCSPC 
C3TR C3Ghost C3HB C3RFEM C3STR C3x C3GC C3SPP C3C2 CTR3
GhostCSPA GhostCSPB GhostCSPC GhostSPPCSPC GhostBottleneck DownC
STCSPA STCSPB STCSPC SPPCSP ST2CSPA ST2CSPB ST2CSPC SPPCSPC SPPCSPTR
efficient involution mobilenet hornet spd addcon cneb convnext replknet frem seam horblock implicit shffule vovcsp HorBlock CNeB ConvNextBlock RepVGGBlockv6 VoVCSP  ImplicitA ImplicitM 
