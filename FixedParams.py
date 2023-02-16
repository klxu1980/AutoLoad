"""
从张方远程序中拿出来的参数
"""
# 基坑标记线
PitMarkL1P1 = (1000, 950)   # 定位线1点1
PitMarkL1P2 = (1230, 810)   # 定位线1点2
PitMarkL2P1 = (1230, 810)   # 定位线2点1
PitMarkL2P2 = (2280, 1260)  # 定位线2点2
PitMarkL3P1 = (2280, 1260)  # 定位线3点1
PitMarkL3P2 = (1880, 1950)  # 定位线3点2

# 带卷运动轨迹
coil_trans_begin = (1900, 500)     # 上卷(大卷)起始点。(1900, 530)
# coil_trans_begin = (1900, 530)     # 上卷(小卷)起始点。(1900, 530)

coil_fall_begin = (1400, 800)      # 带卷下落起始点
coil_fall_begin1 = (1400, 830)     # 全局点：小卷关键点下降点

alignment_keypoint = (1420, 1510)  # 全局点：关键点对准点
end_keypoint = (1210, 1710)		   # 全局点：预测结束点

stage1_up1 = (1900, 460)   # 平移阶段1运动上区间线右点
stage1_up2 = (1365, 781)   # 平移阶段1运动上区间线左点
stage1_down1 = (1910, 585) # 平移阶段1运动下区间线右点
stage1_down2 = (1440, 867) # 平移阶段1运动下区间线左点

stage2_left1 = stage1_up2	# 下降阶段2运动左区间线上点
stage2_left2 = (1385, 1490)     # 下降阶段2运动左区间线下点
stage2_right1 = stage1_down2 # 下降阶段2运动右区间线上点
stage2_right2 = (1455, 1530)    # 下降阶段2运动右区间线下点

stage3_up1 = stage2_left2    # 平移阶段3运动上区间线右点
stage3_up2 = (1200, 1685)	    # 平移阶段3运动上区间线左点
stage3_down1 = stage2_right2 # 平移阶段3运动下区间线右点
stage3_down2 = (1240, 1735)		# 平移阶段3运动下区间线左点


INIT_COIL_CENTER = (1800, 630)   # 带卷在堆料区的初始位置。通常会首先进行带卷的初始位置定位，如果没有执行，则默认使用该位置。
AIMING_BEGIN = (1450, 1450)      # 带卷对准开卷机的检测开始和停止位置
AIMING_END = end_keypoint[0]

# 根据现场状况增加的参数
TRACKING_BEGIN_X = 1800       # 带卷初始定位超过该位置时，进入到带卷跟踪状态
RECORD_ROOT_PATH = "D:\\huada2#data"

# 试验控制用全局参数
CNN_SUB_IMAGE_CNT = 8
SHOW_FITTING_ELLIPSE = True      # 显示椭圆拟合过程
NO_KALMAN = True                 # 不使用卡尔曼滤波
SCREEN_SHOT_IN_PIT = True        # 带卷到达基坑底部时自动截图