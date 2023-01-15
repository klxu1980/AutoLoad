"""
从张方远程序中拿出来的参数
"""
PositionL1P1 = (1000, 950)   # 定位线1点1
PositionL1P2 = (1230, 810)   # 定位线1点2
PositionL2P1 = (1230, 810)   # 定位线2点1
PositionL2P2 = (2280, 1260)  # 定位线2点2
PositionL3P1 = (2280, 1260)  # 定位线3点1
PositionL3P2 = (1880, 1950)  # 定位线3点2

start_keypoint = (1900, 500)  # 全局点:预测大卷起始点
start_keypoint1 = (1900, 530) # 全局点:预测小卷起始点

drop_keypoint = (1400,800)   # 全局点：大卷关键点下降点
drop_keypoint1 = (1400, 830) # 全局点：小卷关键点下降点

alignment_keypoint = (1420, 1510) # 全局点：关键点对准点
end_keypoint = (1210, 1710)		  # 全局点：预测结束点

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