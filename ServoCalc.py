import numpy as np
import math
import matplotlib.pyplot as plt


def calc_L1_L2(j1, j2, xy):
    L1 = np.sqrt((xy[0] - j1[0])**2 + (xy[1] - j1[1])**2)
    L2 = np.sqrt((xy[0] - j2[0]) ** 2 + (xy[1] - j2[1]) ** 2)
    return L1, L2


def calc_sin_cos(j1, j2, xy, L1, L2):
    sin1 = (xy[1] - j1[1]) / L1
    cos1 = (xy[0] - j1[0]) / L1
    sin2 = (xy[1] - j2[1]) / L2
    cos2 = (xy[0] - j2[0]) / L2

    return sin1, cos1, sin2, cos2


def calc_F1_F2(Fh, Fv, sin1, cos1, sin2, cos2):
    FF = cos1 * sin2 - sin1 * cos2
    F1 = (Fh * sin2 - Fv * sin1) / FF
    F2 = (Fv * cos1 - Fh * cos2) / FF

    return F1, F2


def calc_xy(j1, j2, L1_L2):
    L = math.sqrt((j1[0] - j2[0])**2 + (j1[1] - j2[1])**2)
    alpha = math.atan2(j2[1] - j1[1], j1[0] - j2[0])
    beta = np.arccos((L1_L2[0]**2 - L1_L2[1]**2 + L**2) / (2.0 * L * L1_L2[0]))
    theta1 = math.pi - alpha - beta
    x = L1_L2[0] * np.cos(theta1) + j1[0]
    y = L1_L2[0] * np.sin(theta1) + j1[1]

    return x, y


def calc_F1_F2_by_L1_L2(j1, j2, L1_L2, Fh, Fv):
    x, y = calc_xy(j1, j2, L1_L2)

    sin1 = (y - j1[1]) / L1_L2[0]
    cos1 = (x - j1[0]) / L1_L2[0]
    sin2 = (y - j2[1]) / L1_L2[1]
    cos2 = (x - j2[0]) / L1_L2[1]

    F11, F21 = np.fabs(calc_F1_F2(Fh, Fv, sin1, cos1, sin2, cos2))
    F12, F22 = np.fabs(calc_F1_F2(-Fh, Fv, sin1, cos1, sin2, cos2))
    F13, F23 = np.fabs(calc_F1_F2(Fh, -Fv, sin1, cos1, sin2, cos2))
    F14, F24 = np.fabs(calc_F1_F2(-Fh, -Fv, sin1, cos1, sin2, cos2))

    F1 = np.maximum(F11, F12)
    F1 = np.maximum(F1, F13)
    F1 = np.maximum(F1, F14)

    F2 = np.maximum(F21, F22)
    F2 = np.maximum(F2, F23)
    F2 = np.maximum(F2, F24)

    return x, y, F1, F2


def enum_space_and_force(j1, j2, L1_range, L2_range, Fh, Fv):
    L1 = np.arange(L1_range[0], L1_range[1], 1)
    L2 = np.arange(L2_range[0], L2_range[1], 1)
    L1_L2 = np.array(np.meshgrid(L1, L2))

    x, y, F1, F2 = calc_F1_F2_by_L1_L2(j1, j2, L1_L2, Fh, Fv)
    return L1_L2, x, y, F1, F2


def enum_show_space_and_force(j1, j2, L1_range, L2_range, work_rect, Fh, Fv):
    L1_L2, x, y, F1, F2 = enum_space_and_force(j1, j2, L1_range, L2_range, Fh, Fv)

    F1_min = F1.min()
    F1_max = F1.max()
    F1_norm = (F1 - F1_min) / (F1_max - F1_min)
    plt.contourf(x, y, F1, 100, cmap='RdBu_r')
    plt.colorbar()
    draw_plt_rectangle(plt, work_rect)
    plt.title("F1")
    plt.show()

    F2_min = F2.min()
    F2_max = F2.max()
    F2_norm = (F2 - F2_min) / (F2_max - F2_min)
    plt.contourf(x, y, F2, 100, cmap='RdBu_r')
    plt.colorbar()
    draw_plt_rectangle(plt, work_rect)
    plt.title("F2")
    plt.show()

    plt.contourf(x, y, np.maximum(F1_norm, F2_norm), 100, cmap='RdBu_r')
    plt.colorbar()
    draw_plt_rectangle(plt, work_rect)
    plt.title("F1 F2 maximum")
    plt.show()


def draw_plt_rectangle(plt, work_rect):
    plt.plot((work_rect[0], work_rect[0]), (work_rect[1], work_rect[3]), color="green")
    plt.plot((work_rect[2], work_rect[2]), (work_rect[1], work_rect[3]), color="green")

    plt.plot((work_rect[0], work_rect[2]), (work_rect[1], work_rect[1]), color="green")
    plt.plot((work_rect[0], work_rect[2]), (work_rect[3], work_rect[3]), color="green")


L1_neu = 1495
L2_neu = 1140
L1_range = (1400, 1550)
L2_range = (1100, 1200)

j1 = (L2_neu, 0)
j2 = (0, L1_neu)
x1_offset = (-20, 40)
y2_offset = (-75, 40)

work_rect = (j1[0] + x1_offset[0], j2[1] + y2_offset[0], j1[0] + x1_offset[1], j2[1] + y2_offset[1])

enum_show_space_and_force(j1, j2, L1_range, L2_range, work_rect, Fh=6.0, Fv=1.5)





"""
L1_neu = 700      # 垂向电缸中立长度
L2_neu = 450      # 侧向电缸中立长度
Fh = 6.0          # 侧向最大受力
Fv = 1.5          # 垂向最大受力
dx_range = (-20, 40)    # 侧向变形范围
dy_range = (-75, 40)    # 垂向变形范围
step = 1

x_neu = L2_neu
y_neu = L1_neu

x = np.arange(dx_range[0], dx_range[1], step) + x_neu
y = np.arange(dy_range[0], dy_range[1], step) + y_neu
xy = np.array(np.meshgrid(x, y))

j1 = np.array((L2_neu, 0))
j2 = np.array((0, L1_neu))
L1, L2 = calc_L1_L2(j1, j2, xy)
sin1, cos1, sin2, cos2 = calc_sin_cos(j1, j2, xy, L1, L2)

F11, F21 = np.fabs(calc_F1_F2(Fh, Fv, sin1, cos1, sin2, cos2))
F12, F22 = np.fabs(calc_F1_F2(-Fh, Fv, sin1, cos1, sin2, cos2))
F13, F23 = np.fabs(calc_F1_F2(Fh, -Fv, sin1, cos1, sin2, cos2))
F14, F24 = np.fabs(calc_F1_F2(-Fh, -Fv, sin1, cos1, sin2, cos2))

F1 = np.maximum(F11, F12)
F1 = np.maximum(F1, F13)
F1 = np.maximum(F1, F14)

F2 = np.maximum(F21, F22)
F2 = np.maximum(F2, F23)
F2 = np.maximum(F2, F24)

print("L1 range = %1.1f, %1.1f" % (L1.min(), L1.max()))
print("L2 range = %1.1f, %1.1f" % (L2.min(), L2.max()))
print("F1 max = %1.1f" % F1.max())
print("F2 max = %1.1f" % F2.max())

plt.contourf(xy[0, :, :] - x_neu, xy[1, :, :] - y_neu, F1, 100, cmap='RdBu')
plt.colorbar()
plt.title("F1")
plt.show()

plt.contourf(xy[0, :, :] - x_neu, xy[1, :, :] - y_neu, F2, 100, cmap='RdBu')
plt.colorbar()
plt.title("F2")
plt.show()
"""