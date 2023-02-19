import numpy as np

offset = 2
row_cnt = offset * offset * 4

ellipse_x = np.arange(10, 15, 1)
ellipse_y = np.arange(20, 25, 1)
x_idx = np.tile(ellipse_x, row_cnt).reshape((row_cnt, ellipse_x.shape[0]))
y_idx = np.tile(ellipse_y, row_cnt).reshape((row_cnt, ellipse_y.shape[0]))

x_offset = np.repeat(np.arange(-offset, offset, 1), offset * 2 * x_idx.shape[1]).reshape(x_idx.shape)
y_offset = np.tile(np.arange(-offset, offset, 1), offset * 2)
y_offset = np.repeat(y_offset, x_idx.shape[1]).reshape(y_idx.shape)

x_idx += x_offset
y_idx += y_offset

m = np.arange(0, 900, 1).reshape((30, 30))
print(m)
print("")
print(m[x_idx, y_idx])

for i in range(row_cnt):
    print("")
    print(x_idx[i])
    print(y_idx[i])
    print(m[x_idx[i], y_idx[i]])

        # img = image[x_idx, y_idx]
        # row_norm = np.linalg.norm(img, ord=2, axis=1)
        # max_row = np.argmax(row_norm)
        #
        # x = x_offset[max_row, 0]
        # y = y_offset[max_row, 0]