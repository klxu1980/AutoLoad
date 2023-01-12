import numpy as np
import time


ellipse = np.array((1.1, 2.2, 3.3))
row = ["%d" % value for value in ellipse]
row.append("0.01")

print(row)