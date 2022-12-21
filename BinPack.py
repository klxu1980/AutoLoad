import binpacking
import math

"""
b = { '1-1': 4.2,
      '1-2': 4.2,
      '2-1': 18.2,
      '2-2': 18.2,
      '2-3': 18.2,
      '3': 13.6,
      '4': 22.7,
      '5': 36.4,
      '6': 13.6,
      '7': 22.7,
      '8': 22.7,
      '9': 4.5,
      '10': 4.5,
      '11-1': 16.7,
      '11-2': 16.7,
      '11-3': 16.7,
      '12': 13.6,
      '13': 9.0,
      '14-1': 30.3,
      '14-2': 30.3,
      '14-3': 30.3,
      '15': 18.2}
"""
b = { '1': 22.7,
      '2-1': 9.0,
      '2-2': 9.0,
      '2-3': 9.0,
      '3-1': 4.5,
      '3-2': 4.5,
      '3-3': 4.5,
      '3-4': 4.5}


bins = binpacking.to_constant_bin_number(b, 3) # 4 being the number of bins
print("A: \n", bins[0])
print("B: \n", bins[1])
print("C: \n", bins[2])

Ia = 0
for _, I in enumerate(bins[0]):
    Ia += bins[0][I]

Ib = 0
for _, I in enumerate(bins[1]):
    Ib += bins[1][I]

Ic = 0
for _, I in enumerate(bins[2]):
    Ic += bins[2][I]

Iavg = (Ia + Ib + Ic) / 3.0
Ca = math.fabs(Ia - Iavg)
Cb = math.fabs(Ib - Iavg)
Cc = math.fabs(Ic - Iavg)

Conv = max(max(Ca, Cb), Cc) * 100.0 / Iavg
print("Ia = %1.2f, Ib = %1.2f, Ic = %1.2f" % (Ia, Ib, Ic))
print(Conv)