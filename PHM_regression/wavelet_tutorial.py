# -*- coding: utf-8 -*-
import numpy as np
import pywt
import matplotlib.pyplot as plt

# dwt
x = np.linspace(-5,5,100)
y = np.sin(x)
(cA, cD) = pywt.dwt(y, 'db1')

A2,D2,D1 = pywt.wavedec(y,'db4',mode='symmetric',level=2)

plt.subplot(311)
plt.plot(A2)

plt.subplot(312)
plt.plot(D2)

plt.subplot(313)
plt.plot(D1)

plt.show()

print(A2.shape,D2.shape)