import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(2)

# x/domain
x = np.arange(0, np.pi*2, np.pi/40)

# co-domain

# cos
cos_x = np.cos(x)

# dot product
a =  2
b = 3

d_prod = abs(a) * abs(b) * cos_x

plt.plot(x, cos_x, label='y = cos(x)')
plt.plot(x, d_prod, label='y = |a| . |b| . cos(x)')


plt.grid()
plt.legend()
plt.show()

