import numpy as np
import matplotlib.pyplot as plt 


# np.set_printoptions(2)
# rand = np.random.randint

# SIMPLE PLOTTING LINE GRAPH WITH GRID
'''
x = np.arange(-10,10,0.1)
y = x**3 - (2 * (x**2)) + 3

plt.plot(x, y)
plt.grid()
plt.show()
'''

# LEDENDS, LABELS AND DECORATIONS
"""
x = np.arange(-3,3,0.1)
y = x**3 - (2 * (x**2)) + 3
y2 = x**2 + 3*x -1

plt.plot(x, y, label='Curve y1')
plt.plot(x, y2, label='Curve y2')
plt.legend()

plt.title('Charts of y1 and y2')
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.grid()

plt.show()
"""


# BARCHARTS AND HISTOGRAMS #

'''
x = [2,4,6,8,10]
y = [rand(20) for i in range(5)]

x2 = [1,3,5,7,9]
y2 = [rand(20) for i in range(5)]

# colors and legends for multiple bars
plt.bar(x,y, label='Bar1', color='r')
plt.bar(x2,y2, label='Bar2', color='b')
plt.legend()
plt.show()
'''
# histogram: used to condense a lot of data into smaller 'bins'
'''
ages = [rand(130) for i in range(200)]

bins = np.linspace(1,130,10)

plt.hist(ages, bins, histtype='bar', rwidth=0.95)

plt.show()
'''


# SCATTER PLOT #
'''
x = np.arange(-10,10,1)
y = [rand(100) for i in x]

# changeable MARKERS, SIZES & COLORS
plt.scatter(x,y, label='scatter-platter', color='k', s=200, marker="x")
plt.legend()
plt.show()
'''

# STACK PLOTS #
'''
days = [1,2,3,4,5]

sleeping = [5, 6, 5,12,10]
working =  [12,13,14,3,2]
playing =  [7, 5, 5, 9, 12]

y = [sleeping, working, playing]
# for the labels
y_labels = ['sleeping','working','playing']

# for colors
pal = ["b", "g", "r"]

plt.stackplot(days, y, colors=pal, labels=y_labels, alpha=0.4)

plt.legend()
plt.show()
'''
# stack plot 2
'''
x = np.arange(0, 40, 1)

y1 = [rand(9) for i in x]
y2 = [rand(15) for i in x]
y3 = [rand(7) for i in x]

y = [y1,y2,y3]

y_labels = ['y1','y2','y3']
pal = ["b", "g", "r"]

plt.stackplot(x, y, colors=pal, labels=y_labels, alpha=0.4)

plt.legend()
plt.show()
'''

# PIE CHARTS #
'''
values = [rand(33) for i in range(6)]
labels = ['A','B','C','D','E','F']

# explode: pull out a piece
# autopct: adds percentage
plt.pie(
  values, 
  labels=labels,
  autopct='%.2f%%',
  explode=(0,0,0.1,0,0,0),
  shadow=True,
  startangle=90
)

plt.legend()
plt.show()
'''

# WRITING AND LOADING DATA FROM FILE #

# writing/making a random csv file
'''
f = open('./sample1.csv', 'w')

out = ''
for i in range(20):
  for j in range(4):
    out += str(rand(20)) + ','
  out = out[:-1] + '\n'
out = out[:-1]
  
f.write(out)
f.close()
'''

# reading from file csv-module / numpy
'''
import csv
import os
filePath = [os.path.realpath(os.path.join('.',f)) for f in os.listdir('.') if os.path.isfile(f) and f == 'sample.csv']
# with open(filePath,'rb') as f:
#     reader = csvkit.reader(f)
#     print reader

file = os.path.abspath('.') + '/sample.csv'

f = open(file)
  
f.close()

# file = './sample.csv'
# f = open('./sample.csv', 'r')
# f.close()
'''


# 3D PLOTTING #

# first 3d example from lab class
'''
x = np.arange(-2,2,.01)
y = x

xv, yv = np.meshgrid(x, y)
fig = plt.figure()
ax = Axes3D(fig)

z = (np.multiply(x, np.exp(-xv ** 2 - yv ** 2)))
ax.plot_surface(xv, yv, z)
plt.show()

'''
# OR
'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

x = y = np.arange(-2,2,0.25)
xx,yy = np.meshgrid(x,y)

fig = plt.figure()
ax1 = axes3d.Axes3D(fig)


z = (x * np.exp(-xx ** 2 - yy ** 2))

ax1.plot_surface(xx,yy,z, rstride=1, cstride=1, cmap=cm.seismic, alpha=0.6)

plt.show()
'''

# plot default test data
'''
'''



from mpl_toolkits.mplot3d import axes3d

x,y,z = axes3d.get_test_data()

fig = plt.figure()

rst = 5
cst = 10

ax = fig.add_subplot(1,2,1, projection='3d')
ax.plot_surface(x,y,z, rstride=rst, cstride=cst)

ax = fig.add_subplot(1,2,2, projection='3d')
ax.plot_wireframe(x,y,z, rstride=rst, cstride=cst)


plt.show()


# plot complex surface plot : sin( sqrt(x^2 + y^2) ) in -5<=x<=5
'''

from matplotlib import cm
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection='3d')

Y = X = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cm.coolwarm)

fig.colorbar(surf, shrink=0.5, aspect=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')  

plt.show()
'''