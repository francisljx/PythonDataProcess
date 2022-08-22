import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from scipy.linalg import lstsq
from scipy.stats import linregress
pathread  = r"D:\Scientific Research\Programs\python\DataProcess\PythonDataProcess\data\1.dat"
data = np.loadtxt(pathread,usecols=(0,1,2,3,4,5,6,7,8,9))
pathwrite = r"D:\Scientific Research\Programs\python\DataProcess\PythonDataProcess\data\text1.dat"
np.savetxt(pathwrite,data,delimiter=" ",fmt="%4d%3d%3d%3d%3d%7.3f%13.7f%13.7f%13.7f%13.7f")

x = data[:,6]
y = data[:,7]
y1 = poly1d(polyfit(x,y,2))
plt.figure(1)
p1 = plt.plot(x, y, 'rx')
p2 = plt.plot(x, y1(x))
print (y1)
#plt.show()

plt.figure(2)
delta_y = y1(x)-y
p1 = plt.plot(x, delta_y, 'rx')
plt.show()

