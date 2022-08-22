import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from numpy import polyfit, poly1d
from scipy.linalg import lstsq
from scipy.stats import linregress
#pathread  = r"D:\Scientific Research\Programs\python\DataProcess\PythonDataProcess\data\1.dat"
filepath = 'D:\Scientific Research\Programs\python\DataProcess\PythonDataProcess\data\\'
namelist = os.listdir(filepath)
#print(type(namelist[1]))
i = 0
while i < len(namelist):
#while i < 1: #len(namelist):
    pathread = filepath+namelist[i]
    #data = np.loadtxt(pathread,usecols=(0,1,2,3,4,5,6,7,8,9))
    data = np.loadtxt(pathread)

    x = data[:,6]
    y = data[:,7]
    y1 = poly1d(polyfit(x,y,2))
    plt.figure(1)
    p1 = plt.plot(x, y, 'rx')
    p2 = plt.plot(x, y1(x))
    print (y1)

    plt.figure(2)
    delta_y = y1(x)-y
    p1 = plt.plot(x, delta_y, 'rx')

    plt.figure(3)
    plt.boxplot(x = delta_y, # 指定绘制箱线图的数据
            whis = 1.5, # 指定1.5倍的四分位差
            widths = 0.7, # 指定箱线图的宽度为0.8
            patch_artist = True, # 指定需要填充箱体颜色
            showmeans = True, # 指定需要显示均值
            boxprops = {'facecolor':'steelblue'}, # 指定箱体的填充色为铁蓝色
            # 指定异常点的填充色、边框色和大小
            flierprops = {'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':4}, 
            # 指定均值点的标记符号（菱形）、填充色和大小
            meanprops = {'marker':'D','markerfacecolor':'black', 'markersize':4}, 
            medianprops = {'linestyle':'--','color':'orange'}, # 指定中位数的标记符号（虚线）和颜色
            labels = [''] # 去除箱线图的x轴刻度值
            )


    # 计算下四分位数和上四分位
    Q1 = np.quantile(delta_y, .25)
    Q3 = np.quantile(delta_y, .75)
    #Q1 = delta_y.quantile(q = 0.25)
    #Q3 = delta_y.quantile(q = 0.75)

    # 基于1.5倍的四分位差计算上下须对应的值
    low_whisker = Q1 - 1.5*(Q3 - Q1)
    up_whisker = Q3 + 1.5*(Q3 - Q1)

    # 寻找异常点
    #error_y = data[(delta_y < up_whisker) & (delta_y > low_whisker)]

    pathwrite = filepath + "washed\\" + namelist[i]
    np.savetxt(pathwrite,data[(delta_y < up_whisker) & (delta_y > low_whisker)],delimiter=" ",fmt="%4d%3d%3d%3d%3d%7.3f%13.7f%13.7f%13.7f%13.7f%11.3f%11.3f%20.3f%20.3f%10d%10d%13.6f%13.6f%10.3f%5d%10d%12.3f")

    #plt.show()
    i = i + 1