from matplotlib import pyplot as plt
import numpy as np
import make_classification as mc

u = 1

X,labels, testX, testLabels, a = mc.make_classification(2,50,u)
colors = np.where(labels > 0, 'r','b')

x_plot = np.linspace(-u,u,100)
y_plot = -a[0]/a[1] * x_plot
plt.plot(x_plot,y_plot)
plt.scatter(
    X[:,0], X[:,1],
    c = colors)
plt.ylim(-u,u)
plt.title("sample plot")

plt.show()