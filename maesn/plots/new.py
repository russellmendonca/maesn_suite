import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0,12,size=(12,4))
y = np.random.randint(0,8,size=(12,4))

fig, (ax, ax2, ax3) = plt.subplots(1, 3, sharex=True, figsize=(15,5))

l, = ax.plot(x[:,0],y[:,0], marker = 'o', label='1')
l2, =ax2.plot(x[:,1],y[:,1], marker = 'o', label='2',color='r')
l3, =ax2.plot(x[:,2],y[:,2], marker = 'o', label='3',color='turquoise')
l4, =ax3.plot(x[:,3],y[:,3], marker = 'o', label='4',color='g')


plt.legend( handles=[l, l2, l3, l4],loc="upper left", bbox_to_anchor=[-2, 0],
           ncol=4,  title="Legend", fancybox=True)

plt.show()
