from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
import pickle
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
ax.set_xlim(-0.5,0.5)
ax.set_ylim(-0.5, 0.5)

fobj = open("latentLogger.pkl", "rb")
LatentData = pickle.load(fobj)
import ipdb
ipdb.set_trace()
for i in range(100):
    lm = LatentData[i][0]
    lstd = LatentData[i][1]
    #for j in range(self.num_total_tasks):    
    e = Ellipse(xy=lm, width=lstd[0], height=lstd[1], fill = False)
    ax.add_artist(e)
plt.savefig("trpo_withNoise_latents.png")
