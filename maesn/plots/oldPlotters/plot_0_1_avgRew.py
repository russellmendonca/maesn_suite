import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#goalList = range(0, 25)
filePrefix ="/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/maml-randommazes/trpomaml_flr_0.5metalr_0.01_step11kl_weight2latent_dim4/"


with open(filePrefix+"progress.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    
    row = None
    records = []
    for row in reader:
        records.append(row)
    ret0Ind, ret1Ind = 0,0  
    for i in range(len(records[0])):
        if records[0][i]=="0AverageReturn":
            ret0Ind = i
    for j in range(len(records[0])):
        if records[0][j]=="1AverageReturn":
            ret1Ind = j
       
    
    ret0 , ret1= [], []
    for record in records[1:]:
        ret0.append(record[ret0Ind])
        ret1.append(record[ret1Ind])
   
    _iter = np.arange(0, 150)

    plt.title("rMaze_kl2_flr0.5")
    plt.plot(_iter, ret0[:150], '-r', label='0AverageReturn')
    plt.plot(_iter, ret1[:150], '-b', label = "1AverageReturn")
    

plt.legend()
plt.savefig("rMaze_kl2_flr0.5.png")

       # avg_returns.append(returns)
