import csv
import numpy as np
from matplotlib import pyplot as plt
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']
_file = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/blockpush-Baseline/fulltrpomaml1latent_dim2_fbs100_mbs10_flr_0metalr_0.01_step11kl_schemeNonekl_weighting0'


with open(_file+"/progress.csv", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    
    row = None
    records = []
    for row in reader:
        records.append(row)
    

    ret0Ind, ret1Ind = 0,0  
    for i in range(len(records[0])):
        if records[0][i]=="0AverageReturn":
            ret0Ind = i
        if records[0][i]=="1AverageReturn":
            ret1Ind = i

    
    ret0, ret1 = [], []
    for record in records[1:]:
        ret0.append(float(record[ret0Ind]))
        ret1.append(float(record[ret1Ind]))
        
Iterations = range(0,200)
plt.plot(Iterations, ret0, label = "Average Reward (Baseline)")
#plt.plot(Iterations, ret1, label = "Average1 Reward")


#History.append(ret0)

plt.legend()    
plt.savefig("blockPush_ldim_2_Baseline.png")


# Iterations = range(0,100)
# import ipdb
# ipdb.set_trace()
# Mean = np.mean(History, axis = 0)
# Std = np.std(History, axis = 0)

# plt.plot(Iterations, Mean, '-r', label  = 'Normal Obs state')
# plt.fill_between(Iterations, Mean-Std, Mean+Std,facecolor='r',alpha=0.5)
# plt.savefig("Block_pusher_trpo.png")

       # avg_returns.append(returns)