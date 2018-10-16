import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    
    goals = [1,2,3,6,7,8]
    origHis = []
    for goal in goals:
        
        with open(filePrefix+string+str(goal)+"/progress.csv", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            
            row = None
            records = []
            for row in reader:
                records.append(row)
            ret0Ind = 0 
            for i in range(len(records[0])):
                if records[0][i]=="AverageReturn":
                    ret0Ind = i
               
            
            ret0 = []
            for record in records[1:]:
                ret0.append(float(record[ret0Ind]))
                #ret1.append(record[ret1Ind])
           
            origHis.append(ret0[:num_itr])
        
    return origHis

# def buildRL2(_file, total):
#     fobj = open(_file, "rb")
#     list1 = pickle.load(fobj)
#     _len = len(list1)
#     last = list1[_len -1]
#     for i in range(_len, total):
#         list1 = np.concatenate([list1, [last]])
#     return list1
    


#plt.ylim(-1000, -700)
plt.title("SparseBlockPushTest-OptimalRate-ada")

vpg = plot("/home/russellm/rllab_iclr_Baselines/data/local/BlockPush-sparse-vpg-single-rate0.3/", "Task", 50)
#trpo = plot("/home/russellm/rllab_iclr_Baselines/data/local/iclr_exp_info/BlockPushSparse/blockpush-from-Scratch/","Task", 10, 50)
maml = plot("/home/russellm/rllab_iclr_Baselines/data/local/iclr_exp_info/BlockPushSparse/blockpush-maml-test/","test", 50)
ls= plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/SparseblockpushTest-Dec-OptimalRate-adaH-ldim6-kl0.001-itr599/", "", 10)

masen = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/SparseblockpushTest-Dec-OptimalRate-adaH-ldim6-kl0.01step8to0.1-itr599/","", 10)

import ipdb
ipdb.set_trace()

plt.plot(np.arange(0,50), np.mean(vpg, 0),label = "vpg")
plt.plot(np.arange(0,50), np.mean(maml, 0), label = "Plain_MAML")
plt.plot(np.arange(0,10), np.mean(ls, 0), label="Only LS")
plt.plot(np.arange(0,10), np.mean(masen, 0), label = "MASEN_ls")
plt.legend()
plt.savefig("Sparse_BlockPush.png")
    



#plt.plot(Iterations, plainMean, '-r', label  = 'Normal Obs state')
# plt.plot(Iterations, addedNoiseMean, '-b', label = 'Noise added to obs state')

# plt.fill_between(Iterations, plainMean-plainStd, plainMean+plainStd,facecolor='r',alpha=0.5)
# plt.fill_between(Iterations, addedNoiseMean-addedNoiseStd, addedNoiseMean+addedNoiseStd,facecolor='b',alpha=0.5)



# Iterations = range(0,100)
# import ipdb
# ipdb.set_trace()
# Mean = np.mean(History, axis = 0)
# Std = np.std(History, axis = 0)

# plt.plot(Iterations, Mean, '-r', label  = 'Normal Obs state')
# plt.fill_between(Iterations, Mean-Std, Mean+Std,facecolor='r',alpha=0.5)
# plt.savefig("Block_pusher_trpo.png")

       # avg_returns.append(returns)
