import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    #numgoals = 100
    
    origHis = []
    for goal in range(0,50):
        
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
    #import ipdb
    #ipdb.set_trace()
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
plt.title("randPusher-trainedOn100-valSet1")

trpo_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-TRPO-Batch2000-val1-rate1/", "Task", 100)
vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPG-Batch2000-val1-rate1/", "Task", 100)

trpoSPARSE = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-SparseOrig-Batch2000-val1-rate0.01/", "Task", 100)

maml_val_subsample = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher/MamlPusher-valTest1-batch2000-trainedOn100-meta20-fbs20-itr499/","", 10)
#maml_val_direct = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest1-trainedOn100-direct-fbs20-itr499/","",10)
maml_val_subsample2 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher/MamlPusher-valTest1-batch2000-lr0.5-trainedOn100-meta20-fbs20-itr499/","", 10)




masen_val_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenNormalBlocks/MasenPusher-ValSet1-trainedOn100-meta20-ldim2-kl0.05-itr400/","", 10)


import ipdb
ipdb.set_trace()

plt.plot(np.arange(0,10), np.mean(maml_val_subsample, 0), label = "maml_val_trainedOn100_subsample20_lr1")
plt.plot(np.arange(0,10), np.mean(maml_val_subsample2, 0), label = "maml_val_trainedOn100_subsample20_lr0.5")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")

plt.plot(np.arange(0,100), np.mean(trpo_val, 0), label="trpo_val_batch2000")

plt.plot(np.arange(0,100), np.mean(trpoSPARSE,0) , label="trpoSPARSE")

plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg_val_batch2000')
#plt.plot(np.arange(0,100), np.mean(vpg_val_1,0), label='vpg_val_batch20000')



plt.plot(np.arange(0,10), np.mean(masen_val_trainedOn100_ldim2, 0), label = "masen_val_trainedOn100_ldim2_subsample20")
#plt.plot(np.arange(0,100), np.mean(LS_baseline_trainedOn100_ldim2, 0), label = "LS_Baseline_trainedOn100_ldim2_subsample20")
#plt.plot(np.arange(0,10), np.mean(LS_baseline_trainedOn100_ldim10, 0), label = "LS_Baseline_trainedOn100_ldim10_subsample20")
#plt.plot(np.arange(0,100), np.mean(LS_baseline_trainedOn100_ldim2_Direct, 0), label = "LS_Baseline_trainedOn100_ldim2_Direct")

plt.legend()
plt.savefig("pusherANALYSIS.png")
    



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
