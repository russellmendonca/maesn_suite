import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = range(100)
    
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
plt.title("randSparsePusher-trainedOn100")

vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
#vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/SparsePusher-vpg-val1-rate1/", "Task", 100)

trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-SparseOrig-Batch2000-val1-rate0.01/", "Task", 100)

maml_val_subsample = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-trainedOn100-origSparse-meta20-fbs20-itr499-batch2000/","", 100)
#maml_val_direct = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest-trainedOn100-direct-fbs20-itr499/","",10)

maml_ADA = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlSparsePusher-orig-Adaptive-trainedOn100-meta20-fbs20-itr280-batch2000-initRate0.1/","",100)




#masen_val_ldim2_1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl0.5-#itr399/","", 100)
masen_linit = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-algo2-trainedOn100-meta20-ldim2-kl0.5-itr399/","", 100)
masen_STDprior = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-algo2-trainedOn100-meta20-ldim2-kl0.5-itr399-StdPrior/","", 100)

#masen_val2_complete =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-algo2-trainedOn100-meta20-ldim2-kl0.5-#itr399-StdPrior/","", 100) 



LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-origSparse-trainedOn100-meta20-ldim2-kl0.01-itr350/","",100)


#masen_val_ldim2_kl0_1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl0.1-#itr399/","", 100)
#masen_val_ldim2_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl0.5-#itr399/","", 100)
#masen_val_ldim2_kl1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl1-#itr399/","", 100)
#masen_val_ldim2_kl2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl2-#itr399/","", 100)
#LS_baseline_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet-trainedOn100-meta20-ldim2-kl0-#itr400/","", 10)

import ipdb
ipdb.set_trace()



#plt.plot(np.arange(0,100), np.mean(maml_val_subsample, 0), label = "maml")
plt.plot(np.arange(0,100), np.mean(maml_ADA,0), label="mamlADA")

#_trainedOn100_subsample20_batch2000")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

plt.plot(np.arange(0,100), np.mean(trpo,0), label='trpo')
#plt.plot(np.arange(0,100), np.mean(vpg_val_1,0), label='vpg_val_Batch20000')



#plt.plot(np.arange(0,100), np.mean(masen_val2_complete, 0), label = "masen_val_trainedOn100_ldim2_1")
plt.plot(np.arange(0,100), np.mean(masen_STDprior, 0), label = "masen_StPrior")
#plt.plot(np.arange(0,100), np.mean(masen_linit, 0), label = "masen_learnedInit")
plt.plot(np.arange(0,100), np.mean(LS, 0), label = "LS")


#plt.plot(np.arange(0,100), np.mean(lsBase_kl0, 0), label = "LS_kl0")
#plt.plot(np.arange(0,100), np.mean(lsBase_kl0_01, 0), label = "LS_kl0.01")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim2_3, 0), label = "masen_val_trainedOn100_ldim2_3")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim10_kl0_05, 0), label = "masen_val_trainedOn100_ldim10_subsample20")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim2_kl2, 0), label = "masen_val_trainedOn100_ldim2_subsample20")

#plt.plot(np.arange(0,10), np.mean(LS_baseline_trainedOn100_ldim2, 0), label = "LS_Baseline_trainedOn100_ldim2_subsample20")

plt.legend()
plt.savefig("randSparsePusher-trainedOn100-valSet1-final-100Tasks.png")
    



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
