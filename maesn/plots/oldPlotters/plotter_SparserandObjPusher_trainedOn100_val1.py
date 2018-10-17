import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = range(1,50)
    
    origHis = []
    for goal in goals:
        
        with open(filePrefix+string+str(goal)+"/progress.csv", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            
            row = None
            records = []
            for row in reader:
                records.append(row)
            ret0Ind = 0 
            #import ipdb
            #ipdb.set_trace()
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
#     fobj = open(_file, "rb")generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-fancySparse-trainedOn100-meta20-ldim10-kl1-itr450
#     list1 = pickle.load(fobj)
#     _len = len(list1)
#     last = list1[_len -1]
#     for i in range(_len, total):
#         list1 = np.concatenate([list1, [last]])
#     return list1
    


#plt.ylim(-1000, -700)
plt.title("randSparsePusher-trainedOn100")

vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/randObjPusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/randObjPusher-VPGSparse-Batch20000-val1-rate1/", "Task", 100)
trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-SparseFancy-Batch2000-val1-rate0.01/", "Task", 100)
#maml_ada = plot("/home/russellm/maml_rl_baseline_test/data/local/randObj-MamlSparsePusher-Adaptive-trainedOn100-meta20-fbs20-itr499-batch2000-initRate0.1/", "", 10)
#maml_ada_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/randObj-MamlSparsePusher-Adaptive-trainedOn100-meta20-fbs20-itr499-batch2000-initRate0.5/", "", 10)
maml = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-trainedOn100-fancySparse-meta20-fbs20-itr499-batch2000/","", 100)

#lsBaseLine_ldim5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-trainedOn100-meta20-ldim5-kl0.01-itr275/", "",10)
#lsBaseLine_ldim8 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-trainedOn100-meta20-ldim8-kl0.01-itr275/", "",10)
#lsBaseLine_ldim5_kl0 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-trainedOn100-meta20-ldim5-kl0-itr275/", "",10)
#lsBaseLine_ldim8_kl0 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-trainedOn100-meta20-ldim8-kl0-itr275/", "",10)
#maml_20000 = plot("/home/russellm/maml_rl_baseline_test/data/local/randObj-MamlSparsePusher-trainedOn100-meta20-fbs20-itr499-batch20000/","", 10)

#masen_ldim2_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-trainedOn100-meta20-ldim2-kl0.5-itr275/", "", 10)
masen_ldim10_kl1 =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-fancySparse-trainedOn100-meta20-ldim10-kl1-itr450/", "",10)
masen_ldim8_kl1 =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-fancySparse-trainedOn100-meta20-ldim8-kl1-itr450/", "",10)
masen_ldim5_kl1 =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-fancySparse-trainedOn100-meta20-ldim5-kl1-itr450/", "",10)
masen_ldim2_kl1 =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-fancySparse-trainedOn100-meta20-ldim2-kl1-itr450/", "",10)
#masen_ldim5_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-trainedOn100-meta20-ldim5-kl0.5-itr275/", "", 10)
#masen_ldim2_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-trainedOn100-meta20-ldim2-kl0.5-itr275/", "", 10)



#masen_val_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenSparsePusher-ValSet1-trainedOn100-meta20-ldim2-kl0.05-#itr400/","", 10)
#LS_baseline_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet-trainedOn100-meta20-ldim2-kl0-#itr400/","", 10)

import ipdb
ipdb.set_trace()



#plt.plot(np.arange(0,10), np.mean(maml_val_subsample, 0), label = "maml_val_trainedOn100_subsample20")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg_val_Batch2000')
plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg_val_Batch2000')
plt.plot(np.arange(0,100), np.mean(vpg_val_1,0), label='vpg_val_Batch20000')
plt.plot(np.arange(0,100), np.mean(trpo,0), label='trpo_batch2000')

plt.plot(np.arange(0,100), np.mean(maml, 0), label = "maml")
#plt.plot(np.arange(0,10), np.mean(maml_ada, 0), label = "maml_ada_0.1")
#plt.plot(np.arange(0,10), np.mean(maml_ada_1, 0), label = "maml_ada_0.5")
#plt.plot(np.arange(0,10), np.mean(maml_20000, 0), label = "maml_20000")


#plt.plot(np.arange(0,10), np.mean(masen_ldim10_kl1, 0), label = "masen_ldim10_kl1")
plt.plot(np.arange(0,10), np.mean(masen_ldim8_kl1, 0), label = "masen_ldim8_kl0.5")
#plt.plot(np.arange(0,10), np.mean(masen_ldim5_kl1, 0), label = "masen_ldim5_kl0.5")
#plt.plot(np.arange(0,10), np.mean(masen_ldim2_kl1, 0), label = "masen_ldim2_kl0.5")


#plt.plot(np.arange(0,10), np.mean(lsBaseLine_ldim5 , 0), label = "LS_Baseline_ldim5_kl0.01")
#plt.plot(np.arange(0,10), np.mean(lsBaseLine_ldim8 , 0), label = "LS_Baseline_ldim8_kl0.01")
#plt.plot(np.arange(0,10), np.mean(lsBaseLine_ldim5_kl0 , 0), label = "LS_Baseline_ldim5_kl0")
#plt.plot(np.arange(0,10), np.mean(lsBaseLine_ldim8_kl0 , 0), label = "LS_Baseline_ldim8_kl0")


plt.legend()
plt.savefig("randObjSparsePusher-trainedOn100-valSet1-50Tasks.png")
    



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
