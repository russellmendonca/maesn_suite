import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = range(2,80)
    
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

def loadRL2(_file, total):
     fobj = open(_file, "rb")
     list1 = pickle.load(fobj)
     _len = len(list1)
     last = list1[_len -1]
     for i in range(_len, total):
         list1 = np.concatenate([list1, [last]])
     return list1
    


#plt.ylim(-1000, -700)
plt.title("WheeledRobot")

#vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
#vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/SparsePusher-vpg-val1-rate1/", "Task", 100)

trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goalsVal0.01radius2/", "Task", 100)


maml2 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-initlr0.1-lr0.05-proper-trainedOn100-itr450/", "", 50)
mamlHalfStep_450 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-lr0.1-halfStep-trainedOn100-itr450/", "", 50)
mamlHalfStep_163 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-lr0.1-halfStep-trainedOn100-itr163/","", 100)

mamlHalfStep_163_TrainSet = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTrainSetTest-wheeledRad2-lr0.1-halfStep-trainedOn100-itr163/","", 20)
maml1 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-lr0.1-proper-trainedOn100-itr450/", "", 50)

LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaseline-wheeled-trainedOn100-meta20-Val-ldim2-kl0.01-itr320/", "", 50)




rl2 = loadRL2("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/RL2_testResults/RL2_test_wheeled_Dense.pkl", 100)
#maml = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-trainedOn20-itr440/", "", 50)


#MamlPusher-trainedOn100-origSparse-meta20-fbs20-itr499-batch2000/","", 100)
#maml_val_direct = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest-trainedOn100-direct-fbs20-itr499/","",10)

##maml_ADA = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlSparsePusher-orig-Adaptive-trainedOn100-meta20-fbs20-itr280-batch2000-initRate0.1/","",100)
masenTestOnDense = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenWheeledTestDense-Radius2-trainedOn100-ldim2-kl0.5-itr300/", "", 50)
masenTestOnSparse = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenWheeledTestSparse1-Radius2-trainedOn100-ldim2-kl0.5-itr300/", "", 50)

masenDenseSparse = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/DenseSparse-MasenWheeledTestSparse1-Radius2-trainedOn100-meta20-ldim2-kl0.1-itr225/", "", 15)




#masen_val2_complete =plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-algo2-trainedOn100-meta20-ldim2-kl0.5-#itr399-StdPrior/","", 100) 



#LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet1-origSparse-trainedOn100-meta20-ldim2-kl0.01-itr350-PROPER100itr-#algo2/","",100)


#masen_val_ldim2_kl0_1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl0.1-#itr399/","", 100)
#masen_val_ldim2_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl0.5-#itr399/","", 100)
#masen_val_ldim2_kl1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl1-#itr399/","", 100)
#masen_val_ldim2_kl2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet1-normalSparse-trainedOn100-meta20-ldim2-kl2-#itr399/","", 100)
#LS_baseline_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet-trainedOn100-meta20-ldim2-kl0-#itr400/","", 10)

import ipdb
ipdb.set_trace()



#plt.plot(np.arange(0,100), np.mean(maml_val_subsample, 0), label = "maml")
#plt.plot(np.arange(0,50), np.mean(maml,0), label="maml")

#_trainedOn100_subsample20_batch2000")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

plt.plot(np.arange(0,100), np.mean(trpo,0), label='trpo_Dense')

plt.plot(np.arange(0,100), rl2, label="rl2")

#plt.plot(np.arange(0,50), np.mean(maml1,0) ,label = "maml_lr0.1")
#plt.plot(np.arange(0,50), np.mean(maml2,0) ,label = "maml_lr0.05")
plt.plot(np.arange(0,50), np.mean(mamlHalfStep_450,0) ,label = "mamlHalfStep_450")
plt.plot(np.arange(0,100), np.mean(mamlHalfStep_163,0) ,label = "mamlHalfStep_163")
plt.plot(np.arange(0,20), np.mean(mamlHalfStep_163_TrainSet,0) ,label = "mamlHalfStep_163_TrainSet")

plt.plot(np.arange(0,50), np.mean(LS, 0), label = "latentSpace")



plt.plot(np.arange(0,50), np.mean(masenTestOnDense, 0), label = "masen_TrainOnDense_TestOnDense")


#plt.plot(np.arange(0,100), np.mean(lsBase_kl0, 0), label = "LS_kl0")
#plt.plot(np.arange(0,100), np.mean(lsBase_kl0_01, 0), label = "LS_kl0.01")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim2_3, 0), label = "masen_val_trainedOn100_ldim2_3")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim10_kl0_05, 0), label = "masen_val_trainedOn100_ldim10_subsample20")
#plt.plot(np.arange(0,100), np.mean(masen_val_ldim2_kl2, 0), label = "masen_val_trainedOn100_ldim2_subsample20")

#plt.plot(np.arange(0,10), np.mean(LS_baseline_trainedOn100_ldim2, 0), label = "LS_Baseline_trainedOn100_ldim2_subsample20")

plt.legend()
plt.savefig("WheeledRad2-trainedOn100-Dense.png")
    



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
