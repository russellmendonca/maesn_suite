import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr, rewardMarker = "AverageReturn"):
    
    goals = list(range(0,100))
    
    #LS
    goals.remove(40)
    goals.remove(41)
    goals.remove(54)
    goals.remove(99)
    
    
    ##VPG missing goals
    goals.remove(11)
    goals.remove(48)
    goals.remove(69)
    goals.remove(97)
    
    #Vime missing goals
    goals.remove(51)
    
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
                if records[0][i]==rewardMarker:
                    ret0Ind = i
               
            
            ret0 = []
            for record in records[1:]:
                ret0.append(float(record[ret0Ind]))
                #ret1.append(record[ret1Ind])
           
            origHis.append(ret0[:num_itr])
        
    return origHis

def buildRL2(_file, total):
    fobj = open(_file, "rb")
    list1 = pickle.load(fobj)
    _len = len(list1)
    last = list1[_len -1]
    for i in range(_len, total):
        list1 = np.concatenate([list1, [last]])
    return list1
    


#plt.ylim(-1000, -700)
plt.title("Robotic Manipulation")

LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/LSBaselinePusher-Ind2New-ldim2-kl0.01-itr350/", "", 100)


vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/Pusher-VPGSparseInd2-Batch2000-val1-rate1/", "Task", 100)

trpo_10 = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-Sparse-wrtGoal0.1-Batch2000-val1-rate0.01/", "Task", 100)
trpo_20 = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-Sparse-wrtGoal0.2-Batch2000-val1-rate0.01/", "Task", 100)

theanoTRPO = plot("/home/russellm/rllab-vime/data/s3/TheanoTRPO-Pusher0.01/","Task",100)


vime = plot("/home/russellm/rllab-vime/data/s3/VIME-Pusher2-expl-eta0.0001-seed0/", "Task", 100)


trpo_10_oracle = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-Pusher-benchMarks-for-Indicator0.01/", "Task", 100, "AverageReturnSparse1")
trpo_20_oracle = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-Pusher-benchMarks-for-Indicator0.01/", "Task", 100, "AverageReturnSparse2")


masen_S10 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.1-ldim2-kl0.5/", "", 100)
masen_S15 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.15-ldim2-kl0.5/", "", 100)
masen_S20 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.20-ldim2-kl0.5/", "", 100)
masen_S25 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.25-ldim2-kl0.5/", "", 100)



masenTRPOImprove = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/masenTRPOImprovement-PusherIndicator-val1-rate0.01/", "Task", 100)
mamlTRPOImprove = plot("/home/russellm/new_maml_rl_baseline/data/local/mamlBiasBiasADATRPOImprovement-PusherIndicator-val1-rate0.01/", "Task", 100)

mamlBiasBiasADA = plot("/home/russellm/new_maml_rl_baseline/data/local/MamlTest-Bias-BiasADA-Pusher0-2-halfStep-trainedOn100-itr499/", "", 100)

rl2 = buildRL2("/home/russellm/rllab-private-RL2unstable/examples/RL2_Pusher_TrainThresh_TestInd.pkl", 100)

#mamlBiasFullADA = plot("/home/russellm")

def plotMean(startItr, endItr, var, label, marker=None):

    plt.plot(np.arange(startItr,endItr), np.mean(var,0), label=label, marker=marker, markersize=8, markevery=10)
   # plt.fill_between(np.arange(startItr,endItr), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )



plotMean(0, 100, masen_S20, "MAESN","d")
plotMean(0,100, LS, "LatentSpace", "P")
plotMean(0, 100, mamlBiasBiasADA, "MAML", "s")
#plotMeanStd(3, 100, np.array(masenTRPOImprove)[:,:97], "masenTRPO")


plotMean(0,100, vime, "VIME", "^")
plotMean(0, 100, trpo_20, "TRPO", "h")
plotMean(0,100, vpg, "VPG", "x")
plt.plot(np.arange(0,100), rl2, label="RL2",marker="*", markersize=8 ,markevery=10)

#plotMeanStd(3, 100, np.array(mamlTRPOImprove)[:,:97], "masenTRPO")
plt.ylabel("Average Return")
plt.xlabel("Number of Iterations")


#plt.legend()
plt.savefig("randSparsePusher-indicatorTest.png")
    



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
