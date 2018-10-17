import csv
import numpy as np
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr, startGoal, numGoals):
    
    
    goals = range(startGoal,startGoal+numGoals)
    TotalAvg = [[0,0] for i in range(num_itr) ]
    for goal in goals:
        
        with open(filePrefix+string+str(goal)+"/progress.csv", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            
            row = None
            records = []
            for row in reader:
                records.append(row)
            ret0Ind, numTrajsInd = 0 ,0
            for i in range(len(records[0])):
                if records[0][i]=="AverageReturn":
                    ret0Ind = i
                if records[0][i]=="NumTrajs":
                    numTrajsInd = i               
            
            ret0, numTrajs = [], []
            for record in records[1:]:
                ret0.append(float(record[ret0Ind]))
                numTrajs.append(float(record[numTrajsInd]))
                #ret1.append(record[ret1Ind])
        #origHis.append(ret0[:num_itr])
            counter = 0
            for a,b  in zip(ret0[:num_itr], numTrajs[:num_itr]):
                TotalAvg[counter][0]+=a*b
                TotalAvg[counter][1]+=b
                counter+=1
        #print(np.shape(origHis))
    #import ipdb
    #ipdb.set_trace()
    finalResult = []
    for elem in TotalAvg:
        finalResult.append(elem[0]/elem[1])     
    return finalResult

# def buildRL2(_file, total):
#     fobj = open(_file, "rb")
#     list1 = pickle.load(fobj)
#     _len = len(list1)
#     last = list1[_len -1]
#     for i in range(_len, total):
#         list1 = np.concatenate([list1, [last]])
#     return list1
    


#plt.ylim(-1000, -700)
plt.title("Random Mazes")

#vpg = plot("/home/russellm/rllab_iclr_Baselines/data/local/BlockPush-sparse-vpg-single-rate0.3/", "Task", 50)
#trpo = plot("/home/russellm/rllab_iclr_Baselines/data/local/iclr_exp_info/BlockPushSparse/blockpush-from-Scratch/","Task", 10, 50)
#maml = plot("/home/russellm/rllab_iclr_Baselines/data/local/iclr_exp_info/BlockPushSparse/blockpush-maml-test/","test", 50)
#ls= plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/SparseblockpushTest-Dec-OptimalRate-adaH-ldim6-kl0.001-itr599/", "", 10)
#lr1_kl1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-Selftest-RandomMaze-vpg-lr-1-kl-1-itr-120seed1/", "goal_",20) 
#lr1_kl1 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-valtest-RandomMaze-vpg-lr-1-kl-1-itr-170seed1ldim4/", "goal_",10, 50, 20)
lr1_kl0_5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-valtest-RandomMaze-vpg-lr-1-kl-0.5-itr-200seed1ldim4/", "goal_",10, 50, 20)
#lr0_5_kl2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-valtest-RandomMaze-vpg-lr-0.5-kl-2-itr-170seed1ldim4/", "goal_",10, 50, 20)
#lr1_kl2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-valtest-RandomMaze-vpg-lr-1-kl-2-itr-160seed1ldim4/", "goal_",10, 50, 20)
maml = plot("/home/russellm/rllab_iclr_Baselines/data/local/MAMLtest-RandomMaze-vpg-lr-0.1/", "goal_", 10, 50 ,20)
vpg = plot('/home/russellm/rllab_iclr_Baselines/data/local/vpg-randommazes-vpglr-0.1/',"/Goal_", 10, 50, 20)

#lr1_kl2_StandardPrior_160 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-valtest-RandomMaze-vpg-lr-1-kl-2-itr-160seed1/", "goal_",10, 50, 20)
#lr1_kl2_metaLearnedPrior_160 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Masen-Selftest-RandomMaze-vpg-lr-1-kl-2-itr-160seed1/", "goal_",10, 0, 50)
#masen = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/SparseblockpushTest-Dec-OptimalRate-adaH-ldim6-kl0.01step8to0.1-itr599/","", 10)
#self.num_total_tasks,
#import ipdb
#ipdb.set_trace()

#plt.plot(np.arange(0,10), lr1_kl1,label = " masen_lr1_kl1")
plt.plot(np.arange(0,10), lr1_kl0_5,label = " masen_lr1_kl0.5")
#plt.plot(np.arange(0,10), lr0_5_kl2, label = " lr0.5_kl2")
#plt.plot(np.arange(0,10), lr1_kl2,label = " lr1_kl2")
plt.plot(np.arange(0,10), maml, label = "maml")
plt.plot(np.arange(0,10), vpg, label="vpg")
#plt.plot(np.arange(0,10), lr1_kl2_StandardPrior_160, label = "Standard prior on latents itr160")
#plt.plot(np.arange(0,20), np.mean(lr1_kl0_5, 0), label="lr1_kl0_5")
#plt.plot(np.arange(0,50), np.mean(masen, 0), label = "MASEN_ls")
plt.legend()
plt.savefig("randomMaze.png")
    



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
