import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = list(range(0,30))
    
    #LS
    goals.remove(12)
    goals.remove(26)
    goals.remove(29)
    #goals.remove(19)
    #goals.remove(64)
    #goals.remove(86)
    
    #Vime
    goals.remove(13)
    
    #VPG
    goals.remove(3)
    goals.remove(15)
    goals.remove(17)
    
    #Vpg
    
    #goals.remove(12)
    #goals.remove(21)
    #goals.remove(33)
    #goals.remove(39)
    #goals.remove(40)
    #goals.remove(69)
    
    
    
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

def buildRL2(_file, total):
    fobj = open(_file, "rb")
    list1 = pickle.load(fobj)
    _len = len(list1)
    last = list1[_len -1]
    for i in range(_len, total):
     list1 = np.concatenate([list1, [last]])
    return list1
plt.title("Legged Locomotion")

#vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
#vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/SparsePusher-vpg-val1-rate1/", "Task", 100)

#trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goalsVal0.01radius2/", "Task", 200)

#vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/Wheeled-VPGThresh1-Batch10000-val1-rate1/", "Task", 100)
vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/VPG-ant-Ind80.1/", "Task", 100)
vime = plot("/home/russellm/rllab-vime/data/s3/VIME-antInd8-expl-eta0.0001-seed0/","Task",100)
trpo = plot("/home/russellm/maml_rl_baseline_test/data/s3/TRPO-ant-Ind80.01/", "Task", 100)
masen=plot("/home/russellm/antdata/MasenTest-AntDense-Indicator8/", "", 100)
maml = plot("/home/russellm/antdataMaml/MamlTest-Bias-fullADA-Indicator8-Ant/", "",100)

rl2 = buildRL2("/home/russellm/rllab-private-RL2unstable/examples/RL2_Ant_TrainDense_TestInd.pkl", 100)

LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/LSBaselineAnt-Ind8-ldim2/", "",100)


def plotMean(numItrs, var, label, marker=None):

    plt.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )



#plotMean(100, trpoSparse, "trpo")

plotMean(100, masen, "MAESN","d")
plotMean(100, LS, "LatentSpace","P")
plotMean(100, maml, "MAML","s")


plotMean(100,vime, "VIME","^")
plotMean(100, trpo, "TRPO","h")
plotMean(100, vpg, "VPG","x")
plt.plot(np.arange(0, 100), rl2, label="RL2", marker="*", markevery=10, markersize=8)

#plotMean(100, LS, "LS")
#plotMean(100, vime, "vime")
#plotMean(100, vpg, "vpg")

#plotMean(100, vpg, "vpg")

#plotMeanStd(100, mamlBiasBiasADA, "maml+Bias+BiasADA")
#plotMeanStd(100, mamlBiasFullADA, "maml+Bias+FullADA")


plt.ylabel("Average Return")
plt.xlabel("Number of Iterations")

plt.legend()
plt.savefig("Ant-Indicator8.png")
    




