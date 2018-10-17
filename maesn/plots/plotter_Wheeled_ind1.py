import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = list(range(1,30))
    
    #masen
    goals.remove(29)
    
    #LS
    goals.remove(27)
    
    #maml
    goals.remove(16)
    goals.remove(26)
    
    #trpo
    goals.remove(22)
    
    #vpg
    goals.remove(19)
    
    #vime
    goals.remove(20)
    
    
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



plt.title("Wheeled Locomotion")




masen = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/WheeledInd8Redone-Masen-ldim2-kl0.1-itr320/", "", 100)
LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/WheeledInd8Redone-LS-ldim2-kl0.01/", "", 100)
maml = plot("/home/russellm/maml_rl_baseline_test/data/s3/WheeledInd8Redone-Maml-Bias-BiasADA/", "", 100)


trpo = plot("/home/russellm/maml_rl_baseline_test/data/s3/TRPO-Redone-Normalized-Wheeled-Ind80.01/", "Task", 100)
vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/VPG-Redone-Normalized-wheeled-Ind80.1/", "Task", 100)
vime = plot("/home/russellm/rllab-vime/data/s3/VIME-wheeledInd8-expl-eta0.0001-seed0/", "Task",100)

rl2 = buildRL2("/home/russellm/rllab-private-RL2unstable/examples/RL2_wheeled_trainDense_testInd.pkl", 100)
#vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/VPG-ant-Ind80.1/", "Task", 100)




#plt.plot(np.arange(0,100), np.mean(maml_val_subsample, 0), label = "maml")
#plt.plot(np.arange(0,50), np.mean(maml,0), label="maml")

#_trainedOn100_subsample20_batch2000")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

def plotMean(numItrs, var, label, marker=None):

    plt.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )






plotMean(100, masen, \textbf{MAESN (Ours)}", 'd')
plotMean(100, LS, "LatentSpace", 'P')
plotMean(100, maml, "MAML", 's')

plotMean(100, vime, "VIME", '^')
plotMean(100, trpo, "TRPO", 'h')
plotMean(100, vpg, "VPG","x" )
plt.plot(np.arange(0,100), rl2, label="RL2",marker="*", markersize=8, markevery=10)

#plotMean(100, vpg, "vpg")

#plotMeanStd(100, mamlBiasBiasADA, "maml+Bias+BiasADA")
#plotMeanStd(100, mamlBiasFullADA, "maml+Bias+FullADA")



plt.ylabel("Average Return")
plt.xlabel("Number of Iterations")

plt.legend(ncol=3, loc=4)
plt.savefig("Wheeled_Ind8.png")
    




