import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
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


def plotAnt(numItrs, var, label, marker=None):
    return ant.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )


fig, (pusher, wheeled, ant) = plt.subplots(1, 3, sharex=True, figsize=(25,5))


#plotMean(100, trpoSparse, "trpo")

ant.set_title("Legged Locomotion")

ant.set_xlabel("Number of Learning Iterations")
ant.set_ylabel("Average Return")

a1, = plotAnt(100, masen, "MAESN (ours)","d")
a2, = plotAnt(100, LS, "LatentSpace","P")
a3, =plotAnt(100, maml, "MAML","s")


a4, =plotAnt(100,vime, "VIME","^")
a5, =plotAnt(100, trpo, "TRPO","h")
a6, =plotAnt(100, vpg, "VPG","x")
a7, = ant.plot(np.arange(0, 100), rl2, label="RL2", marker="*", markevery=10, markersize=8)



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

def plotWheeled(numItrs, var, label, marker=None):

    wheeled.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )




wheeled.set_title("Wheeled Locomotion")

wheeled.set_xlabel("Number of Learning Iterations")
wheeled.set_ylabel("Average Return")

plotWheeled(100, masen, "MAESN (ours)", 'd')
plotWheeled(100, LS, "LatentSpace", 'P')
plotWheeled(100, maml, "MAML", 's')

plotWheeled(100, vime, "VIME", '^')
plotWheeled(100, trpo, "TRPO", 'h')
plotWheeled(100, vpg, "VPG","x" )
wheeled.plot(np.arange(0,100), rl2, label="RL2",marker="*", markersize=8, markevery=10)


def plot(filePrefix, string,  num_itr, rewardMarker = "AverageReturn"):
    
    goals = list(range(1,100))
    
    #LS
    goals.remove(40)
    goals.remove(41)
    goals.remove(54)
    goals.remove(99)
    
    
    ##VPG missing goals
    goals.remove(11)
    
    goals.remove(3)
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

def plotPusher(numItr, var, label, marker=None):

    return pusher.plot(np.arange(0,numItr), np.mean(var,0), label=label, marker=marker, markersize=8, markevery=10)
   # plt.fill_between(np.arange(startItr,endItr), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )





#plotMean(100, trpoSparse, "trpo")

pusher.set_title("Robotic Manipulation")

pusher.set_xlabel("Number of Learning Iterations")
pusher.set_ylabel("Average Return")


plotPusher(100, masen_S20, "MAESN (Ours)","d")
plotPusher(100, LS, "LatentSpace", "P")
plotPusher(100, mamlBiasBiasADA, "MAML", "s")
#plotMeanStd(3, 100, np.array(masenTRPOImprove)[:,:97], "masenTRPO")

plotPusher(100, vime, "VIME", "^")
plotPusher(100, trpo_20, "TRPO", "h")
plotPusher(100, vpg, "VPG", "x")
pusher.plot(np.arange(0,100), rl2, label="RL2",marker="*", markersize=8 ,markevery=10)
pusher.plot(np.arange(0,100), np.mean(theanoTRPO,0) ,label="TRPO(Theano)",linestyle="dashed", color="purple")

pBox = pusher.get_position()

wBox = wheeled.get_position()

aBox = ant.get_position()



pusher.set_position([pBox.x0, pBox.y0 + pBox.height * 0.1,
                 pBox.width, pBox.height * 0.9])

ant.set_position([aBox.x0, aBox.y0 + aBox.height * 0.1,
                 aBox.width, aBox.height * 0.9])


wheeled.set_position([wBox.x0, wBox.y0 + wBox.height * 0.1,
                 wBox.width, wBox.height * 0.9])

# Put a legend below current axis
plt.legend(loc='center', bbox_to_anchor=(-0.65, -0.2),
          fancybox=True, shadow=True, ncol=8, fontsize="x-large")


plt.savefig("OverallOrdered.png")
    




