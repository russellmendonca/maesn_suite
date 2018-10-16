import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = list(range(6,100))
    
    #Vime
    goals.remove(14)
    goals.remove(19)
    #goals.remove(64)
    #goals.remove(86)
    
    #Vpg
    
    goals.remove(12)
    goals.remove(21)
    goals.remove(33)
    goals.remove(39)
    goals.remove(40)
    goals.remove(69)
    
    
    
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


plt.title("WheeledRobot")

#vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
#vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/SparsePusher-vpg-val1-rate1/", "Task", 100)

#trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goalsVal0.01radius2/", "Task", 200)

#vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/Wheeled-VPGThresh1-Batch10000-val1-rate1/", "Task", 100)

vpg = plot("/home/russellm/maml_rl_baseline_test/data/s3/WheeledNormalized-VPGThresh1-Batch10000-val1-rate0.1/", "Task", 100)
for i in vpg:
    #import ipdb
    #ipdb.set_trace()
    if np.shape(i)!=(100,):
        vpg.remove(i)



trpoSparse = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goals0.01radius2-Sparse1/", "Task", 100)

vime = plot("/home/russellm/rllab-vime/data/s3/VIME-wheeledRad1-expl-eta0.0001-seed0/", "Task", 100)



#maml_rl_baseline_test/data/local/TRPO-wheeled-robot0.01radius2/", "Task", 50)
#TRPO-SparseOrig-Batch2000-val1-rate0.01/", "Task", 100)

#maml2 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-initlr0.1-lr0.05-proper-trainedOn100-itr450/", "", 50)
#mamlHalfStep = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-lr0.1-halfStep-trainedOn100-itr450/", "", 50)
#maml1 = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-lr0.1-proper-trainedOn100-itr450/", "", 50)




maml = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-SparseRad1-wheeledRad2-lr0.1-halfStep-trainedOn100-itr450/", "", 100)

LS = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/TrainDenseTestSparse-LSBaseline-wheeled-trainedOn100-meta20-Val-ldim2-kl0.01-itr320/", "", 100)

#maml = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlTest-wheeledRad2-trainedOn20-itr440/", "", 50)


#MamlPusher-trainedOn100-origSparse-meta20-fbs20-itr499-batch2000/","", 100)
#maml_val_direct = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest-trainedOn100-direct-fbs20-itr499/","",10)

##maml_ADA = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlSparsePusher-orig-Adaptive-trainedOn100-meta20-fbs20-itr280-batch2000-initRate0.1/","",100)

#masen = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenWheeledTestSparse1-Radius2-trainedOn100-ldim2-kl0.5-itr300/", "", 100) 


masen = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/Final-MasenWheeledTestSparse1-100itr-Radius2-trainedOn100-ldim2-kl0.1-itr320/", "",100)

mamlBiasBiasADA = plot("/home/russellm/new_maml_rl_baseline/data/local/MamlTest-Bias-BiasADA-SparseRad1-wheeledRad2-lr0.1-halfStep-trainedOn100-itr499/", "", 100)


mamlBiasFullADA = plot("/home/russellm/new_maml_rl_baseline/data/local/MamlTest-Bias-fullADA-SparseRad1-wheeledRad2-halfStep-trainedOn100-itr499/", "", 100)





import ipdb
ipdb.set_trace()



#plt.plot(np.arange(0,100), np.mean(maml_val_subsample, 0), label = "maml")
#plt.plot(np.arange(0,50), np.mean(maml,0), label="maml")

#_trainedOn100_subsample20_batch2000")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

def plotMean(numItrs, var, label):

    plt.plot(np.arange(0,numItrs), np.mean(var,0), label=label)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )



plotMean(100, trpoSparse, "trpo")
plotMean(100, maml, "maml")
plotMean(100, masen, "masen")
plotMean(100, LS, "LS")
plotMean(100, vime, "vime")
plotMean(100, vpg, "vpg")

#plotMean(100, vpg, "vpg")

#plotMeanStd(100, mamlBiasBiasADA, "maml+Bias+BiasADA")
#plotMeanStd(100, mamlBiasFullADA, "maml+Bias+FullADA")




plt.legend()
plt.savefig("WheeledRad2-trainedOn100-SparseRad1-priorMethods.png")
    




