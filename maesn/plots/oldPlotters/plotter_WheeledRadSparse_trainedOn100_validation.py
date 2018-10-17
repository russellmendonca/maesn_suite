import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    goals = np.array(range(6,30))
    #np.delete(goals,5)
    
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
plt.title("WheeledRobot")

#vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-VPGSparse-Batch2000-val1-rate1/", "Task", 100)
#vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/SparsePusher-vpg-val1-rate1/", "Task", 100)

#trpo = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goalsVal0.01radius2/", "Task", 200)
trpoSparse = plot("/home/russellm/maml_rl_baseline_test/data/local/TRPO-wheeled-robot-100goals0.01radius2-Sparse1/", "Task", 100)
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



#masenDenseSparse = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/DenseSparse-MasenWheeledTestSparse1-Radius2-trainedOn100-meta20-ldim2-kl0.1-itr225/", "", 15)




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

def plotMeanStd(numItrs, var, label):

    plt.plot(np.arange(0,numItrs), np.mean(var,0), label=label)
    plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )



#plotMeanStd(100, trpoSparse, "trpo")
plotMeanStd(100, maml, "maml")
plotMeanStd(100, masen, "masen")
#plotMeanStd(100, LS, "LS")
plotMeanStd(100, mamlBiasBiasADA, "maml+Bias+BiasADA")
plotMeanStd(100, mamlBiasFullADA, "maml+Bias+FullADA")




plt.legend()
plt.savefig("WheeledRad2-trainedOn100-SparseRad1-MamlComparions.png")
    




