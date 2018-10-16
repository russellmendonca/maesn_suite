import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr):
    
    numgoals = 20
    
    origHis = []
    for goal in range(numgoals):
        
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
plt.title("randPusher-trainedOn100")

#vpg_val = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-vpg-val-rate0.05/", "Task", 20)
vpg_val_1 = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-vpg-val-rate1/", "Task", 100)
#vpg_val_2 = plot("/home/russellm/maml_rl_baseline_test/data/local/Pusher-vpg-val-rate0.5/", "Task", 20)


maml_val_subsample = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest-trainedOn100-meta20-fbs20-itr499/","", 10)
maml_val_direct = plot("/home/russellm/maml_rl_baseline_test/data/local/MamlPusher-valTest-trainedOn100-direct-fbs20-itr499/","",10)



masen_val_trainedOn100_ldim5 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet-trainedOn100-meta20-ldim5-kl0.05-itr400/","", 10)

masen_val_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-ValSet-trainedOn100-meta20-ldim2-kl0.05-itr400/","", 10)
LS_baseline_trainedOn100_ldim2 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/LSBaselinePusher-ValSet-trainedOn100-meta20-ldim2-kl0-itr400/","", 10)
import ipdb
ipdb.set_trace()



plt.plot(np.arange(0,10), np.mean(maml_val_subsample, 0), label = "maml_val_trainedOn100_subsample20")
#plt.plot(np.arange(0,10), np.mean(maml_val_direct, 0) , label = "maml_val_trainedOn100_direct")


plt.plot(np.arange(0,100), np.mean(vpg_val_1,0), label='vpg_val_1')


plt.plot(np.arange(0,10), np.mean(masen_val_trainedOn100_ldim5, 0), label = "masen_val_trainedOn100_ldim5_subsample20")
plt.plot(np.arange(0,10), np.mean(masen_val_trainedOn100_ldim2, 0), label = "masen_val_trainedOn100_ldim2_subsample20")
plt.plot(np.arange(0,10), np.mean(LS_baseline_trainedOn100_ldim2, 0), label = "LS_Baseline_trainedOn100_ldim2_subsample20")

plt.legend()
plt.savefig("randPusher-trainedOn100.png")
    



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
