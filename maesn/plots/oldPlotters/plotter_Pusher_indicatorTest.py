import csv
import numpy as np
from matplotlib import pyplot as plt
import pickle
#tasks = range(0,6)
#folders = ['3-itr190-lr0.3', '3-itr190-lr0.4']
#folders = ['3-lr0.1']


def plot(filePrefix, string,  num_itr, rewardMarker="AverageReturn"):
    
    goals = range(0,90)
    
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

# def buildRL2(_file, total):
#     fobj = open(_file, "rb")
#     list1 = pickle.load(fobj)
#     _len = len(list1)
#     last = list1[_len -1]
#     for i in range(_len, total):
#         list1 = np.concatenate([list1, [last]])
#     return list1
    


#plt.ylim(-1000, -700)
plt.title("randSparsePusher-trainedOn100")

masen_D10 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrDense-TestIndicator0.1-ldim10-kl0.5/", "", 100)
masen_D15 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrDense-TestIndicator0.15-ldim10-kl0.5/", "", 100)
masen_D20 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrDense-TestIndicator0.2-ldim10-kl0.5/", "", 100)
masen_D25 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrDense-TestIndicator0.25-ldim10-kl0.5/", "", 100)


masen_S10 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.1-ldim2-kl0.5/", "", 100)
masen_S15 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.15-ldim2-kl0.5/", "", 100)
masen_S20 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.20-ldim2-kl0.5/", "", 100)
masen_S25 = plot("/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/local/MasenPusher-TrSparse-TestIndicator0.25-ldim2-kl0.5/", "", 100)






plt.plot(np.arange(0,100), np.mean(masen_D10,0), label="TrDense_0.1")
plt.plot(np.arange(0,100), np.mean(masen_D15,0), label="TrDense_0.15")
plt.plot(np.arange(0,100), np.mean(masen_D20,0), label="TrDense_0.2")
plt.plot(np.arange(0,100), np.mean(masen_D25,0), label="TrDense_0.25")

plt.plot(np.arange(0,100), np.mean(masen_S10,0), label="TrSparse_0.1")
plt.plot(np.arange(0,100), np.mean(masen_S15,0), label="TrSparse_0.15")
plt.plot(np.arange(0,100), np.mean(masen_S20,0), label="TrSparse_0.2")
plt.plot(np.arange(0,100), np.mean(masen_S25,0), label="TrSparse_0.25")

plt.legend()
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
