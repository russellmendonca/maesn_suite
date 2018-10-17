import numpy as np
import matplotlib.pyplot as plt
import csv





    # allReturns = []
    # for goal in goals:

def one_avg_return(dirName, numItr = 100):


    allReturns = []
    goals = list(range(30))
    goals.remove(16)
    goals.remove(25)

    for goal in goals:
        
        with open(dirName+'/'+str(goal)+"/progress.csv", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            
            row = None
            records = []
            for row in reader:
                records.append(row)
            ret1Ind = 0 
            for i in range(len(records[0])):
                if records[0][i]=="AverageReturn":
                    ret1Ind = i
               
            
            ret1 = np.array([float(record[ret1Ind]) for record in records[1:]])

           

            allReturns.append(ret1[:numItr])
          
           
    
    return allReturns
        
    # return allReturns

def plot(rewArray, label, marker):

    avgReward = np.mean(rewArray, axis = 0)
    stdRew = np.std(rewArray, axis = 0)

    numItr = len(avgReward)
    plt.plot(np.arange(0, numItr), avgReward, label = label, marker=marker, markevery=10, markersize=8)
    plt.fill_between(np.arange(0, numItr), avgReward - stdRew, avgReward + stdRew, alpha = 0.3)



def testingPlot(dirPrefix, label, marker):

    

    returns_across_seeds= []
    for seed in [10, 30, 50]:


        dirName = dirPrefix+str(seed)

        all_returns = one_avg_return(dirName)
        

        returns_across_seeds.append(np.mean(all_returns, axis = 0))

    



    plot(returns_across_seeds, label, marker)



testingPlot('/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/fineTunedMaesnTest-seed', 'MAESN (ours)', marker = 'd')
testingPlot('/home/russellm/new_maml_rl_baseline/data/s3/fineTunedMamlTest-seed', 'MAML', marker = '*')

plt.legend()

plt.savefig('metaTesting.png')
   







