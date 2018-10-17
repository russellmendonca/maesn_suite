import numpy as np
import matplotlib.pyplot as plt
import csv





    # allReturns = []
    # for goal in goals:

def one_avg_return(dirName, numItr = 100):
        
    with open(dirName+"/progress.csv", 'r') as f:
        reader = csv.reader(f, delimiter=',')
        
        row = None
        records = []
        for row in reader:
            records.append(row)
        ret1Ind = 0 
        for i in range(len(records[0])):
            if records[0][i]=="1AverageReturn":
                ret1Ind = i
           
        
        ret1 = np.array([float(record[ret1Ind]) for record in records[1:]])
      
       
        return ret1[:numItr]
    
    # return allReturns

def plot(rewArray, xaxisPoints, label, marker):

    avgReward = np.mean(rewArray, axis = 0)
    stdRew = np.std(rewArray, axis = 0)

    if 'MAESN' in label:
        color = 'b'
    else:
        color = 'g'
    
    plt.plot(xaxisPoints, avgReward, label = label, marker=marker, markevery=10, markersize=8)
    plt.fill_between(xaxisPoints, avgReward - stdRew, avgReward + stdRew, alpha = 0.3)



def plotMethod(prefix, suffix, xaxisPoints, label, marker):

    all_returns = []
    for seed in [10, 30,  50]:

        _dir =  prefix + str(seed) + suffix
        all_returns.append(one_avg_return(_dir, numItr = len(xaxisPoints)))


    all_returns = np.array(all_returns)

   

    plot(all_returns, xaxisPoints, label = label, marker = marker)



# plotMethod('/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/maesn-ant-radius2-tasks100-metaTrain20-seedTest/fulltrpomasen1_seed',

#            '_ldim_2_fbs50_mbs20_flr_1metalr_0.01_step11kl_schemeNonekl_weighting0.1',

#            np.arange(0, 140) , label = 'MAESN')


plotMethod('/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/preTrainDense-trainSparse-ant-maesn-multipleSeeds/fulltrpomasen1seed_', 
         '_ldim_2_fbs50_mbs20metalr_0.01_step11kl_schemeNonekl_weighting0.1',
         np.arange(0, 100) , label = 'MAESN (ours)', marker = 'd')




# plotMethod('/home/russellm/new_maml_rl_baseline/data/s3/ant-maml-baseline-fullyadaptive-biastransformation-100goals-meta20-seedTest/maml_baseline_fullyadaptive_biastransformationseed',
#             'um1_fbs50_mbs20_flr_0.5metalr_0.01_step11ldim8',

#             np.arange(0, 140), label = 'MAML')


plotMethod('/home/russellm/new_maml_rl_baseline/data/s3/pretrainDense-trainSparse-newAnt-maml-multipleSeeds/ant_maml_baseline_fullyadaptive_biastransformationseed',

         'um1_fbs50_mbs20_flr_0.5metalr_0.01_step11ldim8',

           np.arange(0, 100), label = 'MAML', marker = '*')


plt.legend()
plt.savefig('pretrainDense-MetaTraining.png')


# def plotMaesn():

#     #densePrefix = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/maesn-ant-radius2-tasks100-metaTrain20-seedTest/fulltrpomasen1_seed10_ldim_2_fbs50_mbs20_flr_1metalr_0.01_step11kl_schemeNonekl_weighting0.1
#     fineTunePrefix = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/preTrainDense-trainSparse-ant-maesn-multipleSeeds/fulltrpomasen1seed_'



#     allmaesn_returns = []
#     for seed in [10, 30,  50]:

#         _dir = filePrefix + str(seed) + '_ldim_2_fbs50_mbs20metalr_0.01_step11kl_schemeNonekl_weighting0.1'
#         allmaesn_returns.append(one_avg_return(_dir))


#     allmaesn_returns = np.array(allmaesn_returns)

#     plot(allmaesn_returns, label = 'avgMaesnRew')



# def plotMaml():

#     filePrefix = '/home/russellm/new_maml_rl_baseline/data/s3/pretrainDense-trainSparse-newAnt-maml-multipleSeeds/ant_maml_baseline_fullyadaptive_biastransformationseed'
    


#     all_returns = []
#     for seed in [10,  30,  50 ]:

#         _dir = filePrefix+ str(seed)+'um1_fbs50_mbs20_flr_0.5metalr_0.01_step11ldim8'

       
#         all_returns.append(one_avg_return(_dir))


#     all_returns = np.array(all_returns)

#     plot(all_returns, label = 'avgMamlRew')



# #plotFULLMaesn()
# plotMaesn()
# plotMaml()
