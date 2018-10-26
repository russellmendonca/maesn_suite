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
    

    goals = list(range(30))
    goals.remove(5)
    goals.remove(0)
    goals.remove(16)
    goals.remove(17)
    origHis = []
    for goal in goals:
        
        
        with open(filePrefix+string+str(goal)+"/progress.csv", 'r') as f:
            reader = csv.reader(f, delimiter=',')
            
            row = None
            records = []
            for row in reader:
                records.append(row)
            ret0Ind = 0 

            print(len(records))
            for i in range(len(records[0])):
                if records[0][i]=="AverageReturn":
                    ret0Ind = i
               
            
            ret0 = []
            for record in records[1:]:
                ret0.append(float(record[ret0Ind]))
                #ret1.append(record[ret1Ind])
           
            origHis.append(ret0[:num_itr])
    

    return origHis


masen=plot("/home/russellm/data/s3/maesn/rllab/experiments/Maesn-Ant-Test/", "", 100)
LS = plot("/home/russellm/data/s3/maesn/rllab/experiments/LSBaseline-Ant-Test/", "",100)


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





masen = plot("/home/russellm/data/s3/maesn/rllab/experiments/Maesn-Wheeled-Test/", "", 100)
LS = plot("/home/russellm/data/s3/maesn/rllab/experiments/LSBaseline-Wheeled-Test/", "", 100)

#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

def plotWheeled(numItrs, var, label, marker=None):

    wheeled.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )




wheeled.set_title("Wheeled Locomotion")

wheeled.set_xlabel("Number of Learning Iterations")
wheeled.set_ylabel("Average Return")

plotWheeled(100, masen, "MAESN (ours)", 'd')
plotWheeled(100, LS, "LatentSpace", 'P')


plt.savefig('Ant_and_wheeled.png')




masen = plot("/home/russellm/data/s3/maesn/rllab/experiments/Maesn-Pusher-Test/", "", 50)
maesnOld = plot('/home/russellm/data/finalMaesnData/MasenPusher-TrSparse-TestIndicator0.20-ldim2-kl0.5/', '',100)
LS = plot("/home/russellm/data/s3/maesn/rllab/experiments/LSBaseline-Pusher-Test/", "", 100)

#plt.plot(np.arange(0,100), np.mean(vpg_val,0), label='vpg')

def plotPusher(numItrs, var, label, marker=None):

    pusher.plot(np.arange(0,numItrs), np.mean(var,0), label=label, marker=marker, markevery=10, markersize=8)
   # plt.fill_between(np.arange(0,numItrs), np.mean(var, 0) - np.std(var, 0),np.mean(var, 0) + np.std(var, 0), alpha = 0.3 )




pusher.set_title("Pusher")

pusher.set_xlabel("Number of Learning Iterations")
pusher.set_ylabel("Average Return")

plotPusher(100, maesnOld, "MAESN (ours)", 'd')
plotPusher(100, LS, "LatentSpace", 'P')


plt.savefig('Ant_and_wheeled_and_Pusher.png')