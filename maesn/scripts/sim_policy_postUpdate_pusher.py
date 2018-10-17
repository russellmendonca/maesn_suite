import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
import pickle

from matplotlib import pyplot as plt
import numpy as np

configList = pickle.load(open('/home/russellm/generativemodel_tasks/maml_rl_fullversion/rllab/envs/mujoco/pusher_valSet1.pkl', 'rb'))

def plot(path, blockchoice):

    xidx = int(3 + 2*blockchoice)

    obs = path['observations']
  
    block_xs , block_ys = obs[:,xidx] , obs[:,xidx+1]

    gripper_xs, gripper_ys = obs[:,0], obs[:,1]

   

    
   
    ax.plot(block_xs, block_ys, 'g')
    ax.plot(gripper_xs, gripper_ys, 'grey')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--save_video', type=bool, default=True,
                        help='decision to save video')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--num_tries', type=int, default = 1,
                        help='number of tries')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    args = parser.parse_args()

    max_tries = 10
    tri = 0

    simItr = 5
    saveDir = 'Plots_SimItr'+str(simItr)
   
    
    goals = list(range(30))

    import os
    if os.path.isdir(saveDir) == False:
        os.mkdir(saveDir)

    #goals = [  17]
    for goal in goals:

        print('#####GOAL__'+str(goal)+'#####')

        tf.reset_default_graph()
        plt.clf()
        _file = str(goal)+"/itr_"+str(simItr)+".pkl"

        fig, ax = plt.subplots()

        config = configList[goal]
       
        blockchoice = config[0]
       

        targetPos = np.array([config[1], config[2]])
        #objPos = np.array([config[int(6+2*blockchoice)], config[int(7+2*blockchoice)]])

      
        try:
            with tf.Session() as sess:
                data = joblib.load(_file)
                policy = data['policy']
                env = data['env']
                
                for _try in range(max_tries):
                    # path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = goal,
                    #                animated=True, save_video = args.save_video, speedup=args.speedup, video_filename="/Users/russell/Desktop/antPostUpdate/Repeatgoal"+str(goal)+"_try_"+str(_try)+".mp4")
                    path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = goal,
                                    animated=False, save_video = args.save_video, speedup=args.speedup, video_filename="/Users/russell/Desktop/pusherPostUpdate/goal"+str(goal)+"_try_"+str(_try)+".mp4")
                    

                   
                    plot(path, blockchoice = blockchoice)


                plt.xlim(-1, 1.)
                plt.ylim(-0.5, 0.5)

                
             


                blocks = config[6:] 


                for i in range(5):
                    x, y = blocks[2*i] + 0.2*(i+1) , blocks[(2*i) + 1]

                    if i == blockchoice:
                        ax.add_artist(plt.Circle([x, y], 0.02, color='g'))
                    else:
                        ax.add_artist(plt.Circle([x,y], 0.02, color='b'))

                    



               
                ax.add_artist(plt.Circle(targetPos, 0.02, color='r'))
               
                

                plt.savefig(saveDir+'/goal'+str(goal)+'.png')

        except:
            pass