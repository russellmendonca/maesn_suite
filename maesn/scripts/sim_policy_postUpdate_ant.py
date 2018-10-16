import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

from matplotlib import pyplot as plt

import pickle

xyGoals = pickle.load(open('/home/russellm/generativemodel_tasks/maml_rl_fullversion/rllab/envs/mujoco/goals_ant_val.pkl', 'rb'))

def plot(path, goalIdx):

    xs , ys = path['observations'][:,0] , path['observations'][:,1]

   
    ax.plot(xs, ys)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_path_length', type=int, default=200,
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

   
    for goal in goals:

        print('#############GOAL'+str(goal)+"###################")
       
        tf.reset_default_graph()
        plt.clf()
        _file = str(goal)+"/itr_"+str(simItr)+".pkl"

        fig, ax = plt.subplots()

       
        # try:

        with tf.Session() as sess:

            data = joblib.load(_file)


          
            policy = data['policy']
            env = data['env']
            
            for _try in range(max_tries):
              
                path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = goal,
                                animated=False, save_video = args.save_video, speedup=args.speedup, video_filename="/home/russellm/wheeledPostUpdate/goal"+str(goal)+"_try_"+str(_try)+".mp4")

                plot(path, goalIdx = goal)


            plt.xlim(-2.5, 2.5)
            plt.ylim(0, 2.5)

            circle1 = plt.Circle(xyGoals[goal], 0.03, color='r')

           
            ax.add_artist(circle1)
            

            plt.savefig(saveDir+'/goal'+str(goal)+'.png')

        # except:
        #     pass


               
        # except:
        #     pass

                