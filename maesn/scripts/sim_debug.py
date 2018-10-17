import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout


from rllab.envs.mujoco.wheeled_robot_goal_Val_sparse_Ind8 import WheeledEnvGoal

import pickle

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
    
    #goals = [5,0,6,11,19,25,27,28]
    #goals = list(range(20))

    goal = 0
   
   
       
        #tf.reset_default_graph()
    _file = 'goal'+str(goal)+'_itr_199.pkl'

    with tf.Session() as sess:
     
        fobj = open('new.pkl', 'rb')
        #data = pickle.load(fobj)
        data = {}
        
     
     
        #import pickle
       
        # race = 'a'
        #data = pickle.load(fobj)
          
            # policy = data['policy']

        envNew = WheeledEnvGoal()
        for i in range(args.max_path_length):
            action = envNew.action_space.sample()
            envNew.step(action)
            envNew.render()

            #env = data['env']
            
            # for _try in range(max_tries):
            #     # path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = goal,
            #     #                animated=True, save_video = args.save_video, speedup=args.speedup, video_filename="/Users/russell/Desktop/antPostUpdate/Repeatgoal"+str(goal)+"_try_"+str(_try)+".mp4")
            #     path = rollout(env, policy, max_path_length=args.max_path_length, reset_arg = goal,
            #                     animated=True, save_video = args.save_video, speedup=args.speedup, video_filename="/home/russellm/wheeledPostUpdate/goal"+str(goal)+"_try_"+str(_try)+".mp4")


               
        # except:
        #     pass

                