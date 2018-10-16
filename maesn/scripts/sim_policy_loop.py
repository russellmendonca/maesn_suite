import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--save_video', type=bool, default=False,
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
    
        
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        
        for i in range(max_tries):
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, save_video = args.save_video, speedup=args.speedup, video_filename="try.mp4")
        