#from rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnvSparse`
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite


from rllab.envs.mujoco.ant_env_rand_goal_ring_maesn_sparse_Indicator8 import AntEnvRandGoalRing
from sandbox.rocky.tf.algos.trpo import TRPO

import pickle
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())

rate = 0.01

ldim = 2
kl = 0.1
#initial_params_files = [initial_params_file1]

goals = np.array(range(0,1))
stepSizes = [0.1] 

initFolder = "/root/code/rllab/antdata/MasenTest-AntDense-Indicator8/"

for step_size in stepSizes:

    
    for goal in goals:

        env = TfEnv( AntEnvRandGoalRing())
       
        n_itr = 100
        

        baseline = ZeroBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=None,
            load_policy = initFolder + str(goal)+"/itr_5.pkl",
            baseline=baseline,
            batch_size=10000,
            max_path_length=200,
            n_itr=200,
            discount=0.99,
            step_size=step_size,
            reset_arg = np.asscalar(goal),
            improve = True,
            latent_dim = ldim,
            #plot=True,
        )


        run_experiment_lite(
            algo.train(),
            # Number of parallel workers for sampling
            n_parallel=4,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="all",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=1,
            exp_prefix="MasenTest_Ant_improveItr5_stepSize"+str(step_size),
            exp_name=str(goal),
            
            mode = "ec2",
            sync_s3_pkl = True,
        )
       
        
