#from rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnvSparse
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.algos.cem import CEM
#from sandbox.rocky.tf.algos.vpg_StandardPrior import VPG
from sandbox.rocky.tf.algos.vpg_var_stepSize_opt_2 import VPG
#from rllab.envs.mujoco.pusher_env_morerandom_sparse_val1_orig import  PusherEnvRandomized
# from sandbox.rocky.tf.algos.vpg_reparam import VPG
from rllab.envs.mujoco.wheeled_robot_goal_Val_sparse_Ind8 import WheeledEnvGoal
from sandbox.rocky.tf.algos.trpo import TRPO

import pickle
from sandbox.rocky.tf.envs.base import TfEnv

import csv
import joblib
import numpy as np
import pickle
import tensorflow as tf

stub(globals())

ldim = 2
kl = 0.1


goals = np.array(range(0,30))
 

counter = 0
 
initial_params_file = "/root/code/rllab/metaTrainedPolicies/maesn_wheeled.pkl"   #(Meta-Trained Policy)




for goal in goals:
    env = TfEnv( normalize(WheeledEnvGoal()))
    #env = PusherEnvRandomized()
    n_itr = 100
    #env = TfEnv(env)
    if initial_params_file is not None:
        policy = None

    baseline = ZeroBaseline(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=policy,
        load_policy=initial_params_file,
        baseline=baseline,
        batch_size=10000,  # 2x
        max_path_length=200,
        n_itr=n_itr,
        latent_dim=ldim,
        num_total_tasks=100,  #numTotalTasks while training
        noise_opt = True,
        default_step = 1,
        reset_arg=np.asscalar(goal),
        #optimizer_args={'tf_optimizer_args': {'learning_rate': 0.3}, 'tf_optimizer_cls': tf.train.GradientDescentOptimizer}
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
        exp_prefix="WheeledInd8Redone_Masen_ldim"+str(ldim)+"_kl"+str(kl)+"_itr320",
        exp_name=str(counter),
        mode = "local_docker",
        sync_s3_pkl = True,
        pre_commands=["yes | pip install --upgrade pip",
                           "yes | pip install tensorflow=='1.2.0'"]
    )
    counter+=1
    
