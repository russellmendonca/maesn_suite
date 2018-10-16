#from rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnvSparse
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
# from sandbox.rocky.tf.algos.vpg import VPG
#from sandbox.rocky.tf.algos.cem import CEM
#from sandbox.rocky.tf.algos.vpg_StandardPrior import VPG
from sandbox.rocky.tf.algos.vpg_var_stepSize_opt_2 import VPG
#from rllab.envs.mujoco.pusher_env_morerandom_sparse_val1_orig import  PusherEnvRandomized
# from sandbox.rocky.tf.algos.vpg_reparam import VPG
#from rllab.envs.mujoco.wheeled_robot_goal_Val import WheeledEnvGoal
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



ldim = 2
kl = 0.1
#initial_params_files = [initial_params_file1]

goals = np.array(range(0,30))
 
#initial_params_file = "/home/abhigupta/abhishek_sandbox/russell_MAESN/maml_rl_fullversion/data/s3/masen-antrunning-onlylatentsadaptive/ant_fulltrpomasen1_ldim_2_fbs50_mbs20_flr_0.5metalr_0.01_step11kl_schemeNonekl_weighting0.1/itr_134.pkl"

params_file_prefix = '/root/code/rllab/antFineTunedPolicies/maesn_finetunedAnt_seed'
#params_file_prefix = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/antFineTunedPolicies/maesn_finetunedAnt_seed'
#'/home/russellm/generativemodel_tasks/maml_rl_fullversion/metaTrainedPolicies/maesn_ant.pkl'

#counter = 0
seeds = [20, 30, 40, 50]

for seed in seeds :

    initial_params_file =  params_file_prefix + str(seed) + '.pkl'

    counter = 0

    for goal in goals:
        env = TfEnv( AntEnvRandGoalRing())
       
        n_itr = 100
     
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
            #joint_opt=True,
            # center_adv=False,
            # positive_adv=False,
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
            seed=seed,
            exp_prefix="fineTunedMaesnTest_seed"+str(seed),
            exp_name=str(counter),

            mode="ec2",
            pre_commands=["yes | pip install --upgrade pip",
                           "yes | pip install tensorflow=='1.2.0'"]

        )
        counter+=1
        

