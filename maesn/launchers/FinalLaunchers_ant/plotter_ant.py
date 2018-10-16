from sandbox.rocky.tf.algos.maml_trpo_plotter import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaStep import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
#from rllab.envs.mujoco.pusher_env_morerandom_subsample_ec2 import  PusherEnvRandomized


from rllab.envs.mujoco.ant_env_rand_goal_ring_maesn_dense import AntEnvRandGoalRing
# from rllab.envs.mujoco.wheeled_robot_goal_subsample import WheeledEnvGoal

#from rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnvSparse

import tensorflow as tf


fast_batch_size = 50  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 20
num_total_tasks = 100  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 200
num_grad_updates = 1
meta_step_size = 0.01






prefix = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/data/s3/maesn-ant-radius2-tasks100-metaTrain20-seedTest/fulltrpomasen1_seed'


#10_ldim_2_fbs50_mbs20_flr_1metalr_0.01_step11kl_schemeNonekl_weighting0.1

kl_weighting = 0.1
flrs = [ 0.5, 0.1, 1]
ldim = 2
itr = 140
seeds = [10, 20 , 30, 40, 50, 60]

for seed in seeds:
    for flr in flrs:
        
        stub(globals())
        initial_params_file = prefix + str(seed)+'_ldim_2_fbs50_mbs20_flr_'+str(flr)+'metalr_0.01_step11kl_schemeNonekl_weighting0.1/itr_140.pkl'
        #initial_params_file = prefix+'ldim_'+str(ldim)+'_fbs50_mbs20_flr_'+str(flr)+'metalr_0.01_step11kl_schemeNonekl_weighting'+str(kl_weighting)+'/itr_'+str(itr)+'.pkl'

        env = TfEnv( AntEnvRandGoalRing())

      

        
        algo = MAMLTRPO(
            env=env,
            policy=None,
            load_policy = initial_params_file,
            baseline=None,
            batch_size=fast_batch_size, # number of trajs for grad update
            max_path_length=max_path_length,
            meta_batch_size=meta_batch_size,
            num_grad_updates=num_grad_updates,
            n_itr=1,
            use_maml=True,
            step_size=meta_step_size,
            plot=False,
            latent_dim=ldim,
            num_total_tasks=num_total_tasks,
            kl_weighting=kl_weighting,
            #plottingFolder = "Sparse_BP_kl0.05_ldim2",
            visitationFolder = "maesn_seedTest_ant",
            visitationFile = 'maesnAnt'+'seed'+str(seed)+'_flr'+str(flr),
            
            kl_scheme=None,

        )
        run_experiment_lite(
            algo.train(),
            n_parallel=4,
            snapshot_mode="all",
            #python_command='python3',
            seed=1,
            #exp_prefix='plotter_masen_randSparsePush_train100_subsampling20',
            #exp_name='fulltrpomasen'+str(int(use_maml))+'_ldim_'+str(latent_dim)+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates) + "kl_scheme" + str(kl_scheme) + "kl_weighting" + str(kl_weighting),
            plot=False,
            #sync_s3_pkl=True,
            ##mode="ec2",
            #pre_commands=["yes | pip install --upgrade pip",
            #               "yes | pip install tensorflow=='1.2.0'"]
        )
