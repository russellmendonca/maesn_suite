from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_FULLadaStep import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
#from rllab.envs.mujoco.pusher_env_morerandom_subsample_ec2 import  PusherEnvRandomized
#from rllab.envs.mujoco.blockpush_env_sparse import BlockPushEnvSparse
#from rllab.envs.mujoco.wheeled_robot_goal import WheeledEnvGoal
from rllab.envs.mujoco.ant_env_rand_goal_ring_maesn_sparse_Indicator8 import  AntEnvRandGoalRing
import tensorflow as tf
import pickle
reset_step = 'False'
fast_learning_rate = 0.05
baselines = ['linear']
fast_batch_size = 50  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 20
num_total_tasks = 100  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 200
num_grad_updates = 1
meta_step_size = 0.01
kl_weighting = 0.1
#kl_weighting = 0.05
kl_scheme = None
use_maml = True
latent_dim = 2
seeds = [10, 20, 30, 40, 50]
# initFile= '/home/abhigupta/maesn_nips_rebuttal/maml_rl_fullversion/metaTrainedPolicies/maesn_ant.pkl'
# param_loads = "/home/abhigupta/maesn_nips_rebuttal/maml_rl_fullversion/launchers/FinalLaunchers_ant/maesn_ant_vals.pkl"



#param_load_prefix = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/antMetaTrainedPolicies/maesn_ant_seed'
param_load_prefix = "/root/code/rllab/antMetaTrainedPolicies/maesn_ant_seed"


for seed in seeds:

    stub(globals())

    param_loads = param_load_prefix+str(seed)+'valsDict.pkl'

    env = TfEnv(AntEnvRandGoalRing())

    policy = MAMLGaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=fast_learning_rate,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100,100),
                latent_dim=latent_dim,
                num_total_tasks=num_total_tasks,
            )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = MAMLTRPO(
        env=env,
        policy=policy,
        load_policy = None,
        baseline=baseline,
        batch_size=fast_batch_size, # number of trajs for grad update
        max_path_length=max_path_length,
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_grad_updates,
        n_itr=500,
        use_maml=use_maml,
        step_size=meta_step_size,
        plot=False,
        latent_dim=latent_dim,
        num_total_tasks=num_total_tasks,
        kl_weighting=kl_weighting,
        #plottingFolder = "Sparse_BP_kl0.05_ldim2",
        #visitationFolder = "Ant",
        #visitationFile = "antZsFixedtoZero",
        load_policy_vals=param_loads,
        kl_scheme=kl_scheme,
        reset_step=reset_step
    )
    run_experiment_lite(
        algo.train(),
        n_parallel=4,
        snapshot_mode="all",
        #python_command='python3',
        seed=seed,
        exp_prefix='FULLMAESN_REBUTTAL_maesn_ant_preTrainDense_trainSparse_redone',
        exp_name='FULLMAESN_seed'+str(seed) + 'usemaml' + str(int(use_maml))+'_ldim_'+str(latent_dim)+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates) + "kl_scheme" + str(kl_scheme) + "kl_weighting" + str(kl_weighting),
        plot=False,
        sync_s3_pkl=True,
        mode="ec2",
        pre_commands=["yes | pip install --upgrade pip",
                   "yes | pip install tensorflow=='1.2.0'"]

    )
