from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaStep import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv

#If not subsampling use this env:
#from rllab.envs.mujoco.wheeled_robot_goal_Val_sparse_Ind8_noSubsampling import WheeledEnvGoal
#If using subsampling, use this env:

#from rllab.envs.mujoco.wheeled_robot_goal_subsample import WheeledEnvGoal
from rllab.envs.mujoco.wheeled_robot_goal_Val_sparse_Ind8 import WheeledEnvGoal
import tensorflow as tf

fast_batch_size = 50  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 20
num_total_tasks = 100  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 200
num_grad_updates = 1
#meta_step_sizes = [0.001, 0.002, 0.005, 0.0005, 0.0001, 0.01]
meta_step_size = 0.01
#kl_weightings = [1, 0.1, 0.5, 1.5]
kl_weighting = 0.5
kl_scheme = None
use_maml = True
latent_dim = 2

seeds = [31,41]
#init_policy = '/root/code/rllab/rllab/maesnWheeledDense.pkl'

#init_policy = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/metaTrainedPolicies/maesn_wheeled.pkl'


init_policy = '/root/code/rllab/metaTrainedPolicies/maesn_wheeled.pkl'

#init_policy = '/home/russellm/generativemodel_tasks/maml_rl_fullversion/metaTrainedPolicies/maesn_wheeled.pkl'
#init_policy = '/home/russellm/maesn_OLD_codebase/maml_rl_fullversion/data/s3/masen-wheeledRobot-radius2-train100-subsampling/fulltrpomasen1_ldim_2_fbs50_mbs20_flr_1metalr_0.01_step11kl_schemeNonekl_weighting0.5/itr_300.pkl'


#for meta_step_size in meta_step_sizes:

for seed in seeds:
    
    stub(globals())

    env = TfEnv( normalize(WheeledEnvGoal()))



    algo = MAMLTRPO(
        env=env,
        policy=None,
        baseline=None,
        load_policy = init_policy,
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
        kl_scheme=kl_scheme
    )
    run_experiment_lite(
        algo.train(),
        n_parallel=4,
        snapshot_mode="all",
        #python_command='python3',
        seed=seed,
        exp_prefix='preTrainDense-TrainSparse-masen-wheeledRobot-SparseBonus',
        exp_name='fulltrpomasen'+str(int(use_maml))+'_seed'+str(seed)+'_ldim_'+str(latent_dim)+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates) + "kl_scheme" + str(kl_scheme) + "kl_weighting" + str(kl_weighting),
        plot=False,
        sync_s3_pkl=True,
        mode="ec2",
        pre_commands=["yes | pip install --upgrade pip",
                       "yes | pip install tensorflow=='1.2.0'"]
    )
