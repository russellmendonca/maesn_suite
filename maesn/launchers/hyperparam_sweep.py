from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy_adaStep import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):

    
    @variant
    def seed(self):
        return [x for x in range(2)]

    @variant
    def fast_batch_size(self):
        return [20 , 50]
        #return [5]
    @variant
    def fast_learning_rate(self):
        return [0]
        #return [0.5, 1]
    
    @variant
    def meta_batch_size(self):
        return [20]
    @variant
    def meta_learning_rate(self):
        return [0.01]

    @variant
    def kl_weighting(self):
        #return [0.1 , 0.5]
        return [0 , 0.01, 0.05 , 0.1]

    @variant
    def latent_dim(self):
        return [2]

    @variant
    def init_std(self):
        return [1]

    @variant
    def exp_name(self , fast_batch_size , fast_learning_rate , meta_batch_size , meta_learning_rate , kl_weighting , latent_dim , init_std , seed):
        yield  'fbs_'+str(fast_batch_size)+'_flr_'+str(fast_learning_rate)+'_mbs_'+str(meta_batch_size)+'_mlr_'+str(meta_learning_rate)+'_kl_'+str(kl_weighting)+ \
        '_ldim_'+str(latent_dim)+'_initStd_'+str(init_std)+'_seed_'+str(seed)

# best_pusher_hyperParams = {'seed': 1 , "fast_learning_rate": 1 , "meta_step_size" : 0.01  , "fast_batch_size": 50 , "meta_batch_size" : 20 , "kl_weighting" : 0.5 , "latent_dim" : 2}
# best_wheeled_hyperParams = {'seed': 1 , "fast_learning_rate": 1 , "meta_step_size" : 0.01  , "fast_batch_size": 50 , "meta_batch_size" : 20 , "kl_weighting" : 0.1 , "latent_dim" : 2}
# best_ant_hyperParams = {'seed': 1 , "fast_learning_rate": 0.5 , "meta_step_size" : 0.01  , "fast_batch_size": 50 , "meta_batch_size" : 20 , "kl_weighting" : 0.1 , "latent_dim" : 2}