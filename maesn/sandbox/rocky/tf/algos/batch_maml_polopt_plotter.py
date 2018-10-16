import matplotlib
#matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.samplers.vectorized_sampler_noNoise import VectorizedSamplerNoNoise
from sandbox.rocky.tf.spaces import Discrete
from rllab.sampler.stateful_pool import singleton_pool
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # Note that the number of trajectories for grad upate = batch_size
            # Defaults are 10 trajectories of length 500 for gradient update
            batch_size=100,
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            load_policy=None,
            latent_dim=4,
            num_total_tasks=100,
            kl_weighting=0,
            kl_scheme=None,
            plottingFolder = None,
            visitationFolder = None,
            visitationFile = None,
            addedNoise = True,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #
        :param max_path_length: Maximum length of a single rollout.
        :param meta_batch_size: Number of tasks sampled per meta-update
        :param num_grad_updates: Number of fast gradient updates
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.latent_dim = latent_dim
        self.kl_weighting = kl_weighting
        self.kl_scheme = kl_scheme
        self.num_total_tasks = num_total_tasks
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size is the number of trajectories for one fast grad update.
        # self.batch_size is the number of total transitions to collect.
        self.num_trajs = batch_size
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size # number of tasks
        self.num_grad_updates = num_grad_updates # number of gradient steps during training
        self.latent_dist = DiagonalGaussian(self.latent_dim)
        self.plottingFolder = plottingFolder
        self.visitationFolder = visitationFolder
        self.visitationFile = visitationFile
        self.addedNoise = addedNoise

        if sampler_cls is None:
            # if singleton_pool.n_parallel > 1:
            #     sampler_cls = BatchSampler
            # else:
            #sampler_cls = BatchSampler
            if self.addedNoise:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = VectorizedSamplerNoNoise
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        sampler_args['latent_dim'] = self.latent_dim
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # This obtains samples using self.policy, and calling policy.get_actions(obses)
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)
        
        paths = self.sampler.obtain_samples(itr, reset_args=reset_args, task_idxs=reset_args, return_dict=True, log_prefix=log_prefix)
        #paths = self.sampler.obtain_samples(itr, reset_args=reset_args)

        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', noise_opt=False, joint_opt=False, log=True, task_idx=None):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log, task_idx=task_idx, noise_opt=noise_opt, joint_opt=joint_opt)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        
        with tf.Session(config=config) as sess:

        
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if self.load_policy is not None:
                import joblib
                loaded = joblib.load(self.load_policy)
                self.policy = loaded['policy']
                self.baseline = loaded['baseline']
            
            
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            self.start_worker()
            start_time = time.time()
            
            
            assign_op_mean = self.policy.all_params['latent_means'].assign(np.zeros((self.num_total_tasks, self.latent_dim)))
            assign_op_std = self.policy.all_params['latent_stds'].assign(np.zeros((self.num_total_tasks, self.latent_dim)))
            sess.run(assign_op_mean)
            sess.run(assign_op_std)
            
            
            env = self.env

            while 'sample_goals' not in dir(env):
                env = env.wrapped_env
            learner_env_goals = env.sample_goals(self.meta_batch_size)
                   
            
                    #learner_env_goals = np.array([3])
            self.policy.switch_to_init_dist()  # Switch to pre-update policy

         
           
            
                                  
            paths = self.obtain_samples(itr=0, reset_args=learner_env_goals)
            
            
            for task in range(self.meta_batch_size):
                for traj in range(self.num_trajs):
                    x = paths[task][traj]["observations"][:,0]
                    y = paths[task][traj]["observations"][:,1]
                    plt.plot(x,y)
            plt.xlim(-2, 2)
            plt.ylim(-.5,2.5)
        
            plt.savefig("/home/russellm/Plots/"+self.visitationFolder+"/"+self.visitationFile+".png")
    
    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
