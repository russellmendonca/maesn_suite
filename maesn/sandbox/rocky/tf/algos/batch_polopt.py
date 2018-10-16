import time
from rllab.algos.base import RLAlgorithm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Policy
import tensorflow as tf
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.spaces.discrete import Discrete
import os.path as osp

class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
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
            batch_size=5000,
            max_path_length=500,
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
            load_policy=None,
            reset_arg=None,
            latent_dim=4,
            num_total_tasks=10,
            noise_opt=False,
            joint_opt=False,
            improve = False,
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
        :param batch_size: Number of samples per iteration.
        :param max_path_length: Maximum length of a single rollout.
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
        self.noise_opt = noise_opt
        self.joint_opt = joint_opt
        self.latent_dim = latent_dim
        self.num_total_tasks = num_total_tasks
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
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
        self.improve = improve
        if sampler_cls is None:
            #sampler_cls = VectorizedSampler
            sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
        self.sampler = sampler_cls(self, **sampler_args)
        self.reset_arg = reset_arg
        self.latent_dist = DiagonalGaussian(self.latent_dim)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr):
        return self.sampler.obtain_samples(itr, reset_args=self.reset_arg)

    def process_samples(self, itr, paths, noise_opt=False, joint_opt=False):
        return self.sampler.process_samples(itr, paths, task_idx=0, noise_opt=noise_opt, joint_opt=joint_opt)

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        
        with tf.Session(config=config) as sess:

        
            if self.load_policy is not None:
                import joblib
                loaded = joblib.load(self.load_policy)
                self.policy = loaded['policy']
                self.baseline = loaded['baseline']
            
            self.init_opt()
      
            uninit_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.initialize_variables(uninit_vars))


##############################################
            #Code to plot the latents
            # from matplotlib.patches import Ellipse

            # fig = plt.figure(0)
            # ax = fig.add_subplot(111, aspect='equal')
            # ax.set_xlim(-5, 5)
            # ax.set_ylim(-5, 5)

            # lm = sess.run(self.policy.all_params['latent_means'])
            # lstd = np.exp(sess.run(self.policy.all_params['latent_stds']))
            # for j in range(self.num_total_tasks):    
            #     e = Ellipse(xy=lm[j], width=lstd[j][0], height=lstd[j][1], fill = False)
            #     ax.add_artist(e)
#################################################


          
            self.start_worker()
            start_time = time.time()
         
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):

                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                 

                    logger.log("Processing samples...")

                    #if self.joint_opt:
                        #joint_opt handled by only_latents variable in the policy
                    samples_data, samples_data_latent = self.process_samples(itr, paths, noise_opt=self.noise_opt, joint_opt=True)
                    #else:
                        #samples_data = self.process_samples(itr, paths, noise_opt=self.noise_opt)
                    
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                   
                    logger.log("Optimizing policy...")
                    if self.improve:
                        self.optimize_policy(itr, samples_data)
                    elif self.joint_opt:
                        self.optimize_policy(itr, samples_data, samples_data_latent)
                    else:
                        self.optimize_policy(itr, samples_data_latent)
                    
                    logger.log("Saving snapshot...")
                    
                    if self.improve or self.joint_opt:
                        params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    else:
                        params = self.get_itr_snapshot(itr, samples_data_latent)

                    if self.store_paths:
                       params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    #plt.clf()

                    #observations = samples_data["observations"]
                    
                    
                   # plt.title("Trajectory Plot for Iteration "+str(itr)) 
                   # plt.xlim([-.7, .7])
                   # plt.ylim([-.7, .7])
                    
                    
                    #xpaths = np.split(observations[:,1], self.max_path_length)
                    #ypaths = np.split(observations[:,0], self.max_path_length)


                    #for xpath, ypath in zip(xpaths, ypaths):

                     #   plt.plot(xpath, ypath)

                    #plt.savefig(osp.join(logger.get_snapshot_dir(), str(itr)+".png"))

                   
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")


        self.shutdown_worker()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
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
