import numpy as np
from collections import OrderedDict

from rllab.misc import ext
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.spaces.box import Box

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian # This is just a util class. No params.
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.rocky.tf.misc import tensor_utils

from matplotlib import pyplot as plt
#import seaborn as sns

import itertools
import time

import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from sandbox.rocky.tf.core.utils import make_input, _create_param, add_param, make_dense_layer, forward_dense_layer, make_param_layer, forward_param_layer

load_params = True



class MAMLGaussianMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=tf.nn.tanh,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.identity,
            mean_network=None,
            std_network=None,
            std_parametrization='exp',
            grad_step_size=1.0,
            stop_grad=False,
            latent_dim=4,
            param_deg=0,
            num_total_tasks=10,
            trainableLatents = True
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :param grad_step_size: the step size taken in the learner's gradient update, sample uniformly if it is a range e.g. [0.1,1]
        :param stop_grad: whether or not to stop the gradient through the gradient.
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)
        self.only_latents = param_deg
        self.latent_dim = latent_dim
        self.num_total_tasks = num_total_tasks
        obs_dim = env_spec.observation_space.flat_dim
        self.obs_dim = obs_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.input_shape = (None, obs_dim + latent_dim,)
        self.step_size = grad_step_size
        self.stop_grad = stop_grad
        self.trainableLatents = trainableLatents
        if type(self.step_size) == list:
            raise NotImplementedError('removing this since it didnt work well')

        # create network
        if mean_network is None:
            self.all_params = self.create_MLP(  # TODO: this should not be a method of the policy! --> helper
                name="mean_network",
                output_dim=self.action_dim,
                hidden_sizes=hidden_sizes,
                latent_dim=self.latent_dim,
                num_total_tasks=num_total_tasks
            )
            self.input_tensor, self.task_idx, self.noise, self.zs, self.output_tensor_tosample = self.forward_MLP('mean_network', self.all_params,
                reuse=None # Need to run this for batch norm
            )
            forward_mean = lambda x, task_idx, noise, params, is_train: self.forward_MLP('mean_network', params,
                input_tensor=x, task_idx=task_idx, noise=noise, is_training=is_train)[-1]
        else:
            raise NotImplementedError('Not supported.')

        if std_network is not None:
            raise NotImplementedError('Not supported.')
        else:
            if adaptive_std:
                raise NotImplementedError('Not supported.')
            else:
                if std_parametrization == 'exp':
                    init_std_param = np.log(init_std)
                elif std_parametrization == 'softplus':
                    init_std_param = np.log(np.exp(init_std) - 1)
                else:
                    raise NotImplementedError
                self.all_params['std_param'] = make_param_layer(
                    num_units=self.action_dim,
                    param=tf.constant_initializer(init_std_param),
                    name="output_std_param",
                    trainable=learn_std,
                )
                forward_std = lambda x, params: forward_param_layer(x, params['std_param'])
            self.all_param_vals = None

            # unify forward mean and forward std into a single function
            self._forward = lambda obs, task_idx, noise, params, is_train: (
                    forward_mean(obs, task_idx, noise, params, is_train), forward_std(obs, params))

            self.std_parametrization = std_parametrization

            if std_parametrization == 'exp':
                min_std_param = np.log(min_std)
            elif std_parametrization == 'softplus':
                min_std_param = np.log(np.exp(min_std) - 1)
            else:
                raise NotImplementedError

            self.min_std_param = min_std_param

            self._dist = DiagonalGaussian(self.action_dim)

            self._cached_params = {}

            super(MAMLGaussianMLPPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(self.input_tensor, self.task_idx, self.noise, dict(), is_training=False)
            mean_var = dist_info_sym["mean"]
            log_std_var = dist_info_sym["log_std"]

            # pre-update policy
            self._init_f_dist = tensor_utils.compile_function(
                inputs=[self.input_tensor, self.task_idx, self.noise],
                outputs=[mean_var, log_std_var],
            )
            self._cur_f_dist = self._init_f_dist


    @property
    def vectorized(self):
        return True

    def set_init_surr_obj(self, input_list, surr_objs_tensor, surr_objs_latent_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor
        self.surr_objs_latent = surr_objs_latent_tensor

    def compute_updated_dists(self, samples, samples_latent, plot=None):
        """ Compute fast gradients once per iteration and pull them out of tensorflow for sampling with the post-update policy.
        """
        start = time.time()
        num_tasks = len(samples)
        # param_keys = self.all_params.keys()

        param_keys = []
        param_keys_latent = []
        all_keys = list(self.all_params.keys())
        all_keys.remove('latent_means_stepsize')
        all_keys.remove('latent_stds_stepsize')

        for key in all_keys:
            if 'latent' not in key:
                param_keys.append(key)
            else:
                param_keys_latent.append(key)


        update_param_keys = param_keys
        update_param_keys_latent = param_keys_latent

        sess = tf.get_default_session()


        obs_list, action_list, adv_list, noise_list, task_idx_list = [], [], [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples[i],
                    'observations', 'actions', 'advantages', 'noises', 'task_idxs')
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])
            noise_list.append(inputs[3])
            task_idx_list.append(inputs[4])


        adv_list_latent, zs_list_latent, task_idx_list_latent = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples_latent[i],'advantages', 'noises', 'task_idxs')
            means = tf.gather(self.all_params['latent_means'], inputs[-1])
            stds = tf.gather(self.all_params['latent_stds'], inputs[-1])
            zs = sess.run(means + inputs[-2]*tf.exp(stds))
            adv_list_latent.append(inputs[0])
            zs_list_latent.append(zs)
            task_idx_list_latent.append(inputs[2])


        inputs = obs_list + action_list  + adv_list + noise_list + task_idx_list
        inputs += adv_list_latent + zs_list_latent + task_idx_list_latent
        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
        #TODO: Maybe change?
        init_param_values = None
        if self.all_param_vals is not None:
            init_param_values = self.get_variable_values(self.all_params)

        step_size = self.step_size
        for i in range(num_tasks):
            if self.all_param_vals is not None:
                self.assign_params(self.all_params, self.all_param_vals[i])


        step_sizes_sym = {}
        for key in all_keys:
            step_sizes_sym[key] = step_size
        step_sizes_sym['latent_means'] = self.all_params['latent_means_stepsize']
        step_sizes_sym['latent_stds'] = self.all_params['latent_stds_stepsize']

        if 'all_fast_params_tensor' not in dir(self):
            # make computation graph once
            self.all_fast_params_tensor = []
            for i in range(num_tasks):
                gradients = dict(zip(update_param_keys, tf.gradients(self.only_latents*self.surr_objs[i], [self.all_params[key] for key in update_param_keys])))
                gradients_latent = dict(zip(update_param_keys_latent, tf.gradients(self.surr_objs_latent[i], [self.all_params[key] for key in update_param_keys_latent])))
                gradients.update(gradients_latent)
                fast_params_tensor = OrderedDict(zip(all_keys, [self.all_params[key] - step_sizes_sym[key]*tf.convert_to_tensor(gradients[key]) for key in all_keys]))

                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
        
        
        #PLOTTING CODE
        if plot!=None:
            folderName, kl,  itr = plot[0], plot[1], plot[2]
            lmeans_plot , lstd_plot = [], []
            for i in range(len(self.all_param_vals)):
                lmeans_plot.append(self.all_param_vals[i]["latent_means"][i])
                lstd_plot.append(np.exp(self.all_param_vals[i]["latent_stds"][i]))
                
            self.plotLatents(folderName, str(kl), str(itr), "1", lmeans_plot, lstd_plot)   
                

        outputs = []
        self._cur_f_dist_i = {}
        inputs = tf.split(self.input_tensor, num_tasks, 0)
        task_idxs = tf.split(self.task_idx, num_tasks, 0)
        noises = tf.split(self.noise, num_tasks, 0)

        for i in range(num_tasks):
            # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
            task_inp = inputs[i]
            task_idx_inp = task_idxs[i]
            noise_inp = noises[i]
            info, _ = self.dist_info_sym(task_inp, task_idx_inp, noise_inp, dict(), all_params=self.all_param_vals[i],
                    is_training=False)

            outputs.append([info['mean'], info['log_std']])

        self._cur_f_dist = tensor_utils.compile_function(
            inputs = [self.input_tensor, self.task_idx, self.noise],
            outputs = outputs,
        )
        total_time = time.time() - start
        logger.record_tabular("ComputeUpdatedDistTime", total_time)
    
    def heatmapLatents(self, folderName, kl, itr, step, lm, lstd):
    
        #import ipdb
        #ipdb.set_trace()
        plt.clf() 
        sns.set()
       
        
        ax1 = sns.heatmap(lm)
        plt.savefig("/home/russellm/Plots/"+folderName+"/itr_"+itr+"_kl_"+kl+"_step_"+step+"mean.png")
        plt.clf()
        ax2 = sns.heatmap(lstd)
        plt.savefig("/home/russellm/Plots/"+folderName+"/itr_"+itr+"_kl_"+kl+"_step_"+step+"lstd.png")
    
    
    
    
    def plotLatents(self, folderName, kl, itr, step, lm, lstd):
    
        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse
        plt.clf()
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

        
        e = Ellipse(xy=np.zeros(2), width=1, height=1, fill = False, color = 'b')
        ax.add_artist(e)
        
        for j in range(np.shape(lm)[0]):
            
           
            e = Ellipse(xy=lm[j], width=lstd[j][0], height=lstd[j][1], fill = False)
            ax.add_artist(e)
       
        plt.savefig("/home/russellm/Plots/"+folderName+"/itr_"+itr+"_kl_"+kl+"_step_"+step+".png")
    
    
    def get_variable_values(self, tensor_dict):
        sess = tf.get_default_session()
        result = sess.run(tensor_dict)
        return result

    def assign_params(self, tensor_dict, param_values):
        if 'assign_placeholders' not in dir(self):
            # make computation graph, if it doesn't exist; then cache it for future use.
            self.assign_placeholders = {}
            self.assign_ops = {}
            for key in tensor_dict.keys():
                self.assign_placeholders[key] = tf.placeholder(tf.float32)
                self.assign_ops[key] = tf.assign(tensor_dict[key], self.assign_placeholders[key])

        feed_dict = {self.assign_placeholders[key]:param_values[key] for key in tensor_dict.keys()}
        sess = tf.get_default_session()
        sess.run(self.assign_ops, feed_dict)


    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        self._cur_f_dist = self._init_f_dist
        self._cur_f_dist_i = None
        self.all_param_vals = None

    def dist_info_sym(self, obs_var, task_idx, noise, state_info_vars=None, all_params=None, is_training=True):
        # This function constructs the tf graph, only called during beginning of meta-training
        # obs_var - observation tensor
        # mean_var - tensor for policy mean
        # std_param_var - tensor for policy std before output
        return_params=True
        if all_params is None:
            return_params=False
            all_params = self.all_params

        mean_var, std_param_var = self._forward(obs_var, task_idx, noise, all_params, is_training)
        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        if self.std_parametrization == 'exp':
            log_std_var = std_param_var
        elif self.std_parametrization == 'softplus':
            log_std_var = tf.log(tf.log(1. + tf.exp(std_param_var)))
        else:
            raise NotImplementedError
        if return_params:
            return dict(mean=mean_var, log_std=log_std_var), all_params
        else:
            return dict(mean=mean_var, log_std=log_std_var)

    def updated_dist_info_sym(self, task_id, surr_obj, surr_obj_latent, new_obs_var, new_task_idx_var, new_noise_var, params_dict=None, params_dict_latent=None, is_training=True):
        """ symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one grad step.
        """
        old_params_dict = params_dict
        old_params_dict_latent = params_dict_latent

        step_size = self.step_size

        param_keys = []
        param_keys_latent = []

        all_keys = list(self.all_params.keys())
        all_keys.remove('latent_means_stepsize')
        all_keys.remove('latent_stds_stepsize')

        for key in all_keys:
            if 'latent' not in key:
                param_keys.append(key)
            else:
                param_keys_latent.append(key)

        update_param_keys = param_keys
        update_param_keys_latent = param_keys_latent

        no_update_param_keys = []
        no_update_param_keys_latent = []

        unconverted_grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
        unconverted_grads_latent = tf.gradients(surr_obj_latent, [old_params_dict_latent[key] for key in update_param_keys_latent])
        grads = []
        grads_latent = []
        for grad in unconverted_grads:
            if grad is not None:
                grad = tf.convert_to_tensor(grad)
            grads.append(grad)
        for grad in unconverted_grads_latent:
            if grad is not None:
                grad = tf.convert_to_tensor(grad)
            grads_latent.append(grad)

        gradients = dict(zip(update_param_keys, grads))
        gradients_latent = dict(zip(update_param_keys_latent, grads_latent))
        params_dict = dict(zip(update_param_keys, [old_params_dict[key] - self.only_latents*step_size*gradients[key] for key in update_param_keys]))
        
        step_sizes_sym = {}
        step_sizes_sym['latent_means'] = self.all_params['latent_means_stepsize']
        step_sizes_sym['latent_stds'] = self.all_params['latent_stds_stepsize']


        params_dict_latent = dict(zip(update_param_keys_latent, [old_params_dict_latent[key] - step_sizes_sym[key]*gradients_latent[key] for key in update_param_keys_latent]))
        params_dict.update(params_dict_latent)
        return self.dist_info_sym(new_obs_var, new_task_idx_var, new_noise_var, all_params=params_dict, is_training=is_training)

    # def updated_dist_info_sym(self, task_id, surr_obj, new_obs_var, new_task_idx_var, new_noise_var, params_dict=None, is_training=True):
    #     """ symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
    #     Called more than once if you want to do more than one grad step.
    #     """
    #     old_params_dict = params_dict

    #     step_size = self.step_size

    #     if old_params_dict == None:
    #         old_params_dict = self.all_params
    #     param_keys = self.all_params.keys()
    #     update_param_keys = param_keys
    #     no_update_param_keys = []

    #     unconverted_grads = tf.gradients(surr_obj, [old_params_dict[key] for key in update_param_keys])
    #     grads = []
    #     for grad in unconverted_grads:
    #         if grad is not None:
    #             grad = tf.convert_to_tensor(grad)
    #         grads.append(grad)
    #     if self.stop_grad:
    #         grads = [tf.stop_gradient(grad) for grad in grads]

    #     gradients = dict(zip(update_param_keys, grads))
    #     params_dict = dict(zip(update_param_keys, [old_params_dict[key] - step_size*gradients[key] for key in update_param_keys]))
    #     for k in no_update_param_keys:
    #         params_dict[k] = old_params_dict[k]

    #     return self.dist_info_sym(new_obs_var, new_task_idx_var, new_noise_var, all_params=params_dict, is_training=is_training)


    @overrides
    def get_action(self, observation, task_idx, noise, idx=None):
        # this function takes a numpy array observations and outputs randomly sampled actions.
        # idx: index corresponding to the task/updated policy.
        flat_obs = self.observation_space.flatten(observation)
        f_dist = self._cur_f_dist
        mean, log_std = [x[0] for x in f_dist([flat_obs], [task_idx], [noise])]
        rnd = np.random.normal(size=mean.shape)
        action = mean + rnd * np.exp(log_std)
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations, task_idxs, noises):
        # this function takes a numpy array observations and outputs sampled actions.
        # Assumes that there is one observation per post-update policy distr
        flat_obs = self.observation_space.flatten_n(observations)
        result = self._cur_f_dist(flat_obs, task_idxs, noises)

        if len(result) == 2:
            # NOTE - this code assumes that there aren't 2 meta tasks in a batch
            means, log_stds = result
        else:
            means = np.array([res[0] for res in result])[:,0,:]
            log_stds = np.array([res[1] for res in result])[:,0,:]

        rnd = np.random.normal(size=means.shape)
        actions = means + rnd * np.exp(log_stds)
        return actions, dict(mean=means, log_std=log_stds)

    @property
    def distribution(self):
        return self._dist
        
    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
        
            if self.trainableLatents:
                params = tf.trainable_variables()
            else:
               
                params = [ p for p in tf.trainable_variables() if 'latent' not in p.name]
              
               
                        
        else:
            params = tf.global_variables()
        
        params = [p for p in params if p.name.startswith('mean_network') or p.name.startswith('output_std_param')]
        params = [p for p in params if 'Adam' not in p.name]
        ##import ipdb
        #ipdb.set_trace()
        return params

    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, latent_dim, num_total_tasks, hidden_sizes,
                   hidden_W_init=tf_layers.xavier_initializer(), hidden_b_init=tf.zeros_initializer(),
                   output_W_init=tf_layers.xavier_initializer(), output_b_init=tf.zeros_initializer(),
                   weight_normalization=False,
                   ):
        all_params = OrderedDict()

        cur_shape = self.input_shape
        with tf.variable_scope(name):
            for idx, hidden_size in enumerate(hidden_sizes):
                W, b, cur_shape = make_dense_layer(
                    cur_shape,
                    num_units=hidden_size,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                    weight_norm=weight_normalization,
                )
                all_params['W' + str(idx)] = W
                all_params['b' + str(idx)] = b
            W, b, _ = make_dense_layer(
                cur_shape,
                num_units=output_dim,
                name='output',
                W=output_W_init,
                b=output_b_init,
                weight_norm=weight_normalization,
            )
            all_params['W' + str(len(hidden_sizes))] = W
            all_params['b'+str(len(hidden_sizes))] = b
            all_params['latent_means'] = tf.get_variable("latent_means", shape=(num_total_tasks, latent_dim), initializer=tf.random_normal_initializer)
            all_params['latent_stds'] = tf.get_variable("latent_stds", shape=(num_total_tasks, latent_dim), initializer=tf.zeros_initializer)
            all_params['latent_means_stepsize'] = tf.Variable(self.step_size*tf.ones((latent_dim,)), name="latent_means_stepsize")
            all_params['latent_stds_stepsize'] = tf.Variable(self.step_size*tf.ones((latent_dim,)), name="latent_stds_stepsize")
            
        return all_params

    def forward_MLP(self, name, all_params, input_tensor=None, task_idx=None, noise=None,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                l_in = make_input(shape=(None, self.obs_dim,), input_var=None, name='input')
                l_tasks = tf.placeholder(tf.int32, shape=(None,), name="task_idxs")
                l_noise = make_input(shape=(None, self.latent_dim), input_var=None, name='noise')
            else:
                l_in = input_tensor
                l_tasks = task_idx
                l_noise = noise

            chosen_latent_means = tf.gather(all_params['latent_means'], l_tasks)
            chosen_latent_stds = tf.gather(all_params['latent_stds'], l_tasks)
            zs = chosen_latent_means+ l_noise*tf.exp(chosen_latent_stds)
            
           
            
            l_hid = tf.concat([l_in, zs], axis=1)

            for idx in range(self.n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W'+str(idx)], all_params['b'+str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=self.hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W'+str(self.n_hidden)], all_params['b'+str(self.n_hidden)],
                                         batch_norm=False, nonlinearity=self.output_nonlinearity,
                                         )
            return l_in, l_tasks, l_noise, zs, output
            
    def get_params(self, all_params=False, **tags):
        """
        Get the list of parameters (symbolically), filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(all_params, **tags)
        return self._cached_params[tag_tuple]

    def get_param_values(self, all_params=False, **tags):
        params = self.get_params(all_params, **tags)
        param_values = tf.get_default_session().run(params)
        return flatten_tensors(param_values)

    def log_diagnostics(self, paths, prefix=''):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular(prefix+'AveragePolicyStd', np.mean(np.exp(log_stds)))

    #### code largely not used after here except when resuming/loading a policy. ####
    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        # Not used
        import pdb; pdb.set_trace()
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def get_param_dtypes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, all_params=False, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(all_params, **tags)
            param_values = tf.get_default_session().run(params)
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def set_param_values(self, flattened_params, all_params=False, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(all_params, **tags))
        ops = []
        feed_dict = dict()
        for param, dtype, value in zip(
                self.get_params(all_params, **tags),
                self.get_param_dtypes(all_params, **tags),
                param_values):
            if param not in self._cached_assign_ops:
                assign_placeholder = tf.placeholder(dtype=param.dtype.base_dtype)
                assign_op = tf.assign(param, assign_placeholder)
                self._cached_assign_ops[param] = assign_op
                self._cached_assign_placeholders[param] = assign_placeholder
            ops.append(self._cached_assign_ops[param])
            feed_dict[self._cached_assign_placeholders[param]] = value.astype(dtype)
            if debug:
                print("setting value of %s" % param.name)
        tf.get_default_session().run(ops, feed_dict=feed_dict)

    def flat_to_params(self, flattened_params, all_params=False, **tags):
        return unflatten_tensors(flattened_params, self.get_param_shapes(all_params, **tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values(all_params=True)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            tf.get_default_session().run(tf.variables_initializer(self.get_params(all_params=True)))
            self.set_param_values(d["params"], all_params=True)


