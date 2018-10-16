from contextlib import contextmanager
import itertools
import numpy as np
import sandbox.rocky.tf.core.layers as L
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import tensorflow as tf
from sandbox.rocky.tf.core.utils import make_input, _create_param, add_param, make_dense_layer, forward_dense_layer, make_param_layer, forward_param_layer
from collections import OrderedDict

tf_layers = None
load_params = True

@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class MAMLCategoricalMLPPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
            prob_network=None,
            grad_step_size=1.0,
            latent_dim=4,
            num_total_tasks=10,
            only_latent=0
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :param grad_step_size: the step size taken in the learner's gradient update, sample uniformly if it is a range e.g. [0.1,1]
        :return:
        """
        Serializable.quick_init(self, locals())
        self.only_latent = only_latent
        assert isinstance(env_spec.action_space, Discrete)
        obs_dim = env_spec.observation_space.flat_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_total_tasks = num_total_tasks
        self.action_dim = env_spec.action_space.n
        self.n_hidden = len(hidden_sizes)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.input_shape = (None, obs_dim + self.latent_dim,)
        self.step_size = grad_step_size

        if prob_network is None:
            self.all_params = self.create_MLP(
                output_dim=self.action_dim,
                hidden_sizes=hidden_sizes,
                name="prob_network",
                latent_dim=latent_dim,
                num_total_tasks=num_total_tasks
            )
        self._l_obs, self._l_tasks, self._l_noise, self.zs, self._l_prob = self.forward_MLP('prob_network', self.all_params,
            n_hidden=len(hidden_sizes), input_shape=(obs_dim,),
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=tf.nn.softmax, reuse=None)

        # if you want to input your own tensor.
        self._forward_out = lambda x, task_idx, noise, params, is_train: self.forward_MLP('prob_network', params,
            n_hidden=len(hidden_sizes), hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=tf.nn.softmax, input_tensor=x, task_idx=task_idx, noise=noise, is_training=is_train)[-1]


        self._init_f_prob = tensor_utils.compile_function(
            [self._l_obs, self._l_tasks, self._l_noise],
            [self._l_prob])
        self._cur_f_prob = self._init_f_prob

        self._dist = Categorical(self.action_dim)
        self._cached_params = {}
        super(MAMLCategoricalMLPPolicy, self).__init__(env_spec)


    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, task_idx_var, noise_var, state_info_vars=None, all_params=None, is_training=True):
        # sym means symbolic here.
        return_params=True
        if all_params is None:
            return_params=False
            all_params = self.all_params

        output = self._forward_out(tf.cast(obs_var,tf.float32),tf.cast(task_idx_var,tf.int32),tf.cast(noise_var,tf.float32),all_params, is_training)

        if return_params:
            return dict(prob=output), all_params
        else:
            return dict(prob=output)

    def updated_dist_info_sym(self, task_id, surr_obj, surr_obj_latent, new_obs_var, new_task_idx_var, new_noise_var, params_dict=None, params_dict_latent=None, is_training=True):
        """ symbolically create MAML graph, for the meta-optimization, only called at the beginning of meta-training.
        Called more than once if you want to do more than one grad step.
        """
        old_params_dict = params_dict
        old_params_dict_latent = params_dict_latent

        step_size = self.step_size

        if old_params_dict == None:
            print("SHOULDN'T be happening, old params dict is None")
            old_params_dict = self.all_params


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

        unconverted_grads = tf.gradients(surr_obj, [old_params_dict[key] for key in param_keys])
        unconverted_grads_latent = tf.gradients(surr_obj_latent, [old_params_dict_latent[key] for key in param_keys_latent])
        grads = []
        for grad in unconverted_grads:
            if grad is not None:
                grad = tf.convert_to_tensor(grad)
            grads.append(grad)

        grads_latent = []
        for grad in unconverted_grads_latent:
            if grad is not None:
                grad = tf.convert_to_tensor(grad)
            grads_latent.append(grad)

        gradients = dict(zip(param_keys, grads))
        gradients_latent = dict(zip(param_keys_latent, grads_latent))

        params_dict = dict(zip(param_keys, [old_params_dict[key] - self.only_latent*step_size*gradients[key] for key in param_keys]))
        step_sizes_sym = {}
        step_sizes_sym['latent_means'] = self.all_params['latent_means_stepsize']
        step_sizes_sym['latent_stds'] = self.all_params['latent_stds_stepsize']
        
        params_dict_latent = dict(zip(param_keys_latent, [old_params_dict_latent[key] - step_sizes_sym[key]*gradients_latent[key] for key in param_keys_latent]))
        params_dict.update(params_dict_latent)

        return self.dist_info_sym(new_obs_var, new_task_idx_var, new_noise_var, all_params=params_dict, is_training=is_training)

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self._f_prob(obs))

    def switch_to_init_dist(self):
        # switch cur policy distribution to pre-update policy
        self._cur_f_prob = self._init_f_prob
        self.all_param_vals = None

    def set_init_surr_obj(self, input_list, surr_objs_tensor, surr_objs_latent_tensor):
        """ Set the surrogate objectives used the update the policy
        """
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor
        self.surr_objs_latent = surr_objs_latent_tensor

    def compute_updated_dists(self, samples, samples_latent, plot=None):
        """ Compute fast gradients once and pull them out of tensorflow for sampling.
        """
        num_tasks = len(samples)
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


        adv_list_latent, z_list_latent, task_idx_list_latent = [], [], []
        for i in range(num_tasks):
            inputs = ext.extract(samples_latent[i],'advantages', 'noises', 'task_idxs')
            means = tf.gather(self.all_params['latent_means'], inputs[-1])
            stds = tf.gather(self.all_params['latent_stds'], inputs[-1])
            zs = sess.run(means + inputs[-2]*tf.exp(stds))
            adv_list_latent.append(inputs[0])
            z_list_latent.append(zs)
            task_idx_list_latent.append(inputs[2])

        inputs = obs_list + action_list + adv_list + noise_list + task_idx_list
        inputs += adv_list_latent + z_list_latent + task_idx_list_latent

        # To do a second update, replace self.all_params below with the params that were used to collect the policy.
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
                gradients = dict(zip(param_keys, tf.gradients(self.only_latent*self.surr_objs[i], [self.all_params[key] for key in param_keys])))
                gradients_latent = dict(zip(param_keys_latent, tf.gradients(self.surr_objs_latent[i], [self.all_params[key] for key in param_keys_latent])))
                gradients.update(gradients_latent)
                fast_params_tensor = OrderedDict(zip(all_keys, [self.all_params[key] - step_sizes_sym[key]*tf.convert_to_tensor(gradients[key]) for key in all_keys]))
                self.all_fast_params_tensor.append(fast_params_tensor)

        # pull new param vals out of tensorflow, so gradient computation only done once
        self.all_param_vals = sess.run(self.all_fast_params_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
        
        #PLOTTING CODE
        if plot!=None:
            folderName, itr = plot[0], plot[1]
            lmeans_plot , lstd_plot = [], []
            for i in range(len(self.all_param_vals)):
                lmeans_plot.append(self.all_param_vals[i]["latent_means"][i])
                lstd_plot.append(np.exp(self.all_param_vals[i]["latent_stds"][i]))
                
            self.visualizePaths(folderName, str(itr), "1", lmeans_plot, lstd_plot)  
        

        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)
        if init_param_values is not None:
            self.assign_params(self.all_params, init_param_values)

        outputs = []
        inputs = tf.split(self._l_obs, num_tasks, 0)
        task_inputs = tf.split(self._l_tasks, num_tasks, 0)
        noise_inputs = tf.split(self._l_noise, num_tasks, 0)
        for i in range(num_tasks):
            # TODO - use a placeholder to feed in the params, so that we don't have to recompile every time.
            task_inp = inputs[i]
            task_idx_inp = task_inputs[i]
            noise_inp = noise_inputs[i]
            info, _ = self.dist_info_sym(task_inp, task_idx_inp, noise_inp, dict(), all_params=self.all_param_vals[i],
                    is_training=False)

            outputs.append([info['prob']])

        self._cur_f_prob = tensor_utils.compile_function(
            inputs = [self._l_obs, self._l_tasks, self._l_noise],
            outputs = outputs,
        )
    def visualizePaths(self, folderName, itr, step, lm, lstd):
    
        from matplotlib import pyplot as plt
        from matplotlib.patches import Ellipse
        plt.clf()
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

        
        e = Ellipse(xy=np.zeros(2), width=1, height=1, fill = False, color = 'b')
        ax.add_artist(e)
        for j in range(np.shape(lm)[0]):
            
           
            e = Ellipse(xy=lm[j], width=lstd[j][0], height=lstd[j][1], fill = False)
            ax.add_artist(e)
       
        plt.savefig("/home/russellm/Plots/"+folderName+"/itr_"+itr+"_step_"+step+".png")
    
        
    
    
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

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation, task_idx, noise):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._cur_f_prob([flat_obs], [task_idx], [noise])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations, task_idxs, noises):
        flat_obs = self.observation_space.flatten_n(observations)
        result = self._cur_f_prob(flat_obs, task_idxs, noises)
        if len(result) == 1:
            probs = result[0]
        else:
            #import pdb; pdb.set_trace()
            # TODO - I think this is correct but not sure.
            probs = np.array(result)[:,0,0,:]
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist


    # This makes all of the parameters.
    def create_MLP(self, name, output_dim, latent_dim, num_total_tasks, hidden_sizes,
                   hidden_W_init=L.XavierUniformInitializer(), hidden_b_init=tf.zeros_initializer,
                   output_W_init=L.XavierUniformInitializer(), output_b_init=tf.zeros_initializer,
                   weight_normalization=False,
                   ):
        input_shape = self.input_shape
        cur_shape = input_shape
        with tf.variable_scope(name):
            all_params = {}
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

    def forward_MLP(self, name, all_params,input_tensor=None, task_idx=None, noise=None, 
                    input_shape=None, n_hidden=-1,
                    hidden_nonlinearity=tf.identity, output_nonlinearity=tf.identity,
                    batch_normalization=False, reuse=True, is_training=False):
        # is_training and reuse are for batch norm, irrelevant if batch_norm set to False
        # set reuse to False if the first time this func is called.
        with tf.variable_scope(name):
            if input_tensor is None:
                assert input_shape is not None
                l_in = make_input(shape=(None,)+input_shape, input_var=None, name='input')
                l_tasks = tf.placeholder(tf.int32, shape=(None,), name="task_idxs")
                l_noise = make_input(shape=(None, self.latent_dim), input_var=None, name='noise')
            else:
                l_in = input_tensor
                l_tasks = task_idx
                l_noise = noise

            chosen_latent_means = tf.gather(all_params['latent_means'], l_tasks)
            chosen_latent_stds = tf.gather(all_params['latent_stds'], l_tasks)
            zs = chosen_latent_means + l_noise*tf.exp(chosen_latent_stds)
            # l_hid = l_in
            l_hid = tf.concat([l_in, zs], axis=1)


            for idx in range(n_hidden):
                l_hid = forward_dense_layer(l_hid, all_params['W'+str(idx)], all_params['b'+str(idx)],
                                            batch_norm=batch_normalization,
                                            nonlinearity=hidden_nonlinearity,
                                            scope=str(idx), reuse=reuse,
                                            is_training=is_training
                                            )
            output = forward_dense_layer(l_hid, all_params['W'+str(n_hidden)], all_params['b'+str(n_hidden)],
                                         batch_norm=False, nonlinearity=output_nonlinearity,
                                         )
            return l_in, l_tasks, l_noise, zs, output


    def get_params_internal(self, all_params=False, **tags):
        if tags.get('trainable', False):
            params = tf.trainable_variables()
        else:
            params = tf.all_variables()

        # TODO - this is hacky...
        params = [p for p in params if p.name.startswith('prob_network')]
        params = [p for p in params if 'Adam' not in p.name]

        return params

    def log_diagnostics(self, paths, prefix=''):
        pass

