from matplotlib.patches import Ellipse
from collections import OrderedDict
from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt_ada import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

class VPG(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            default_step,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.default_step_size = default_step
        self.opt_info = None
        super(VPG, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)

    def make_vars(self):
        
        
        obs_var = self.env.observation_space.new_tensor_variable(
                'obs' ,
                extra_dims=1,
            )
        action_var = self.env.action_space.new_tensor_variable(
                'action' ,
                extra_dims=1,
            )
        adv_var = tensor_utils.new_tensor(
                name='advantage' ,
                ndim=1 , dtype=tf.float32,
            )
        noise_var = tf.placeholder(dtype=tf.float32, shape=[None, self.latent_dim], name='noise' )
        
        task_idx_var = tensor_utils.new_tensor(
                name='task_idx' ,
                ndim=1 , dtype=tf.int32,
            )

        return obs_var, action_var, adv_var, noise_var, task_idx_var


    def make_vars_latent(self):
        # lists over the meta_batch_size
       
       
        adv_var = tensor_utils.new_tensor(
            name='advantage_latent' ,
            ndim=1 , dtype=tf.float32,
        )
        
        z_var = tf.placeholder(dtype=tf.float32, shape=[None, self.latent_dim], name='zs_latent' )
        task_idx_var = tensor_utils.new_tensor(
            name='task_idx_latent' ,
            ndim=1 , dtype=tf.int32,
        )
        return adv_var, z_var, task_idx_var


    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent  # not supported

       

        #obs_var, action_var, adv_var, noise_var, task_idx_var = self.make_vars()
        adv_var_latent, z_var_latent, task_idx_var_latent = self.make_vars_latent()

        #self.input_list_for_grad = [obs_var,  action_var, adv_var, noise_var, task_idx_var, adv_var_latent, z_var_latent, task_idx_var_latent]
        self.input_list_for_grad = [ adv_var_latent, z_var_latent, task_idx_var_latent]

        
        #dist_info_vars, _ = self.policy.dist_info_sym(obs_var, task_idx_var, noise_var,  all_params=self.policy.all_params)
        #logli = self.policy._dist.log_likelihood_sym(action_var, dist_info_vars)
        #self.surr_obj = - tf.reduce_mean(logli * adv_var)
      

        means = tf.gather(self.policy.all_params['latent_means'], task_idx_var_latent)
        log_stds = tf.gather(self.policy.all_params['latent_stds'], task_idx_var_latent)
        dist_info_vars_latent = {"mean": means, "log_std": log_stds}

        logli_latent = self.latent_dist.log_likelihood_sym(z_var_latent, dist_info_vars_latent)
        self.surr_obj_latent = - tf.reduce_mean(logli_latent * adv_var_latent)

        all_keys = list(self.policy.all_params.keys())
       
        sess = tf.get_default_session()
        self.policy.all_param_vals = OrderedDict()
        for key in all_keys:
            self.policy.all_param_vals[key] = sess.run(self.policy.all_params[key]) 


       

    @overrides
    def optimize_policy(self, itr, samples_latent):
        logger.log("optimizing policy")
        # inputs = ext.extract(samples,
        #             'observations', 'actions', 'advantages', 'noises', 'task_idxs')
        
        # obs=inputs[0]
        # actions=inputs[1]
        # advantages=inputs[2]
        # noises=inputs[3]
        # task_idxs = inputs[4]

        latent_inputs = ext.extract(
            samples_latent,
            "advantages", "noises", "task_idxs"
        )
        latent_advantages = latent_inputs[0]
        latent_noises = latent_inputs[1]
        latent_task_idxs = latent_inputs[2]

        sess = tf.get_default_session()

        means = sess.run(tf.gather(self.policy.all_params['latent_means'], latent_task_idxs))
        logstds = sess.run(tf.gather(self.policy.all_params['latent_stds'], latent_task_idxs))
        #import ipdb
        #ipdb.set_trace()
        
        zs = means + latent_noises*np.exp(logstds)
        # self.num_top = 10
        # best_indices = advantages.argsort()[-self.num_top:][::-1]
        # good_noises = np.asarray([zs[ind] for ind in best_indices])
       # inputs = [obs,  actions, advantages, noises, task_idxs, latent_advantages, zs, latent_task_idxs]
        inputs = [latent_advantages, zs, latent_task_idxs]

      

        self.optimize(inputs, sess, itr)



        #loss_after = self.optimizer.loss(inputs)
        #logger.record_tabular("LossBefore", loss_before)
        #logger.record_tabular("LossAfter", loss_after)
        #return good_noises, zs


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

    def optimize(self, inputs, sess, itr):


        param_keys = []
        param_keys_latent = []
        all_keys = list(self.policy.all_params.keys())
        all_keys.remove('latent_means_stepsize')
        all_keys.remove('latent_stds_stepsize')

        for key in all_keys:
            if 'latent' not in key:
                param_keys.append(key)
            else:
                param_keys_latent.append(key)


        update_param_keys = param_keys
        update_param_keys_latent = param_keys_latent

        step_sizes_sym = {}
        for key in all_keys:
            step_sizes_sym[key] = self.default_step_size
        step_sizes_sym['latent_means'] = self.policy.all_params['latent_means_stepsize']
        step_sizes_sym['latent_stds'] = self.policy.all_params['latent_stds_stepsize']

     
        # if 'all_fast_params_tensor' not in dir(self):
        #     # make computation graph once
            #self.all_fast_params_tensor = []
            #for i in range(num_tasks):
        #gradients = dict(zip(update_param_keys, tf.gradients(self.policy.only_latents*self.surr_obj, [self.policy.all_params[key] for key in update_param_keys])))
        gradients_latent = dict(zip(update_param_keys_latent, tf.gradients(self.surr_obj_latent, [self.policy.all_params[key] for key in update_param_keys_latent])))
        #gradients.update(gradients_latent)
        
        update_tensor = OrderedDict(zip(update_param_keys_latent, [self.policy.all_params[key] - step_sizes_sym[key]*tf.convert_to_tensor(gradients_latent[key]) for key in update_param_keys_latent]))

                
       
        # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
        # these are the updated values of the params after the gradient step
        result = sess.run(update_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))

        
        self.policy.all_param_vals['latent_means'] = result['latent_means']
        self.policy.all_param_vals['latent_stds'] = result['latent_stds'] 

        #import ipdb
        #ipdb.set_trace()
        if itr>=1 :
            
            #if min( self.policy.all_param_vals['latent_means_stepsize']) >= 1:
            self.policy.all_param_vals['latent_means_stepsize'] /= 2
            #if min (self.policy.all_param_vals['latent_stds_stepsize']) >= 1:
            self.policy.all_param_vals['latent_stds_stepsize'] /= 2



        self.policy.assign_params(self.policy.all_params, self.policy.all_param_vals)
       

     




