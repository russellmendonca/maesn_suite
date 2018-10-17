
import tensorflow as tf

def optimize(surr_obj, surr_obj_latent, inputs):


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
        step_sizes_sym[key] = step_size
    step_sizes_sym['latent_means'] = self.policy.all_params['latent_means_stepsize']
    step_sizes_sym['latent_stds'] = self.policy.all_params['latent_stds_stepsize']

    # if 'all_fast_params_tensor' not in dir(self):
    #     # make computation graph once
        #self.all_fast_params_tensor = []
        #for i in range(num_tasks):
    gradients = dict(zip(update_param_keys, tf.gradients(self.policy.only_latents*surr_obj, [self.policy.all_params[key] for key in update_param_keys])))
    gradients_latent = dict(zip(update_param_keys_latent, tf.gradients(surr_obj_latent, [self.policy.all_params[key] for key in update_param_keys_latent])))
    gradients.update(gradients_latent)
    
    update_tensor = OrderedDict(zip(all_keys, [self.policy.all_params[key] - step_sizes_sym[key]*tf.convert_to_tensor(gradients[key]) for key in all_keys]))

            

    # pull new param vals out of tensorflow, so gradient computation only done once ## first is the vars, second the values
    # these are the updated values of the params after the gradient step
    self.policy.all_param_vals = sess.run(update_tensor, feed_dict=dict(list(zip(self.input_list_for_grad, inputs))))
        


