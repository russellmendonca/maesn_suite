


from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.algos.batch_maesn_polopt import BatchMAESNPolopt
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from collections import OrderedDict


class MAESN_NPO(BatchMAESNPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            use_maml=True,
            **kwargs):
        assert optimizer is not None  # only for use with MAML TRPO
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        if not use_maml:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            optimizer = FirstOrderOptimizer(**default_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.use_maml = use_maml
        self.kl_constrain_step = -1  # needs to be 0 or -1 (original pol params, or new pol params)
        super(MAESN_NPO, self).__init__(**kwargs)

    def make_vars(self, stepnum='0'):
        # lists over the meta_batch_size
        obs_vars, action_vars, adv_vars, noise_vars, task_idx_vars = [], [], [], [], []
        for i in range(self.meta_batch_size):
            obs_vars.append(self.env.observation_space.new_tensor_variable(
                'obs' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            action_vars.append(self.env.action_space.new_tensor_variable(
                'action' + stepnum + '_' + str(i),
                extra_dims=1,
            ))
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
            noise_vars.append(tf.placeholder(dtype=tf.float32, shape=[None, self.latent_dim], name='noise' + stepnum + '_' + str(i)))
            task_idx_vars.append(tensor_utils.new_tensor(
                name='task_idx' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.int32,
            ))
        return obs_vars, action_vars, adv_vars, noise_vars, task_idx_vars


    def make_vars_latent(self, stepnum='0'):
        # lists over the meta_batch_size
        adv_vars, z_vars, task_idx_vars = [], [], []
        for i in range(self.meta_batch_size):
            adv_vars.append(tensor_utils.new_tensor(
                name='advantage_latent' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.float32,
            ))
            z_vars.append(tf.placeholder(dtype=tf.float32, shape=[None, self.latent_dim], name='zs_latent' + stepnum + '_' + str(i)))
            task_idx_vars.append(tensor_utils.new_tensor(
                name='task_idx_latents' + stepnum + '_' + str(i),
                ndim=1, dtype=tf.int32,
            ))
        return adv_vars, z_vars, task_idx_vars


    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        assert not is_recurrent  # not supported

        dist = self.policy.distribution
        self.kl_weighting_ph = tf.placeholder(dtype=tf.float32, shape=[1], name='kl_weighting_ph')
        old_dist_info_vars, old_dist_info_vars_list = [], []
        for i in range(self.meta_batch_size):
            old_dist_info_vars.append({
                k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s_%s' % (i, k))
                for k, shape in dist.dist_info_specs
                })
            old_dist_info_vars_list += [old_dist_info_vars[i][k] for k in dist.dist_info_keys]

        state_info_vars, state_info_vars_list = {}, []

        all_surr_objs, input_list = [], []
        all_surr_objs_latent = []
        new_params = None
        new_params_latent = None
        for j in range(self.num_grad_updates):
            obs_vars, action_vars, adv_vars, noise_vars, task_idx_vars = self.make_vars(str(j))
            adv_vars_latent, z_vars_latent, task_idx_vars_latent = self.make_vars_latent(str(j))

            surr_objs = []
            surr_objs_latent = []

            cur_params = new_params
            new_params = []  # if there are several grad_updates the new_params are overwritten
            new_params_latent = []
            kls = []

            for i in range(self.meta_batch_size):
                dist_info_vars, params_temp = self.policy.dist_info_sym(obs_vars[i], task_idx_vars[i], noise_vars[i], state_info_vars, all_params=self.policy.all_params)
                params = OrderedDict()
                for param_key in params_temp.keys():
                    if 'latent' not in param_key:
                        params[param_key] = params_temp[param_key]
                means = tf.gather(self.policy.all_params['latent_means'], task_idx_vars_latent[i])
                log_stds = tf.gather(self.policy.all_params['latent_stds'], task_idx_vars_latent[i])
                dist_info_vars_latent = {"mean": means, "log_std": log_stds}
                params_latent = OrderedDict()
                params_latent['latent_means'] = self.policy.all_params['latent_means']
                params_latent['latent_stds'] = self.policy.all_params['latent_stds']

                new_params.append(params)
                new_params_latent.append(params_latent)

                logli = dist.log_likelihood_sym(action_vars[i], dist_info_vars)


                logli_latent = self.latent_dist.log_likelihood_sym(z_vars_latent[i], dist_info_vars_latent)

                # formulate as a minimization problem
                # The gradient of the surrogate objective is the policy gradient
                surr_objs.append(- tf.reduce_mean(logli * adv_vars[i]))
                surr_objs_latent.append(- tf.reduce_mean(logli_latent * adv_vars_latent[i]))


            input_list += obs_vars + action_vars + adv_vars + noise_vars + task_idx_vars + state_info_vars_list
            input_list += adv_vars_latent + z_vars_latent + task_idx_vars_latent

            #TODO: FIX THIS!!!!!!
            if j == 0:
                # For computing the fast update for sampling
                self.policy.set_init_surr_obj(input_list, surr_objs, surr_objs_latent)
                init_input_list = input_list

            all_surr_objs.append(surr_objs)
            all_surr_objs_latent.append(surr_objs_latent)


        obs_vars, action_vars, adv_vars, noise_vars, task_idx_vars = self.make_vars('test')

        surr_objs = []
        for i in range(self.meta_batch_size):
            dist_info_vars, _ = self.policy.updated_dist_info_sym(i, all_surr_objs[-1][i], all_surr_objs_latent[-1][i], obs_vars[i], task_idx_vars[i], noise_vars[i], params_dict=new_params[i], params_dict_latent=new_params_latent[i])

            if self.kl_constrain_step == -1:  # if we only care about the kl of the last step, the last item in kls will be the overall
                kl = dist.kl_sym(old_dist_info_vars[i], dist_info_vars)
                kls.append(kl)


            lr = dist.likelihood_ratio_sym(action_vars[i], old_dist_info_vars[i], dist_info_vars)
            curr_obj = - tf.reduce_mean(lr*adv_vars[i])

            #Computation of the KL divergence for tasks which have been selected
            curr_mean = tf.gather(self.policy.all_params["latent_means"], task_idx_vars[i])
            curr_logstd = tf.gather(self.policy.all_params["latent_stds"], task_idx_vars[i])
            curr_latent_dist = {"mean": curr_mean, "log_std": curr_logstd}
            unit_gaussian_dist = {"mean": tf.zeros_like(curr_mean), "log_std": tf.zeros_like(curr_logstd)}
            kl_regularization = tf.reduce_mean(self.latent_dist.kl_sym(curr_latent_dist, unit_gaussian_dist))
            curr_obj += self.kl_weighting_ph[0]*kl_regularization
            # curr_obj += dist.kl_sym(self.policy.all_params['latent_means'][task_idx_vars[i]], self.policy.all_params['latent_means'][task_idx_vars[i]], 0 , 0)
            surr_objs.append(curr_obj)

        if self.use_maml:
            surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
            input_list += obs_vars + action_vars + adv_vars + noise_vars + task_idx_vars + old_dist_info_vars_list
        else:
            surr_obj = tf.reduce_mean(tf.stack(all_surr_objs[0], 0)) # if not meta, just use the first surr_obj
            input_list = init_input_list

        input_list += [self.kl_weighting_ph]
        if self.use_maml:
            mean_kl = tf.reduce_mean(tf.concat(kls, 0))  ##CF shouldn't this have the option of self.kl_constrain_step == -1?
            max_kl = tf.reduce_max(tf.concat(kls, 0))

            self.optimizer.update_opt(
                loss=surr_obj,
                target=self.policy,
                leq_constraint=(mean_kl, self.step_size),
                inputs=input_list,
                constraint_name="mean_kl"
            )
        else:
            self.optimizer.update_opt(
                loss=surr_obj,
                target=self.policy,
                inputs=input_list,
            )
        return dict()

    @overrides
    def optimize_policy(self, itr, all_samples_data, all_samples_data_latent):
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!
        sess = tf.get_default_session()
        if not self.use_maml:
            all_samples_data = [all_samples_data[0]]

        input_list = []
        for step in range(len(all_samples_data)):  # these are the gradient steps
            obs_list, action_list, adv_list, noise_list, task_idx_list = [], [], [], [], []
            for i in range(self.meta_batch_size):

                inputs = ext.extract(
                    all_samples_data[step][i],
                    "observations", "actions", "advantages", "noises", "task_idxs"
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
                noise_list.append(inputs[3])
                task_idx_list.append(inputs[4])
            input_list += obs_list + action_list + adv_list + noise_list + task_idx_list # [ [obs_0], [act_0], [adv_0], [obs_1], ... ]
            if step == 0:
                adv_list_latent, z_list_latent, task_idx_list_latent = [], [], []
                for i in range(self.meta_batch_size):

                    inputs = ext.extract(
                        all_samples_data_latent[step][i],
                        "advantages", "noises", "task_idxs"
                    )
                    #import ipdb
                    #ipdb.set_trace()
                    means = tf.gather(self.policy.all_params['latent_means'], inputs[-1])
                    stds = tf.gather(self.policy.all_params['latent_stds'], inputs[-1])
                    zs = sess.run(means + inputs[-2]*tf.exp(stds))
                    adv_list_latent.append(inputs[0])
                    z_list_latent.append(zs)
                    task_idx_list_latent.append(inputs[2])
                input_list += adv_list_latent + z_list_latent + task_idx_list_latent
            #import ipdb
            #ipdb.set_trace()
            if step == 0:  ##CF not used?
                init_inputs = input_list

        if self.use_maml:
            dist_info_list = []
            for i in range(self.meta_batch_size):
                agent_infos = all_samples_data[self.kl_constrain_step][i]['agent_infos']
                dist_info_list += [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
            input_list += tuple(dist_info_list)
            logger.log("Computing KL before")
            mean_kl_before = self.optimizer.constraint_val(input_list)

        if self.kl_scheme is None:
            curr_kl_weighting = self.kl_weighting
        elif self.kl_scheme == "0.01step4to0.05":
            curr_kl_weighting = min(0.05, 0.01 + (itr//10)*0.01)
        elif self.kl_scheme == "0.01step8to0.1":
            curr_kl_weighting = min(0.1, 0.01 + (itr//5)*0.001)

        elif self.kl_scheme == "0.01step8to0.05":
            curr_kl_weighting = min(0.05, 0.01 + (itr//5)*0.0005)

        elif self.kl_scheme == "0.01step8to0.2":
            curr_kl_weighting = min(0.2, 0.01 + (itr//5)*0.002)

        elif self.kl_scheme == "0.002step100to0.1":
            curr_kl_weighting = min(0.1, 0.002 + (itr//5)*0.001)

        elif self.kl_scheme == "0.002step100to0.02":
            curr_kl_weighting = min(0.02, 0.002 + (itr//5)*0.0002)

        elif self.kl_scheme == "0.002step100to0.05":
            curr_kl_weighting = min(0.05, 0.002 + (itr//5)*0.0005)

    


        elif self.kl_scheme == "0.01stepcontto0.05":
            curr_kl_weighting = min(0.05, 0.01 + (itr)*0.001)
        elif self.kl_scheme == "0.01stepcontto0.1":
            curr_kl_weighting = min(0.1, 0.01 + (itr)*0.001)
        elif self.kl_scheme == "0.01step8to0.3":
            curr_kl_weighting = min(0.3, 0.01 + (itr)*0.003)
        else:
            print("ERROR")
            import IPython
            IPython.embed()
        input_list += ([curr_kl_weighting],)
        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list)
        logger.log("Optimizing")
        self.optimizer.optimize(input_list)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list)
        if self.use_maml:
            logger.log("Computing KL after")
            mean_kl = self.optimizer.constraint_val(input_list)
            logger.record_tabular('MeanKLBefore', mean_kl_before)  # this now won't be 0!
            logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
