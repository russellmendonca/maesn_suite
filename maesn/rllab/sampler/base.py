

import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths, prefix='', log=True, task_idx=0, noise_opt=False, joint_opt=False, sess = None):
        baselines = []
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
        if log:
            logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths, log=log)
        if log:
            logger.log("fitted")


        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        if joint_opt is True:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            noises = tensor_utils.concat_tensor_list([path["noises"] for path in paths])
            task_idxs = task_idx*np.ones((len(noises),), dtype=np.int32)
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]
            debug_avg_ret = np.mean(undiscounted_returns)
            #mean = sess.run(self.algo.policy.all_params["latent_means"])
            #std = sess.run(self.algo.policy.all_params["latent_stds"])
            #import ipdb
            #ipdb.set_trace()
            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                noises=noises,
                task_idxs=task_idxs,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )


            observations_latent = tensor_utils.concat_tensor_list([path["observations"][0:1] for path in paths])
            noises_latent = tensor_utils.concat_tensor_list([path["noises"][0:1] for path in paths])
            task_idxs_latent = task_idx*np.ones((len(noises_latent),), dtype=np.int32)
            actions_latent = tensor_utils.concat_tensor_list([path["actions"][0:1] for path in paths])
            rewards_latent = tensor_utils.concat_tensor_list([path["rewards"][0:1] for path in paths])
            returns_latent = tensor_utils.concat_tensor_list([path["returns"][0:1] for path in paths])
            advantages_latent = tensor_utils.concat_tensor_list([path["advantages"][0:1] for path in paths])
            env_infos_latent = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos_latent = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages_latent = util.center_advantages(advantages_latent)

            if self.algo.positive_adv:
                advantages_latent = util.shift_advantages_to_positive(advantages_latent)

            samples_data_latent = dict(
                observations=observations_latent,
                noises=noises_latent,
                task_idxs=task_idxs_latent,
                actions=actions_latent,
                rewards=rewards_latent,
                returns=returns_latent,
                advantages=advantages_latent,
                env_infos=env_infos_latent,
                agent_infos=agent_infos_latent,
                paths=paths,
            )
        elif noise_opt is False:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            noises = tensor_utils.concat_tensor_list([path["noises"] for path in paths])
            task_idxs = task_idx*np.ones((len(noises),), dtype=np.int32)
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])

            for path in paths:
                for key in path['agent_infos']:
                    if key == 'prob' and len(path['agent_infos'][key].shape) == 3:
                        path['agent_infos'][key] = path['agent_infos'][key][:,0]
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])


            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                noises=noises,
                task_idxs=task_idxs,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        elif noise_opt is True:
            observations = tensor_utils.concat_tensor_list([path["observations"][0:1] for path in paths])
            noises = tensor_utils.concat_tensor_list([path["noises"][0:1] for path in paths])
            task_idxs = task_idx*np.ones((len(noises),), dtype=np.int32)
            actions = tensor_utils.concat_tensor_list([path["actions"][0:1] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"][0:1] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"][0:1] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"][0:1] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                noises=noises,
                task_idxs=task_idxs,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
     
        if log:
            #logger.record_tabular('Iteration', itr)
            #logger.record_tabular('AverageDiscountedReturn',
            #                      average_discounted_return)

            for key in path['env_infos']:


                info_returns = [sum(path["env_infos"][key]) for path in paths]
                logger.record_tabular(prefix+'Average'+key, np.mean(info_returns))
                logger.record_tabular(prefix+'Max'+key, np.max(info_returns))



            logger.record_tabular(prefix+'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(prefix+'ExplainedVariance', ev)
            logger.record_tabular(prefix+'NumTrajs', len(paths))
            logger.record_tabular(prefix+'Entropy', ent)
            logger.record_tabular(prefix+'Perplexity', np.exp(ent))
            logger.record_tabular(prefix+'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(prefix+'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(prefix+'MinReturn', np.min(undiscounted_returns))
        if joint_opt is True:
            return samples_data, samples_data_latent
        else:
            return samples_data
