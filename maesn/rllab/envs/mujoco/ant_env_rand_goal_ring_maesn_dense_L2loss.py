from .mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
import pickle

def generate_goals(num_goals):
    radius = 2.0 
    angle = np.random.uniform(0, np.pi, size=(num_goals,))
    xpos = radius*np.cos(angle)
    ypos = radius*np.sin(angle)
    return np.concatenate([xpos[:, None], ypos[:, None]], axis=1)

# num_goals = 100
# goals = generate_goals(num_goals)
# import pickle
# pickle.dump(goals, open("goals_ant_val.pkl", "wb"))
# import IPython
# IPython.embed()

class AntEnvRandGoalRing(MujocoEnv, Serializable):

    FILE = 'low_gear_ratio_ant.xml'

    def __init__(self, goal= None, *args, **kwargs):
        self._goal_idx = goal
        self.goals = pickle.load(open("/root/code/rllab/rllab/envs/mujoco/goals_ant.pkl", "rb"))
        #self.goals = pickle.load(open("/home/russellm/generativemodel_tasks/maml_rl_fullversion/rllab/envs/mujoco/goals_ant.pkl", "rb"))
        super(AntEnvRandGoalRing, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def sample_goals(self, num_goals):
        return np.random.choice(np.array(range(100)), num_goals)

    @overrides
    def reset(self, init_state=None, reset_args=None, **kwargs):
        goal_idx = reset_args
        if goal_idx is not None:
            self._goal_idx = goal_idx
        elif self._goal_idx is None:
            self._goal_idx = np.random.randint(1)


        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        obs = self.get_current_obs()
        return obs


    def step(self, action):
        self.forward_dynamics(action)
        com = self.get_body_com("torso")
    


        goal_reward = -np.linalg.norm(com[:2] - self.goals[self._goal_idx]) + 4.0 # make it happy, not suicidal

        distance_reward = -np.linalg.norm(com[:2] - self.goals[self._goal_idx])
    
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done, distance_reward = distance_reward)

    @overrides
    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))

