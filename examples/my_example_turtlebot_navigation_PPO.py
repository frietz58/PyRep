"""
A turtlebot reaches for 4 randomly places targets.
This script contains examples of:
    - Non-linear mobile paths to reach a target with collision avoidance
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

from gym import GoalEnv, Env
from gym import spaces

import sys
sys.path.insert(0, '/home/finn/Desktop/webots_rl/spinningup')
from spinup.algos.pytorch.sac.sac import sac
from spinup.algos.pytorch.ppo.ppo import ppo


class PyRepNavGoalEnv(Env):
    def __init__(self,
                 n_obs=7,
                 n_act=3,
                 render=False,
                 seed=1337,
                 scene_file="my_scene_turtlebot_navigation.ttt",
                 dist_reach_thresh=0.3):

        self.n_obs = n_obs
        self.n_act = n_act

        # PPO
        self.action_space = spaces.Discrete(n_act)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[n_obs,], dtype='float32')

        # self.observation_space = spaces.Box(
        #     low=np.array([-0.8, -0.8, -0.8, -0.8, -3.1415]),
        #     high=np.array([0.8, 0.8, 0.8, 0.8, 3.1415]),
        #     dtype=np.float32)
        # self.action_space = spaces.Box(low=-5, high=5, shape=(4,), dtype=np.float32)


        self.scene_file = join(dirname(abspath(__file__)), scene_file)
        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=not render)
        self.pr.start()

        self.agent = TurtleBot()

        # We could have made this target in the scene, but lets create one dynamically
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)
        self.position_min, self.position_max = [-1.5, -1.5, 0.1], [1.5, 1.5, 0.1]
        self.target_pos = []  # initialized in reset

        self.starting_pose = [0.0, 0.0, 0.061]

        self.agent.set_motor_locked_at_zero_velocity(True)

        self.trajectory_step_counter = 0
        self.done_dist_thresh = dist_reach_thresh

    def step(self, action):
        self._set_action(action)
        self.pr.step()
        self.trajectory_step_counter += 1

        o = self._get_obs()
        r = self._get_reward()
        d = self._is_done()
        i = {}

        return o, r, d, i  # obs, reward, done, info..

    def reset(self):
        # Get a random position within a cuboid for the goal and set the target position
        self.target_pos = list(np.random.uniform(self.position_min, self.position_max))
        # make sure it doesn*t spawn too close to robot...
        if self.target_pos[0] > 0 and self.target_pos[0] < 0.7:
            self.target_pos[0] = 0.7
        elif self.target_pos[0] < 0  and self.target_pos[0] > -0.7:
            self.target_pos[0] = 0.7

        if self.target_pos[1] > 0 and self.target_pos[1] < 0.7:
            self.target_pos[1] = 0.7
        elif self.target_pos[1] < 0  and self.target_pos[1] > -0.7:
            self.target_pos[1] = 0.7

        # hard goal goal for now:
        self.target_pos[0] = 1.1
        self.target_pos[1] = 0

        self.target.set_position(self.target_pos)

        # Reset robot position to starting position
        self.agent.set_position(self.starting_pose)
        agent_rot = self.agent.get_orientation()
        # agent_rot[2] = 0
        agent_rot[2] = np.random.uniform(-3.1415, 3.1415)

        self.agent.set_orientation(agent_rot)

        self.agent.set_joint_target_velocities([0, 0])  # reset velocities

        self.trajectory_step_counter = 0
        self.pr.step()  # think we need this for first obs to already return the values we set here above...

        return self._get_obs()

    def render(self, mode='human'):
        pass

    def close(self):
        self.pr.stop()
        self.pr.shutdown()

    def seed(self, seed=None):
        pass

    def _get_obs(self):
        agent_pos = self.agent.get_2d_pose()
        agent_rot_rads = self.agent.get_orientation()

        # obs = [agent_pos[0], agent_pos[1], agent_rot_rads[2]]  # x_pos, y_pos, abs z_orientation
        # achieved_goal = [agent_pos[0], agent_pos[1]]
        # desired_goal = [self.target_pos[0], self.target_pos[1]]
        # obs = {'observation': obs.copy(), 'achieved_goal': achieved_goal.copy(),
        #        'desired_goal': np.array(desired_goal.copy()),
        #        'non_noisy_obs': agent_pos.copy()}

        agent_joint_vels = self.agent.get_joint_velocities()

        obs = [self.target_pos[0], self.target_pos[1], agent_pos[0], agent_pos[1], agent_rot_rads[2]]
        for v in agent_joint_vels:
            obs.append(v)

        obs = [round(o, 3) for o in obs]
        return obs

    def _set_action(self, action):
        # set motor velocities
        target_vels = []
        frac = 0.5
        if action == 0:
            target_vels = [3.1415*frac, 3.1415*frac]
        elif action == 1:
            target_vels = [3.1415*frac, -3.1415*frac]
        elif action == 2:
            target_vels = [-3.1415*frac, 3.1415*frac]

        self.agent.set_joint_target_velocities(target_vels)

    def _get_reward(self):
        agent_pos = self.agent.get_2d_pose()
        dist = np.sqrt((self.target_pos[0] - agent_pos[0]) ** 2 + (self.target_pos[1] - agent_pos[1]) ** 2)
        if dist <= self.done_dist_thresh:
            return 0
        else:
            return -1

    def _is_done(self):
        agent_pos = self.agent.get_2d_pose()
        dist = np.sqrt((self.target_pos[0] - agent_pos[0]) ** 2 + (self.target_pos[1] - agent_pos[1]) ** 2)
        # print(dist)
        if dist <= self.done_dist_thresh:
            print("done because close")
            return True
        else:
            return False


# for i in range(LOOPS):
#     agent.set_2d_pose(starting_pose)
#
#
#     path = agent.get_nonlinear_path(position=pos, angle=0)
#
#     path.visualize()
#     done = False
#
#     while not done:
#         done = path.step()
#         pr.step()
#
#     path.clear_visualization()
#
#     print('Reached target %d!' % i)

def test_env():
    env = PyRepNavGoalEnv()

    LOOPS = 100
    STEPS = 400

    for loop in range(LOOPS):
        obs = env.reset()
        for step in range(STEPS):
            action = np.random.choice([0, 1, 2])
            action = 1
            obs, reward, done, info = env.step(action)
            # print("s: {} | a: {} | o: {} | r: {}".format(step, action, obs, reward))

            if done:
                break


if __name__ == "__main__":
    # test_env()

    ppo(PyRepNavGoalEnv,
        ac_kwargs=dict(hidden_sizes=[256] * 2),
        gamma=0.99, epochs=100, steps_per_epoch=10000, max_ep_len=2000)
