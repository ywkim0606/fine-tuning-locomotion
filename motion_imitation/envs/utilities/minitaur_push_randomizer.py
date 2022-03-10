"""Adds random forces to the base of Minitaur during the simulation steps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
parentdir = os.path.dirname(os.path.dirname(parentdir))
os.sys.path.insert(0, parentdir)

import functools
import math
import gin
import numpy as np
from motion_imitation.envs.utilities import env_randomizer_base
from motion_imitation.robots.minitaur import Minitaur
import csv

# _PERTURBATION_START_STEP = 100
# _PERTURBATION_INTERVAL_STEPS = 200
# _PERTURBATION_DURATION_STEPS = 10
# _HORIZONTAL_FORCE_UPPER_BOUND = 120
# _HORIZONTAL_FORCE_LOWER_BOUND = 240
# _VERTICAL_FORCE_UPPER_BOUND = 300
# _VERTICAL_FORCE_LOWER_BOUND = 500

_PERTURBATION_START_STEP = 0
_PERTURBATION_INTERVAL_STEPS = 1
_PERTURBATION_DURATION_STEPS = 99999999999999999999
_HORIZONTAL_FORCE_UPPER_BOUND = 120000
_HORIZONTAL_FORCE_LOWER_BOUND = 24000
_VERTICAL_FORCE_UPPER_BOUND = 300000
_VERTICAL_FORCE_LOWER_BOUND = 50000


@gin.configurable
class MinitaurPushRandomizer(env_randomizer_base.EnvRandomizerBase):
  """Applies a random impulse to the base of Minitaur."""

  def __init__(
      self,
      perturbation_start_step=_PERTURBATION_START_STEP,
      perturbation_interval_steps=_PERTURBATION_INTERVAL_STEPS,
      perturbation_duration_steps=_PERTURBATION_DURATION_STEPS,
      horizontal_force_bound=None,
      vertical_force_bound=None,
  ):
    """Initializes the randomizer.
    Args:
      perturbation_start_step: No perturbation force before the env has advanced
        this amount of steps.
      perturbation_interval_steps: The step interval between applying
        perturbation forces.
      perturbation_duration_steps: The duration of the perturbation force.
      horizontal_force_bound: The lower and upper bound of the applied force
        magnitude when projected in the horizontal plane.
      vertical_force_bound: The z component (abs value) bound of the applied
        perturbation force.
    """
    self._perturbation_start_step = perturbation_start_step
    self._perturbation_interval_steps = perturbation_interval_steps
    self._perturbation_duration_steps = perturbation_duration_steps
    self._horizontal_force_bound = (horizontal_force_bound if horizontal_force_bound else
                                    [_HORIZONTAL_FORCE_LOWER_BOUND, _HORIZONTAL_FORCE_UPPER_BOUND])
    self._vertical_force_bound = (vertical_force_bound if vertical_force_bound else
                                  [_VERTICAL_FORCE_LOWER_BOUND, _VERTICAL_FORCE_UPPER_BOUND])
    self._perturbation_parameter_dict = None
    self.myenv = None
    self.xyz_acc = self.read_csv('/home/yoonwoo/motion_imitation/ab13.csv')
  
  def read_csv(self, filename):
    with open(filename, 'r') as csvfile:
      datareader = list(csv.reader(csvfile))
    datareader = np.array(datareader[1:],dtype=np.float)
    np_data = datareader[:,1:4]
    return np_data
  
  def _get_robot_from_env(self, env):
    if hasattr(env, "minitaur"):
      return env.minitaur
    elif hasattr(env, "robot"):
      return env.robot
    else:
      return None

  def randomize_env(self, env):
    """Randomizes the simulation environment.
    Args:
      env: The Minitaur gym environment to be randomized.
    """
    self.myenv = self._get_robot_from_env(env)
  

  def randomize_step(self, env):
    """Randomizes env steps.
    Will be called at every env step. Called to generate randomized  force and
    torque to apply. Application of forces are done in randomize_sub_step.
    Args:
      env: The Minitaur gym environment to be randomized.
    """
    robot = self.myenv
    base_link_ids = robot.chassis_link_ids
    # print("********************", base_link_ids)
    # env_step_counter --> step_counter
    if env.env_step_counter % self._perturbation_interval_steps == 0:
      print(env.env_step_counter)
      self._applied_link_id = base_link_ids[np.random.randint(0, len(base_link_ids))]
      # horizontal_force_magnitude = np.random.uniform(self._horizontal_force_bound[0],
      #                                                self._horizontal_force_bound[1])
      horizontal_force_magnitude = 1000
      x_acc, y_acc, z_acc = self.xyz_acc[env.env_step_counter+14000]
      theta = -math.pi/2 #np.random.uniform(0, 2 * math.pi)
      vertical_force_magnitude = 0
      # vertical_force_magnitude = np.random.uniform(self._vertical_force_bound[0],
      #                                              self._vertical_force_bound[1])
      self._applied_force = 960 * np.array([y_acc, z_acc, x_acc])
      # self._applied_force = horizontal_force_magnitude * np.array(
          # [math.cos(theta), math.sin(theta), 0]) + np.array([0, 0, -vertical_force_magnitude])
      #print('FORCE: ', self._applied_force)
      # print(robot.GetMotorTorques())
    if (env.env_step_counter % self._perturbation_interval_steps <
        self._perturbation_duration_steps) and (env.env_step_counter >=
                                                self._perturbation_start_step):
      # Parameter of pybullet_client.applyExternalForce()
      self._perturbation_parameter_dict = dict(#objectUniqueId=env.minitaur.quadruped,
                                              objectUniqueId=robot.quadruped,
                                               linkIndex=self._applied_link_id,
                                               forceObj=self._applied_force,
                                               posObj=[0.0, 0.0, 0.0],
                                               flags=env.pybullet_client.LINK_FRAME)
      env.pybullet_client.applyExternalForce(**self._perturbation_parameter_dict)
    else:
      self._perturbation_parameter_dict = None

  def randomize_sub_step(self, env, sub_step_index, num_sub_steps):
    """Randomize simulation steps per sub steps (simulation step).
    Will be called at every simulation step. This is the correct place to add
    random forces/torques to Minitaur.
    Args:
      env: The Minitaur gym environment to be randomized.
      sub_step_index: Index of sub step, from 0 to N-1. N is the action repeat.
      num_sub_steps: Number of sub steps, equals to action repeat.
    """
    print("sub_step ??????????????????????????")
    if self._perturbation_parameter_dict is not None:
      env.pybullet_client.applyExternalForce(**self._perturbation_parameter_dict)