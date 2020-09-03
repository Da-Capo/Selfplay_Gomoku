# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
"""V-trace based SEED learner."""

import collections
import math
import os
import time

from absl import flags
from absl import logging

from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import utils
from seed_rl.common import vtrace
from seed_rl.common.parametric_distribution import get_parametric_distribution_for_action_space

import tensorflow as tf
import numpy as np


# Training.
flags.DEFINE_integer('task', 0, 'Task ID')
flags.DEFINE_integer('save_checkpoint_secs', 1800,
                     'Checkpoint save period in seconds.')
flags.DEFINE_integer('total_environment_frames', int(1e11),
                     'Total environment frames to train for.')
flags.DEFINE_integer('batch_size', 1, 'Batch size for training.')
flags.DEFINE_integer('inference_batch_size', 1, 'Batch size for inference.')
flags.DEFINE_integer('unroll_length', 10, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_training_tpus', 1, 'Number of TPUs for training.')
flags.DEFINE_string('init_checkpoint', None,
                    'Path to the checkpoint used to initialize the agent.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('target_entropy', None, 'If not None, the entropy cost is '
                   'automatically adjusted to reach the desired entropy level.')
flags.DEFINE_float('entropy_cost_adjustment_speed', 10., 'Controls how fast '
                   'the entropy cost coefficient is adjusted.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('kl_cost', 0., 'KL(old_policy|new_policy) loss multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('lambda_', 1., 'Lambda.')
flags.DEFINE_float('max_abs_reward', 0.,
                   'Maximum absolute reward when calculating loss.'
                   'Use 0. to disable clipping.')

# Logging
flags.DEFINE_integer('log_batch_frequency', 100, 'We average that many batches '
                     'before logging batch statistics like entropy.')
flags.DEFINE_integer('log_episode_frequency', 100, 'We average that many episodes'
                     ' before logging average episode return and length.')

FLAGS = flags.FLAGS


Unroll = collections.namedtuple(
    'Unroll', 'agent_state prev_actions env_outputs agent_outputs')


def validate_config():
  assert FLAGS.num_actors >= FLAGS.inference_batch_size, (
      'Inference batch size is bigger than the number of actors.')


def learner_loop(create_env_fn, create_agent_fn):
  """Main learner loop.

  Args:
    create_env_fn: Callable that must return a newly created environment. The
      callable takes the task ID as argument - an arbitrary task ID of 0 will be
      passed by the learner. The returned environment should follow GYM's API.
      It is only used for infering tensor shapes. This environment will not be
      used to generate experience.
    create_agent_fn: Function that must create a new tf.Module with the neural
      network that outputs actions and new agent state given the environment
      observations and previous agent state. See dmlab.agents.ImpalaDeep for an
      example. The factory function takes as input the environment action and
      observation spaces and a parametric distribution over actions.
    create_optimizer_fn: Function that takes the final iteration as argument
      and must return a tf.keras.optimizers.Optimizer and a
      tf.keras.optimizers.schedules.LearningRateSchedule.
  """
  logging.info('Starting learner loop')
  # validate_config()
  settings = utils.init_learner(FLAGS.num_training_tpus)
  strategy, inference_devices, training_strategy, encode, decode = settings
  env = create_env_fn(0)
  parametric_action_distribution = get_parametric_distribution_for_action_space(
      env.action_space)
  env_output_specs = utils.EnvOutput(
      tf.TensorSpec([], tf.float32, 'reward'),
      tf.TensorSpec([], tf.bool, 'done'),
      tf.TensorSpec(env.observation_space.shape, env.observation_space.dtype,
                    'observation'),
  )
  action_specs = tf.TensorSpec(env.action_space.shape,
                               env.action_space.dtype, 'action')
  agent_input_specs = (action_specs, env_output_specs)

  # Initialize agent and variables.
  agent = create_agent_fn(env.action_space, env.observation_space,
                          parametric_action_distribution)
  initial_agent_state = agent.initial_state(1)
  agent_state_specs = tf.nest.map_structure(
      lambda t: tf.TensorSpec(t.shape[1:], t.dtype), initial_agent_state)
  input_ = tf.nest.map_structure(
      lambda s: tf.zeros([1] + list(s.shape), s.dtype), agent_input_specs)
  input_ = encode(input_)


  # Setup checkpointing and restore checkpoint.
  ckpt = tf.train.Checkpoint(agent=agent)
  if FLAGS.init_checkpoint is not None:
    tf.print('Loading initial checkpoint from %s...' % FLAGS.init_checkpoint)
    ckpt.restore(FLAGS.init_checkpoint) #.assert_consumed()
  manager = tf.train.CheckpointManager(
      ckpt, FLAGS.logdir, max_to_keep=1, keep_checkpoint_every_n_hours=6)
  last_ckpt_time = 0  # Force checkpointing of the initial model.
  if manager.latest_checkpoint:
    logging.info('Restoring checkpoint: %s', manager.latest_checkpoint)
    ckpt.restore(manager.latest_checkpoint) #.assert_consumed()
    last_ckpt_time = time.time()

  actor_run_ids = utils.Aggregator(FLAGS.num_actors,
                                   tf.TensorSpec([], tf.int64, 'run_ids'))

  # Current agent state and action.
  agent_states = utils.Aggregator(
      FLAGS.num_actors, agent_state_specs, 'agent_states')
  actions = utils.Aggregator(FLAGS.num_actors, action_specs, 'actions')

  def add_batch_size(ts):
    return tf.TensorSpec([FLAGS.inference_batch_size] + list(ts.shape),
                         ts.dtype, ts.name)

  inference_iteration = tf.Variable(-1)
  inference_specs = (
      tf.TensorSpec([], tf.int32, 'actor_id'),
      tf.TensorSpec([], tf.int64, 'run_id'),
      env_output_specs,
      tf.TensorSpec([], tf.float32, 'raw_reward'),
  )
  inference_specs = tf.nest.map_structure(add_batch_size, inference_specs)
  @tf.function(input_signature=inference_specs)
  def inference(actor_ids, run_ids, env_outputs, raw_rewards):
    # Reset the actors that had their first run or crashed.
    previous_run_ids = actor_run_ids.read(actor_ids)
    actor_run_ids.replace(actor_ids, run_ids)
    reset_indices = tf.where(tf.not_equal(previous_run_ids, run_ids))[:, 0]
    actors_needing_reset = tf.gather(actor_ids, reset_indices)
    if tf.not_equal(tf.shape(actors_needing_reset)[0], 0):
      tf.print('Actor ids needing reset:', actors_needing_reset)
    initial_agent_states = agent.initial_state(
        tf.shape(actors_needing_reset)[0])
    # tf.print("agent initial_agent_states",tf.reduce_mean(initial_agent_states))
    agent_states.replace(actors_needing_reset, initial_agent_states)
    actions.reset(actors_needing_reset)

    # Inference.
    prev_actions = parametric_action_distribution.postprocess(
        actions.read(actor_ids))
    input_ = encode((prev_actions, env_outputs))
    prev_agent_states = agent_states.read(actor_ids)
    tf.print("agent states",tf.reduce_mean(prev_agent_states),tf.reduce_max(prev_agent_states))
    # def make_inference_fn(inference_device):
    #   def device_specific_inference_fn():
    #     with tf.device(inference_device):
    #       @tf.function
    #       def agent_inference(*args):
    #         return agent(*decode(args), is_training=False,
    #                      postprocess_action=False)

    #       return agent_inference(*input_, prev_agent_states)

    #   return device_specific_inference_fn

    # # Distribute the inference calls among the inference cores.
    # branch_index = inference_iteration.assign_add(1) % len(inference_devices)
    # agent_outputs, curr_agent_states = tf.switch_case(branch_index, {
    #     i: make_inference_fn(inference_device)
    #     for i, inference_device in enumerate(inference_devices)
    # })
    with tf.device(inference_devices[0]):
      agent_outputs, curr_agent_states = agent(*(decode((*input_, prev_agent_states))))
    # Update current state.
    agent_states.replace(actor_ids, curr_agent_states)
    actions.replace(actor_ids, agent_outputs.action)

    # Return environment actions to actors.
    return parametric_action_distribution.postprocess(agent_outputs.action)


  summary_writer = tf.summary.create_noop_writer()
  timer_cls = utils.nullcontext

  actor_step = 0
  with summary_writer.as_default():
    while True:
      try:

        env = create_env_fn(FLAGS.task)

        # Unique ID to identify a specific run of an actor.
        run_id = np.random.randint(np.iinfo(np.int64).max)
        observation = env.reset()
        reward = 0.0
        raw_reward = 0.0
        done = False

        episode_step = 0
        episode_return = 0
        episode_raw_return = 0

        while True:
          tf.summary.experimental.set_step(actor_step)
          env_output = utils.EnvOutput([tf.cast(reward, tf.float32)], [done], tf.cast(observation[None], tf.float32))
          with timer_cls('actor/elapsed_inference_s', 1000):
            action = inference(
                [FLAGS.task], [run_id], env_output, [raw_reward])[0]
          with timer_cls('actor/elapsed_env_step_s', 1000):
            observation, reward, done, info = env.step(action.numpy())
            
          # env.render()
          episode_step += 1
          episode_return += reward
          raw_reward = float((info or {}).get('score_reward', reward))
          episode_raw_return += raw_reward
          if done:
            logging.info('Return: %f Raw return: %f Steps: %i', episode_return,
                          episode_raw_return, episode_step)
            env.render()
            with timer_cls('actor/elapsed_env_reset_s', 10):
              observation = env.reset()
              episode_step = 0
              episode_return = 0
              episode_raw_return = 0
              
          actor_step += 1
      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        logging.exception(e)
        env.close()


if __name__ == "__main__":
  import env
  import networks
  from absl import app

  def create_agent(action_space, unused_env_observation_space,
                 unused_parametric_action_distribution):
    return networks.ImpalaDeep(action_space.n)

  def main(argv):
    learner_loop(env.create_play_environment, create_agent)
  
  # app.run(main, "--init_checkpoint agent/1588419442/ckpt-112 --num_actors 1".split(" "))
  app.run(main)