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

r"""SEED actor."""

import os
import time

from absl import flags
from absl import logging
import numpy as np
from seed_rl import grpc
from seed_rl.common import common_flags  
from seed_rl.common import profiling
from seed_rl.common import utils
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_integer('task', 199, 'Task id.')
flags.DEFINE_integer('num_actors_with_summaries', 4,
                     'Number of actors that will log debug/profiling TF '
                     'summaries.')
flags.DEFINE_bool('render', False,
                  'Whether the first actor should render the environment.')


def are_summaries_enabled():
  return FLAGS.task < FLAGS.num_actors_with_summaries


def is_rendering_enabled():
  return FLAGS.render and FLAGS.task == 0


def actor_loop(create_env_fn):
  """Main actor loop.

  Args:
    create_env_fn: Callable (taking the task ID as argument) that must return a
      newly created environment.
  """
  logging.info('Starting actor eval loop')

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.logdir, 'actor_embryo'.format(FLAGS.task)),
      flush_millis=20000, max_queue=1000)
  timer_cls = profiling.ExportingTimer

  actor_step = 0
  with summary_writer.as_default():
    while True:
      try:
        # Client to communicate with the learner.
        client = grpc.Client(FLAGS.server_address)

        env = create_env_fn(FLAGS.task, color='black')
        env1 = create_env_fn(FLAGS.task, color='white')

        # Unique ID to identify a specific run of an actor.
        run_id = np.random.randint(np.iinfo(np.int64).max)
        observation = env.reset()
        reward = 0.0
        raw_reward = 0.0
        done = False

        episode_step = 0
        episode_return = 0
        episode_raw_return = 0

        eval_times = 0
        eval_state = 'black'
        print("starting eval: ", eval_state)

        while True:
          tf.summary.experimental.set_step(actor_step)
          env_output = utils.EnvOutput(tf.cast(reward, tf.float32), done, tf.cast(observation, tf.float32))
          with timer_cls('actor/elapsed_inference_s', 1000):
            action = client.inference_eval(
                FLAGS.task, run_id, env_output, raw_reward)

          if eval_state=='black':
            with timer_cls('actor/elapsed_env_step_s', 1000):
              observation, reward, done, info = env.step(action.numpy())
          else:
            with timer_cls('actor/elapsed_env_step_s', 1000):
              observation, reward, done, info = env1.step(action.numpy())

          if is_rendering_enabled():
            env.render()
          episode_step += 1
          episode_return += reward
          raw_reward = float((info or {}).get('score_reward', reward))
          episode_raw_return += raw_reward

          if done:
            eval_times+=1
            if eval_times>=50:
              tf.summary.scalar('actor/eval_return_'+eval_state, episode_return)
              logging.info('%s win/all: %d/%d Raw return: %f Steps: %i', eval_state, (episode_return+eval_times)/2, eval_times,
                          episode_raw_return, episode_step)
              episode_step = 0
              episode_return = 0
              episode_raw_return = 0

              time.sleep(100)
              eval_times = 0
              eval_state = 'white' if eval_state=='black' else 'black'
              print("starting eval: ", eval_state)

            if eval_state=='black':
              with timer_cls('actor/elapsed_env_reset_s', 10):
                observation = env.reset()
            else:
              with timer_cls('actor/elapsed_env_reset_s', 10):
                observation = env1.reset()

          actor_step += 1
      except (tf.errors.UnavailableError, tf.errors.CancelledError) as e:
        logging.exception(e)
        env.close()


if __name__ == "__main__":
  import env
  from absl import app
  def main(argv):
    actor_loop(env.create_embryo_environment)
  
  app.run(main, "--logtostderr --pdb_post_mortem --server_address=localhost:8686 --num_actors=200 --task=198".split(" "))