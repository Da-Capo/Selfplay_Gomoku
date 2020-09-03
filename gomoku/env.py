from envs.gomoku_vec_run import GomokuEnv
from envs.gomoku import GomokuEnv as EvalGomokuEnv
from envs.gomoku_embryo import GomokuEnv as EmbryoGomokuEnv
from absl import logging
from absl import flags

flags.DEFINE_string('game', 'Pong', 'Game name.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('max_random_noops', 30,
                     'Maximal number of random no-ops at the beginning of each '
                     'episode.')
flags.DEFINE_boolean('sticky_actions', False,
                     'When sticky actions are enabled, the environment repeats '
                     'the previous action with probability 0.25, instead of '
                     'playing the action given by the agent. Used to introduce '
                     'stochasticity in ATARI-57 environments, see '
                     'Machado et al. (2017).')

def create_environment(task):
    logging.info('creating environment: gomoku-%s', task)
    env = GomokuEnv(15)
    return env

def create_eval_environment(task, color='black'):
    logging.info('creating rule environment: gomoku-%s', task)
    env = EvalGomokuEnv(color, 15, "rule")
    return env

def create_embryo_environment(task, color='black'):
    logging.info('creating embryo environment: gomoku-%s', task)
    env = EmbryoGomokuEnv(color, 15, "rule")
    return env

def create_play_environment(task, color='black'):
    logging.info('creating play environment: gomoku-%s', task)
    env = EvalGomokuEnv(color, 15, "human")
    return env
def create_random_environment(task, color='black'):
    logging.info('creating random environment: gomoku-%s', task)
    env = EvalGomokuEnv(color, 15, "random")
    return env

if __name__ == "__main__":
    env = create_environment(0)
    obs = env.reset()
    import time
    t = time.time()
    for _ in range(10):
        action = env.sample()
        new_obs, reward, done, info = env.step(action)
        print(action, reward, done)
        obs = new_obs
        env.render()
        if done:
            print ("Game is Over")
            obs = env.reset()
            env.render()
            break

    print(time.time()-t)