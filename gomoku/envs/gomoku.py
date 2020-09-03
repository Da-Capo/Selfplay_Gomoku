from ctypes import c_int, c_void_p, c_double, c_char_p, POINTER
import numpy.ctypeslib as npct
import numpy as np
import gym
from envs.gomoku_ai import strategy

def rule_step(obs):
    board_size = obs[0].shape[0]

    if np.sum(obs)<=0:
        return int((board_size**2)/2)+1

    x_stones = set([(x+1,y+1) for x,y in zip(*np.where(obs[0][...,0]==1))])
    o_stones = set([(x+1,y+1) for x,y in zip(*np.where(obs[0][...,1]==1))])
    board = (x_stones, o_stones)
    state = (board, None, 1, board_size)
    a = strategy(state)
    actions = np.array([(a[0]-1)*board_size+a[1]-1])
    return actions
    
def move_to_pos(move):
    try:
        px = ord(move[0])-97
        py = int(move[1:])-1
    except:
        print(move)
    return px, py

lib = npct.load_library('envs/build/libgomoku.so', ".")
lib.G_new.argtypes = [c_int, c_int, c_int]
lib.G_new.restype = c_void_p
lib.G_move.argtypes = [c_void_p, c_int]
lib.G_move.restype  = c_int
lib.G_board.argtypes = [c_void_p, npct.ndpointer(dtype=np.float32, ndim=2)]
lib.G_board.restype  = c_int
lib.G_display.argtypes = [c_void_p]
lib.G_display.restype  = c_int
lib.G_status.argtypes = [c_void_p, npct.ndpointer(dtype=np.float32, ndim=1)]
lib.G_status.restype  = c_int
lib.G_clean.argtypes = [c_void_p]
lib.G_clean.restype  = c_int

class GomokuEnv(gym.Env):    
    def __init__(self, player_color=1, board_size=15, oppo_type="random"):
        # player_color ---- 1:X-black, 0:O-white
        # boartd_size  ---- int
        if player_color=='black': player_color=1
        if player_color=='white': player_color=-1
        self.player_color = player_color
        self.board_size = board_size
        shape = (self.board_size, self.board_size, 2) 
        self.observation_space = gym.spaces.Box(np.zeros(shape), np.ones(shape))
        self.action_space = gym.spaces.Discrete(self.board_size**2)
        self.oppo_type = oppo_type
        self.g = None

    def move(self, action):
        lib.G_move(self.g, action)
        self._get_state()

    def oppo_move(self):
        if self.oppo_type==None: 
            pass
        else:
            if self.oppo_type=='rule':
                action = rule_step([self.obs])
            elif self.oppo_type=='random':
                action = self.sample()
            elif self.oppo_type=='human':
                self.render()
                while 1:
                    try:
                        move = input("Your move(a~o,0~15):")
                        pos = move_to_pos(move)
                        assert 0<=pos[0]<=15
                        assert 0<=pos[1]<=15
                        break
                    except:
                        if move=="exit":exit(0)
                        print("illegal move",pos,"enter 'exit' to quit")
                action = pos[1]*self.board_size +pos[0]
            else:
                action = self.sample()
            
            lib.G_move(self.g, action)
            self._get_state()

    def reset(self):
        if not self.g is None:
            lib.G_clean(self.g)
        self.g = lib.G_new(c_int(self.board_size), c_int(5), c_int(1))
        self._get_state()
        if self.player_color==-1:
            self.oppo_move()
        return self.obs
    
    
    def step(self, action):
        self.move(action)

        if not self.done:
            self.oppo_move()

        return self.obs, self.reward, self.done, {}

    def _get_state(self):
        board = np.zeros((self.board_size,self.board_size), dtype=np.float32)
        lib.G_board(self.g, board)
        
        color = 1 if np.sum(board==-1)==np.sum(board==1) else -1

        obs = np.zeros((self.board_size, self.board_size, 2))
        if color == -1:
            obs[...,0] = (board==-1)*1
            obs[...,1] = (board==1)*1
        else:
            obs[...,0] = (board==1)*1
            obs[...,1] = (board==-1)*1
        
        status = np.zeros(2, dtype=np.float32)
        lib.G_status(self.g, status)
        done = status[0]==1
        win_color = status[1]

        reward = 0.
        if done:
            if win_color == self.player_color: 
                reward = 1.
            else:
                reward = -1.
        
        if done and np.random.random()>0.999:
            self.render()
        
        self.done = done
        self.win_color = win_color
        self.obs = obs
        self.reward = reward
        self.color = color
        self.board = board
    
    def sample(self):
        vp = np.where(self.board.reshape(-1)==0)[0]
        if len(vp) == 0:
            print ("Space is empty")
            return 0
        # np_random, _ = seeding.np_random()
        return vp[np.random.randint(len(vp))]

    def render(self, mode="human", close=False):
        lib.G_display(self.g)

if __name__=="__main__":  
    env = GomokuEnv('white', 15, "rule")
    obs = env.reset()
    import time
    t = time.time()
    for _ in range(1024):
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
    # import pdb; pdb.set_trace()