
from ctypes import c_int, c_void_p, c_double, c_char_p, POINTER
import numpy.ctypeslib as npct
import numpy as np
import gym
from .embryo.new_protocol import new_protocol
import os

ROOTDIR = os.path.dirname(__file__)+'/'

lib = npct.load_library(ROOTDIR+'build/libgomoku.so', ".")
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
        self.last_move = (-1, -1)
        self.engine = None

    def move(self, action):
        lib.G_move(self.g, action)
        x = int(action/self.board_size)
        y = int(action%self.board_size)
        self.last_move = (x,y)
        self._get_state()

    def oppo_move(self):
        if self.oppo_type==None: 
            pass
        else:
            if self.oppo_type=='rule':
                if self.move_num<=1:
                    if self.move_num==1:
                        board = [[(0,0) for i in range(self.board_size)] for j in range(self.board_size)]
                        board[self.last_move[0]][self.last_move[1]] = (1, 2)
                    else:
                        board = [[(0,0) for i in range(self.board_size)] for j in range(self.board_size)]
                    if not self.engine is None:
                        self.engine.clean()
                    self.engine = new_protocol(
                            cmd = ROOTDIR+"./embryo/pbrain-embryo-1.0.4-33da365a-s",
                            board = board,
                            timeout_turn = 1000,
                            timeout_match = 100000,
                            max_memory = 450*1024*1024,
                            game_type = 2,
                            rule = 0,
                            folder = ROOTDIR+"./embryo/",
                            working_dir = ROOTDIR+"./embryo/",
                            tolerance = 1000)
                    msg, x, y = self.engine.start()
                else:
                    msg, x, y = self.engine.turn(self.last_move[0], self.last_move[1])
                action = x*self.board_size + y
            elif self.oppo_type=='random':
                action = self.sample()
            else:
                action = self.sample()
            
            lib.G_move(self.g, action)
            self._get_state()

    def reset(self):
        self.g = lib.G_new(c_int(self.board_size), c_int(5), c_int(1))
        self.move_num = -1
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
        
        self.done = done
        self.win_color = win_color
        self.obs = obs
        self.reward = reward
        self.color = color
        self.board = board

        self.move_num += 1
    
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