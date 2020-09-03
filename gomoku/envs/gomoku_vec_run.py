from ctypes import c_int, c_void_p, c_double, c_char_p, POINTER
import numpy.ctypeslib as npct
import numpy as np
import gym

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
    def __init__(self, board_size=15):
        # player_color ---- 1:X-black, 0:O-white
        # boartd_size  ---- int
        self.board_size = board_size
        shape = (self.board_size, self.board_size, 2) 
        self.observation_space = gym.spaces.Box(np.zeros(shape), np.ones(shape))
        self.action_space = gym.spaces.Discrete(self.board_size**2)
        self.g=None

    def reset(self):
        if not self.g is None:
            lib.G_clean(self.g)
        self.g = lib.G_new(c_int(self.board_size), c_int(5), c_int(1))
        self.color = 1.
        self._get_state()
        
        # 开局随机初始化
        # if np.random.random()>0.9:
        #     rand_n_step = (np.random.randint(3)+1)*2
        # else:
        #     rand_n_step=0
        # for _ in range(rand_n_step):
        #     lib.G_move(self.g, self.sample())
        #     self._get_state()
        
        return self.obs
    
    
    def step(self, action):
        action = self.rot_dict[action]
        lib.G_move(self.g, action)
        self._get_state()

        if not self.done:
            assert self.reward==0, "reward is %s"%self.reward


        if self.done and np.random.random()>0.9995:
            print("(vec render)", self.reward)
            self.render()

        return self.obs, self.reward, self.done, {}

    # 获取当前状态 切换 颜色
    def _get_state(self):
        board = np.zeros((self.board_size,self.board_size), dtype=np.float32)
        lib.G_board(self.g, board)
        
        # color *= -1
        color = 1 if np.sum(board==-1)==np.sum(board==1) else -1

        obs = np.zeros((self.board_size, self.board_size, 2))
        if self.color == -1:
            obs[...,0] = (board==-1)*1
            obs[...,1] = (board==1)*1
        else:
            obs[...,0] = (board==1)*1
            obs[...,1] = (board==-1)*1
        
        status = np.zeros(2, dtype=np.float32)
        lib.G_status(self.g, status)
        done = status[0]==1 or np.sum(board==0)<=0
        win_color = status[1]

        reward = 0.
        if done:
            reward = win_color

        self.done = done 
        self.win_color = win_color 
        self.obs = obs
        self.reward = reward
        self.color *= -1.
        self.board = board

        # 随机旋转 翻转
        rot_dict = np.arange(self.board_size*self.board_size).reshape(self.board_size, self.board_size)
        # k = np.random.randint(4)
        # self.obs = np.rot90(self.obs,k=k,axes=(0,1))
        # rot_dict = np.rot90(rot_dict,k=k,axes=(0,1))
        # if np.random.random()>0.9:
        #     self.obs = self.obs[::-1,:,:]
        #     rot_dict = rot_dict[::-1,:]
        # if np.random.random()>0.9:
        #     self.obs = self.obs[:,::-1,:]
        #     rot_dict = rot_dict[:,::-1]
        # print(rot_dict)
        self.rot_dict = rot_dict.reshape(-1)
    
    def sample(self):
        vp = np.where(self.board.reshape(-1)==0)[0]
        if len(vp) == 0:
            print ("Space is empty")
            return 0
        # np_random, _ = seeding.np_random()
        return vp[np.random.randint(len(vp))]

    def render(self, mode="human", close=False):
        lib.G_display(self.g)        

        
def test_gomoku():
    env = GomokuEnv(6)
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

if __name__=="__main__":  
    test_gomoku()