import subprocess, shlex
import psutil
import copy
import time
import sys
from .utility import *
from queue import Queue, Empty
from threading  import Thread

class new_protocol(object):
    def __init__(self,
                 cmd,
                 board,
                 timeout_turn = 30*1000,
                 timeout_match = 180*1000,
                 max_memory = 350*1024*1024,
                 game_type = 1,
                 rule = 0,
                 folder = "./",
                 working_dir = "./",
                 tolerance = 1000):
        self.cmd = cmd
        self.board = copy.deepcopy(board)
        self.timeout_turn = timeout_turn
        self.timeout_match = timeout_match
        self.max_memory = max_memory
        self.game_type = game_type
        self.rule = rule
        self.folder = folder
        self.working_dir = working_dir
        self.tolerance = tolerance
        self.timeused = 0

        self.vms_memory = 0

        self.color = 1
        self.piece = {}
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j][0] != 0:
                    self.piece[board[i][j][0]] = (i,j)

        self.process = subprocess.Popen(shlex.split(cmd),
                                        shell=False,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        bufsize=1,
                                        close_fds='posix' in sys.builtin_module_names,
                                        cwd=self.working_dir)
        def enqueue_output(out, queue):
            for line in iter(out.readline, b''):
                queue.put(line)
            out.close()
        self.queue = Queue()
        queuethread = Thread(target=enqueue_output, args=(self.process.stdout, self.queue))
        queuethread.daemon = True # thread dies with the program
        queuethread.start()
        
        self.pp = psutil.Process(self.process.pid)
        self.write_to_process("START " + str(len(self.board)) + "\n")
        self.write_to_process("INFO timeout_turn " + str(self.timeout_turn) + "\n")
        self.write_to_process("INFO timeout_match " + str(self.timeout_match) + "\n")
        self.write_to_process("INFO max_memory " + str(self.max_memory) + "\n")
        self.write_to_process("INFO game_type " + str(self.game_type) + "\n")
        self.write_to_process("INFO rule " + str(self.rule) + "\n")
        self.write_to_process("INFO folder " + str(self.folder) + "\n")
    def init_board(self, board):
        self.board = copy.deepcopy(board)
        self.piece = {}
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j][0] != 0:
                    self.piece[board[i][j][0]] = (i,j)

    def write_to_process(self, msg):
        #print '===>', msg
        #sys.stdout.flush()
        self.process.stdin.write(msg.encode('utf-8'))
        self.process.stdin.flush()

    def suspend(self):
        try:
            self.pp.suspend()
        except:
            pass

    def resume(self):
        try:
            self.pp.resume()
        except:
            pass

    def update_vms(self):
        try:
            m = 0
            ds = list(self.pp.children(recursive=True))
            ds = ds + [self.pp]
            for d in ds:
                try:
                    m += d.memory_info()[1]
                except:
                    pass
            self.vms_memory = max(self.vms_memory, m)
        except:
            pass

    def wait(self, special_rule = ""):
        self.resume()
        
        msg = ''
        x, y = -1, -1
        timeout_sec = (self.tolerance + min((self.timeout_match - self.timeused), self.timeout_turn)) / 1000.
        start = time.time()
        while True:
            try: buf = self.queue.get_nowait().decode('utf-8')
            except Empty:
                if time.time() - start > timeout_sec:
                    break
                time.sleep(0.01)
            else:
                #print '<===', buf
                #sys.stdout.flush()
                # print(buf)
                if buf.lower().startswith("message"):
                    msg += buf
                else:
                    try:
                        if special_rule == "swap2":
                            if buf.lower().startswith("swap"):
                                x = -1
                                y = []
                            else:
                                x = -1
                                y = []
                                buf = buf.split()
                                for xy in buf:
                                    xi, yi = xy.split(",")
                                    xi, yi = int(xi), int(yi)
                                    y += [(xi, yi)]
                                assert(len(y) > 0)
                        else:
                            x, y = buf.split(",")
                            x, y = int(x), int(y)
                        break
                    except:
                        pass
        end = time.time()
        self.timeused += int(max(0, end-start-0.01)*1000)
        if end-start >= timeout_sec:
            raise Exception("TLE")

        self.update_vms()
        self.suspend()
        if self.vms_memory > self.max_memory and self.max_memory != 0:
            raise Exception("MLE")
        if get_dir_size(self.folder) > 70*1024*1024:
            raise Exception("FLE")

        if special_rule == "":
            self.piece[len(self.piece)+1] = (x,y)
            self.board[x][y] = (len(self.piece), self.color)
        return msg, x, y
        
    def turn(self, x, y):
        self.piece[len(self.piece)+1] = (x,y)
        self.board[x][y] = (len(self.piece), 3 - self.color)

        self.write_to_process("INFO time_left " + str(self.timeout_match - self.timeused) + "\n")
        self.write_to_process("TURN " + str(x) + "," + str(y) + "\n")

        return self.wait()

    def start(self):
        self.write_to_process("INFO time_left " + str(self.timeout_match - self.timeused) + "\n")
        self.write_to_process("BOARD\n")
        for i in range(1, len(self.piece)+1):
            self.process.stdin.write((str(self.piece[i][0]) + "," + str(self.piece[i][1]) + "," + str(self.board[self.piece[i][0]][self.piece[i][1]][1]) + "\n").encode('utf-8'))
        self.write_to_process("DONE\n")

        return self.wait()

    def swap2board(self):
        self.write_to_process("INFO time_left " + str(self.timeout_match - self.timeused) + "\n")
        self.write_to_process("SWAP2BOARD\n")
        for i in range(1, len(self.piece)+1):
            self.process.stdin.write(str(self.piece[i][0]) + "," + str(self.piece[i][1]) + "\n")
        self.write_to_process("DONE\n")

        return self.wait(special_rule = "swap2")

    def clean(self):
        self.resume()
        
        self.write_to_process("END\n")
        time.sleep(0.5)
        if self.process.poll() is None:
            #self.process.kill()
            for pp in self.pp.children(recursive=True):
                pp.kill()
            self.pp.kill()


def main():
    import numpy as np
    board_1 = [[(0,0) for i in range(15)] for j in range(15)]
    board_2 =  [[(0,0) for i in range(15)] for j in range(15)]
    last_move = (-1, -1)
    move_num = 0
    for step in range(20):
        if step==0:
            engine = new_protocol(
                cmd = "/media/duoyi/H/Downloads/gomoku/GomocupJudge-master/client/pbrain-embryo-1.0.4-33da365a-s",
                board = board_1,
                timeout_turn = 1000,
                timeout_match = 100000,
                max_memory = 450*1024*1024,
                game_type = 2,
                rule = 0,
                folder = "./",
                working_dir = "./",
                tolerance = 1000)
            msg, x, y = engine.start()
        elif step==1:
            engine1 = new_protocol(
                cmd = "/media/duoyi/H/Downloads/gomoku/GomocupJudge-master/client/pbrain-embryo-1.0.4-33da365a-s",
                board = board_2,
                timeout_turn = 1000,
                timeout_match = 100000,
                max_memory = 450*1024*1024,
                game_type = 2,
                rule = 0,
                folder = "./",
                working_dir = "./",
                tolerance = 1000)
            msg, x, y = engine1.start()
        

        elif step==6:
            print('restart black')
            board_1 = [[(0,0) for i in range(15)] for j in range(15)]
            board_2 =  [[(0,0) for i in range(15)] for j in range(15)]
            move_num=0
            last_move = (-1, -1)
            engine = new_protocol(
                cmd = "/media/duoyi/H/Downloads/gomoku/GomocupJudge-master/client/pbrain-embryo-1.0.4-33da365a-s",
                board = board_1,
                timeout_turn = 1000,
                timeout_match = 100000,
                max_memory = 450*1024*1024,
                game_type = 2,
                rule = 0,
                folder = "./",
                working_dir = "./",
                tolerance = 1000)
            msg, x, y = engine.start()

        elif step==7:
            print('restart white')
            engine1 = new_protocol(
                cmd = "/media/duoyi/H/Downloads/gomoku/GomocupJudge-master/client/pbrain-embryo-1.0.4-33da365a-s",
                board = board_2,
                timeout_turn = 1000,
                timeout_match = 100000,
                max_memory = 450*1024*1024,
                game_type = 2,
                rule = 0,
                folder = "./",
                working_dir = "./",
                tolerance = 1000)
            msg, x, y = engine1.start()

        else:
            if move_num % 2 == 0:
                msg, x, y = engine.turn(last_move[0], last_move[1])
            else:
                msg, x, y = engine1.turn(last_move[0], last_move[1])
            # print(np.array(board_1)[...,1] if else np.array(board_2)[...,1])
            # engine.init_board(board_1 if move_num%2==0 else board_2)
        # print(board_1(i,j) for i in range(20)] for j in range(20))
        # print(move_num, x,y)
        # print("board")

        print(step, move_num, np.array(board_1)[...,1])
        board_1[x][y] = (move_num+1, (move_num + 0) % 2 + 1)
        board_2[x][y] = (move_num+1, (move_num + 1) % 2 + 1)
        last_move = (x,y)
        move_num+=1
    engine.clean()
    engine1.clean()


if __name__ == '__main__':
    main()
