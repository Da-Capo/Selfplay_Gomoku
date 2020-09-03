from ctypes import c_int, c_void_p, c_double, c_char_p, POINTER
import numpy.ctypeslib as npct
import numpy as np

lib = npct.load_library('./build/libgomoku.so', ".")
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

g = lib.G_new(c_int(15), c_int(5), c_int(1))
board = np.zeros((15,15), dtype=np.float32)
status = np.zeros(2, dtype=np.float32)

for a in [3,13,4,14,5,15,6,16,7,17,8,18,9,19]:
    lib.G_move(g,a)
    lib.G_board(g,board)
    lib.G_display(g)
    lib.G_status(g, status)
    print(status)
    print(board)

lib.G_clean(g)