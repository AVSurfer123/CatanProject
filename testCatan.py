import numpy as np
import catan
from catan import *

from catan_bot import action, expected_gain, planBoard, dumpPolicy

num_trials = 10
width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)
import time
start = time.time()
print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))
#print("game finished?")
print("Time for {} games:".format(num_trials), time.time() - start)
board.draw()
plt.show()