import numpy as np
import catan
from catan import *

from catanPlanBoard import expected_gain
from naiveCatanAction import action

def planBoard(board):
    return expected_gain(board)

# sample dump policy function: takes in the "Player" and ROBBER_MAX_RESOURCES
# and returns a resource array which indicates the number of each resource to dump.
# self.resources - dumpPolicy(self, max_resources) must sum up to less than or equal ROBBER_MAX_RESOURCES
def dumpPolicy(self, max_resources):
    settlementCount = 0
    for id in self.board.settlements.values():
        if id == self.player_id:
            settlementCount += 1

    cityCount = 0
    for id in self.board.settlements.values():
        if id == self.player_id:
            cityCount += 1

    # Cumulative value of cities and settlements before switching to card buying strategy.
    optimumSettlements = 3

    new_resources = np.copy(self.resources)

    # Checking what strategy to use.
    if settlementCount < optimumSettlements:
        # Optimizing for buying settlements.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if 2 * new_resources[1] > new_resources[0]:
                new_resources[1] -= 1
            elif 2 * new_resources[2] > new_resources[0]:
                new_resources[2] -= 1
            else:
                new_resources[0] -= 1
    elif cityCount < optimumSettlements:
        # Optimising for buying cities.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if new_resources[0] > 0:
                new_resources[0] -= 1
            elif new_resources[2] > new_resources[1]:
                new_resources[2] -= 1
            else:
                new_resources[1] -= 1
    else:
        # Optimizing for cards.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if new_resources[2] > 2:
                new_resources[2] -= 1
            elif new_resources[1] > 2:
                new_resources[1] -= 1
            else:
                new_resources[0] -= 1
    return self.resources - new_resources


num_trials = 100
width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)
import time
start = time.time()
print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))
#print("game finished?")
print("Time for {} games:".format(num_trials), time.time() - start)
