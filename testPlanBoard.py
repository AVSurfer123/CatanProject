from catan import *
from catanPlanBoard import planBoard
from basicCatanAction import action, dumpPolicy
import matplotlib.pyplot as plt

num_trials = 100
print("computing")

width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)

player = Player("Player 1", action, dumpPolicy, planBoard)
player.join_board(board)

player2 = Player("Player 2", action, dumpPolicy, planBoard)
#player2.join_board(board)

p = planBoard(board)
expected_gain = p[3]
num_turns = 10

settlement_v, settlement_c = p[0](player, board, expected_gain)
print(player.name+": Settlement 1 at", settlement_v, settlement_c)
board.settlements[settlement_v] = player.player_id

"""
settlement_v2, settlement_c2 = p[0](player2, board, expected_gain)
print(player2.name+": Settlement 1 at", settlement_v2, settlement_c2)
board.settlements[settlement_v2] = player2.player_id
"""

for i in range(2, num_turns+1):
    settlement_v, settlement_c = p[0](player, board, expected_gain)
    print(player.name+": Settlement", i,  "at", settlement_v, settlement_c)
    road, road_vc, road_nc = p[2](player, board, settlement_v)
    board.roads[road] = player.player_id

    """
    settlement_v2, settlement_c2 = p[0](player2, board, expected_gain)
    print(player2.name+": Settlement", i,  "at", settlement_v2, settlement_c2)
    road2, road_vc2, road_nc2 = p[2](player2, board, settlement_v2)
    board.roads[road2] = player2.player_id
    """


    while(road[1] != settlement_v):
        if road[1] != settlement_v:
            road, road_vc, road_nc = p[2](player, board, settlement_v)
            board.roads[road] = player.player_id

        """
        if road2[1] != settlement_v2:
            road2, road_vc2, road_nc2 = p[2](player2, board, settlement_v2)
            board.roads[road2] = player2.player_id
        """

    board.settlements[settlement_v] = player.player_id
    #board.settlements[settlement_v2] = player2.player_id



city_v, city_c = p[1](player,board,expected_gain)
print("First city to be upgrade should be", city_v, city_c)

board.draw()
plt.show()
#print("finished game", i)

print("done")

#print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))

#settlements, cities, roads, hands, live_points, dice_rolls = simulate_1p_game_with_data(action, dumpPolicy, planBoard, board)
