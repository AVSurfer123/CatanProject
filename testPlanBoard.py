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
#print(board.resources)
#print(board.dice)
p = planBoard(board)
expected_gain = p[3]
settlement_v, settlement_c = p[0](player, board, expected_gain)

print("First settlement at", settlement_v, settlement_c)

board.build(*(settlement_c), "settlement", player.player_id)

settlement_v, settlement_c = p[0](player,board, expected_gain)
print("Second settlement should be at", settlement_v, settlement_c)

road, road_vc, road_nc = p[2](player, board, settlement_v)
board.build_road(road_vc, road_nc, player.player_id)
road, road_vc, road_nc = p[2](player, board, settlement_v)
board.roads[road] = player.player_id


board.settlements[settlement_v] = player.player_id

city_v, city_c = p[1](player,board,expected_gain)
print("First city to be upgrade should be", city_v, city_c)

board.draw()
plt.show()
#print("finished game", i)

print("done")

#print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))

#settlements, cities, roads, hands, live_points, dice_rolls = simulate_1p_game_with_data(action, dumpPolicy, planBoard, board)
