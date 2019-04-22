from catan import *
from catanPlanBoard import planBoard
from basicCatanAction import action, dumpPolicy

num_trials = 100
print("computing")

width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)

player = Player("Player 1", action, dumpPolicy, planBoard)
player.join_board(board)
print(board.resources)
print(board.dice)
p = planBoard(board)
v, c = p[0](player,board)
print(v, c)
x_b, y_b = c
board.build(x_b, y_b,"settlement", player.player_id)

v, c = p[0](player,board)
print(v,c)
r_v, r_n = p[2](player, board, v)

board.build_road(r_v, r_n, player.player_id)
#print("finished game", i)

print("done")

#print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))

#settlements, cities, roads, hands, live_points, dice_rolls = simulate_1p_game_with_data(action, dumpPolicy, planBoard, board)
