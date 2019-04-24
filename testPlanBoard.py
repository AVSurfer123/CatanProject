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

player1 = Player("Player 1", action, dumpPolicy, planBoard)
player1.join_board(board)

player2 = Player("Player 2", action, dumpPolicy, planBoard)
player2.join_board(board)

player3 = Player("God", action, dumpPolicy, planBoard)
player3.join_board(board)


player1.resources = np.array([1000,1000,1000])
player2.resources = np.array([1000,1000,1000])

player1.buy("settlement", 1, 1)
player1.buy("road", (1,1), (2,1))


player1.buy("road", (2,1), (3,1))
player1.buy("road", (3,1), (4,1))

player1.buy("road", (3,1), (3,2))


player1.buy("road", (4,1), (4,2))
player2.buy("road", (3,1), (3,2))
player2.buy("settlement", 4,1)
player2.buy("road", (3,1), (4,1))
player1.buy("settlement", 4, 2)
player1.buy("road", (3,1), (3,0))

player1.buy("settlement", 1,3)

board.draw()

#board.settlements[board.get_vertex_number(1,1)] = player3.player_id
#board.settlements[board.get_vertex_number(1,3)] = player3.player_id
#board.settlements[board.get_vertex_number(3,3)] = player3.player_id
#board.settlements[board.get_vertex_number(3,1)] = player3.player_id


p = planBoard(board)
expected_gain = p[3]
num_turns = 100

settlement_v, settlement_c = p[0](player1, board, expected_gain)
print(player1.name+": Settlement 1 at", settlement_v, settlement_c)
player1.buy("settlement", *settlement_c)


settlement_v2, settlement_c2 = p[0](player2, board, expected_gain)
print(player2.name+": Settlement 1 at", settlement_v2, settlement_c2)
player2.buy("settlement", *settlement_c2)


for i in range(2, num_turns+1):
    for player in [player1, player2]:
        settlement_v, settlement_c = p[0](player, board, expected_gain)
        if settlement_v:
            print("what")
            if board.if_can_build("settlement", *settlement_c, player.player_id):
                player.buy("settlement", *settlement_c)
                print(player.name+": Settlement", i,  "at", settlement_v, settlement_c)
            else:
                road_vc, road_nc = p[2](player, board, settlement_v)
                player.buy("road", road_vc, road_nc)
                print(player.name+": Road", i,  "at", road_vc, road_nc)

        """
        while(road[1] != settlement_v):
            if road[1] != settlement_v:
                road, road_vc, road_nc = p[2](player, board, settlement_v)
                board.roads[road] = player.player_id
        """

        #board.settlements[settlement_v] = player.player_id



city_v, city_c = p[1](player,board,expected_gain)
print("First city to be upgrade should be", city_v, city_c)

board.draw()
plt.show()
#print("finished game", i)

print("done")

#print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))

#settlements, cities, roads, hands, live_points, dice_rolls = simulate_1p_game_with_data(action, dumpPolicy, planBoard, board)
