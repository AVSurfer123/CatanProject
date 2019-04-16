from catan import *
from mdpCatanAction import MDP, action, dumpPolicy, planBoard, ValueIteration

num_trials = 100

width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)

player = Player("Player 1", action, dumpPolicy, planBoard)
player.join_board(board)

mdp = MDP(player)
valueIter = ValueIteration(mdp, discount=0.9, iterations=100)
print(valueIter.values[(3,3,3,4)])
print(valueIter.values[(1,2,2,5)])
for a in mdp.getPossibleActions((3,3,3,0)):
    print("Action", a, ":", mdp.getTransitionStatesAndProbs((3,3,3,0), a))
    print("Q-Value with", a, ":", valueIter.getQValue((3,3,3,0), a))
print(valueIter.getAction((3,3,3,4))) 
print(player.player_id) 
print(board.num_players) 

#print(mdp.getStates())
print(mdp.getPossibleActions((2,2,4,4)))
print(mdp.getTransitionStatesAndProbs((2,2,2,1), "Buy settlement"))
print(mdp.getReward((2,2,2,1), None, (4,4,1,6)))
print(mdp.getStartState())

input()

print("average turns to win: {}".format(simulate_1p_game(action, dumpPolicy, planBoard, board, num_trials)))

settlements, cities, roads, hands, live_points, dice_rolls = simulate_1p_game_with_data(action, dumpPolicy, planBoard, board)

def draw(t):
    t = int(t)
    live_board = Catan(board.dice, board.resources, [], [])
    live_board.settlements = settlements[t]
    live_board.cities = cities[t]
    live_board.roads = roads[t]
    print("turn:", t)
    print("points:", live_points[t])
    print("dice roll:", dice_rolls[t])
    print("resources:", hands[t])
    live_board.draw()

from ipywidgets import *
interact(draw, t=(0, len(live_points) - 1, 1))