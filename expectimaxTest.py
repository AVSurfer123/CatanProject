from catan import *

from expectimaxCatanAction import *

num_trials = 100

width, height = 4, 4
dice = get_random_dice_arrangement(width, height)
resources = np.random.randint(0, 3, (height, width))
board = Catan(dice, resources)

player = Player("Player 1", action, dumpPolicy, planBoard)
player.join_board(board)

gameTree = Expectimax(depth=2, evalFunction=boardHeuristic)
startState = State(player)
print(gameTree.getAction(startState))
print(gameTree.getValue(startState))
print(boardHeuristic(startState))
state = State(startState, copy=True)
state.resources = np.array([0,3,3])
#print(gameTree.getAction(state))
#print(gameTree.getValue(state))
#print(boardHeuristic(state))
state.resources = np.array([6,2,2])
print(state.getLegalActions(1))
print(gameTree.getAction(state))
print(gameTree.getValue(state))
print(boardHeuristic(state))

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
