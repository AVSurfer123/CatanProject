import catan
import numpy as np
from naiveCatanAction import manhattan_distance

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}
goal_list = {"settlement": 0, "card": 1, "city": 2, "road": 3}
costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

def planBoard(board):
    resource_scarcity = get_resource_scarcity(board)
    def opt_settlements(player, board, goal="settlement"):
        """
        Want to return a list of optimal settlements given the current state of the board.
        Should iterate through every viable vertex and pick the most optimal one relative to the player's
        current game state. Emphasis on diversifying resources/increasing odds for a resource.
        """
        optimal_vertex = -1
        max_score = 0
        goal_index = goal_list.get(goal, 0)
        for v in range(board.max_vertex+1):
            x,y = board.get_vertex_location(v)
            if board.if_can_build("settlement", x, y, player.player_id):
                vertex_score = vertex_eval(player,x,y,"settlement",board, resource_scarcity, goal_index) #formula calculations go here
                if vertex_score > max_score:
                    optimal_vertex = v
                    max_score = vertex_score
        return (optimal_vertex, board.get_vertex_location(optimal_vertex))

    def opt_cities(player, board):
        """ Same thing as opt_settlements but for cities. Emphasis on maximizing resource gain per turn. """
        return 0 #for now

    return [opt_settlements, opt_cities]

def vertex_eval(player, x, y, building, board, resource_scarcity, goal=0):
    id = player.player_id
    h1,h2,h3,h4,h5 = 0.1,1,1,1,1 #hyperparameters to tune
    w = costs[goal] # weight given to resource based on goal as determined by player action (very naive implementation right now)
    prox_score = get_proximity_score([x,y], board, id) # should be the number of settlements nearby
    vertex_score = 0
    resource_check = np.array([0,0,0]) # should check and see if there is a type of resource existing around vertex, which will be denoted by 1
    multiplier = 2 if building == "city" else 1
    for dx in [-1, 0]:
        for dy in [-1,0]:
            xx = x + dx
            yy = y + dy
            if board.is_tile(xx, yy):
                die = board.dice[yy, xx]
                if board.is_tile(xx, yy) and die != 7:
                    resource = board.resources[xx,yy]
                    if resource_check[resource] != 1 and w[resource] != 0: #check to see if we have a resource that is amicable to our goal
                        resource_check[resource] = 1
                    vertex_score += multiplier*rollProb.get(die, 0)*resource_scarcity[resource]*w[resource]+h5*prox_score
    diversity = np.sum(resource_check)
    return h1*diversity+h2*h3*h4*vertex_score+h5*prox_score

def get_resource_scarcity(board):
    resource_check = np.zeros(3)
    for x in range(board.resources.shape[0]):
        for y in range(board.resources.shape[1]):
            if board.is_tile(x,y):
                r = board.resources[x,y]
                if r != -1:
                    resource_check[r] += 1;
    total_resources = board.width*board.height
    return total_resources/resource_check #lower resources should yield a higher score.

def get_proximity_score(vertex1, board, player_id):
    num_buildings = 0
    total_dist = 0
    if board.num_players == 1 or len(board.settlements)+len(boad.cities) == 0:
        return 0
    for s in board.settlements:
        if board.settlements[s] != player_id:
            vertex2 = s
            total_dist+= manhattan_distance(vertex1, vertex2)
            num_buildings+=1

    for c in board.cities:
        if board.cities[c] != player_id:
            vertex2 = c
            total_dist+= manhattan_distance(vertex1, vertex2)
            num_buildings+=1

    return total_dist/num_buildings
