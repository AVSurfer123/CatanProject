import catan
import numpy as np
from naiveCatanAction import manhattan_distance

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}
goal_list = {"default": 0,"settlement": 1, "card": 2, "city": 3, "road": 4}
costs = np.array([[1, 1, 1],
                  [2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])

def opt_settlement(player, board, gains, goal="default"):
    """
    Want to return a list of optimal settlements given the current state of the board.
    Should iterate through every viable vertex and pick the most optimal one relative to the player's
    current game state. Emphasis on diversifying resources/increasing odds for a resource.
    """
    goal_index = goal_list.get(goal, 0)
    vertex_score = lambda t: vertex_eval(player, board, t[0], gains, goal_index)
    vertex_list = [(v, board.get_vertex_location(v)) for v in range(board.max_vertex+1) \
                    if board.if_can_build("settlement", *(board.get_vertex_location(v)))]
    return max(vertex_list, key = vertex_score)

def opt_city(player, board, goal = "default"):
    """ Same thing as opt_settlements but for cities."""
    goal_index = goal_list.get(goal, 0)
    vertex_score = lambda t: settlement_eval(player, board, t[0], gains, goal_index)
    vertex_list = [(v, board.get_vertex_location(v)) for v in board.get_player_settlements(player.player_id) \
                    if board.if_can_build("city", *(board.get_vertex_location(v)))]
    return max(vertex_list, key = vertex_score)


def opt_road(player, board, building_vertex):
    """ Given some sort of target settlement/building, determines the optimal place to put a road. Currently only adapted for singleplayer. """
    player_buildings = board.get_player_settlements(player.player_id) + board.get_player_cities(player.player_id)
    player_roads = board.get_player_roads(player.player_id)
    accessible_vertices = sorted(set(player_buildings+ [v for pair in player_roads for v in pair]), key = lambda v: manhattan_distance(v,building_vertex,board))
    for v in accessible_vertices:
        neighbor_vertices = []
        x,y = board.get_vertex_location(v)
        for dx in [-1, 0]:
            for dy in [-1,0]:
                xx = x + dx
                yy = y + dy
                if board.is_tile(xx, yy):
                    neighbor_vertices.append(board.get_vertex_number(xx,yy))
        neighbor_vertices = sorted(neighbor_vertices, key = lambda v: manhattan_distance(v,building_vertex,board))
        for n in neighbor_vertices:
            if board.if_can_build_road(v, n, player.player_id):
                v = list(board.get_vertex_location(v))
                n = list(board.get_vertex_location(n))
                return v,n


def planBoard(board):
    return [opt_settlement, opt_city, opt_road, expected_gain(board)]

def expected_gain(board):
    gains = np.zeros((board.max_vertex+1, 5))
    resource_scarcity = get_resource_scarcity(board)
    for v in range(board.max_vertex+1):
        vertex_score = 0
        x,y = board.get_vertex_location(v)
        resource_check = np.array([0,0,0])
        for dx in [-1, 0]:
            for dy in [-1,0]:
                xx = x + dx
                yy = y + dy
                if board.is_tile(xx, yy):
                    die = board.dice[yy, xx]
                    if board.is_tile(xx, yy) and die != 7:
                        resource = board.resources[xx,yy]
                        resource_check[resource] += 1
                        vertex_score += rollProb.get(die, 0)/resource_scarcity[resource]

        diversity = np.count_nonzero(gains[v, 1:4])
        gains[v,0] = vertex_score
        gains[v,1] = diversity
        gains[v,2:5] = resource_check

    return gains


def settlement_eval(player, board, v, gains, goal=0):
    h1,h2,h3,h4,h5 = 0.1,1,1,1,0.5 #hyperparameters to tune
    w = 0.5*costs[goal] # weight given to resource based on goal as determined by player action (very naive implementation right now)
    vertex_score = gains[v,0]
    diversity = gains[v, 1]
    resource_count = gains[v, 2:5]
    resource_weights = sum([w[resource]*resource_count[resource] for resource in range(len(resource_count))])

    return h1*diversity+h2*h3*vertex_score+h4*resource_weights

def vertex_eval(player, board, v, gains, goal=0):
    h1,h2,h3,h4,h5 = 0.1,1,1,0.2,0.5 #hyperparameters to tune
    w = 0.5*costs[goal] # weight given to resource based on goal as determined by player action (very naive implementation right now)
    prox_score = get_proximity_score(v, board, player) # should be the number of settlements nearby
    vertex_score = gains[v, 0]
    diversity = gains[v, 1]
    resource_count = gains[v, 2:5]
    resource_weights = sum([w[resource]*resource_count[resource] for resource in range(len(resource_count))])
    return h1*diversity+h2*h3*vertex_score+h4*resource_weights-h5*prox_score

def get_resource_scarcity(board):
    resource_check = np.zeros(3)
    for x in range(board.resources.shape[0]):
        for y in range(board.resources.shape[1]):
            if board.is_tile(x,y):
                r = board.resources[x,y]
                if r != -1:
                    resource_check[r] += 1;
    total_resources = board.width*board.height
    return resource_check/total_resources #lower resources should yield a higher score.

def get_proximity_score(vertex1, board, player): #implement preference for closer settlements
    """ Want to see how close the vertex is to our closest settlement """
    num_buildings = 0
    total_dist = 0
    player_buildings = board.get_player_settlements(player.player_id) + board.get_player_cities(player.player_id)

    if len(player_buildings) == 0: #if it is our first turn
        return 0

    player_roads = board.get_player_roads(player.player_id)
    accessible_vertices = list(set(player_buildings+ [vertex for pair in player_roads for vertex in pair]))
    get_distance = lambda v: manhattan_distance(v, vertex1, board)
    min_distance = min(map(get_distance, accessible_vertices))


    """
    for s in board.settlements:
        if board.settlements[s] != player_id:
            vertex2 = s
            total_dist_enemies += manhattan_distance(vertex1, vertex2, board)
            num_buildings+=1

    for c in board.cities:
        if board.cities[c] != player_id:
            vertex2 = c
            total_dist_enemies += manhattan_distance(vertex1, vertex2, board)
            num_buildings+=1

    """
    return min_distance
