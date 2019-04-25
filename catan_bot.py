import numpy as np

settlement_threshold = 5
rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}
goal_list = {"default": 4,"settlement": 0, "card": 1, "city": 2, "road": 3}
costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0],
                  [1, 1, 1]])

SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3
ROBBER_MAX_RESOURCES = 7

def action(self):
    if self.points == 9:
        if self.if_can_buy("card"):
            self.buy("card")
    total_resources = np.sum(self.resources)
    num_settlements = len(self.get_settlements())
    num_cities = len(self.get_cities())
    #print("resources at beginning of turn ", self.resources)
    #print("num_settlements", num_settlements)
    #print("num_cities", num_cities)
    #print("points", self.points)
    #print("roads", self.board.roads)
    if num_settlements < settlement_threshold:
        self.optimal_settlement, op_settlement = opt_settlement(self, self.board, self.preComp)
        #print("optimal settlement", op_settlement)
        if op_settlement:
            #print("dice", self.board.dice)
            if self.board.if_can_build("settlement", op_settlement[0], op_settlement[1], self.player_id) and num_settlements < settlement_threshold:
                if self.if_can_buy("settlement"):
                    #print("buying settlement")
                    self.buy("settlement", op_settlement[0], op_settlement[1])
            elif self.if_can_buy("road"):
                self.to_build_road = opt_road(self, self.board, self.optimal_settlement)
                if self.to_build_road is not None:
                    #print("buying road")
                    self.buy("road", self.to_build_road[0], self.to_build_road[1])
    if num_cities < settlement_threshold:
        self.optimal_city, op_city  = opt_city(self, self.board, self.preComp)
        if op_city:
            if self.if_can_buy("city") and self.optimal_city is not None:
                #print("buying city")
                self.buy("city", op_city[0], op_city[1])
    if self.points > 7 and self.resources[0] > 2*costs[CARD][0] and self.resources[1] > 2*costs[CARD][1] and self.resources[2] > 2*costs[CARD][2]:
        #print("buying card")
        self.buy("card")
    if self.resources[np.argmax(self.resources)] >= 4 and total_resources > 7:
        #print("trying to trade")
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
   # print("resources at end of turn", self.resources)
    return


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
    return max(vertex_list, key = vertex_score, default=(None, None))

def opt_city(player, board, gains, goal = "default"):
    """ Same thing as opt_settlements but for cities."""
    goal_index = goal_list.get(goal, 0)
    vertex_score = lambda t: settlement_eval(player, board, t[0], gains, goal_index)
    vertex_list = [(v, board.get_vertex_location(v)) for v in board.get_player_settlements(player.player_id) \
                    if board.if_can_build("city", *(board.get_vertex_location(v)), player.player_id)]

    return max(vertex_list, key = vertex_score, default=(None,None))


def opt_road(player, board, building_vertex):
    """ Given some sort of target settlement/building, determines the optimal place to put a road. Currently only adapted for singleplayer. """
    player_buildings = board.get_player_settlements(player.player_id) + board.get_player_cities(player.player_id)
    player_roads = board.get_player_roads(player.player_id)
    accessible_vertices = sorted(set(player_buildings+ [v for pair in player_roads for v in pair]), \
                                    key = lambda v: manhattan_distance(v,building_vertex,board))
    if building_vertex in accessible_vertices:
        print("Error: Building vertex already accessible, do not need road.")
        return None, None
    for v in accessible_vertices:
        neighbor_vertices = []
        x,y = board.get_vertex_location(v)
        for dx, dy in [[0,1],[0,-1],[1,0],[-1,0]]:
            xx = x + dx
            yy = y + dy
            if board.get_vertex_number(xx,yy) in range(board.max_vertex+1):
                neighbor_vertices.append(board.get_vertex_number(xx,yy))
        neighbor_vertices = sorted(neighbor_vertices, key = lambda v: manhattan_distance(v,building_vertex,board))
        for n in neighbor_vertices:
            if board.if_can_build_road(v, n, player.player_id):
                v_t = list(board.get_vertex_location(v))
                n_t = list(board.get_vertex_location(n))
                return v_t, n_t
    print("need to implement default behavior")
    return None,None

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
    dist_key = lambda t: t[1]
    player_dist = distance_score(v, board, player.player_id)
    enemies = set([id for id in board.settlements.values() if id != player.player_id])
    player_distances = [(id, distance_score(v, board, id)) for id in enemies] + [(player.player_id, player_dist)]
    closest_player = min(player_distances, key=dist_key)
    if closest_player[0] == player.player_id:
        dist_score = player_dist
    else:
        dist_score = closest_player[1] + player_dist
    vertex_score = gains[v, 0]
    diversity = gains[v, 1]
    resource_count = gains[v, 2:5]
    resource_weights = sum([w[resource]*resource_count[resource] for resource in range(len(resource_count))])
    return h1*diversity+h2*h3*vertex_score+h4*resource_weights-h5*dist_score

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

def distance_score(vertex1, board, player_id): #implement preference for closer settlements
    """ Want to see how close the vertex is to our closest settlement """
    num_buildings = 0
    total_dist = 0
    player_buildings = board.get_player_settlements(player_id) + board.get_player_cities(player_id)

    if len(player_buildings) == 0: #if it is our first turn
        return 0

    player_roads = board.get_player_roads(player_id)
    accessible_vertices = list(set(player_buildings+ [vertex for pair in player_roads for vertex in pair]))
    get_distance = lambda v: manhattan_distance(v, vertex1, board)
    min_distance = min(map(get_distance, accessible_vertices))

    enemy_buildings = [v for v in board.settlements if board.settlements[v] != player_id]
    enemy_roads = [r for r in board.roads if board.roads[r] != player_id]


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
    

def manhattan_distance(vertex_1, vertex_2, board):
    x1, y1 = board.get_vertex_location(vertex_1)
    x2, y2 = board.get_vertex_location(vertex_2)
    return abs(x1 - x2) + abs(y1 - y2)


def planBoard(board):
    return expected_gain(board)

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
    optimumSettlements = settlement_threshold

    new_resources = np.copy(self.resources)

    # Checking what strategy to use.
    if settlementCount < optimumSettlements:
        # Optimizing for buying settlements.
        while sum(new_resources) > ROBBER_MAX_RESOURCES:
            if 2 * new_resources[1] > new_resources[0]:
                new_resources[1] -= 1
            elif 2 * new_resources[2] > new_resources[0]:
                new_resources[2] -= 1
            else:
                new_resources[0] -= 1
    elif cityCount < optimumSettlements:
        # Optimising for buying cities.
        while sum(new_resources) > ROBBER_MAX_RESOURCES:
            if new_resources[0] > 0:
                new_resources[0] -= 1
            elif new_resources[2] > new_resources[1]:
                new_resources[2] -= 1
            else:
                new_resources[1] -= 1
    else:
        # Optimizing for cards.
        while sum(new_resources) > ROBBER_MAX_RESOURCES:
            if new_resources[2] > 2:
                new_resources[2] -= 1
            elif new_resources[1] > 2:
                new_resources[1] -= 1
            else:
                new_resources[0] -= 1
    return self.resources - new_resources