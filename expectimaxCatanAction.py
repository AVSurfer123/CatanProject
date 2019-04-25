import numpy as np
import random
import time
#from catanPlanBoard import opt_city, opt_road, opt_settlement, expected_gain

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

class State:

    def __init__(self, player, copy=False):
        if copy:
            self.copy(player)
            return
        self.player = player
        self.board = player.board
        self.id = player.player_id
        self.resources = np.array(player.resources)
        self.settlements = self.player.get_settlements()[:]
        self.cities = self.player.get_cities()[:]
        self.roads = self.player.get_roads()[:]
        self.points =  player.points
        #self.opponent = State()
        self.trade_req = [4,4,4]
        self.updateTradeReq()

    def copy(self, state):
        self.player = state.player
        self.board = state.board
        self.resources = np.array(state.resources)
        self.id = state.id
        self.settlements = state.settlements[:]
        self.cities = state.cities[:]
        self.roads = state.roads[:]
        self.points = state.points
        self.trade_req = state.trade_req[:]

    def updateTradeReq(self):
        ports = []
        for e in self.player.get_settlements():
            if self.player.board.is_port(e):
                ports.append(self.player.board.which_port(e))
        for e in self.player.get_cities():
            if self.player.board.is_port(e):
                ports.append(self.player.board.which_port(e))
        for i in range(2):
            if i in ports:
                self.trade_req[i] = 2
        if 3 in ports:
            self.trade_req = np.minimum(self.trade_req, [3,3,3])

    def generateSuccessor(self, id, action):
        resources = np.array(self.resources)
        successor = State(self, copy=True)
        if action == "Roll dice":
            states = {}
            roll_resources = np.array(self.player.board.get_resources(self.id), dtype=np.int32)
            for roll in rollProb:
                resources = roll_resources[roll-2, :]
                successor = State(self, copy=True)
                # print(successor.resources)
                # print(successor.resources.dtype)
                # print(resources)
                # print(resources.dtype)
                successor.resources += resources
                if roll == 7 and np.sum(successor.resources) > 7:
                    successor.resources = dumpState(self, 7)
                states[roll] = successor
            return states
        if action == "Buy settlement":
            resources = np.subtract(resources, costs[SETTLEMENT, :])
            successor.settlements.append(opt_settlement(self.player, self.board, self.player.preComp)[0])
            successor.points += 1
        elif action == "Buy card":
            resources = np.subtract(resources, costs[CARD, :])
            successor.points += 1
        elif action == "Buy city":
            resources = np.subtract(self.resources, costs[CITY, :])
            v = opt_city(self.player, self.board, self.player.preComp)[0]
            successor.settlements.remove(v)
            successor.cities.append(v)
            successor.points += 1
        elif action == "Buy road":
            resources = np.subtract(resources, costs[ROAD, :])
            successor.roads.append("Road")
        elif action == "Trade wood for brick":
            resources[0] -= self.trade_req[0]
            resources[1] += 1
        elif action == "Trade wood for grain":
            resources[0] -= self.trade_req[0]
            resources[2] += 1
        elif action == "Trade brick for wood":
            resources[1] -= self.trade_req[1]
            resources[0] += 1
        elif action == "Trade brick for grain":
            resources[1] -= self.trade_req[1]
            resources[2] += 1
        elif action == "Trade grain for wood":
            resources[2] -= self.trade_req[2]
            resources[0] += 1
        elif action == "Trade grain for brick":
            resources[2] -= self.trade_req[2]
            resources[1] += 1
        successor.resources = resources
        return successor

    def getLegalActions(self, id):
        actions = []
        if self.isTerminal():
            return []
        bestSettlement = opt_settlement(self.player, self.board, self.player.preComp)[1]
        bestCity = opt_city(self.player, self.board, self.player.preComp)[1]
        if bestSettlement:
            canBuildSettlement = self.board.if_can_build('settlement', *bestSettlement, self.id)
            canBuildRoad = not canBuildSettlement
        else:
            canBuildSettlement = False
            canBuildRoad = False
        if bestCity:
            canBuildCity = self.board.if_can_build('city', *bestCity, self.id)
        else:
            canBuildCity = False
        if np.all(self.resources >= costs[SETTLEMENT,:]) and canBuildSettlement:
            actions.append("Buy settlement")
        if np.all(self.resources >= costs[CARD,:]):
            actions.append("Buy card")
        if np.all(self.resources >= costs[CITY,:]) and len(self.settlements) > 0 and canBuildCity:
            actions.append("Buy city")
        if np.all(self.resources >= costs[ROAD,:]) and canBuildRoad:
            actions.append("Buy road")
        if self.resources[0] >= self.trade_req[0]:
            actions.append("Trade wood for brick")
            actions.append("Trade wood for grain")
        if self.resources[1] >= self.trade_req[1]:
            actions.append("Trade brick for wood")
            actions.append("Trade brick for grain")
        if self.resources[2] >= self.trade_req[2]:
            actions.append("Trade grain for wood")
            actions.append("Trade grain for brick")
        return actions

    def isTerminal(self):
        return self.points == 10


class Expectimax:

    def __init__(self, depth, evalFunction, opponent=False):
        self.depth = depth
        self.evalFunction = evalFunction
        self.total = 1 # 2 if opponent else 1  Fix for multiplayer

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.stateValue(gameState, 0, 1)[1]

    def getValue(self, gameState):
        return self.stateValue(gameState, 0, 1)[0]

    def stateValue(self, state, depth, index):
        if state.isTerminal() or depth == self.depth:
            return self.evalFunction(state), None
        elif index == 1:
            return self.maxValue(state, depth)
        elif index == self.total+1:
            return self.expectedValue(state, depth)
        elif index == 2:
            return self.minValue(state, depth)

    def maxValue(self, state, depth):
        v = float("-inf")
        actions = state.getLegalActions(1)
        best = None
        for a in actions:
            nextState = state.generateSuccessor(1, a)
            val = self.stateValue(nextState, depth, 2)[0]
            if val > v:
                v = val
                best = a
        if not actions:
            return self.stateValue(state, depth, 2)[0], None
        return v, best

    # Fix for multiplayer
    def minValue(self, state, depth):
        v = float("inf")
        actions = state.getLegalActions(2)
        best = None
        for a in actions:
            nextState = state.generateSuccessor(2, a)
            val = self.stateValue(nextState, depth, 3)[0]
            if val < v:
                v = val
                best = a
        if not actions:
            return self.stateValue(state, depth, 3)[0], None
        return v, best

    def expectedValue(self, state, depth):
        best = None
        value = 0
        depth = depth + 1
        successors = state.generateSuccessor(3, "Roll dice")
        for a in rollProb:
            nextState = successors[a]
            val = self.stateValue(nextState, depth, 1)[0] * rollProb[a]
            value += val
        return value, best


def boardHeuristic(state):
    resourceGain = stateResources(state)
    averageGain = averageResourceGain(resourceGain)
    resourceWeights = resourceWeighting(averageGain)
    points = state.points
    resources = state.resources
    num_settlements = len(state.settlements)
    num_cities = len(state.cities)
    num_roads = len(state.roads)
    trade = 12 - np.sum(state.trade_req)
    return points*10 + num_settlements*4 + num_cities*6 + 2*np.sum(resources.dot(resourceWeights)) \
            + np.sum(averageGain.dot(resourceWeights)) + 0.25*num_roads + trade/2

def averageResourceGain(resourceGain):
    return np.mean(resourceGain, axis=0)

def resourceWeighting(gain):
    v = 1/(gain+1)
    return v/np.sum(v)

def stateResources(state):
    r = np.zeros((11, 3))
    for vertex in state.settlements:
        x, y = state.board.get_vertex_location(vertex)
        for dx in [-1, 0]:
            for dy in [-1, 0]:
                xx = x + dx
                yy = y + dy
                if state.board.is_tile(xx, yy):
                    die = state.board.dice[yy, xx]
                    if die != 7:
                        resource = state.board.resources[yy, xx]
                        r[die - 2, resource] += 1
    for vertex in state.cities:
        x, y = state.board.get_vertex_location(vertex)
        for dx in [-1, 0]:
            for dy in [-1, 0]:
                xx = x + dx
                yy = y + dy
                if state.board.is_tile(xx, yy):
                    die = state.board.dice[yy, xx]
                    if die != 7:
                        resource = state.board.resources[yy, xx]
                        r[die - 2, resource] += 2
    return r


gameTree = Expectimax(depth=2, evalFunction=boardHeuristic)

def action(self):
    self.state = State(self)
    action = gameTree.getAction(self.state)
    if not hasattr(self, 'i'):
        self.i = 0
    self.i += 1
    # print("Turn:", self.i)
    # print("Resources:", self.resources)
    # print("Points:", self.points)
    if action:
        pass
        #print("Action:", action)
        #print("Points:", self.points)
        #print("Resources:", self.resources)
    if action == "Buy settlement":
        s = opt_settlement(self, self.board, self.preComp)[1]
        self.buy("settlement", *s)
        #print("Settlements:", self.get_settlements())
    elif action == "Buy card":
        self.buy("card")
    elif action == "Buy road":
        r = opt_road(self, self.board, opt_settlement(self, self.board, self.preComp)[0])
        self.buy("road", *r)
        #print("Roads:", self.get_roads())
    elif action == "Buy city":
        city = opt_city(self, self.board, self.preComp)[1]
        self.buy("city", *city)
        #print("Cities:", self.get_cities())
    elif action == "Trade wood for brick":
        self.trade(0, 1)
    elif action == "Trade wood for grain":
        self.trade(0, 2)
    elif action == "Trade brick for wood":
        self.trade(1, 0)
    elif action == "Trade brick for grain":
        self.trade(1, 2)
    elif action == "Trade grain for wood":
        self.trade(2, 0)
    elif action == "Trade grain for brick":
        self.trade(2, 1)
    return

def planBoard(board):
    #print("**********************NEW GAME***********************")
    return expected_gain(board)

def dumpPolicy(self, max_resources):
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources

def dumpState(state, max_resources):
    new_resources = np.minimum(state.resources, max_resources // 3)
    return new_resources



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