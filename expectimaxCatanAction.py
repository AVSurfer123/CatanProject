import numpy as np
import random 
import time

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}

costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])


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
        self.resources = player.resources[:]
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
        self.resources = state.resources[:]
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
            self.trade_req = np.min(self.trade_req, [3,3,3])

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
            successor.settlements.append(optimalSettlement(self.player))
            successor.points += 1
        elif action == "Buy card":
            resources = np.subtract(resources, costs[CARD, :])
            successor.points += 1
        elif action == "Buy city":
            resources = np.subtract(self.resources, costs[CITY, :])
            city = successor.settlements.pop()
            successor.cities.append(city)
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
        if np.all(self.resources >= costs[SETTLEMENT,:]):
            actions.append("Buy settlement")
        if np.all(self.resources >= costs[CARD,:]):
            actions.append("Buy card")
        if np.all(self.resources >= costs[CITY,:]) and len(self.settlements) > 0:
            actions.append("Buy city")
        if np.all(self.resources >= costs[ROAD,:]):
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
        self.total = 2 if opponent else 1

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
    resourceWeights = resourceWeighting(averageGain, state.board)
    points = state.points
    resources = state.resources
    num_settlements = len(state.settlements)
    num_cities = len(state.cities)
    num_roads = len(state.roads)
    trade = 12 - np.sum(state.trade_req)
    return points*10 + num_settlements*4 + num_cities*8.2 + 2*np.sum(resources.dot(resourceWeights)) \
            + np.sum(averageGain.dot(resourceWeights)) + 0.5*num_roads + trade/2

def averageResourceGain(resourceGain):
    return np.mean(resourceGain, axis=0)

def resourceWeighting(gain, board):
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

# Need working optimal functions so we can build other settlements. Need to create optimalRoad function as well
def optimalSettlement(player):
    if player.get_settlements() == []:
        return player.board.get_vertex_number(*planBoard(player.board))
    return np.random.randint(player.board.max_vertex)

def optimalCity(player):
    return player.get_settlements()[0]

def optimalRoad(player):
    pass

gameTree = Expectimax(depth=1, evalFunction=boardHeuristic)

def action(self):
    self.state = State(self)
    start = time.time()
    action = gameTree.getAction(self.state)
    if not hasattr(self, 'i'):
        self.i = 0
    if self.i < 10:
        self.i += 1
        print("Time for expectimax:", time.time()-start)
        print("Current points:", self.points)
        print("Resources:", self.resources)
        print("Action:", action)
        print("Value:", gameTree.getValue(self.state))
        print("Heursitic:", boardHeuristic(self.state))
    if action == "Buy settlement":
        self.buy("settlement", *self.board.get_vertex_location(optimalSettlement(self)))
    elif action == "Buy card":
        self.buy("card")
    elif action == "Buy road":
        if self.get_settlements():
            v = random.choice(self.get_settlements())
            x, y = self.board.get_vertex_location(v)
            s = x + np.random.randint(2)
            t = y + np.random.randint(2)
            self.buy("road", (x,y), (s,t))
    elif action == "Buy city":
        self.buy("city", *self.board.get_vertex_location(optimalCity(self)))
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
    if self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)

def planBoard(baseBoard):
    x = genRand(1,baseBoard.width)
    y = genRand(1,baseBoard.height)
    return x, y

def dumpPolicy(self, max_resources):
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources

def dumpState(state, max_resources):
    new_resources = np.minimum(state.resources, max_resources // 3)
    return new_resources

def genRand(low,high):
    return np.random.randint(low, high)

