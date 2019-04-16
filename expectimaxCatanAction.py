import numpy as np

class Expectimax:

    def __init__(self, depth, evalFunction):
        self.depth = depth
        self.evalFunction = evalFunction

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
       
        return self.stateValue(gameState, 0, 0)[1]
        
    def stateValue(self, state, depth, index):
        if self.isTerminal(state) or depth == self.depth:
            return self.evaluationFunction(state), None
        elif index == 0:
            return self.maxValue(state, depth)
        else:
            return self.expectedValue(state, depth, index)

    def maxValue(self, state, depth):
        v = float("-inf")
        actions = state.getLegalActions(0)
        total = state.getNumAgents()
        best = None
        for a in actions:
            nextState = state.generateSuccessor(0, a)
            val = self.stateValue(nextState, depth, 1 % total)[0]
            if val > v:
                v = val
                best = a
        return v, best

    def expectedValue(self, state, depth, index):
        v = float("inf")
        actions = state.getLegalActions(index)
        total = state.getNumAgents()
        best = None
        value = 0
        if index == total - 1:
            depth = depth + 1
        for a in actions:
            nextState = state.generateSuccessor(index, a)
            val = self.stateValue(nextState, depth, (index + 1) % total)[0] / len(actions)
            if val < v:
                v = val
                best = a
            value += val
        return value, a

    def isTerminal(self, state):
        pass


def boardHeuristic(player, board):
    pass


def action(self):
    if self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)

def planBoard(baseBoard):
    x = genRand(0,baseBoard.width+1)
    y = genRand(0,baseBoard.height+1)
    optimalSettlements = (x,y)
    return optimalSettlements

def dumpPolicy(self, max_resources):
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources

def genRand(low,high):
    return np.random.randint(low, high)
