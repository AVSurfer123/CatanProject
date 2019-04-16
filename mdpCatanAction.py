import numpy as np
import itertools
import time
from collections import defaultdict

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}

costs = np.array([[2, 1, 1],
                  [1, 2, 2],
                  [0, 3, 3],
                  [1, 1, 0]])


SETTLEMENT = 0
CARD = 1
CITY = 2
ROAD = 3

WOOD = 0
BRICK = 1
GRAIN = 2

class MDP:

    def __init__(self, player, start_state=(3,3,3,0), state_max=3):
        self.player = player
        resource_states = itertools.product(range(state_max+1), repeat=3)
        self.states = []
        for r in resource_states:
            for p in range(11):
                self.states.append(r+(p,))
        #self.states = list(itertools.product(resource_states, range(11)))
        self.start_state = start_state
        self.trade_req = [4, 4, 4]
        self.updateTradeReq()

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

    def getStates(self,):
        """
        Return a list of all states in the MDP.
        Not generally possible for large MDPs.
        """
        return self.states

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        return self.start_state

    def getPossibleActions(self, state):
        """
        Return list of possible actions from 'state'.
        """
        if self.isTerminal(state):
            return []
        actions = ["Roll dice"]
        if np.all(state[:3] >= costs[SETTLEMENT,:]):
            actions.append("Buy settlement")
        if np.all(state[:3] >= costs[CARD,:]):
            actions.append("Buy card")
        if np.all(state[:3] >= costs[CITY,:]) and len(self.player.get_settlements()) > 0:
            actions.append("Buy city")
        if np.all(state[:3] >= costs[ROAD,:]):
            actions.append("Buy road")
        if state[0] >= self.trade_req[0]:
            actions.append("Trade wood for brick")
            actions.append("Trade wood for grain")
        if state[1] >= self.trade_req[1]:
            actions.append("Trade brick for wood")
            actions.append("Trade brick for grain")
        if state[2] >= self.trade_req[2]:
            actions.append("Trade grain for wood")
            actions.append("Trade grain for brick")
        return actions
        

    def getTransitionStatesAndProbs(self, state, action):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.

        Note that in Q-Learning and reinforcment
        learning in general, we do not know these
        probabilities nor do we directly model them.
        """
        nextState = np.array(state) 
        if action == "Roll dice":
            transitions = []
            roll_resources = self.player.board.get_resources(self.player.player_id)
            for roll in rollProb:
                nextState = np.zeros(4)
                nextState[:3] = np.add(state[:3], roll_resources[roll-2, :])
                nextState[3] = state[3]
                transitions.append((tuple(nextState), rollProb[roll]))
            return transitions
        elif action == "Buy settlement":
            nextState[:3] = np.subtract(state[:3], costs[SETTLEMENT, :])
            nextState[3] = state[3] + 1
        elif action == "Buy card":
            nextState[:3] = np.subtract(state[:3], costs[CARD, :])
            nextState[3] = state[3] + 1
        elif action == "Buy city":
            nextState[:3] = np.subtract(state[:3], costs[CITY, :])
            nextState[3] = state[3] + 1
        elif action == "Buy road":
            nextState[:3] = np.subtract(state[:3], costs[ROAD, :])
            nextState[3] = state[3]
        elif action == "Trade wood for brick":
            nextState[0] -= self.trade_req[0]
            nextState[1] += 1
        elif action == "Trade wood for grain":
            nextState[0] -= self.trade_req[0]
            nextState[2] += 1
        elif action == "Trade brick for wood":
            nextState[1] -= self.trade_req[1]
            nextState[0] += 1
        elif action == "Trade brick for grain":
            nextState[1] -= self.trade_req[1]
            nextState[2] += 1
        elif action == "Trade grain for wood":
            nextState[2] -= self.trade_req[2]
            nextState[0] += 1
        elif action == "Trade grain for brick":
            nextState[2] -= self.trade_req[2]
            nextState[1] += 1
        else:
            raise ValueError("Invalid action")
        nextState = tuple(nextState)
        return [(nextState, 1.0)]

    def getReward(self, state, action, nextState):
        """
        Get the reward for the state, action, nextState transition.

        Not available in reinforcement learning.
        """
        return nextState[3] - state[3]
        

    def isTerminal(self, state):
        """
        Returns true if the current state is a terminal state.  By convention,
        a terminal state has zero future rewards.  Sometimes the terminal state(s)
        may have no possible actions.  It is also common to think of the terminal
        state as having a self-loop action 'pass' with zero reward; the formulations
        are equivalent.
        """
        return state[3] == 10

class ValueIteration:
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, mdp, discount = 1.0, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = defaultdict(float) 
        start = time.time()
        self.runValueIteration()
        print("Value Iteration on", iterations, "iterations took", time.time()-start, "seconds.")

    def runValueIteration(self):
        # Write value iteration code here
        for i in range(self.iterations):
            oldVals = self.values.copy()
            for state in self.mdp.getStates():    
                all_q = []
                actions = self.mdp.getPossibleActions(state)
                if(len(actions) == 0):
                    self.values[state] = 0.0
                    continue
                for a in actions:
                    transition = self.mdp.getTransitionStatesAndProbs(state, a)
                    q = 0
                    for nextState, prob in transition:
                        q += prob*(self.mdp.getReward(state, a, nextState) + self.discount*oldVals[nextState])
                    all_q.append(q)
                self.values[state] = max(all_q)


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        total = 0
        for nextState, prob in transition:
            total += prob*(self.mdp.getReward(state, action, nextState) + self.discount*self.values[nextState])
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return 'exit'
        actions = self.mdp.getPossibleActions(state)
        vals = {}
        for a in actions:
            vals[a] = self.computeQValueFromValues(state, a)
        return max(vals.keys(), key=lambda k: vals[k])

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)



def action(self):
    mdp = MDP(self)
    valueIter = ValueIteration(mdp, discount=1.0, iterations=10)
    print(max(valueIter.values.values()))
    state = np.concatenate((self.resources, [self.points]))
    state = tuple(state)
    print(state)
    best = valueIter.getAction(state)
    print(best)
    print(self.board.settlements)
    if best == "Buy settlement":
        x = np.random.randint(0, self.board.width+1)
        y = np.random.randint(0, self.board.height+1)
        self.buy("settlement", x, y)
    elif best == "Buy card":
        self.buy("card")
    elif best == "Buy road":
        x, y = random.choice(self.get_settlements())
        x += np.random.randint(2)
        y += np.random.randint(2)
        self.buy("road", )
    elif best == "Buy city":
        x, y = random.choice(self.get_settlements())
        self.buy("city", x, y)
    elif best == "Trade wood for brick":
        self.trade(0, 1)
    elif best == "Trade wood for grain":
        self.trade(0, 2)
    elif best == "Trade brick for wood":
        self.trade(1, 0)
    elif best == "Trade brick for grain":
        self.trade(1, 2)
    elif best == "Trade grain for wood":
        self.trade(2, 0)
    elif best == "Trade grain for brick":
        self.grade(2, 1)
    elif self.resources[np.argmax(self.resources)] >= 4:
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
