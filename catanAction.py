import numpy as np

# GLOBALS

# Current goal, Growth or Points
goal = "Growth"


def action(self):
    if self.board.settlements == []:
        (x,y) = self.preComp 
        self.buy("settlement", x, y)
    elif self.if_can_buy("card"):
        self.buy("card")
    elif self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    return

def dumpPolicy(self, max_resources):
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources

def planBoard(baseBoard):
    settlements = baseBoard.settlements
    x = genRand(0,baseBoard.width+1)
    y = genRand(0,baseBoard.height+1)
    optSettlementLoc = (x,y)
    return optSettlementLoc

def genRand(low,high):
    return np.random.randint(low, high)
