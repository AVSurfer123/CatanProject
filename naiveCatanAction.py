import numpy as np

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}

def action(self):
    self.optimal_settlement = planBoard(baseBoard)[0]
    self.closest = self.closest_settlement_to_optimal(self.optimal_settlement)
    if self.points == 9:
        if self.if_can_buy("card"):
            self.buy("card")

    if self.points > 7 and self.resources[0] > 2*costs[CARD][0] and self.resources[1] > 2*costs[CARD][1] and self.resources[2] > 2*costs[CARD][2]:
        self.buy("card")
    if self.board.if_can_build("settlement", optimal_settlement[0], optimal_settlement[1]):
        if self.if_can_buy("settlement"):
            self.board.build(optimal_settlement[0], optimal_settlement[1], "settlement")
    elif self.if_can_buy("road"):
        roads = self.get_roads()




    elif self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    return

def planBoard(baseBoard):
    x = genRand(0,baseBoard.width+1)
    y = genRand(0,baseBoard.height+1)
    optSettlementLoc = (x,y)
    return optSettlementLoc


def genRand(low,high):
    return np.random.randint(low, high)

def manhattan_distance(vertex_1, vertex_2, board):
    x1, y1 = board.get_vertex_location(vertex_1)
    x2, y2 = board.get_vertex_location(vertex_2)
    return abs(x1 - x2) + abs(y1 - y2)

def expected_resources_gain(board):
    resources = board.get_resources()
    expected_gain = [None, None, None]
    for i in range(11):
        for j in range(3):
            expected_gain[j] += rollProb[i + 2] * resources[i][j]

    return expected_gain

def closest_settlement_to_optimal(self):
    distances = []
    for settlement in self.get_settlements():
        distances.append(manhattan_distance(settlement, self.optimal_settlement))

    return settlement[distances.index(min(distances))]
