import numpy as np

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}
settlement_threshold = 5

def action(self):
    if self.points == 9:
        if self.if_can_buy("card"):
            self.buy("card")
    total_resources = np.sum(self.resources)
    num_settlements = len(self.get_settlements)
    num_cities = len(self.get_cities())
    if num_settlements < settlement_threshold:
        self.optimal_settlement, op_settlement = opt_settlement(self, self.board, self.preComp)
        if self.board.if_can_build("settlement", opt_settlement[0], opt_settlement[1]) and num_settlements < settlement_threshold:
            if self.if_can_buy("settlement"):
                self.board.build(opt_settlement[0], opt_settlement[1], "settlement")
        elif self.if_can_buy("road"):
            self.to_build_road = opt_road(self, self.board, self.optimal_settlement)
            self.board.build(self.to_build_road[1], self.to_build_road[2])
    elif num_cities < settlement_threshold:
        self.optimal_city, op_city  = opt_city(self, self.board, self.preComp)
        if self.if_can_buy("city"):
            self.board.build(op_city[0], op_city[1], "city")
    elif self.points > 7 and self.resources[0] > 2*costs[CARD][0] and self.resources[1] > 2*costs[CARD][1] and self.resources[2] > 2*costs[CARD][2]:
        self.buy("card")
    elif self.resources[np.argmax(self.resources)] >= 4 and total_resources > ROBBER_MAX_RESOURCES:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    return

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
