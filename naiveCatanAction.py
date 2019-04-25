import numpy as np
from catanPlanBoard import *
import catan
from catan import *

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}
settlement_threshold = 5

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
        distances.append(manhattan_distance(settlement, self.optimal_settlement, self.board))

    return settlement[distances.index(min(distances))]


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
    optimumSettlements = 3

    new_resources = np.copy(self.resources)

    # Checking what strategy to use.
    if settlementCount < optimumSettlements:
        # Optimizing for buying settlements.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if 2 * new_resources[1] > new_resources[0]:
                new_resources[1] -= 1
            elif 2 * new_resources[2] > new_resources[0]:
                new_resources[2] -= 1
            else:
                new_resources[0] -= 1
    elif cityCount < optimumSettlements:
        # Optimising for buying cities.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if new_resources[0] > 0:
                new_resources[0] -= 1
            elif new_resources[2] > new_resources[1]:
                new_resources[2] -= 1
            else:
                new_resources[1] -= 1
    else:
        # Optimizing for cards.
        while sum(new_resources) > catan.ROBBER_MAX_RESOURCES:
            if new_resources[2] > 2:
                new_resources[2] -= 1
            elif new_resources[1] > 2:
                new_resources[1] -= 1
            else:
                new_resources[0] -= 1
    return self.resources - new_resources
