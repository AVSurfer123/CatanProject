import catan
import numpy as np

rollProb = {2: 1/36, 12: 1/36, 3: 1/18, 11: 1/18, 4: 1/12, 10: 1/12, 5: 1/9, 9: 1/9, 6: 5/36, 8: 5/36, 7: 1/6}

def planBoard(board):
    resource_scarcity = get_resource_scarcity(board)
    def opt_settlements(player, board):
        """
        Want to return a list of optimal settlements given the current state of the board.
        Should iterate through every viable vertex and pick the most optimal one relative to the player's
        current game state. Emphasis on diversifying resources/increasing odds for a resource.
        """
        optimal_vertex = -1
        max_score = 0
        for v in range(board.max_vertex+1):
            x,y = board.get_vertex_location(v)
            if board.if_can_build("settlement", x, y, player.player_id):
                vertex_score = vertex_eval(x,y,"settlement",board, resource_scarcity) #formula calculations go here
                if vertex_score > max_score:
                    print(v, vertex_score)
                    optimal_vertex = v
                    max_score = vertex_score
        return (optimal_vertex, board.get_vertex_location(optimal_vertex))

    def opt_cities(player, board):
        """ Same thing as opt_settlements but for cities. Emphasis on maximizing resource gain per turn. """
        optimal_vertex = -1
        max_score = 0
        for v in range(board.get_player_settlements(player.player_id)):
            x,y = board.get_vertex_location(v)
            if board.if_can_build("settlement", x, y, player.player_id):
                vertex_score = vertex_eval(x,y,"city",board, resource_scarcity) #formula calculations go here
                if vertex_score > max_score:
                    optimal_vertex = v
                    max_score = vertex_score
        return board.get_vertex_location(v)

    return [opt_settlements, opt_cities]

def vertex_eval(x, y, building, board, resource_scarcity):
    expectation = 0
    multiplier = 2 if building == "city" else 1
    for dx in [-1, 0]:
        for dy in [-1,0]:
            xx = x + dx
            yy = y + dy
            if board.is_tile(xx, yy):
                die = board.dice[yy, xx]
                if board.is_tile(xx, yy) and die != 7:
                    expectation += multiplier*rollProb.get(die, 0)/resource_scarcity[board.resources[xx,yy]]
    return expectation

def get_resource_scarcity(board):
    resource_count = np.zeros(3)
    for x in range(board.resources.shape[0]):
        for y in range(board.resources.shape[1]):
            if board.is_tile(x,y):
                r = board.resources[x,y]
                if r != -1:
                    resource_count[r] += 1;
    total_resources = board.width*board.height
    return resource_count/total_resources
