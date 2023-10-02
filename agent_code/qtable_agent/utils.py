import numpy as np
import settings
from items import Bomb
from collections import deque
import copy

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = ['RIGHT_UP', 'RIGHT', 'RIGHT_DOWN', 'DOWN', 'LEFT_DOWN', 'LEFT', 'LEFT_UP', 'UP', 'NO_ITEM']
REDUCED_ACTIONS = ['UP', 'WAIT', 'BOMB']

POSITION_MAPPING = ['FREE', 'BREAKABLE', 'OBSTACLE', 'DANGER_ZONE']
BOMB_MAPPING = ['NOTHING', 'BOMB_DROPPABLE', 'SHOULD_ESCAPE']
FEATURE_SHAPE = [4, 4, 4, 4, 9, 3]


def game_state_to_features(self, game_state: dict) -> np.ndarray:
    """
    Extracts features from the game state. The idea is to check the surrounding area of the player given by the
    feature array size. The corresponding output is a grid of features for each field in the surrounding area.

    Parameters:
        game_state (dict): A dictionary containing game state information.

    Returns:
        np.ndarray: A 1D NumPy array representing the extracted features with shape (,FEATURE_ARRAY_SIZE^2)
    """

    field = game_state['field']
    _, _, bomb_possible, (self_x, self_y) = game_state['self']  # Agent's current position
    others = game_state['others']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    coins = game_state['coins']

    self.logger.debug(f"Current position: {(self_x, self_y)}")
    feature_array = []
    feature_array.extend(get_position_feature(self, bombs, explosion_map, field, others, self_x, self_y))

    # Get direction to closest , if no coin visible check for crates and afterwards for oppenents
    closest_item_feature = get_closest_coin_feature(self_x, self_y, field, coins)

    # check for crates
    if closest_item_feature == DIRECTIONS.index("NO_ITEM"):
        temp_field_remove_crates, crates = get_crates_field(field)
        closest_item_feature = get_closest_coin_feature(self_x, self_y, temp_field_remove_crates, crates)
        # check for opponents
        if closest_item_feature == DIRECTIONS.index("NO_ITEM"):
            closest_item_feature = get_closest_coin_feature(self_x, self_y, field, [coords for _, _, _,
                                                                                               coords in others])

    feature_array.append(closest_item_feature)

    # Should drop bomb feature, especially check whether a escape option is possible
    bomb_feature = get_should_drop_bomb(bombs, bomb_possible, explosion_map, field, others, self_x, self_y)
    feature_array.append(bomb_feature)
    return tuple(feature_array)


def get_crates_field(field):
    temp_field_remove_crates = np.copy(field)
    crates = []
    for x in range(temp_field_remove_crates.shape[1]):
        for y in range(temp_field_remove_crates.shape[0]):
            if temp_field_remove_crates[x, y] == 1:
                temp_field_remove_crates[x, y] = 0
                crates.append((x, y))

    return temp_field_remove_crates, crates


def get_should_drop_bomb(bombs, bomb_possible, explosion_map, field, others, self_x, self_y):
    """
    Check whether it might be sensful to drop a bomb: Focus on having an impact
    Args:
        bombs:
        bomb_possible:
        explosion_map:
        field:
        others:
        self_x:
        self_y:

    Returns:

    """
    bomb_blasts = get_bomb_blasts(bombs, field)

    if (self_x, self_y) in [coords for coords, _, _, _ in bomb_blasts]:
        return 2
    if not bomb_possible:
        return 0
    else:
        others_coords = [coords for _, _, _, coords in others]
        trapped_opponents, trapped_opponents_predicted = get_trapped_opponents(self_x, self_y, others,
                                                                               bomb_blasts, field)
        bomb = Bomb((self_x, self_y), "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER, "lorem ipsum")
        blast_coords = [(coords, 0, (self_x, self_y), settings.BOMB_TIMER) for coords in bomb.get_blast_coords(field)]
        bomb_blasts_with_possible_bomb = copy.deepcopy(bomb_blasts)
        bomb_blasts_with_possible_bomb.extend(blast_coords)

        escape_routes = get_escape_routes(self_x, self_y, field, bomb_blasts_with_possible_bomb)
        check_for_impact = get_impact_of_possible_bomb(self_x, self_y, field, bomb_blasts, others_coords)

        should_drop_bomb = len(escape_routes) > 0 and (len(check_for_impact) > 0 or len(trapped_opponents) > 0)
        return 1 if should_drop_bomb else 0

def get_trapped_opponents(self_x, self_y, others, bomb_blasts, field):
    others_coords = [coords for _, _, _, coords in others]
    trapped_opponents = []
    trapped_opponents_predicted = []

    bomb = Bomb((self_x, self_y), "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER, "lorem ipsum")
    blast_coords = [(coords, 0, (self_x, self_y), settings.BOMB_TIMER) for coords in bomb.get_blast_coords(field)]

    bomb_blasts_with_possible_bomb = copy.deepcopy(bomb_blasts)
    bomb_blasts_with_possible_bomb.extend(blast_coords)

    for other_coord in others_coords:
        # idea: consider the case when all others than the selected one chooses to drop a bomb, whether this would
        # yield in a trapped opponent
        bomb_blasts_with_possible_bomb_all_others = copy.deepcopy(bomb_blasts_with_possible_bomb)
        for others_coords_not_considered in [coords for _, _, can_drop_bomb, coords in others if coords !=
                                                                                                 other_coord and can_drop_bomb]:
            bomb_others = Bomb(others_coords_not_considered, "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER,
                               "lorem ipsum")
            blast_coords = [(coords, 0, others_coords_not_considered, settings.BOMB_TIMER) for coords in
                            bomb_others.get_blast_coords(field)]
            bomb_blasts_with_possible_bomb_all_others.extend(blast_coords)

        escape_routes_without_possible_bomb = get_escape_routes(other_coord[0], other_coord[1], field, bomb_blasts)
        escape_routes_with_all_possible_bombs = get_escape_routes(other_coord[0], other_coord[1], field,
                                                                  bomb_blasts_with_possible_bomb_all_others)
        escape_routes_with_possible_bomb = get_escape_routes(other_coord[0], other_coord[1], field,
                                                             bomb_blasts_with_possible_bomb)

        if len(escape_routes_with_possible_bomb) == 0 and len(escape_routes_without_possible_bomb) > 0:
            trapped_opponents.append(other_coord)
        if len(escape_routes_without_possible_bomb) > 0 and len(escape_routes_with_all_possible_bombs) == 0:
            trapped_opponents_predicted.append(other_coord)

    return trapped_opponents, trapped_opponents_predicted


def get_impact_of_possible_bomb(self_x, self_y, field, bomb_blasts, others_coords):
    bomb = Bomb((self_x, self_y), "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER, "lorem ipsum")
    blast_coords = bomb.get_blast_coords(field)
    impact_fields = [(coords, get_manhatten_distance(coords, (self_x, self_y))) for coords in blast_coords if field[
        coords] == 1 or coords in others_coords]
    return impact_fields


def get_symmetrical_states(state):
    symmetrical_states = []
    for i in range(1, 4):
        temp_state = get_symmetrical_state_single(state, i)
        symmetrical_states.append(temp_state)
    return symmetrical_states


def get_symmetrical_state_single(state, index):
    position_state = shift_right(list(state[:4]), index)
    direction_state = (state[4] + index * 2) % 8 if state[4] != 8 else 8
    should_drop_bomb_state = state[5]
    return tuple(position_state + [direction_state, should_drop_bomb_state])


def get_escape_routes(x, y, field, bomb_blasts, step_limit=5):
    """
    Finds escape routes from the given position (x, y) to empty fields with a maximum step limit.

    Args:
        x: Starting x-coordinate.
        y: Starting y-coordinate.
        field: The game field.
        step_limit: Maximum number of steps for an escape route (default is 4).

    Returns:
        List of escape routes, where each route is a list of coordinates.
    """

    bomb_blast_coords = [coords for coords, _, _, _ in bomb_blasts]

    def is_valid_position(pos):
        return (
                0 <= pos[0] < field.shape[0] and
                0 <= pos[1] < field.shape[1] and
                field[pos] == 0
        )

    # Define possible moves (up, down, left, right)
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Create a visited matrix to keep track of visited tiles
    visited = np.zeros_like(field)

    # Initialize a queue for BFS
    queue = deque()
    queue.append([(x, y)])

    escape_routes = []

    while queue:
        current_route = queue.popleft()
        current_pos = current_route[-1]

        if len(current_route) > step_limit:
            continue  # Skip routes that exceed the step limit

        # Check adjacent tiles
        for move in moves:
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])

            if is_valid_position(new_pos):
                if new_pos not in current_route:
                    new_route = current_route + [new_pos]
                    if field[new_pos] == 0 and new_pos not in bomb_blast_coords:
                        escape_routes.append(new_route)
                    else:
                        queue.append(new_route)

    return escape_routes


def get_closest_item_bfs(x, y, items, field):
    """
    Uses breath first search

    Args:
        x:
        y:
        items:

    Returns:

    """

    # Define possible moves (up, down, left, right)
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # Create a visited matrix to keep track of visited tiles
    visited = np.zeros_like(field)

    # Initialize a queue for BFS
    queue = deque()
    queue.append(((x, y), 0))

    while queue:
        current_pos = queue.popleft()

        # Check if the current position is one of the coin positions
        if current_pos[0] in items:
            return current_pos

        # Check adjacent tiles
        for move in moves:
            new_pos = (current_pos[0][0] + move[0], current_pos[0][1] + move[1])

            # Check if the new position is within bounds and is a free tile
            if (
                    0 <= new_pos[0] < field.shape[0] and
                    0 <= new_pos[1] < field.shape[1] and
                    field[new_pos] == 0 and
                    not visited[new_pos]
            ):
                visited[new_pos] = 1
                queue.append((new_pos, current_pos[1] + 1))

    # If no path is found, return False
    return None, -1


def get_manhatten_distance(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])


def get_closest_coin_feature(self_x, self_y, field, coins):
    """

    Args:
        self_x:
        self_y:
        field:
        coins:

    Returns:

    """

    # Get closest coin direction
    coins_filtered = [coin for coin in coins if coin != (self_x, self_y)]
    closest_coin, _ = get_closest_item_bfs(self_x, self_y, coins_filtered, field)

    # Also consider second closest coin
    coins_filtered = [coin for coin in coins if coin != closest_coin]
    second_closest_coin, _ = get_closest_item_bfs(self_x, self_y, coins_filtered, field)

    # if no coin is existent treat separately
    if closest_coin == None:
        return 8

    if not second_closest_coin is None:
        coin_distance = get_manhatten_distance(closest_coin, second_closest_coin)
        player_coin_distance = get_manhatten_distance(closest_coin, (self_x, self_y))
        if (coin_distance < player_coin_distance):
            mean_x = (closest_coin[0] + second_closest_coin[0]) / 2
            mean_y = (closest_coin[1] + second_closest_coin[1]) / 2
            closest_coin = (mean_x, mean_y)

    dx = closest_coin[0] - self_x
    dy = closest_coin[1] - self_y

    # Orientate on the position of ACTIONS array
    if dx > 0 and dy < 0:  # RIGHT, UP
        return DIRECTIONS.index('RIGHT_UP')
    elif dx > 0 and dy == 0:  # RIGHT
        return DIRECTIONS.index('RIGHT')
    elif dx > 0 and dy > 0:  # RIGHT, DOWN
        return DIRECTIONS.index('RIGHT_DOWN')
    elif dx == 0 and dy > 0:  # DOWN
        return DIRECTIONS.index('DOWN')
    elif dx < 0 and dy > 0:  # LEFT, DOWN
        return DIRECTIONS.index('LEFT_DOWN')
    elif dx < 0 and dy == 0:  # LEFT
        return DIRECTIONS.index('LEFT')
    elif dx < 0 and dy < 0:  # LEFT, UP
        return DIRECTIONS.index('LEFT_UP')
    elif dx == 0 and dy < 0:  # UP
        return DIRECTIONS.index('UP')
    else:
        return DIRECTIONS.index('NO_ITEM')


def get_position_feature(self, bombs, explosion_map, field, others, self_x, self_y) -> int:
    """

    Mapping:
    - 0: Safe zone
    - 1: Breakable
    - 2: Danger zone

    Args:
        bombs:
        explosion_map:
        field:
        i:
        others:
        x:
        y:

    Returns:

    """

    # Check direct environment to avoid obstacles

    position_features = [0] * 4
    position_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

    blast_fields = []
    bomb_blasts = get_bomb_blasts(bombs, field)
    bomb_blasts_coords = [coords for coords, _, _, _ in bomb_blasts]
    count_bomb_blast_fields = 0
    for i, (x_offset, y_offset) in enumerate(position_offsets):
        x = self_x + x_offset
        y = self_y + y_offset

        # Check if position extends game field
        if x >= settings.COLS or y >= settings.ROWS:
            position_features[i] = 2
        # Check for stones
        elif field[x, y] == -1:
            position_features[i] = 2
        # Check for danger zones for blasted bombs
        elif explosion_map[x, y] > 0:
            position_features[i] = 3
        # Check for crates
        elif field[x, y] == 1:
            position_features[i] = 1
        # Check for opponent
        elif any((x, y) == (ox, oy) for _, _, _, (ox, oy) in others):
            position_features[i] = 1
        # Check danger zones for incoming bombs
        if (x, y) in bomb_blasts_coords and position_features[i] == 0:
            count_bomb_blast_fields += 1
            blast_fields.append((x_offset, y_offset))
            position_features[i] = 3

    count_empty_fields = position_features.count(0)

    if count_bomb_blast_fields >= 1 and count_empty_fields == 0:
        bomb_blasts_with_possible_bombs = copy.deepcopy(bomb_blasts)

        for coords in [coords for _, _, bomb_possible, coords in others if bomb_possible]:
            bomb = Bomb(coords, "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER, "lorem ipsum")
            blast_coords = [(coords, 0, coords, settings.BOMB_TIMER) for coords in bomb.get_blast_coords(field)]
            bomb_blasts_with_possible_bombs.extend(blast_coords)

        escape_routes_with_possible_bombs = get_escape_routes(self_x, self_y, field, bomb_blasts_with_possible_bombs)
        escape_routes = get_escape_routes(self_x, self_y, field, bomb_blasts)

        for escape_routes_temp, bomb_blasts_temp in zip([escape_routes_with_possible_bombs, escape_routes],
                                                        [bomb_blasts_with_possible_bombs, bomb_blasts]):

            escape_routes_first_items = [route[1] for route in escape_routes_temp]

            min_steps_to_escape = np.inf
            min_steps_index = -1
            for (x_offset, y_offset) in blast_fields:
                x = self_x + x_offset
                y = self_y + y_offset

                if ((x, y) in escape_routes_first_items):
                    temp_index = escape_routes_first_items.index((x, y))
                    steps_to_escape = len(escape_routes_temp[temp_index]) - 1

                    initial_bombs = [(bomb_coords, countdown) for blast_coords, _, bomb_coords, countdown in bomb_blasts
                                     if blast_coords == (x, y)]
                    min_bomb_countdown = min(initial_bombs, key=lambda x: x[1])

                    if (steps_to_escape < min_steps_to_escape and steps_to_escape <= min_bomb_countdown[1] + 1):
                        min_steps_to_escape = steps_to_escape
                        min_steps_index = position_offsets.index((x_offset, y_offset))

            if (min_steps_index != -1):
                position_features[min_steps_index] = 0
                self.logger.debug(f"Blast coords: {bomb_blasts}")
                self.logger.debug(f"Escape routes: {escape_routes}")
                self.logger.debug(f"Initial bomb: {initial_bombs}")
                self.logger.debug(f"Steps to Escape: {steps_to_escape}")
                break

    # check whether a potential trap could occur if the others drop a bomb
    # if count_empty_fields > 1:
    #    possible_traps = []
    #    for i, (x_offset, y_offset) in enumerate(position_offsets):
    #       if position_features[i] == 0:
    #          x = self_x + x_offset
    #         y = self_y + y_offset
    #        is_possible_trap, _ = get_possible_trap((x,y),others, bomb_blasts, field)
    #       if is_possible_trap:
    #          possible_traps.append((is_possible_trap, i))
    # if len(possible_traps) < count_empty_fields and len(possible_traps) > 0:
    #    for trap, index in possible_traps:
    #        self.logger.debug(f"Avoid possible traps: {possible_traps}")
    #        position_features[index] == 3

    return position_features


def get_possible_trap(self_coords, others, bomb_blasts, field):
    others_coords = [coords for _, _, _, coords in others]
    bomb_blasts_with_possible_bombs = copy.deepcopy(bomb_blasts)

    for other_coord in [coords for _, _, can_drop_bomb, coords in others if can_drop_bomb]:
        bomb_others = Bomb(other_coord, "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER,
                           "lorem ipsum")
        blast_coords = [(coords, 0, other_coord, settings.BOMB_TIMER) for coords in
                        bomb_others.get_blast_coords(field)]
        bomb_blasts_with_possible_bombs.extend(blast_coords)

    escape_routes_without_possible_bombs = get_escape_routes(self_coords[0], self_coords[1], field, bomb_blasts)
    escape_routes_with_possible_bombs = get_escape_routes(self_coords[0], self_coords[1], field,
                                                          bomb_blasts_with_possible_bombs)

    if (len(escape_routes_with_possible_bombs) == 0):
        return (True, self_coords)
    else:
        return (False, self_coords)


def get_bomb_blasts(bombs, field):
    """

    Args:
        bombs: List of coordinates and countdowns for bombs
        field: Game board

    Returns:
        A list of tuples with the following structure (blast coordinates, danger level, bomb source, countdown for bomb)

    """

    blast_fields = []
    for coord, countdown in bombs:
        bomb = Bomb(coord, "lorem ipsum", countdown, settings.BOMB_POWER, "lorem ipsum")
        blast_coords = bomb.get_blast_coords(field)
        for (x, y) in blast_coords:
            blast_fields_temp = [field for field, _, _, _ in blast_fields]
            danger_level = get_manhatten_distance((x, y), coord)

            if ((x, y) in blast_fields_temp):
                index = blast_fields_temp.index((x, y))
                _, min_value, _, min_countdown = blast_fields[index]
                blast_fields[index] = ((x, y), max(min_value, danger_level), coord, min(countdown, min_countdown))
            else:
                blast_fields.append(((x, y), danger_level, coord, countdown))

    return list(set(blast_fields))


def get_closest_bomb(bombs_coordinates, player_coordinates):
    bombs_coordinates = [coordinates for (coordinates, _) in bombs_coordinates]
    closest_index = np.argmin(np.linalg.norm(np.array(bombs_coordinates) - np.array(player_coordinates), axis=1))
    return bombs_coordinates[closest_index]


def shift_right(arr, num_positions):
    num_positions %= len(arr)  # Ensure the shift amount is within the length of the array
    return arr[-num_positions:] + arr[:-num_positions]


def convert_symmetric_action(action, index):
    """
    Return a symmetric action

    Args:
        action:
        index:

    Returns: index of action

    """
    if action in [1, 2]:
        return action + 3
    else:
        return index % 4


def convert_to_base_action(action):
    """
    Method to convert a given action to its generic version of having a single state for all movements.

    Args:
        action: Index of action

    Returns:
        The index of the action itself and the number of turns which have to be made to get to the base action. For
        ACTIONS 'BOMB' and 'WAIT' there must be no changes.

    """
    if action in [4, 5]:
        return action - 3, 0
    else:
        return 0, (4 - action) % 4
