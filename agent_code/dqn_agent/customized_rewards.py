import numpy as np
import events as e
import settings
from items import Bomb
from collections import Counter, deque

WAIT_LIMIT = 3
POSITION_MAPPING = ['FREE', 'BREAKABLE', 'OBSTACLE', 'DANGER_ZONE']
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
DIRECTIONS = ['RIGHT_UP', 'RIGHT', 'RIGHT_DOWN', 'DOWN', 'LEFT_DOWN', 'LEFT', 'LEFT_UP', 'UP', 'NO_ITEM']

# Custom events
WAITED_TOO_LONG = "WAITED_TOO_LONG"
WAITED_TOO_LONG_JUMPING = "WAITED_TOO_LONG_JUMPING"
ESCAPING_BOMB_FULLY = "ESCAPING_BOMB_FULLY"
ESCAPING_BOMB_PARTLY = "ESCAPING_BOMB_PARTLY"
RUNNING_INTO_BOMB = "RUNNING_INTO_BOMB"
RUNNING_INTO_BOMB_PARTLY = "RUNNING_INTO_BOMB_PARTLY"
INVALID_MOVE_MANUAL = "INVALID_MOVE_MANUAL"
VALID_MOVE_MANUAL = "VALID_MOVE_MANUAL"
COIN_CLOSER = "COIN_CLOSER"
COIN_FURTHER = "COIN_FURTHER"
NOT_LEAVING_BLAST = "NOT_LEAVING_BLAST"
DROPPED_CORNER_BOMB = "DROPPED_CORNER_BOMB"
RUNNING_INTO_EXPLOSION = "RUNNING_INTO_EXPLOSION"


####################################
##### running_into_explosion #######
####################################

def running_into_explosion(old_game_state, events, last_action):
    field = old_game_state['field']
    player_coord_old = old_game_state['self'][-1]

    player_x_coord_old , player_y_coord_old = old_game_state['self'][-1]

    bombs = old_game_state['bombs']
    blast_coords = []
    blast_coords_with_timer = []
    for bomb_coord, t in bombs:
        bomb = Bomb(bomb_coord, "lorem ipsum", t, settings.BOMB_POWER, "lorem ipsum")
        blast_coords_bomb = bomb.get_blast_coords(field)
        blast_coords.append(blast_coords_bomb)
        blast_coords_and_timer = [blast_coords_bomb + [t]]
        blast_coords_with_timer.append(blast_coords_and_timer)
    
    blast_coords_with_timer = [sublist[0] for sublist in blast_coords_with_timer]

    def new_position(x, y, direction):
        # Define movements for each direction
        movements = {
            "RIGHT": (1, 0),
            "LEFT": (-1, 0),
            "UP": (0, -1),
            "DOWN": (0, 1)
        }

        dx, dy = movements.get(direction, (0, 0))  # Get the movement corresponding to the direction
        new_x, new_y = x + dx, y + dy  # Calculate the new position

        # Check if the new position has a value of 1 in the matrix
        return (new_x, new_y)

    player_coord_new = new_position(player_x_coord_old, player_y_coord_old, last_action)

    runs_into_explosion = False
    # Iterate through the array
    for entry in blast_coords_with_timer:
        # Check if the coordinates player_coord_new are in the entry
        if player_coord_new in entry[:-1]:  # Exclude the last element (timer)
            # Check if the timer is 0
            if entry[-1] == 0 or entry[-1] == 1:
                runs_into_explosion = True
   
    if (runs_into_explosion) and (player_coord_old not in blast_coords) and ("KILLED_SELF" in events or "GOT_KILLED" in events):
       events.append(RUNNING_INTO_EXPLOSION)

####################################
######### dropped_corner_bomb ######
####################################

def dropped_corner_bomb(old_game_state, events):
    player_coord = old_game_state['self'][-1]
    corner_coordinates = [(1, 1), (1, 15), (15, 1), (15, 15)]
    if ("BOMB_DROPPED" in events) and (player_coord in corner_coordinates):
        events.append(DROPPED_CORNER_BOMB)

####################################
######### stayed_in_blast ##########
####################################

def stayed_in_blast(old_game_state, new_game_state, events):
    field = old_game_state['field']
    bomb_coords = [coords for coords, _ in old_game_state['bombs']]
    blast_coords = []
    for bomb_coord in bomb_coords:
        bomb = Bomb(bomb_coord, "lorem ipsum", 2, settings.BOMB_POWER, "lorem ipsum")
        blast_coords_bomb = bomb.get_blast_coords(field)
        blast_coords.append(blast_coords_bomb)
        
    player_coord = new_game_state['self'][-1]
    if ("WAITED" in events or "INVALID_ACTION" in events) and any(player_coord in sublist for sublist in blast_coords):
        events.append(NOT_LEAVING_BLAST)

####################################
######### prevent_long_wait ########
####################################

def prevent_long_wait(self, events):
    """
    Method trying tro prevent the agent to wait for longer times. Adds the WAITED_TOO_LONG event, if the agent
    waited more than wait_limit steps.
    Args:
        self:
        events:

    Returns:

    """
    if e.WAITED in events:
        self.waited_times += 1
    else:
        self.waited_times = 0
    if self.waited_times > WAIT_LIMIT:
        events.append(WAITED_TOO_LONG)

    # also if distance didn't change too much punish agent (so prevent from jumping always into the back again always)
    count_positions = Counter(self.positions)
    count_greater_than_2 = 0

    for _, count in count_positions.items():
        if count >= 3:
            count_greater_than_2 += 1

    if count_greater_than_2 >= 2:
        events.append(WAITED_TOO_LONG_JUMPING)

####################################
######### escape_bomb ##############
####################################

def escape_bomb(self, old_game_state, new_game_state, events):
    old_bomb_blasts_coord = [coords for coords, _ in get_bomb_blasts(old_game_state['bombs'], old_game_state['field'])]
    new_bomb_blasts_coord = [coords for coords, _ in get_bomb_blasts(new_game_state['bombs'], new_game_state['field'])]

    old_bombs = [coords for coords, _ in old_game_state['bombs']]
    new_bombs = [coords for coords, _ in new_game_state['bombs']]

    player_coord_new = new_game_state['self'][-1]
    player_coord_old = old_game_state['self'][-1]

    if(
            player_coord_old in old_bomb_blasts_coord and
            player_coord_new not in new_bomb_blasts_coord
    ):
        events.append(ESCAPING_BOMB_FULLY)
    elif(
            player_coord_old not in old_bomb_blasts_coord and
            player_coord_new in new_bomb_blasts_coord and
            e.BOMB_DROPPED not in events
    ):
        events.append(RUNNING_INTO_BOMB)
    elif(player_coord_old in old_bomb_blasts_coord and player_coord_new in new_bomb_blasts_coord):
        closest_bomb_old, _ = get_closest_item_bfs(player_coord_old[0],
                                                   player_coord_old[1],
                                                   old_bombs, old_game_state['field'])
        closest_bomb_new, _ = get_closest_item_bfs(player_coord_new[0],
                                                player_coord_new[1],
                                                new_bombs, new_game_state['field'])
        if not closest_bomb_old == None and not closest_bomb_new == None:
            if get_manhatten_distance(closest_bomb_new, player_coord_new) > \
                    get_manhatten_distance(closest_bomb_old, player_coord_old):
                events.append(ESCAPING_BOMB_PARTLY)
            elif get_manhatten_distance(closest_bomb_new, player_coord_new) < \
                    get_manhatten_distance(closest_bomb_old, player_coord_old) and \
                    not e.BOMB_DROPPED in events:
                events.append(RUNNING_INTO_BOMB_PARTLY)


####################################
##### invalid_action_for_state #####
####################################

def invalid_action_for_state(self, position_feature, action, events):    
    if e.INVALID_ACTION not in events and action < 4 and (position_feature[action] == POSITION_MAPPING.index(\
            'DANGER_ZONE') or position_feature[action] == POSITION_MAPPING.index('BREAKABLE') or position_feature[action] ==
                                                          POSITION_MAPPING.index('OBSTACLE')):
        events.append(INVALID_MOVE_MANUAL)
    elif action < 4 and position_feature[action] == POSITION_MAPPING.index('FREE'):
        events.append(VALID_MOVE_MANUAL)

####################################
##### closer_to_coin ###############
####################################

def closer_to_coin(self, old_game_state, closest_item_feature, new_game_state, new_state_features, events, action):
    player_coord_new = new_game_state['self'][-1]
    player_coord_old = old_game_state['self'][-1]

    old_closest_coin_coord, _ = get_closest_item_bfs(player_coord_old[0], player_coord_old[1], old_game_state['coins'],
                                                  old_game_state['field'])
    new_closest_coin_coord, _ = get_closest_item_bfs(player_coord_new[0], player_coord_new[1], new_game_state['coins'],
                                                  new_game_state['field'])

    # Check whether the way to the closest coin was blocked (only for up, left, right, down). If so, don't punish
    # movement which steps away the player.
    coin_was_blocked = False
    if (closest_item_feature == DIRECTIONS.index('UP') and action == ACTIONS.index('UP')) or \
            (closest_item_feature == DIRECTIONS.index('RIGHT') and action == ACTIONS.index('RIGHT')) or \
            (closest_item_feature == DIRECTIONS.index('LEFT') and action == ACTIONS.index('LEFT')) or \
            (closest_item_feature == DIRECTIONS.index('DOWN') and action == ACTIONS.index('DOWN')):
        coin_was_blocked = True

    if new_closest_coin_coord is None:
        modified_field_old, crates_old = get_crates_field(old_game_state['field'])
        modified_field_new, crates_new = get_crates_field(new_game_state['field'])

        old_closest_coin_coord, _ = get_closest_item_bfs(player_coord_old[0], player_coord_old[1],
                                                         crates_old,
                                                         modified_field_old)
        new_closest_coin_coord, _ = get_closest_item_bfs(player_coord_new[0], player_coord_new[1],
                                                         crates_new,
                                                         modified_field_new)

    if old_closest_coin_coord is None or new_closest_coin_coord is None:
        return

    if(get_manhatten_distance(player_coord_old, old_closest_coin_coord)
       < get_manhatten_distance(player_coord_new, new_closest_coin_coord)) and \
            not e.COIN_COLLECTED in events and \
            not coin_was_blocked:
        events.append(COIN_FURTHER)
    elif(get_manhatten_distance(player_coord_old, old_closest_coin_coord)
         > get_manhatten_distance(player_coord_new, new_closest_coin_coord)):
        events.append(COIN_CLOSER)


####################################
##### get_position_feature #########
####################################

def get_position_feature(bombs, explosion_map, field, others, self_x, self_y) -> int:
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

    position_features = [0]*4
    position_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0)] # Up, Right, Down, Left

    blast_fields = []
    bomb_blasts = get_bomb_blasts(bombs, field)
    bomb_blasts_coords = [coords for coords, _ in bomb_blasts]
    escape_routes = get_escape_routes(self_x, self_y, field, bomb_blasts_coords)
    escape_routes_first_items = [route[0] for route in escape_routes]
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
        # Check danger zones for incoming bombs, don't give a value
        if (x,y) in bomb_blasts_coords and position_features[i] == 0:
            count_bomb_blast_fields += 1
            blast_fields.append((x_offset,y_offset))
            position_features[i] = 3

    count_empty_fields = position_features.count(0)


    if count_bomb_blast_fields >= 1 and count_empty_fields == 0:
        min_steps_to_escape = np.inf
        min_steps_index = -1
        for (x_offset, y_offset) in blast_fields:
            x = self_x + x_offset
            y = self_y + y_offset

            if((x,y) in escape_routes_first_items):
                temp_index = escape_routes_first_items.index((x,y))
                steps_to_escape = len(escape_routes[temp_index])
                if(steps_to_escape < min_steps_to_escape):
                    min_steps_to_escape = steps_to_escape
                    min_steps_index = position_offsets.index((x_offset,y_offset))

        if(min_steps_index != -1):
            position_features[min_steps_index] = 0

    return position_features


###############################
##### get_bomb_blasts #########
###############################

def get_bomb_blasts(bombs, field):
    """
    Finds all fields which will be affected by blasting bombs for the given list of bombs based on horizontal and
    vertical explosion impact.

    Args:
        bombs: List of bomb entries

    Returns:
        Distinct list of all affected fields

    """
    blast_fields = []
    for coord, countdown in bombs:
        bomb = Bomb(coord, "lorem ipsum", countdown, settings.BOMB_POWER, "lorem ipsum")
        blast_coords = bomb.get_blast_coords(field)
        for (x,y) in blast_coords:
            blast_fields_temp = [field for field, _ in blast_fields]
            danger_level = get_manhatten_distance((x, y), coord)

            if((x,y) in blast_fields_temp):
                index = blast_fields_temp.index((x,y))
                _, min_value = blast_fields[index]
                blast_fields[index] = ((x,y), max(min_value, danger_level))
            else:
                blast_fields.append(((x,y), danger_level))
    return list(set(blast_fields))

###############################
##### get_manhatten_distance ##
###############################

def get_manhatten_distance(x1, x2):
    return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

###############################
##### get_closest_item_bfs ####
###############################

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
    queue.append(((x,y), 0))

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
                queue.append((new_pos, current_pos[1]+1))

    # If no path is found, return False
    return None, -1

############################
##### get_escape_routes ####
############################

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
                    if field[new_pos] == 0 and new_pos not in bomb_blasts:
                        escape_routes.append(new_route)
                    else:
                        queue.append(new_route)

    for route in escape_routes:
        if route:
            route.pop(0)

    return escape_routes


###################################
##### get_closest_coin_feature ####
###################################

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
        if(coin_distance < player_coin_distance):
            mean_x = (closest_coin[0] + second_closest_coin[0]) / 2
            mean_y = (closest_coin[1] + second_closest_coin[1]) / 2
            closest_coin = (mean_x, mean_y)

    dx = closest_coin[0] - self_x
    dy = closest_coin[1] - self_y

    # Orientate on the position of ACTIONS array
    if dx > 0 and dy < 0: #RIGHT, UP
        return DIRECTIONS.index('RIGHT_UP')
    elif dx > 0 and dy == 0: #RIGHT
        return DIRECTIONS.index('RIGHT')
    elif dx > 0 and dy > 0: #RIGHT, DOWN
        return DIRECTIONS.index('RIGHT_DOWN')
    elif dx == 0 and dy > 0: #DOWN
        return DIRECTIONS.index('DOWN')
    elif dx < 0 and dy > 0: #LEFT, DOWN
        return DIRECTIONS.index('LEFT_DOWN')
    elif dx < 0 and dy == 0: #LEFT
        return DIRECTIONS.index('LEFT')
    elif dx < 0 and dy < 0: #LEFT, UP
        return DIRECTIONS.index('LEFT_UP')
    elif dx == 0 and dy < 0: #UP
        return DIRECTIONS.index('UP')
    else:
        return DIRECTIONS.index('NO_ITEM')

###########################
##### get_crates_field ####
###########################

def get_crates_field(field):
    temp_field_remove_crates = np.copy(field)
    crates = []
    for x in range(temp_field_remove_crates.shape[1]):
        for y in range(temp_field_remove_crates.shape[0]):
            if temp_field_remove_crates[x, y] == 1:
                temp_field_remove_crates[x, y] = 0
                crates.append((x, y))

    return temp_field_remove_crates, crates

###############################
##### get_should_drop_bomb ####
###############################

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
    bomb_blasts = [coords for coords, _ in get_bomb_blasts(bombs, field)]

    if (self_x, self_y) in bomb_blasts:
        return 2
    if not bomb_possible:
        return 0
    else:
        others_coords = [coords for _,_,_,coords in others]
        escape_routes = get_escape_routes(self_x, self_y, field, bomb_blasts)
        check_for_impact = get_impact_of_possible_bomb(self_x, self_y, field, bomb_blasts, others_coords)
        return 1 if len(escape_routes) > 0 and len(check_for_impact) > 0 else 0

######################################
##### get_impact_of_possible_bomb ####
######################################

def get_impact_of_possible_bomb(self_x, self_y, field, bomb_blasts, others_coords):
    bomb = Bomb((self_x, self_y), "lorem ipsum", settings.BOMB_TIMER, settings.BOMB_POWER, "lorem ipsum")
    blast_coords = bomb.get_blast_coords(field)
    impact_fields = [(coords, get_manhatten_distance(coords, (self_x, self_y))) for coords in blast_coords if field[
        coords] == 1 or coords in others_coords]
    return impact_fields