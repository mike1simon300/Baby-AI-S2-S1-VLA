import random
from VLA2Systems.pyhop.hop import State, Goal, declare_operators, declare_methods, plan, get_operators, get_methods
from VLA2Systems.knowledge_base import KnowledgeBase


class RobotPlanner:
    def __init__(self, knowledge_base: KnowledgeBase, start_location, holding=None, verbose=0):
        self.kb = knowledge_base
        self.verbose = verbose
        self.state = self.create_state(start_location, holding)
        self.declare()

    def create_state(self, start_location, holding):
        state = State('robot_state')
        state.holding = holding
        state.room_objects = {}
        state.doors = {}
        # Populate room objects and doors from the knowledge base
        
        for room, objects in self.kb.KB['rooms'].items():
            state.room_objects[room] = []
            for obj in objects:
                if obj['type'] == 'door':
                    state.room_objects[room].append(
                        [obj['type'], obj['color'], 
                         obj['position'], obj['state']])
                else:
                    state.room_objects[room].append(
                        (obj['type'], obj['color'], obj['position']))

        for r1, r2, door in self.kb.KB['connections']:
            state.doors[(r1, r2)] = [door['color'], door['position'], door['state']]
            # state.doors[(r2, r1)] = [door['color'], door['position'], door['state']]

        # Set the robot's starting location
        if start_location:
            state.robot_location = start_location

        return state

    def current_room(self, robot_location):
        _, pos = robot_location
        room_id = self.kb.room_map[pos]
        return room_id

    def plan_go_to(self, obj_color, obj_type="", obj_location=None):
        tasks = [('go_to_object', obj_type, obj_color, obj_location)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        return self.plan

    def plan_pick_up(self, obj_color, obj_type):
        """
        Plan to go to an object and pick it up.
        """
        tasks = [('pick_up_object', obj_type, obj_color)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        return self.plan

    def declare(self):
        declare_operators(self.go_to, self.pick_up, self.open)
        declare_methods('go_to_object', self.go_to_object)
        declare_methods('pick_up_object', self.pick_up_object)
        # print("Operators: ", get_operators())
        # print("Methods: ", get_methods())
    # Operator
    def pick_up(self, state, obj_type, obj_color):
        # Case: Robot already holding an object
        if state.holding is not None:
            return False
        # # ! Case: Robot Can move in this command (I will think about it later)
        # current_room = self.current_room(state.robot_location)
        # for obj in state.room_objects[current_room]:
        #     # If object is in the room.
        #     if obj[0] == obj_type and obj[1] == obj_color:
        #         state.holding = (obj_type, obj_color)
        #         state.room_objects[current_room].remove(obj)
        #         return state
        
        # Case: operator only pick up the object if the robot is around the object.
        current_room = self.current_room(state.robot_location)
        robot_rot, _ = state.robot_location
        surroundings = get_robot_surroundings(state.robot_location,
                                            self.kb.grid_data,
                                            state.room_objects[current_room])
        # No objects nearby the robot
        if len(surroundings) == 0:
            return False
        for rot, obj in surroundings:
            # Check if one of the nearby objects is the object we are searching for,
            # and the robot is now facing it. 
            if obj_type == obj[0] and obj_color == obj[1] \
                and rot == robot_rot:
                state.holding = (obj_type, obj_color)
                state.room_objects[current_room].remove(obj)
                return state
        # The we don't have the current object or rotation
        return False
    # Operator
    def go_to(self, state, obj_type, obj_color="", obj_location=None, current_room=None):
        if current_room is None:
            current_room = self.current_room(state.robot_location)
        for obj in state.room_objects[current_room]:
            if obj_color != "" and obj[1] != obj_color:
                continue
            # print("ENTERED GOTO, LOCATION IS: ", obj_location, " and obj location is: ", obj[2])
            if obj_location and obj[2] != obj_location:
                continue
            if obj[0] != obj_type:
                continue
            possible_robot_locations = \
                get_object_empty_nearbys(obj[2], self.kb.grid_data, 
                                            state.room_objects[current_room], 
                                            current_room, self.kb.room_map)
            if len(possible_robot_locations) == 0:
                continue
            # There is at least one solution:
            # Get a random choice of a solution
            solution = random.choice(possible_robot_locations)
            # # Get the first choice
            # solution = possible_robot_locations[0]
            # # Check if a path to solution is valid
            # Not Implemented yet (Path plan to possible_robot_location)
            state.robot_location = solution
            return state
        # print("DID NOT FIND ANY OBJECT")
        return False
    # Operator
    def open(self, state, obj_type, obj_color):
        current_room = self.current_room(state.robot_location)
        # print("ROBOT LOCATION INSIDE OPEN: ", state.robot_location)
        surroundings = get_robot_surroundings(state.robot_location, 
                               self.kb.grid_data, 
                               state.room_objects[current_room])
        for _, obj in surroundings:
            if obj[0] == 'door':
                # Look for the door that the robot is standing in front of.
                if obj[3] == 'closed':
                    index, _ = get_object_in_pos(obj[2], 
                                                 state.room_objects[current_room])
                    state.room_objects[current_room][index][3] = 'open'
                    door_location = state.room_objects[current_room][index][2]
                    key, _ = get_door_in_pos(door_location, state)
                    state.doors[key][2] = 'open'
                    # print("Going in lock and opening state: ", state)
                    return state
                elif obj[3] == 'locked' and state.holding is not None \
                    and state.holding[0] == 'key' and state.holding[1] == obj_color:
                    index, _ = get_object_in_pos(obj[2], 
                                                 state.room_objects[current_room])
                    state.room_objects[current_room][index][3] = 'open'
                    door_location = state.room_objects[current_room][index][2]
                    key, _ = get_door_in_pos(door_location, state)
                    state.doors[key][2] = 'open'
                    # print("Going in lock and opening state: ", state)
                    return state
        # print("Variables: ", current_room, surroundings, state.robot_location)
        # print("Not going inside and returning False")
        return False


    # Helper
    def unblock_object(self, state, obj_type, obj_color):
        current_room = self.current_room(state.robot_location)
        room_objects = state.room_objects[current_room]
        objects_in_room = get_objects(obj_type, obj_color, room_objects)
        for target_obj in objects_in_room:
            if is_blocked_by_object(target_obj[2], self.kb.grid_data, room_objects,
                                    current_room, self.kb.room_map):
                nearbys = get_robot_surroundings((0, target_obj[2]), self.kb.grid_data, 
                                    room_objects)
                for _, other_object in nearbys:
                    subplan = plan(state, [('pick_up_object', other_object[0], other_object[1])],
                       get_operators(), get_methods(), verbose=self.verbose)
                    if subplan:
                        break
                if subplan:
                    return subplan + [('go_to', obj_type, obj_color)]
        return False
    # Helper
    def object_not_in_room(self, state, obj_type, obj_color, obj_location=None):
        current_room = self.current_room(state.robot_location)
        room_objects = state.room_objects[current_room]
        objects_in_room = get_objects(obj_type, obj_color, room_objects)
        # If no instance of the object is found in this room, we might consider exploring other rooms.
        if obj_location is None and len(objects_in_room) == 0:
            # For now, return failure (or implement exploration)
            return True
        for obj in objects_in_room:
            if obj[2] == obj_location:
                return False
        return True

    def search_rooms(self, state, obj_type, obj_color, object_location):
        current_room = self.current_room(state.robot_location)
        # Start the recursive search with an empty visited set.
        return self.search_rooms_for_object(state, current_room, obj_type, obj_color, 
                                            object_location, visited=set())

    def search_rooms_for_object(self, state, current_room, obj_type, obj_color, object_location, visited):
        # Mark the current room as visited.
        visited.add(current_room)
        
        # Iterate over all doors in the state.
        for (r1, r2), (door_color, position, door_state) in state.doors.items():
            # Check if current_room is one end of the door.
            if current_room == r1:
                neighbor = r2
            elif current_room == r2:
                neighbor = r1
            else:
                continue

            if neighbor in visited:
                continue

            # Prepare the tasks to handle the door based on its state.
            if door_state == 'open':
                _, robot_position = state.robot_location
                dist = dist_manhatten(robot_position, position)
                # print("MANHATEN DISTANCE", dist)
                # print("STATE IS: ", state.robot_location)
                # print("STATE IS: ", state.doors)
                result = is_infront_of_door(self, state, position)
                # print("is_infront_of_door: ", result)
                if dist > 1:
                    # print(f"ROBOT IS IN {robot_position} not infront of open door, going to go to door {door_color} {position} {door_state}")
                    door_tasks = [('go_to_object', 'door', door_color, position)]
                else:
                    # print(f"ROBOT IS IN {robot_position} infront of open door, do nothing")
                    door_tasks = []
                    # print("AFTER SETTING door_tasks ", door_tasks)
            elif door_state == 'closed':
                _, robot_position = state.robot_location
                dist = dist_manhatten(robot_position, position)
                result = is_infront_of_door(self, state, position)
                # print("is_infront_of_door: ", result)
                if dist > 1:
                    # print(f"ROBOT IS IN {robot_position} not infront of closed door, going to go to door {door_color} {position} {door_state} then open it")
                    door_tasks = [('go_to_object', 'door', door_color, position), ('open', 'door', door_color)]
                else:
                    # print(f"ROBOT IS IN {robot_position} infront of closed door, open door {door_color} {position} {door_state}")
                    door_tasks = [('open', 'door', door_color)]
            elif door_state == 'locked':
                # For locked doors, we pick up the key first.
                door_tasks = [('pick_up_object', 'key', door_color), 
                            ('go_to_object', 'door', door_color, position), 
                            ('open', 'door', door_color)]
            else:
                # If the door state is unknown, skip it.
                continue
            
            # print("AFTER DOOR CONDITIONS door_tasks ", door_tasks)
            if object_location:
                # print(f"CHECK IF THE OBJECT WHICH HAS A LOCATION IS IN A ROOM, object {obj_type} {obj_color} {object_location}")
                # Check if the target object exists in the neighbor room.
                if any(o[0] == obj_type and o[1] == obj_color and o[2] == object_location for o in state.room_objects[neighbor]):
                    # print(f"THE TARGET OBJECT {obj_type} {obj_color} has been found in the neighbor room, returning: {door_tasks + [('go_to_object', obj_type, obj_color, object_location, neighbor)]}")
                    # Return the plan: handle door then go to the object in neighbor.
                    return door_tasks + [('go_to_object', obj_type, obj_color, object_location, neighbor)]
            else:
                # print(f"CHECK IF THE OBJECT WHICH dont have A LOCATION IS IN A ROOM, object {obj_type} {obj_color}")
                # Check if the target object exists in the neighbor room.
                if any(o[0] == obj_type and o[1] == obj_color for o in state.room_objects[neighbor]):
                    # print(f"THE TARGET OBJECT {obj_type} {obj_color} has been found in the neighbor room, returning: {door_tasks + [('go_to_object', obj_type, obj_color, object_location, neighbor)]}")
                    # Return the plan: handle door then go to the object in neighbor.
                    return door_tasks + [('go_to_object', obj_type, obj_color, object_location, neighbor)]

            # print(f"THE OBJECT WAS NOT FOUND IN THE neighbor room, doing recursion. current_room {current_room} neighbor room {neighbor}")
            # Otherwise, try to find the object recursively in the neighbor room.
            sub_tasks = self.search_rooms_for_object(state, neighbor, obj_type, obj_color, 
                                                     object_location, visited)
            # print("RECURSION RETURNED THIS SUBTASKS: ", sub_tasks)
            if sub_tasks:
                # If a plan was found deeper in the graph, prepend the door tasks.
                return door_tasks + sub_tasks

        # If no connected room (or chain of rooms) contains the object, return False.
        return False

    # Method to plan how to reach an object
    def go_to_object(self, state, obj_type, obj_color="", obj_location=None, room=None):
        if self.go_to(state, obj_type, obj_color, obj_location, room):
            return [('go_to', obj_type, obj_color, obj_location, room)]
        # print("FAILED DIRECTLY GOTO, gonna try with searching for the object")
        if self.object_not_in_room(state, obj_type, obj_color, obj_location):
            # print("GOING IN search_rooms")
            plan = self.search_rooms(state, obj_type, obj_color, obj_location)
            # print(plan)
            if plan:
                # print(f"going to return plan {plan}")
                return plan
        # print("FAILED searching for the object")
        plan = self.unblock_object(state, obj_type, obj_color)
        if plan:
            return plan
        return False
    # TODO: Change to add position
    # Method to plan how to pick up object
    def pick_up_object(self, state, obj_type, obj_color):
        """
        Method to plan to pick up an object.
        If the robot is already near and facing the object, it just returns the pick_up action.
        Otherwise, it tries to plan a sequence to get to the object and then pick it up.
        """

        if self.pick_up(state, obj_type, obj_color):
            return [('pick_up', obj_type, obj_color)]
        
        # Otherwise, try to plan a path to reach the object first.
        # TODO: Change to add position
        subplan = plan(state, [('go_to_object', obj_type, obj_color)],
                       get_operators(), get_methods(), verbose=self.verbose)
        if subplan:
            # If a plan is found, append the pick_up action to it.
            return subplan + [('pick_up', obj_type, obj_color)]
        
        # If no plan to reach the object is found, the task fails.
        return False

    def __str__(self):
        plan = self.plan
        steps = []
        for i, action in enumerate(plan, start=1):
            if not action:  # Skip if the action is empty
                continue

            action_type = action[0]

            if action_type == 'go_to' and len(action) > 2:
                steps.append(f"Step {i}: Go to {action[2]} {action[1]}")
            elif action_type == 'pick_up' and len(action) > 2:
                steps.append(f"Step {i}: Pick up {action[2]} {action[1]}")
            elif action_type == 'open' and len(action) > 0:
                steps.append(f"Step {i}: Open door")
            else:
                steps.append(f"Step {i}: Invalid action {action}")

        return "\n".join(steps)

def get_robot_surroundings(robot_location, grid_data, room_objects):
    surroundings = []
    _, robot_position = robot_location
    # These directions are ordered such that it gives
    # The rotation of the robot in order to face the nearby object.
    # starting from 0, 1, 2, 3
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    filled_locations = []
    for obj in room_objects:
        filled_locations.append(obj[2])

    for rot, d in enumerate(directions):
        pos = robot_position[0] + d[0], robot_position[1] + d[1]
        # Check if pos is out of bound
        if pos[0] < 0 or pos[1] < 0 or \
            pos[0] >= len(grid_data[0]) or \
            pos[1] >= len(grid_data):
            continue
        # Checks if the position is occupied by another object in the room.
        occupied = (pos[0], pos[1]) in filled_locations
        if occupied:
            _, obj = get_object_in_pos(pos, room_objects)
            surroundings.append((rot, obj))
    return surroundings

def get_object_in_pos(pos, room_objects):
    for index, obj in enumerate(room_objects):
        if pos == obj[2]:
            return index, obj
    return None, None

def get_door_in_pos(pos, state):
    for key, door in state.doors.items():
        if pos == door[1]:
            return key, door
    return None, None

def blocked_objects():
    pass

def is_blocked(position, grid_data, 
               room_objects, current_room, room_map):
    nearbys = get_object_empty_nearbys(position, grid_data, 
                                       room_objects, current_room,
                                       room_map)
    if len(nearbys) == 0:
        return True
    return False

def is_blocked_by_object(position, grid_data, 
               room_objects, current_room, room_map):
    nearbys = get_object_empty_nearbys(position, grid_data, 
                                       room_objects, current_room,
                                       room_map)
    nearbys_objects = get_object_empty_nearbys(position, grid_data, 
                                       room_objects, current_room,
                                       room_map, True)
    if len(nearbys) == 0 and len(nearbys_objects) > 0:
        return True
    return False

def get_objects(object_type, object_color, room_objects):
    filled_locations = []
    for obj in room_objects:
        if obj[0] == object_type and obj[1] == object_color:
            filled_locations.append(obj)
    return filled_locations

def get_object_empty_nearbys(position, grid_data, 
                             room_objects, current_room, 
                             room_map, ignore_objects=False):
    nearbys = []
    # These directions are ordered such that it gives
    # The rotation of the robot in order to face the object.
    # starting from 0, 1, 2, 3
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    filled_locations = []
    for obj in room_objects:
        filled_locations.append(obj[2])

    for rot, d in enumerate(directions):
        pos = position[0] + d[0], position[1] + d[1]
        # Checks if the position is occupied by another object in the room.
        occupied = (pos[0], pos[1]) in filled_locations
        if not ignore_objects and occupied:
            continue
        elif pos[0] < 0 or pos[1] < 0 or \
            pos[0] >= len(grid_data[0]) or \
            pos[1] >= len(grid_data):
            continue
        # Check if cell is not a wall
        elif grid_data[pos[1]][pos[0]] and grid_data[pos[1]][pos[0]].type == 'wall':
            continue
        # check if the empty space is in another room (if it's a door)
        elif room_map[pos] != current_room:
            continue
        else:
            nearbys.append((rot, pos))
    return nearbys

def dist_manhatten(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    dist = x+y
    return dist

def is_infront_of_door(self, state, door_location):
    current_room = self.current_room(state.robot_location)
    surroundings = get_robot_surroundings(state.robot_location, 
                            self.kb.grid_data, 
                            state.room_objects[current_room])
    for _, obj in surroundings:
        if obj[0] != 'door':
            continue
        # Look for the door that the robot is standing in front of.
        for (r1, r2), (door_color, position, door_state) in state.doors.items():
            if r1==current_room and position == door_location:
                # print(f"robot is in {state.robot_location} in front of door: ", door_color, position, door_state)
                return True
            elif r2==current_room and position == door_location:
                # print(f"robot is in {state.robot_location} in front of door: ", door_color, position, door_state)
                return True
    return False