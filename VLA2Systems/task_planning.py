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
                        (obj['type'], obj['color'], 
                         obj['position'], obj['state']))
                else:
                    state.room_objects[room].append(
                        (obj['type'], obj['color'], obj['position']))

        for r1, r2, door in self.kb.KB['connections']:
            state.doors[(r1, r2)] = (door['color'], door['position'], door['state'])

        # Set the robot's starting location
        if start_location:
            state.robot_location = start_location

        return state

    def current_room(self, robot_location):
        _, pos = robot_location
        room_id = self.kb.room_map[pos]
        return room_id

    def plan_go_to(self, obj_color, obj_type):
        tasks = [('go_to_object', obj_type, obj_color)]
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
        declare_operators(self.go_to, self.pick_up)
        declare_methods('go_to_object', self.go_to_object)
        declare_methods('pick_up_object', self.pick_up_object)

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

    def go_to(self, state, obj_type, obj_color):
        current_room = self.current_room(state.robot_location)
        for obj in state.room_objects[current_room]:
            if obj[0] == obj_type and obj[1] == obj_color:

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
        return False

    # Method to plan how to reach an object in the same room
    def go_to_object(self, state, obj_type, obj_color):
        if self.go_to(state, obj_type, obj_color):
            return [('go_to', obj_type, obj_color)]
        current_room = self.current_room(state.robot_location)
        room_objects = state.room_objects[current_room]
        objects_in_room = get_objects(obj_type, obj_color, room_objects)
        # If no instance of the object is found in this room, we might consider exploring other rooms.
        if len(objects_in_room) == 0:
            # For now, return failure (or implement exploration)
            return False
        # print(objects_in_room)
        for target_obj in objects_in_room:
            # print(target_obj)
            if is_blocked_by_object(target_obj[2], self.kb.grid_data, room_objects,
                                    current_room, self.kb.room_map):
                # print("is_blocked_by_object YES")
                # nearbys = get_object_empty_nearbys(target_obj[2], self.kb.grid_data, 
                #                                    room_objects, current_room, 
                #                                    self.kb.room_map, True)
                nearbys = get_robot_surroundings((0, target_obj[2]), self.kb.grid_data, 
                                    room_objects)
                # print(nearbys)
                for _, other_object in nearbys:
                    # print([('pick_up_object', other_object[0], other_object[1])])
                    subplan = plan(state, [('pick_up_object', other_object[0], other_object[1])],
                       get_operators(), get_methods(), verbose=self.verbose)
                    # print("preformed sub-plan")
                    # print(subplan)
                    if subplan:
                        break
                if subplan:
                    return subplan + [('go_to', obj_type, obj_color)]
                
                # print("Exited Loop")
        # print("HERE")
        return False

    def pick_up_object(self, state, obj_type, obj_color):
        """
        Method to plan to pick up an object.
        If the robot is already near and facing the object, it just returns the pick_up action.
        Otherwise, it tries to plan a sequence to get to the object and then pick it up.
        """
        if self.pick_up(state, obj_type, obj_color):
            return [('pick_up', obj_type, obj_color)]
        
        # Otherwise, try to plan a path to reach the object first.
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
            elif action_type == 'toggle_door' and len(action) > 2:
                steps.append(f"Step {i}: Open {action[2]} door")
            elif action_type == 'pass_door' and len(action) > 3:
                steps.append(f"Step {i}: Pass through {action[2]} door from Room {action[1]} to Room {action[3]}")
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
            obj = get_object_in_pos(pos, room_objects)
            surroundings.append((rot, obj))
    return surroundings

def get_object_in_pos(pos, room_objects):
    for obj in room_objects:
        if pos == obj[2]:
            return obj
    return None

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
    x = pos1[0] - pos2[0]
    y = pos1[1] - pos2[1]
    dist = x+y
    return dist


# # Operators
# def go_to(state, room, obj_pos):
#     state.robot_location = (room, obj_pos)
#     return state

# def pick_up(state, obj_type, obj_color):
#     current_room, _ = state.robot_location
#     for obj in state.room_objects[current_room]:
#         if obj[0] == obj_type and obj[1] == obj_color:
#             state.holding = (obj_type, obj_color)
#             state.room_objects[current_room].remove(obj)
#             return state
#     return False

# def toggle_door(state, door_color):
#     return state

# def pass_door(state, door_color, current_room, next_room):
#     if ((current_room, next_room) in state.doors and state.doors[(current_room, next_room)][0] == door_color):
#         position = state.doors[(current_room, next_room)][1]
#         state.robot_location = (next_room, position)
#         return state
#     return False

# # declare_operators(go_to, pick_up, toggle_door, pass_door)

# # Method: Goal Achievement
# def achieve_goal(state, goal):
#     subtasks = []

#     for task in goal:
#         action, args = task
#         if action == 'pick':
#             obj_type = args['type']
#             obj_color = args['color']

#             # Locate the object in any room
#             obj_location = None
#             for room, objects in state.room_objects.items():
#                 for obj in objects:
#                     if obj[0] == obj_type and obj[1] == obj_color:
#                         obj_location = (room, obj[2])
#                         break
#                 if obj_location:
#                     break

#             if obj_location is None:
#                 print(f"Object {obj_color} {obj_type} not found!")
#                 return False

#             current_room, _ = state.robot_location

#             if current_room != obj_location[0]:
#                 # Find door path to the object
#                 path = find_path_to_room(state, current_room, obj_location[0])
#                 subtasks.extend(path)

#             # Move and pick up the object
#             subtasks.append(('go_to', obj_location[0], obj_location[1]))
#             subtasks.append(('pick_up', obj_type, obj_color))

#     return subtasks

# declare_methods('achieve_goal', achieve_goal)

# # Helper: Find Path to Room
# def find_path_to_room(state, start_room, target_room):
#     path = []
#     visited = set()

#     def dfs(current_room):
#         if current_room == target_room:
#             return True

#         visited.add(current_room)
#         for (r1, r2), (door_color, position) in state.doors.items():
#             if r1 == current_room and r2 not in visited:
#                 path.append(('go_to', r1, position))
#                 path.append(('toggle_door', door_color))
#                 path.append(('pass_door', door_color, r1, r2))
#                 if dfs(r2):
#                     return True
#                 path.pop()
#                 path.pop()
#                 path.pop()
#         return False

#     dfs(start_room)
#     return path

# # Example usage:
# kb = {
#     'rooms': {
#         0: [{'type': 'door', 'color': 'red', 'position': (3, 2)}],
#         1: [{'type': 'ball', 'color': 'green', 'position': (8, 5)}, {'type': 'door', 'color': 'red', 'position': (3, 2)}, {'type': 'door', 'color': 'grey', 'position': (1, 8)}],
#         2: [{'type': 'key', 'color': 'grey', 'position': (1, 8)}, {'type': 'door', 'color': 'grey', 'position': (1, 8)}]
#     },
#     'connections': [
#         (0, 1, {'type': 'door', 'color': 'red', 'position': (3, 2)}),
#         (1, 2, {'type': 'door', 'color': 'grey', 'position': (1, 8)})
#     ]
# }

# # Example: Random start or specific location
# planner = RobotPlanner(kb, start_location=(0, (3, 2)))
# goal = [('pick', {'type': 'ball', 'color': 'green'})]
# plan = planner.plan(goal)

# if plan:
#     print(planner.__str__(plan))
# else:
#     print("No valid plan found.")