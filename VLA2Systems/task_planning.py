import random
from typing import List
from VLA2Systems.pyhop.hop import State, Goal, declare_operators, declare_methods, plan, get_operators, get_methods
from VLA2Systems.reasoning_msgs import (
    directly_action, 
    connecting_action,
    pick_up_to_drop_next_to_reason,
    search_room_reason,
    found_door_reason,
    search_next_room_door_reason,
    object_next_room_door_reason,
    locked_search_key_reason,
    locked_door_have_key_reason,
    reach_room_reason,
    unlock_door_reason,
    unblock_reason,
    door_discription_reason
)

from VLA2Systems.knowledge_base import KnowledgeBase
from copy import deepcopy

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

    def get_current_room(self, robot_location):
        _, pos = robot_location
        room_id = self.kb.room_map[pos]
        return room_id

    def plan_go_to(self, obj_color, obj_type, obj_location=None):
        tasks = [('go_to_object', obj_type, obj_color, obj_location)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        if self.plan:
            last_task = None
            refined_plan: List = deepcopy(self.plan)
            for index, task in enumerate(self.plan):
                if task == last_task:
                    refined_plan.pop(index)
                last_task = task
            self.plan = refined_plan
        return self.plan

    def plan_open_door(self, obj_color, obj_location=None):
        tasks = [('open_door', "door", obj_color, obj_location)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        if self.plan:
            last_task = None
            refined_plan: List = deepcopy(self.plan)
            for index, task in enumerate(self.plan):
                if task == last_task:
                    refined_plan.pop(index)
                last_task = task
            self.plan = refined_plan
        return self.plan

    def plan_pick_up(self, obj_color, obj_type, obj_location=None):
        """
        Plan to go to an object and pick it up.
        """
        tasks = [('pick_up_object', obj_type, obj_color, obj_location)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        if self.plan:
            last_task = None
            refined_plan: List = deepcopy(self.plan)
            for index, task in enumerate(self.plan):
                if task == last_task:
                    refined_plan.pop(index)
                last_task = task
            self.plan = refined_plan
        return self.plan

    def plan_drop_next_to(self, obj_color, obj_type, 
                          next_to_obj_color="", next_to_obj_type="", 
                          next_to_obj_location=None):
        """
        Plan to go to an object and pick it up then go to the other object
        and place it next to it.
        Or if it already have the object, go to the other object and place it next to it.
        """
        tasks = [('drop_next_to_object', obj_type, obj_color, 
                  next_to_obj_type, next_to_obj_color, next_to_obj_location)]
        self.plan = plan(self.state, tasks, get_operators(), get_methods(), verbose=self.verbose)
        if self.plan:
            last_task = None
            refined_plan: List = deepcopy(self.plan)
            for index, task in enumerate(self.plan):
                if task == last_task:
                    refined_plan.pop(index)
                last_task = task
            self.plan = refined_plan
        return self.plan

    def declare(self):
        declare_operators(self.go_to, self.pick_up, self.open, self.drop, self.drop_next_to)
        declare_methods('go_to_object', self.go_to_object)
        declare_methods('pick_up_object', self.pick_up_object)
        declare_methods('open_door', self.open_door)
        declare_methods('drop_next_to_object', self.drop_next_to_object)
        if self.verbose > 0:
            print("Operators: ", get_operators())
            print("Methods: ", get_methods())

    # Operator
    def pick_up(self, state, obj_type, obj_color="", obj_location=None, 
                current_room=None, msg=""):
        # Case: Robot already holding the currect object
        if state.holding is not None and \
            state.holding[0] == obj_type and \
                (state.holding[1] == obj_color or obj_color == ""):
            return state
        # When holding an incorrect object return failure
        # ! Could be removed for more flexible planning
        if state.holding is not None:
            return False
        # Perform location change to go to the object if it was in the room
        self.go_to(state, obj_type, obj_color, obj_location, current_room)
        # if goto worked the robot should be around the object.
        current_room = self.get_current_room(state.robot_location)
        robot_rot, _ = state.robot_location
        surroundings = get_robot_surroundings(state.robot_location,
                                            self.kb.grid_data,
                                            state.room_objects[current_room])
        # if No objects nearby the robot return failure
        if len(surroundings) == 0:
            return False
        for rot, obj in surroundings:
            # Check if one of the nearby objects is the object we are searching for,
            # and the robot is now facing it. 
            if obj_type == obj[0] and rot == robot_rot and \
                (obj_color == obj[1] or obj_color == ""):
                state.holding = (obj_type, obj_color)
                state.room_objects[current_room].remove(obj)
                # print("GOING TO RETURN STATE")
                return state
        # The we don't have the current object or rotation
        return False

    # Operator
    def go_to(self, state, obj_type, obj_color="", obj_location=None, 
              current_room=None, get_all=False, msg=""):
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        # Look for the object in the current room
        for obj in state.room_objects[current_room]:
            if obj_color != "" and obj[1] != obj_color:
                continue
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
            # return all solutions for options
            if get_all:
                return possible_robot_locations
            # There is at least one solution:
            # Get a random choice of a solution
            solution = random.choice(possible_robot_locations)
            # # Get the first choice
            # solution = possible_robot_locations[0]
            # # Check if a path to solution is valid
            # ! Not Implemented yet (Path plan to possible_robot_location)
            state.robot_location = solution
            return state
        # the object is not found in the current room return failure
        return False

    # Operator
    def open(self, state, obj_type, obj_color, obj_location=None, 
             current_room=None, msg=""):
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        self.go_to(state, obj_type, obj_color, obj_location, current_room)
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
                    return state
                elif obj[3] == 'locked' and state.holding is not None \
                    and state.holding[0] == 'key' and state.holding[1] == obj_color:
                    index, _ = get_object_in_pos(obj[2], 
                                                 state.room_objects[current_room])
                    state.room_objects[current_room][index][3] = 'open'
                    door_location = state.room_objects[current_room][index][2]
                    key, _ = get_door_in_pos(door_location, state)
                    state.doors[key][2] = 'open'
                    return state
        return False

    # Operator
    def drop(self, state, obj_type="", obj_color="", obj_location=None, 
             current_room=None, msg=""):
        if not state.holding:
            return False
        if obj_type == "":
            obj_type = state.holding[0]
        if obj_color == "":
            obj_color = state.holding[1]
        if not (state.holding[0] == obj_type and state.holding[1] == obj_color):
            return False
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        # Drop next to the robot.
        possible_drop_locations = \
            get_object_empty_nearbys(state.robot_location[1], self.kb.grid_data, 
                                        state.room_objects[current_room], 
                                        current_room, self.kb.room_map)
        if len(possible_drop_locations) == 0:
            return False
        # There is at least one solution:
        # Get a random choice of a solution
        solution = random.choice(possible_drop_locations)
        # # Get the first choice
        # solution = possible_robot_locations[0]
        state.robot_location[0] = solution[0]
        state.room_objects[current_room].append(
                        (obj_type, obj_color, solution[1]))
        state.holding = None
        return state

    # Operator
    def drop_next_to(self, state, obj_type="", obj_color="", 
                     other_obj_type="", other_obj_color="", other_obj_location=None, 
                     current_room=None, msg=""):
        if not state.holding or other_obj_type == "":
            return False
        if obj_type == "":
            obj_type = state.holding[0]
        if obj_color == "":
            obj_color = state.holding[1]
        if not (state.holding[0] == obj_type and state.holding[1] == obj_color):
            return False
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        # GO to other object:
        self.go_to(state, other_obj_type, other_obj_color, other_obj_location, current_room)
        if other_obj_location is None:
            surroundings = get_robot_surroundings(state.robot_location,
                                            self.kb.grid_data,
                                            state.room_objects[current_room])
            # No objects nearby the robot (go to failed)
            if len(surroundings) == 0:
                return False
            for _, obj in surroundings:
                # Check if one of the nearby objects is the object we are searching for,
                # and the robot is now facing it. 
                if other_obj_type == obj[0] and \
                    (other_obj_color == obj[1] or other_obj_color==""):
                    other_obj_location = obj[2]
        # Didn't find the correct object (go to also failed)
        if other_obj_location is None:
            return False
        
        # Drop object next to the other object.
        possible_drop_locations = \
            get_object_empty_nearbys(other_obj_location, self.kb.grid_data, 
                                        state.room_objects[current_room], 
                                        current_room, self.kb.room_map)
        if len(possible_drop_locations) == 0:
            return False
        robot_object_locations = []

        for _, drop_location in possible_drop_locations:
            # check if the robot can go to one of these empty location:
            test_state = deepcopy(state)
            test_state.room_objects[current_room].append(
                    ("drop_point", "test", drop_location))
            robot_possible_locations = self.go_to(test_state, "drop_point", 
                                                    "test", drop_location, 
                                                    current_room, get_all=True)
            if robot_possible_locations is False:
                continue
            
            for robot_location in robot_possible_locations:
                if robot_location[1] == other_obj_location:
                    continue
                else:
                    robot_object_locations.append((robot_location, drop_location))
        if len(robot_object_locations) == 0:
            return False
        # There is at least one solution:
        # Get a random choice of a solution
        robot_location, drop_location = random.choice(robot_object_locations)
        # # Get the first choice
        # solution = possible_robot_locations[0]
        state.robot_location = robot_location
        state.room_objects[current_room].append(
                        (obj_type, obj_color, drop_location))
        state.holding = None
        return state

    # Helper
    def unblock_object(self, state, obj_type, obj_color, obj_location, action, current_room=None, msg=""):
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        room_objects = state.room_objects[current_room]
        objects_in_room = get_objects(obj_type, obj_color, room_objects)
        for target_obj in objects_in_room:
            if is_blocked_by_object(target_obj[2], self.kb.grid_data, room_objects,
                                    current_room, self.kb.room_map):
                nearbys = get_robot_surroundings((0, target_obj[2]), self.kb.grid_data, 
                                    room_objects)
                for _, other_object in nearbys:
                    unblock_msg = unblock_reason("", obj_type, obj_color, other_object[2], current_room)
                    action = [('pick_up_object', other_object[0], other_object[1], other_object[2], 
                               current_room, unblock_msg)]
                    subplan = plan(state, action,
                       get_operators(), get_methods(), verbose=self.verbose)
                    if subplan:
                        break
                if subplan:
                    task = [(action, obj_type, obj_color, obj_location, current_room, msg)]
                    if action == 'go_to':
                        task = [(action, obj_type, obj_color, obj_location, current_room, False, msg)]
                    return subplan + task
        return False
    # Helper
    def object_not_in_room(self, state, obj_type, obj_color, 
                           obj_location=None, current_room=None):
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        room_objects = state.room_objects[current_room]
        objects_in_room = get_objects(obj_type, obj_color, room_objects)
        # print(f"objects_in_room {objects_in_room}")
        # If no instance of the object is found in this room, we might consider exploring other rooms.
        if obj_location is None and len(objects_in_room) == 0:
            # For now, return failure (or implement exploration)
            return True
        elif obj_location is None:
            return False
        # print(f"obj_location {obj_location}")
        for obj in objects_in_room:
            if obj[2] == obj_location:
                return False
        return True

    def search_rooms(self, state, obj_type, obj_color, object_location, 
                     current_room, action, msg=""):
        if current_room is None:
            current_room = self.get_current_room(state.robot_location)
        # Start the recursive search with an empty visited set.
        return self.search_rooms_for_object(state, current_room, obj_type, obj_color, 
                                            object_location, action, visited=set(), msg=msg)

    def search_rooms_for_object(self, state, current_room, obj_type, obj_color, 
                                object_location, action, visited, msg=""):
        # Mark the current room as visited.
        visited.add(current_room)
        # print(f"ADDED {current_room} to visited")
        # print(f"msg = {msg}")
        # if msg == "" or msg is None:
        #     msg = search_room_reason("because", obj_type, obj_color, 
        #                              object_location, current_room)
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

            # msg += found_door_reason("After searching the connections,", door_color, position, 
            #                         door_state, current_room, neighbor, obj_color, obj_type)
            reach_msg = reach_room_reason("", neighbor)
            # Prepare the tasks to handle the door based on its state.
            if door_state == 'open':
                if is_infront_of_door(self, state, position):
                    door_tasks = []
                else:
                    door_tasks = [['go_to_object', 'door', door_color, position, current_room, reach_msg]]
            elif door_state == 'closed':
                if is_infront_of_door(self, state, position):
                    door_tasks = [['open_door', 'door', door_color, position, current_room, reach_msg]]
                else:
                    door_tasks = [['open_door', 'door', door_color, position, current_room, reach_msg]]
            elif door_state == 'locked':
                # ! Multiple options 
                # ! FIRST (let the RL figure how to drop and pick up)
                if state.holding and state.holding[0] == 'key' and state.holding[1] == door_color:
                    # msg += locked_door_have_key_reason(" However,", door_color, position, current_room, neighbor)
                    door_tasks = [['open_door', 'door', door_color, position, current_room, reach_msg]]
                else:
                    # msg1 = msg + locked_search_key_reason(" However,", door_color, position, 
                    #                                       current_room, neighbor)
                    # msg2 = msg + locked_door_have_key_reason(" However,", door_color, position, current_room, neighbor)
                    unlock_msg = unlock_door_reason("", door_color, position, current_room)
                    door_tasks = [['pick_up_object', 'key', door_color, None, None, unlock_msg], 
                                  ['open_door', 'door', door_color, position, current_room, reach_msg]]
                
                # ! SECOND (manually tell a sub-step of drop and pick up)
                # if state.holding:
                #     if state.holding[0] == 'key' and state.holding[1] == door_color:
                #         door_tasks = [('open_door', 'door', door_color, position)]
                #     else:
                #         # ! here we want to add to drop what it is holding 
                #         # ! and then pick up the key.
                #         door_tasks = [('drop_next_to', state.holding[0], state.holding[1], 
                #                        'key', door_color, position), 
                #                       ('pick_up_object', 'key', door_color, None), 
                #                       ('open_door', 'door', door_color, position)]
                # else:
                #     door_tasks = [('pick_up_object', 'key', door_color, None), 
                #                   ('open_door', 'door', door_color, position)]
                # For locked doors, we pick up the key first.
                # door_tasks = [['pick_up_object', 'key', door_color, None], 
                #             ['open_door', 'door', door_color, position]]
            else:
                # If the door state is unknown, skip it.
                continue
            
            # Check if the target object exists in the neighbor room.
            if any(o[0] == obj_type and (o[1] == obj_color or obj_color=="") for o in state.room_objects[neighbor]) or \
                any(o[0] == obj_type and (o[1] == obj_color or obj_color=="") and o[2] == object_location for o in state.room_objects[neighbor]):
                # Return the plan: handle door then go to the object in neighbor.
                messege = f"{obj_color} {obj_type} {object_location} is in Room {neighbor}, in order to reach this room the robot has to"
                phrase = ""
                for task in door_tasks:
                    messege += connecting_action(phrase, task[0], task[1], door_color, position, state)
                    phrase = "then"

                    # task[-1] += object_next_room_door_reason(" and,",current_room, neighbor, obj_color, obj_type)
                    # task = tuple(task)
                # print("MESSEGE IS: ", messege)
                final_task = [(action, obj_type, obj_color, object_location, neighbor, msg)]
                if action == 'go_to':
                    final_task = [(action, obj_type, obj_color, object_location, neighbor, False, msg)]
                return door_tasks + final_task
            # print(f"THE OBJECT WAS NOT FOUND IN THE neighbor room, doing recursion. current_room {current_room} neighbor room {neighbor}")
            # Otherwise, try to find the object recursively in the neighbor room.
            sub_tasks = self.search_rooms_for_object(state, neighbor, obj_type, obj_color, 
                                                     object_location, action, visited, msg)
            # print("RECURSION RETURNED THIS SUBTASKS: ", sub_tasks)
            if sub_tasks:
                # print("LIST OF SUB TASKS: ", sub_tasks)

                # for task in door_tasks:
                #     print(sub_tasks[-1])
                #     print(task[-1])
                #     task[-1] = sub_tasks[-1][-1]
                #     task = tuple(task)
                # If a plan was found deeper in the graph, prepend the door tasks.
                return door_tasks + sub_tasks

        # If no connected room (or chain of rooms) contains the object, return False.
        return False

    def lookup_ulternative(self, state, obj_type, obj_color, obj_location, room, action, msg=""):
        # print("FAILED DIRECTLY GOTO, gonna try with searching for the object")
        # print(f"object_not_in_room \n {state.room_objects} \n {obj_type} {obj_color} {obj_location} {room}")
        object_is_not_in_the_room = self.object_not_in_room(
            state, obj_type, obj_color, obj_location, room)
        # print(f"object_is_not_in_the_room ", object_is_not_in_the_room)
        if self.object_not_in_room(state, obj_type, obj_color, obj_location, room):
            # print("GOING IN search_rooms")
            plan = self.search_rooms(state, obj_type, obj_color, obj_location, 
                                     room, action, msg)
            # print(plan)
            if plan:
                # print(f"going to return plan {plan}")
                return plan
        # This means that the object is in the room, but there is something wrong
        # This could be that the robot is currently holding another object.
        elif action == 'pick_up_object' and state.holding:
            # ! Multiple options 
            # ! FIRST (let the RL figure how to drop and pick up)
            if obj_location is None:
                objects_in_room = get_objects(obj_type, obj_color, 
                                            state.room_objects[room])
                object = random.choice(objects_in_room)
                new_object = (state.holding[0], state.holding[1], object[2])
                state.holding = (object[0], object[1])
                state.room_objects[room].remove(object)
                state.room_objects[room].append(new_object)
                if msg == "":
                    msg += directly_action('pick_up', obj_type, obj_color, 
                                           obj_location, room=room)
                else:
                    msg += connecting_action("", 'pick_up', obj_type, obj_color, 
                                             obj_location, room=room)
                return [('pick_up', obj_type, obj_color, obj_location, room, msg)]
                                          
            # ! SECOND (manually tell a sub-step of drop and pick up)
            # ! not implemented
        # print("FAILED searching for the object")
        plan = self.unblock_object(state, obj_type, obj_color, obj_location, action, room, msg)
        if plan:
            return plan
        return False

    # Method to plan how to reach an object
    def go_to_object(self, state, obj_type, obj_color="", obj_location=None, 
                     room=None, msg=""):
        if msg is None:
            msg = ""
        test_state = deepcopy(state)
        if self.go_to(test_state, obj_type, obj_color, obj_location, room):
            if room is None:
                room = self.get_current_room(state.robot_location)
            if msg == "":
                msg += directly_action('go_to', obj_type, obj_color, obj_location, room=room)
            elif obj_type == "door" and obj_location:
                key, door = get_door_in_pos(obj_location, state)
                if key:
                    msg += door_discription_reason("", 'go_to', obj_color, obj_location, door[2], key[0], key[1])
                else:
                    msg += connecting_action("", 'go_to', obj_type, obj_color, obj_location, room=room)
            else:
                msg += connecting_action("", 'go_to', obj_type, obj_color, obj_location, room=room)
            return [('go_to', obj_type, obj_color, obj_location, room, False, msg)]
        return self.lookup_ulternative(state, obj_type, obj_color, obj_location, 
                                       room, 'go_to_object', msg)

    # Method to plan how to pick up object
    def pick_up_object(self, state, obj_type, obj_color="", obj_location=None, 
                       room=None, msg=""):
        """
        Method to plan to pick up an object.
        If the robot is already near and facing the object, it just returns the pick_up action.
        Otherwise, it tries to plan a sequence to get to the object and then pick it up.
        """
        if msg is None:
            msg = ""
        # print("PICK UP MESSEGE: ", msg)
        test_state = deepcopy(state)
        if self.pick_up(test_state, obj_type, obj_color, obj_location, room):
            if room is None:
                room = self.get_current_room(state.robot_location)
            if msg == "":
                msg += directly_action('pick_up', obj_type, obj_color, obj_location, room=room)
            else:
                msg += connecting_action("", 'pick_up', obj_type, obj_color, 
                                         obj_location, room=room)
            # print("RETURNED PICK UP WITH THIS MSG: ", msg)
            return [('pick_up', obj_type, obj_color, obj_location, room, msg)]
        return self.lookup_ulternative(state, obj_type, obj_color, obj_location, 
                                       room, 'pick_up_object', msg)

    def drop_next_to_object(self, state, obj_type, obj_color="",
                            other_obj_type="", other_obj_color="", other_obj_location="", 
                            room=None, msg=""):
        if msg is None:
            msg = ""
        if other_obj_type == "":
            return False
        tasks = []
        if state.holding is None:
            msg1 = pick_up_to_drop_next_to_reason("In order to,", obj_type, obj_color, None,
                                                  other_obj_type, other_obj_color, other_obj_location)
            tasks.append(('pick_up_object', obj_type, obj_color, None, None, msg1))
        elif state.holding[0] != obj_type or (state.holding[1] != obj_color and obj_color != ""):
            msg1 = pick_up_to_drop_next_to_reason("In order to,", obj_type, obj_color, None,
                                                  other_obj_type, other_obj_color, other_obj_location)
            return [('pick_up_object', obj_type, obj_color, None, None, msg1),
                    ('drop_next_to_object', obj_type, obj_color, 
                          other_obj_type, other_obj_color, other_obj_location, room)]
        test_state = deepcopy(state)
        if self.drop_next_to(test_state, obj_type, obj_color, 
                             other_obj_type, other_obj_color, other_obj_location, room):
            if len(tasks) == 0:
                msg += directly_action('drop_next_to', obj_type, obj_color, None, None,
                          other_obj_type, other_obj_color, other_obj_location, room)
            else:
                msg += connecting_action('', 'drop_next_to', obj_type, 
                                        obj_color, None, None, other_obj_type, 
                                        other_obj_color, other_obj_location, room)
            tasks.append(('drop_next_to', obj_type, obj_color, 
                          other_obj_type, other_obj_color, other_obj_location, room, msg))
            return tasks
        # If it fails the other object might be in another room.
        # Find the other object (by using the lookup_ulternative to go to the other object) 
        plan = self.lookup_ulternative(state, other_obj_type, other_obj_color, 
                                       other_obj_location, room, 'go_to_object', msg)
        # If it was able to go the other object
        if plan:
            # remove the last task (which will be go to other object) and replace it with
            # drop object next to other object but using the same room as the last task.
            last_task = plan.pop(-1)
            other_obj_location = last_task[3]
            other_obj_room = last_task[4]
            # print(f"last_task {last_task}")
            tasks.extend(plan)
            tasks.append(('drop_next_to_object', obj_type, obj_color, 
                          other_obj_type, other_obj_color, other_obj_location, other_obj_room))
            return tasks
        if len(tasks) == 1 and tasks[0][0] == 'pick_up_object':
            return [('pick_up_object', obj_type, obj_color),
                    ('drop_next_to_object', obj_type, obj_color, 
                          other_obj_type, other_obj_color, other_obj_location, room)]

        return False

    def open_door(self, state, obj_type, obj_color="", obj_location=None, room=None, msg=""):
        if msg is None:
            msg = ""
        test_state = deepcopy(state)
        if self.open(test_state, obj_type, obj_color, obj_location, room):
            if msg == "":
                msg += directly_action('open', obj_type, obj_color, obj_location, room=room)
            else:
                key, door = None, None
                if obj_location:
                    key, door = get_door_in_pos(obj_location, state)
                if key:
                    msg += door_discription_reason("", 'open', obj_color, obj_location, door[2], key[0], key[1])
                else:
                    msg += connecting_action("", 'open', obj_type, obj_color, obj_location, room=room)
            return [('open', obj_type, obj_color, obj_location, room, msg)]
        return self.lookup_ulternative(state, obj_type, obj_color, obj_location, 
                                       room, 'open_door', msg)

    def __str__(self, plan=None):
        if plan is None:
            plan = self.plan
        steps = []
        for i, action in enumerate(plan, start=1):
            if not action:  # Skip if the action is empty
                continue

            action_type = action[0]
            color = action[2]
            if color == "":
                color = random.choice(["a","the"])

            if action_type == 'go_to' and len(action) > 2:
                steps.append(f"Step {i}: Go to {color} {action[1]}")
            elif action_type == 'pick_up' and len(action) > 2:
                steps.append(f"Step {i}: Pick up {color} {action[1]}")
            elif action_type == 'open' and len(action) > 0:
                steps.append(f"Step {i}: Open {color} {action[1]}")
            elif action_type == 'drop_next_to' and len(action) > 0:
                color2 = action[4]
                if color2 == "":
                    color = random.choice(["a","the"])
                steps.append(f"Step {i}: Drop {color} {action[1]} next to {color2} {action[3]}")
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
        if obj[0] == object_type and \
            (obj[1] == object_color or object_color==""):
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
        elif grid_data[pos[1]][pos[0]] and (
            grid_data[pos[1]][pos[0]].type == 'wall' or 
            grid_data[pos[1]][pos[0]].type == 'door'):
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
    current_room = self.get_current_room(state.robot_location)
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