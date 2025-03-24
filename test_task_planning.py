import random
from pyhop.hop import State, Goal, declare_operators, declare_methods, plan, get_operators, get_methods

class RobotPlanner:
    def __init__(self, knowledge_base, start_location=None):
        self.kb = knowledge_base
        self.state = self.create_state(start_location)

    def create_state(self, start_location):
        state = State('robot_state')
        state.holding = None
        state.room_objects = {}
        state.doors = {}

        # Populate room objects and doors from the knowledge base
        for room, objects in self.kb['rooms'].items():
            state.room_objects[room] = [(obj['type'], obj['color'], obj['position']) for obj in objects]

        for r1, r2, door in self.kb['connections']:
            state.doors[(r1, r2)] = (door['color'], door['position'])

        # Set the robot's starting location (input or random)
        if start_location:
            state.robot_location = start_location
        else:
            random_room = random.choice(list(self.kb['rooms'].keys()))
            random_position = random.choice(
                [obj['position'] for obj in self.kb['rooms'][random_room]]
            ) if self.kb['rooms'][random_room] else (0, 0)
            state.robot_location = (random_room, random_position)

        return state

    def plan(self, goal):
        tasks = [('achieve_goal', goal)]
        print(f"Starting at: {self.state.robot_location}")
        return plan(self.state, tasks, get_operators(), get_methods(), verbose=3)

    def __str__(self, plan):
        steps = []
        for i, action in enumerate(plan, start=1):
            if not action:  # Skip if the action is empty
                continue

            action_type = action[0]

            if action_type == 'move_to' and len(action) > 2:
                steps.append(f"Step {i}: Go to {action[2]} in Room {action[1]}")
            elif action_type == 'pick_up' and len(action) > 2:
                steps.append(f"Step {i}: Pick up {action[2]} {action[1]}")
            elif action_type == 'toggle_door' and len(action) > 2:
                steps.append(f"Step {i}: Open {action[2]} door")
            elif action_type == 'pass_door' and len(action) > 3:
                steps.append(f"Step {i}: Pass through {action[2]} door from Room {action[1]} to Room {action[3]}")
            else:
                steps.append(f"Step {i}: Invalid action {action}")

        return "\n".join(steps)

# Operators
def move_to(state, room, obj_pos):
    state.robot_location = (room, obj_pos)
    return state

def pick_up(state, obj_type, obj_color):
    current_room, _ = state.robot_location
    for obj in state.room_objects[current_room]:
        if obj[0] == obj_type and obj[1] == obj_color:
            state.holding = (obj_type, obj_color)
            state.room_objects[current_room].remove(obj)
            return state
    return False

def toggle_door(state, door_color):
    return state

def pass_door(state, door_color, current_room, next_room):
    if ((current_room, next_room) in state.doors and state.doors[(current_room, next_room)][0] == door_color):
        position = state.doors[(current_room, next_room)][1]
        state.robot_location = (next_room, position)
        return state
    return False

declare_operators(move_to, pick_up, toggle_door, pass_door)

# Method: Goal Achievement
def achieve_goal(state, goal):
    subtasks = []

    for task in goal:
        action, args = task
        if action == 'pick':
            obj_type = args['type']
            obj_color = args['color']

            # Locate the object in any room
            obj_location = None
            for room, objects in state.room_objects.items():
                for obj in objects:
                    if obj[0] == obj_type and obj[1] == obj_color:
                        obj_location = (room, obj[2])
                        break
                if obj_location:
                    break

            if obj_location is None:
                print(f"Object {obj_color} {obj_type} not found!")
                return False

            current_room, _ = state.robot_location

            if current_room != obj_location[0]:
                # Find door path to the object
                path = find_path_to_room(state, current_room, obj_location[0])
                subtasks.extend(path)

            # Move and pick up the object
            subtasks.append(('move_to', obj_location[0], obj_location[1]))
            subtasks.append(('pick_up', obj_type, obj_color))

    return subtasks

declare_methods('achieve_goal', achieve_goal)

# Helper: Find Path to Room
def find_path_to_room(state, start_room, target_room):
    path = []
    visited = set()

    def dfs(current_room):
        if current_room == target_room:
            return True

        visited.add(current_room)
        for (r1, r2), (door_color, position) in state.doors.items():
            if r1 == current_room and r2 not in visited:
                path.append(('move_to', r1, position))
                path.append(('toggle_door', door_color))
                path.append(('pass_door', door_color, r1, r2))
                if dfs(r2):
                    return True
                path.pop()
                path.pop()
                path.pop()
        return False

    dfs(start_room)
    return path

# Example usage:
kb = {
    'rooms': {
        0: [{'type': 'door', 'color': 'red', 'position': (3, 2)}],
        1: [{'type': 'ball', 'color': 'green', 'position': (8, 5)}, {'type': 'door', 'color': 'red', 'position': (3, 2)}, {'type': 'door', 'color': 'grey', 'position': (1, 8)}],
        2: [{'type': 'key', 'color': 'grey', 'position': (1, 8)}, {'type': 'door', 'color': 'grey', 'position': (1, 8)}]
    },
    'connections': [
        (0, 1, {'type': 'door', 'color': 'red', 'position': (3, 2)}),
        (1, 2, {'type': 'door', 'color': 'grey', 'position': (1, 8)})
    ]
}

# Example: Random start or specific location
planner = RobotPlanner(kb, start_location=(0, (3, 2)))
goal = [('pick', {'type': 'ball', 'color': 'green'})]
plan = planner.plan(goal)

if plan:
    print(planner.__str__(plan))
else:
    print("No valid plan found.")