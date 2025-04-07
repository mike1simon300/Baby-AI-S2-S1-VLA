from collections import defaultdict, deque

def extract_grid_data(env):
    """Extracts the grid data into a 2D array."""
    grid = env.unwrapped.grid
    grid_data = [[grid.get(i, j) for i in range(grid.width)] for j in range(grid.height)]
    return grid_data

def flood_fill(grid_data, start, visited):
    """Perform flood-fill to find a connected room."""
    room_cells = []
    queue = deque([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()

        if (x, y) in visited or not (0 <= x < len(grid_data[0])) or not (0 <= y < len(grid_data)):
            continue

        cell = grid_data[y][x]
        if cell and (cell.type == 'wall' or cell.type == 'door'):
            continue

        visited.add((x, y))
        room_cells.append((x, y))

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited:
                queue.append((nx, ny))

    return room_cells

def detect_rooms(grid_data):
    """Detects and labels all rooms in the grid."""
    visited = set()
    rooms = []

    for y in range(len(grid_data)):
        for x in range(len(grid_data[0])):
            if (x, y) not in visited and (not grid_data[y][x] or grid_data[y][x].type != 'wall'):
                room_cells = flood_fill(grid_data, (x, y), visited)
                if room_cells:
                    rooms.append(room_cells)

    return rooms

def find_connections(grid_data, rooms):
    """Finds connections (doors) between rooms."""
    connections = []
    room_map = {}

    # Map each cell to its room ID
    for room_id, room in enumerate(rooms):
        for x, y in room:
            room_map[(x, y)] = room_id

    # Check for doors between rooms
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(len(grid_data)):
        for x in range(len(grid_data[0])):
            cell = grid_data[y][x]

            if cell and cell.type == 'door':
                adjacent_rooms = set()
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) in room_map:
                        adjacent_rooms.add(room_map[(nx, ny)])

                if len(adjacent_rooms) == 2:
                    room1, room2 = sorted(adjacent_rooms)
                    if cell.is_locked:
                        door_state = 'locked'
                    elif cell.is_open:
                        door_state = 'open'
                    else:
                        door_state = 'closed'
                    door_info = {'type': cell.type, 'color': cell.color,
                                 'position': (x, y),'state': door_state}
                    connections.append((room1, room2, door_info))
    
    return connections, room_map

def build_knowledge_base(grid_data, rooms, connections):
    """Builds a knowledge base of objects organized by room and connections."""
    knowledge_base = {
        "rooms": defaultdict(list),
        "connections": connections
    }

    for room_id, room in enumerate(rooms):
        knowledge_base["rooms"][room_id] = []
        for x, y in room:
            cell = grid_data[y][x]
            if cell is None or cell.type in ['wall', 'empty']:
                continue
            elif cell.type == 'door':
                if cell.is_locked:
                    door_state = 'locked'
                elif cell.is_open:
                    door_state = 'open'
                else:
                    door_state = 'closed'

                knowledge_base["rooms"][room_id].append({
                    'type': cell.type,
                    'color': cell.color,
                    'position': (x, y),
                    'state': door_state
                })
            else:
                knowledge_base["rooms"][room_id].append({
                    'type': cell.type,
                    'color': cell.color,
                    'position': (x, y)
                })
    # Add doors to both connected rooms
    for room1, room2, door in connections:
        knowledge_base["rooms"][room1].append(door)
        knowledge_base["rooms"][room2].append(door)

    return knowledge_base

def format_knowledge_base(knowledge_base):
    """Formats the knowledge base as a text representation."""
    text_output = []

    for room_id, objects in knowledge_base["rooms"].items():
        if not objects:
            text_output.append(f"Room {room_id} is empty")
        else:
            text_output.append(f"Room {room_id}:")
            for obj in objects:
                if obj['type'] == 'door':
                    text_output.append(
                        f"  {obj['color']} {obj['type']} is at {obj['position']}"+
                        f" in Room {room_id} and is currently {obj['state']}")
                else:
                    text_output.append(
                        f"  {obj['color']} {obj['type']} is at {obj['position']}"+
                        f" in Room {room_id}")
    if len(knowledge_base["connections"]) == 0:
        text_output.append("There is no Connections.")
    else:
        text_output.append("\nConnections:")
    for room1, room2, door in knowledge_base["connections"]:
        text_output.append(f"Room {room1} connect to Room {room2} by {door['color']} {door['type']} at {door['position']} which is currently {door['state']}")

    return "\n".join(text_output)

class KnowledgeBase:
    def __init__(self, env) -> None:
        self.env = env
        self.grid_data = extract_grid_data(self.env)
        self.KB, self.room_map = self.build_KB(self.grid_data)

    def build_KB(self, grid_data):
        # Detect rooms
        rooms = detect_rooms(grid_data)
        # Find connections (doors) between rooms
        connections, room_map = find_connections(grid_data, rooms)
        return build_knowledge_base(grid_data, rooms, connections), room_map
    
    def __str__(self) -> str:
        return format_knowledge_base(self.KB)