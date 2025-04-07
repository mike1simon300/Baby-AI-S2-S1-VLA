

def connecting_action(connecting_phrase, action, object, color, 
                      position=None, state=None, obj2=None, color2=None, 
                      pos2=None, room=None):
    color = "" if color is None else color
    color = f"{color} " if color != "" else color
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f" which is in Room {room}" if room != "" else room

    msg = ""
    if action == 'go_to':
        msg = f'{connecting_phrase} go to {color}{object}{position}{room}'
    elif action == 'pick_up':
        msg = f'{connecting_phrase} pick up {color}{object}{position}{room}'
    elif action == 'open':
        state = "" if state is None else state
        state = f" which is currently {state}" if state != "" else state
        msg = f'{connecting_phrase} open {color}{object}{position}{state}'
    elif action == 'drop_next_to':
        color2 = "" if color2 is None else color2
        color2 = f"{color2} " if color2 != "" else color2
        pos2 = "" if pos2 is None else pos2
        pos2 = f" at ({pos2[0]}, {pos2[1]})" if pos2 != "" else pos2
        msg = f'{connecting_phrase} drop holded {color}{object}{position} next to {color2}{obj2}{pos2}{room}'
    
    return msg

def directly_action(action, object, color, position=None, state=None,
                    obj2=None, color2=None, pos2=None, room=None):
    return connecting_action("directly", action, object, color, 
                      position, state, obj2, color2, 
                      pos2, room)

def reason_of_action(reason_phrase, action, object, color, 
                      position=None, state=None, obj2=None, color2=None, 
                      pos2=None, room=None):
    color = "" if color is None else color
    color = f"{color} " if color != "" else color
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f" which is in Room {room}" if room != "" else room

    msg = ""
    if action == 'go_to':
        msg = f'{reason_phrase} {color}{object}{position} is in {room}'

def pick_up_to_drop_next_to_reason(reason_phrase, object, color, 
                      position=None, obj2=None, color2=None, 
                      pos2=None, room=None):
    color = "" if color is None else color
    color = f"{color} " if color != "" else "the "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f" which is in Room {room}" if room != "" else room
    color2 = "" if color2 is None else color2
    color2 = f"{color2} " if color2 != "" else "the "
    pos2 = "" if pos2 is None else pos2
    pos2 = f" at ({pos2[0]}, {pos2[1]})" if pos2 != "" else pos2
    msg = f'{reason_phrase} drop the {color}{object}{position} next to {color2}{obj2}{pos2}'
    msg += f' the robot need to pick up {color}{object}{position} first.'
    return msg

def search_room_reason(reason_phrase, object, color, position=None, room=None):
    color = "" if color is None else color
    color = f"{color} " if color != "" else "the "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f"was not found in Room {room}" if room != "" else "was not found robot current Room"
    msg = f'{reason_phrase} {color}{object}{position} {room}'
    msg += f' I will look in other rooms through doors. '
    return msg

def found_door_reason(connecting_phrase, door_color, position, state, r1, r2, obj_color, obj_type):
    door_color = "" if door_color is None else door_color
    door_color = f"{door_color} " if door_color != "" else "a "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    obj_color = "" if obj_color is None else obj_color
    obj_color = f"{obj_color} " if obj_color != "" else "the "
    state = "" if state is None else state
    state = f" which is currently {state}" if state != "" else state

    msg = f"{connecting_phrase} I found {door_color}door{position}{state} connecting Room {r1} and Room {r2}. "
    return msg

def object_next_room_door_reason(connecting_phrase, r1, r2, obj_color, obj_type):
    # door_color = "" if door_color is None else door_color
    # door_color = f"{door_color} " if door_color != "" else "the "
    # position = "" if position is None else position
    # position = f" at ({position[0]}, {position[1]})" if position != "" else position
    obj_color = "" if obj_color is None else obj_color
    obj_color = f"{obj_color} " if obj_color != "" else "the "
    # state = "" if state is None else state
    # state = f" which is currently {state}" if state != "" else state

    msg = f"{connecting_phrase} the {obj_color} {obj_type} is in Room {r2} which is connected to Room {r1} through this door."
    msg += f" So the correct current step in order to reach {obj_color}{obj_type} should be,"
    return msg

def search_next_room_door_reason(connecting_phrase, r1, r2, obj_color, obj_type):
    obj_color = "" if obj_color is None else obj_color
    obj_color = f"{obj_color} " if obj_color != "" else "the "

    msg = f"{connecting_phrase} {obj_color}{obj_type} is in a connected Room to Room {r2}."
    msg += f" So the correct current step in order to reach {obj_color}{obj_type} should be,"
    return msg

def locked_door_have_key_reason(connecting_phrase, door_color, position, r1, r2):
    door_color = "" if door_color is None else door_color
    door_color = f"{door_color} " if door_color != "" else "a "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position

    msg = f"{connecting_phrase} The {door_color}door{position} is locked"
    msg += f", but the robot is holding the {door_color}key so it can be opened in this step."
    return msg

def locked_search_key_reason(connecting_phrase, door_color, position, r1, r2):
    door_color = "" if door_color is None else door_color
    door_color = f"{door_color} " if door_color != "" else "a "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position

    msg = f"{connecting_phrase} The {door_color}door{position} is locked"
    msg += f", and the robot does not have the {door_color}key, and the robot needs this key to open the {door_color}door."
    return msg

def reach_room_reason(connecting_phrase, room):
    return f"{connecting_phrase}In order to reach room {room}"

def unlock_door_reason(connecting_phrase, color, position, room):
    color = "" if color is None else color
    color = f"{color} " if color != "" else color
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f" which is in Room {room}" if room != "" else room

    return f"{connecting_phrase}In order to unlock {color}door{position}{room}"

def unblock_reason(connecting_phrase, obj, color, position, room):
    color = "" if color is None else color
    color = f"{color} " if color != "" else color
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    room = "" if room is None else room
    room = f" which is in Room {room}" if room != "" else room

    return f"{connecting_phrase}In order to unblock {color}{obj}{position}{room}"

def door_discription_reason(connecting_phrase, action, door_color, position, state, r1, r2):
    door_color = "" if door_color is None else door_color
    door_color = f"{door_color} " if door_color != "" else "a "
    position = "" if position is None else position
    position = f" at ({position[0]}, {position[1]})" if position != "" else position
    state = "" if state is None else state
    state = f" which is currently {state}" if state != "" else state
    if action == 'go_to':
        msg = f'{connecting_phrase} go to {door_color}door{position}{state} and connect Room {r1} and Room {r2}'
    elif action == 'open':
        msg = f'{connecting_phrase} open {door_color}door{position}{state} and connect Room {r1} and Room {r2}'
    return msg
