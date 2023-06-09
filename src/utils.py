import re


def get_position(observation):
        return observation[0]

def extract_map(input_string):
    env_map = remove_color(input_string)
    # env_map = env_map.strip().split("\n")

    # S = start
    # G = goal
    # F = frozen
    # H = hole

    # parse 

    return env_map

def remove_color(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def get_position_based_on_color(map):
    # get size of map
    map = map.split("\n")

def extract_player_position(input_string):

    lines = input_string.split('\n')
    # remove empty array elements

    # remove empty lines
    lines = list(filter(lambda x: x != '', lines))

    # remove all lines with parenthesis
    lines = list(filter(lambda x: '(' not in x, lines))


    # for i in range(len(lines)):
    #     if lines[i] != '' and lines[i] is not None:
    #         lines_new.append(lines[i])
        


    size = len(lines)
    
    for i in range(size):
        line = lines[i]
        if '\x1b' in line:
            row = i
            col = line.index('\x1b')
            return row, col
    
    raise Exception("Could not find player position")