import json

def read_file(filename):
    with open(filename, 'r') as json_file:
        loaded_dict = json.load(json_file)
        
    return loaded_dict
    

