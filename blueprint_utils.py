from base64 import b64decode, b64encode
import zlib
import json


def load_bp(bp_file):
    with open(bp_file, "r") as f:
        return json.loads(zlib.decompress(b64decode(f.read()[1:])))


def save_bp(bp_file, bp_data):
    with open(bp_file, "w") as f:
        json_str = json.dumps(bp_data)
        compressed = zlib.compress(json_str.encode("utf-8"))
        encoded = b64encode(compressed)

        f.write("0" + encoded.decode("utf-8"))


def step(coords, distance, direction):
    vector_map = {
        0: (0, 1),
        2: (-1, 0),
        4: (0, -1),
        6: (1, 0),
    }
    v = vector_map[direction]
    return (coords[0] + distance * v[0], coords[1] + distance * v[1])


def get_coords(entity):
    return (entity["position"]["x"], entity["position"]["y"])


def get_ent_at_pos(pos, bp_data):
    for entity in bp_data["blueprint"]["entities"]:
        if get_coords(entity) == pos:
            return entity
    return None
