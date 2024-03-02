import random
import sys

from blueprint_utils import get_coords, get_ent_at_pos, load_bp, save_bp, step
from sxp import Factory


def get_inserters(bp_data):
    for entity in bp_data["blueprint"]["entities"]:
        if not entity.get("name") == "stack-filter-inserter":
            continue
        if not entity.get("control_behavior"):
            continue
        if not entity.get("filter_mode") == "blacklist":
            continue
        if not entity["control_behavior"].get("circuit_condition"):
            continue
        yield entity


def get_associated_assembler(inserter, bp_data):
    x_pos, y_pos = step(get_coords(inserter), -2, inserter.get("direction", 0))
    for entity in bp_data["blueprint"]["entities"]:
        if entity["position"]["x"] == x_pos and entity["position"]["y"] == y_pos:
            return entity


def update_inserter_conditions(bp_data, factory):
    for inserter in get_inserters(bp_data):
        assembler = get_associated_assembler(inserter, bp_data)
        if not assembler:
            continue
        if not assembler["name"] == "assembling-machine-3":
            continue
        if "recipe" not in assembler:
            signal_type = "virtual-item"
            signal = "signal-X"
            continue
        elif assembler["recipe"] not in factory.recipes:
            signal_type = "virtual-item"
            signal = "signal-X"
            inserter["filters"] = [{"index": 1, "name": "deconstruction-planner"}]
            continue
        else:
            recipe = factory.recipes[assembler["recipe"]]
            output_obj = list(recipe.outputs.keys())[0]
            signal_type = "item"
            signal = output_obj._id
        inserter["control_behavior"]["circuit_condition"]["first_signal"][
            "name"
        ] = signal
        inserter["control_behavior"]["circuit_condition"]["first_signal"][
            "type"
        ] = signal_type


def update_inserter_layout(inserter, mode, bp_data):
    target_pos = step(get_coords(inserter), 1, inserter.get("direction", 0))
    target_ent = get_ent_at_pos(target_pos, bp_data)
    if not target_ent:
        return
    if not target_ent.get("name") == "fast-underground-belt":
        return
    if mode == 0:
        inserter["name"] = "fast-inserter"
        return
    elif mode == 1:
        inserter["name"] = "long-handed-inserter"
        return
    elif mode == 2:
        belt_direction = 1 if target_ent["type"] == "input" else -1
        neighbor_pos = step(target_pos, belt_direction, target_ent.get("direction", 0))
        neighbor_ent = get_ent_at_pos(neighbor_pos, bp_data)
        target_ent["position"]["x"], target_ent["position"]["y"] = neighbor_pos
        inserter["position"]["x"], inserter["position"]["y"] = target_pos
        inserter["name"] = "long-handed-inserter"
        return neighbor_ent["entity_number"]


def update_all_inserter_layouts(bp_data):
    ents_to_remove = []
    for inserter in get_inserters(bp_data):
        if "control_behavior" not in inserter:
            continue
        if "circuit_condition" not in inserter["control_behavior"]:
            continue
        if "first_signal" not in inserter["control_behavior"]["circuit_condition"]:
            continue
        if (
            "name"
            not in inserter["control_behavior"]["circuit_condition"]["first_signal"]
        ):
            continue
        if (
            inserter["control_behavior"]["circuit_condition"]["first_signal"]["name"]
            == "signal-X"
        ):
            continue
        mode = random.randint(0, 2)
        remove_neighbor = update_inserter_layout(inserter, mode, bp_data)
        if remove_neighbor:
            ents_to_remove.append(remove_neighbor)
    new_ents = [
        e
        for e in bp_data["blueprint"]["entities"]
        if e.get("entity_number") not in ents_to_remove
    ]
    bp_data["blueprint"]["entities"] = new_ents
    return bp_data


if __name__ == "__main__":
    fp = sys.argv[1]
    factory = Factory.from_file("sxp.json")
    bp_data = load_bp(fp)
    update_inserter_conditions(bp_data, factory)
    bp_data = update_all_inserter_layouts(bp_data)
    save_bp("updated_inserters.bp", bp_data)
