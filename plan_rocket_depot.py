from config import Config, load_parameters
from layout import BasicHeuristic, HeuristicStrategy, Layout, LayoutPlanner, Node
from opinions import OpinionatedPartition
from planner import ProductionPlanner
from productivity import ProductivityPlanner  # TODO rename this guy
from sxp import Factory

"""
Usage:
python3 plan_rocket_depot.py ../factorio/sxp.json
"""


class RocketDepotPartition(OpinionatedPartition):
    def allow_recipe(self, recipe):
        return not recipe.is_space_only()


def calculate_production(factory, conf):
    from datetime import datetime
    from pathlib import Path

    import yaml

    """
    returns a production planner that knows all the recipes, how much
    of each recipe to make, what machine each recipe should be made in,
    and what productivity bonus is assumed
    """
    partition = RocketDepotPartition(set(conf.bus_inputs.keys()))
    module_manager = ProductivityPlanner(
        factory.limitations,
        partition.recipes_to_machines,
        conf.get_productivity_module(),
    )
    plan = ProductionPlanner(
        conf.bus_inputs, conf.base_outputs, partition, module_manager
    )
    print(
        f"Machines required: {sum([v[1] for v in plan.machine_requirements.values()])}"
    )
    in_layout_fp = Path("./generated.yml")
    out_layout_fp = Path("./generated.yml")
    if in_layout_fp.exists():
        print(f"Reading from {in_layout_fp}")
        with open(in_layout_fp, "r") as f:
            data = yaml.safe_load(f)
        initial_node = Node.from_layout(
            conf.bus_inputs,
            plan,
            plan.recipe_rates.keys(),
            Layout.from_yaml(data, factory, plan),
        )
        searcher = LayoutPlanner(
            conf.bus_inputs.keys(),
            plan.recipe_rates.keys(),
            plan,
            initial_state=initial_node,
        )
    else:
        searcher = LayoutPlanner(conf.bus_inputs.keys(), plan.recipe_rates.keys(), plan)
    heuristic = BasicHeuristic(
        set(conf.bus_inputs.keys()), plan.recipe_rates.keys(), partition
    )
    strategy = HeuristicStrategy(heuristic)

    start = datetime.now()
    layout = searcher.plan_layout(strategy)
    duration = datetime.now() - start
    with open(out_layout_fp, "w") as f:
        yaml.safe_dump(layout.to_yaml(plan), f)
    print(
        f"Found layout in {duration.total_seconds()} seconds. Score: {layout.get_score()}. See {out_layout_fp}"
    )


def create_sushi_layout(
    base_inputs, desired_outputs, production_plan, layout_file=None, interactive=True
):
    """
    TODO:
    - Create a layout planner class
      -
    - Create a few decier adapters:
      - Interactive input, which prompts on the command line for decisions
      - file input which reads a file for answers to decision prompts
      - automated decider which follows some heuristic to make decisions
    """
    pass


def main(factory_data_file, *args):
    factory = Factory.from_file(factory_data_file)
    parameters = load_parameters("parameters.yml")
    conf = Config.of(parameters, factory)

    calculate_production(factory, conf)


if __name__ == "__main__":
    from sys import argv

    main(*argv[1:])
