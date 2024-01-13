from config import Config, load_parameters
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

    print("Unmet demands:")
    print(
        [
            k
            for k in plan._unmet_demand
            if partition.items_to_recipes[k] not in plan.recipe_rates
        ]
    )

    print("Partially met demands:")
    print(
        {
            k: v
            for k, v in plan._unmet_demand.items()
            if partition.items_to_recipes[k] in plan.recipe_rates and v > 1e-10
        }
    )
    print("Surplus production")
    print(
        {
            k: v
            for k, v in plan._unmet_demand.items()
            if partition.items_to_recipes[k] in plan.recipe_rates and v < -1e-10
        }
    )


def main(factory_data_file, *args):
    factory = Factory.from_file(factory_data_file)
    parameters = load_parameters("parameters.yml")
    conf = Config.of(parameters, factory)

    plan = calculate_production(factory, conf)


if __name__ == "__main__":
    from sys import argv

    main(*argv[1:])
