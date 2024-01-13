from config import Config, load_parameters
from planner import ProductionPlanner
from productivity import ProductivityPlanner  # TODO rename this guy
from sxp import Factory
from terrestrial import TerrestrialPartition

"""
Usage:
python3 plan_rocket_depot.py ../factorio/sxp.json
"""


def calculate_production(factory, conf):
    """
    returns a production planner that knows all the recipes, how much
    of each recipe to make, what machine each recipe should be made in,
    and what productivity bonus is assumed
    """
    terrestrial_partition = TerrestrialPartition(factory)
    module_manager = ProductivityPlanner(
        factory.limitations,
        terrestrial_partition.recipes_to_machines,
        conf.get_productivity_module(),
    )
    return ProductionPlanner(
        conf.bus_inputs, conf.base_outputs, terrestrial_partition, module_manager
    )


def main(factory_data_file, *args):
    factory = Factory.from_file(factory_data_file)
    parameters = load_parameters("parameters.yml")
    conf = Config.of(parameters, factory)

    plan = calculate_production(factory, conf)


if __name__ == "__main__":
    from sys import argv

    main(*argv[1:])
