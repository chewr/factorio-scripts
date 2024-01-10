from config import Config, load_parameters
from sxp import Factory


def calculate_production(
    factory, bus_inputs, desired_outputs, productivity_module=None
):
    """
    returns a production planner that knows all the recipes, how much
    of each recipe to make, what machine each recipe should be made in,
    and what productivity bonus is assumed

    TODO: likely move this to its own module
    """
    pass


def main(factory_data_file, *args):
    factory = Factory.from_file(factory_data_file)
    parameters = load_parameters("parameters.yml")
    conf = Config(parameters, factory)
    # TODO: we need a lookup for preferred recipes
    production = calculate_production(
        factory,
        conf.bus_inputs,
        conf.base_outputs,
        conf.modules.get("productivity"),
    )


if __name__ == "__main__":
    from sys import argv

    main(*argv[1:])
