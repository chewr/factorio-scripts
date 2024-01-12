import math

"""
from productivity import ProductivityPlanner
from terrestrial import TerrestrialPartition

tp = TerrestrialPartition(factory)

pp = ProductivityPlanner(factory.limitations, tp.recipes_to_machines, cfg.modules.get("productivity"))
"""


class ProductivityPlanner:
    def __init__(self, limitations, recipes_to_machines, productivity_module):
        self._productivity_module = productivity_module
        self._recipes_to_machines = recipes_to_machines
        self._limitations = limitations

    def get_machine_requirements(self, recipe, required_output):
        machine = self._recipes_to_machines[recipe]
        speed = machine.speed
        productivity = 1.0
        if self._productivity_module is not None:
            use_productivity = (
                machine.allows_productivity
                and recipe in self._limitations[self._productivity_module.limitation]
            )
            if use_productivity:
                modules = machine.modules
                productivity = 1.0 + self._productivity_module.speed * modules
                speed *= max(
                    1.0 + self._productivity_module.productivity * modules, 0.2
                )  # TODO calculate beacon effects before applying the speed penalty cap

        actual_parallelism = max(
            [
                required / (recipe.get_yield(item, productivity) / recipe.time)
                for item, required in required_output.items()
            ]
        )
        machines_required = math.ceil(actual_parallelism / speed)
        return machine, machines_required, actual_parallelism, productivity
