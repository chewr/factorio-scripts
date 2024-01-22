import math


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
                productivity = 1.0 + self._productivity_module.productivity * modules
                speed *= max(
                    1.0 + self._productivity_module.speed * modules, 0.2
                )  # TODO calculate beacon effects before applying the speed penalty cap

        recipes_per_second = max(
            [
                required / recipe.get_yield(item, productivity)
                for item, required in required_output.items()
            ]
        )

        actual_parallelism = recipes_per_second * recipe.time

        machines_required = math.ceil(actual_parallelism / speed)

        return machine, machines_required, recipes_per_second, productivity
