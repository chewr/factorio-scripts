import math

class ProductivityPlanner:
    def __init__(self, recipes_to_machines, productivity_module):
        self._productivity_module = productivity_module
        self._recipes_to_machines = recipes_to_machines

    def get_machine_requirements(self, recipe, required_output):
        machine = self._recipes_to_machines[recipe]
        speed = machine.speed
        productivity = 1.0
        if recipe_uses_productivity and self._productivity_module is not None:  # TODO figure out if the recipe uses productivity
            modules = machine.modules
            productivity = 1.0 + self._productivity_module.speed * modules
            speed *= max(1.0 + self._productivity_module.productivity * modules, 0.2)  # TODO calculate beacon effects before capping the speed penalty

        minimum_parallelism = max([required / (recipe.get_yield(item, productivity)/recipe.time) for item, required in required_output.items()])
        count = max(1.0, math.ceil(minimum_parallelism / speed))
        return machine, count, productivity
