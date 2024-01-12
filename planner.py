from functool import cmp_to_key

def compare_recipes(r1, r2):
    """
    r1 < r2 if r1 must be done before r2
    """
    if r1.outputs.keys() & r2.ingredients.keys():
        return -1
    if r1.ingredients.keys() & r2.outputs.keys():
        return 1
    return 0


class ProductionPlanner:
    def __init__(self, provided_input, desired_outputs, partition, productivity):
        self._input = provided_input
        self._output = desired_outputs
        self._constraints = partition
        self._machines = productivity

        # Gather recipes
        recipes = set()
        q = [self._constraints.items_to_recipes[recipe] for recipe in desired_outputs.keys() - provided_input.keys()]
        seen = set()
        while q:
            cur = q.pop()
            if cur in seen:
                continue
            seen.add(cur)
            recipes.add(cur)
            for ingredient in cur.ingredients:
                if ingredient in self._input:
                    continue
                q.append(self._constraints.items_to_recipes[ingredient])

        # Collect production statistics
        linearized_recipes = sorted(recipes, key=cmp_to_key(compare_recipes), reverse=True)
        unmet_demand = self._outputs.copy()
        self.recipe_rates = {}
        self.productivity_by_recipe = {}
        self.machine_requirements = {}
        for recipe in linearized_recipes:
            # Calculate production
            items_requested = {itm: unmet_demand[itm] for itm in recipe.outputs.keys() & unmet_demand.keys()}
            machine_type, count, recipes_per_second, productivity = self._machines.get_machine_requirements(recipe, items_requested)

            # Update unmet_demand
            for ingredient, quantity in recipe.ingredients:
                if ingredient in self._inputs:
                    continue
                unmet_demand[ingredient] += unmet_demand.get(ingredient, 0) + quantity * recipes_per_second
            for item in items_requested.keys():
                unmet_demand[item] -= recipe.get_yield(item, productivity)

            self._unmet_demand = unmet_demand

            # Update statistics
            self.recipe_rates[recipe] = recipes_per_second
            self.productivity_by_recipe[recipe] = productivity
            self.machine_requirements[recipe] = (machine_type, count)
        
