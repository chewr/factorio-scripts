def linearize_recipes(available_recipes, base_ingredients, reverse=False):
    linearized_recipes = []
    available_ingredients = base_ingredients.copy()
    q = [r for r in available_recipes if r.ingredients.keys() <= available_ingredients]
    while q:
        cur = q.pop()
        linearized_recipes.append(cur)
        available_ingredients.update(cur.outputs.keys())
        for output in cur.outputs:
            for usage in output.usages.keys() & available_recipes:
                if usage.ingredients.keys() <= available_ingredients:
                    # We are guaranteed to see each producible recipe I times where
                    # I is the number of ingredients in that recipe, as we will
                    # reach it by traversing the edge from each ingredient. We are
                    # guaranteed that exactly once (on the final visit) all
                    # ingredients will be in available_ingredients and we can add it to
                    # our toposort
                    q.append(usage)
    if reverse:
        linearized_recipes.reverse()
    return linearized_recipes


class ProductionPlanner:
    def __init__(self, provided_input, desired_outputs, partition, productivity):
        self._input = provided_input
        self._output = desired_outputs
        self._constraints = partition
        self._machines = productivity

        # Gather recipes
        recipes = set()
        q = [
            self._constraints.items_to_recipes[recipe]
            for recipe in desired_outputs.keys() - provided_input.keys()
        ]
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

        # Topologically sort recipes
        linearized_recipes = linearize_recipes(
            recipes, set(self._input.keys()), reverse=True
        )

        unmet_demand = self._output.copy()
        self.recipe_rates = {}
        self.productivity_by_recipe = {}
        self.machine_requirements = {}
        for recipe in linearized_recipes:
            # Calculate production
            items_requested = {
                itm: unmet_demand[itm]
                for itm in recipe.outputs.keys() & unmet_demand.keys()
            }
            (
                machine_type,
                count,
                recipes_per_second,
                productivity,
            ) = self._machines.get_machine_requirements(recipe, items_requested)

            # Update unmet_demand
            for ingredient, quantity in recipe.ingredients.items():
                if ingredient in self._input:
                    continue
                unmet_demand[ingredient] = (
                    unmet_demand.get(ingredient, 0) + quantity * recipes_per_second
                )
            for item in items_requested.keys():
                unmet_demand[item] -= recipe.get_yield(item, productivity)

            self._unmet_demand = unmet_demand

            # Update statistics
            self.recipe_rates[recipe] = recipes_per_second
            self.productivity_by_recipe[recipe] = productivity
            self.machine_requirements[recipe] = (machine_type, count)

    def get_recipe_rate(self, recipe):
        return self.recipe_rates[recipe]

    def get_productivity(self, recipe):
        return self.productivity_by_recipe[recipe]

    def get_machine_requirements(self, recipe):
        return self.machine_requirements[recipe]
