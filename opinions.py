from partition import Partition

ETA = 1e-10


class OpinionatedPartition(Partition):
    def __init__(self, base_ingredients):
        super().__init__(base_ingredients)
        # Opinion: Maximize productivity, then minimize # of machines
        self.recipes_to_machines = {
            r: max(
                r.producers,  # TODO can we safely limit this to just terrestrially-produced facilities?
                key=lambda m: (m.allows_productivity, m.modules, m.speed),
            )
            for r in self.recipes
        }

        # Opinion: Prefer simple recipes, then fast ones
        preferred_recipes = {}
        for item in self.items:
            if item in base_ingredients:
                continue
            available_recipes = item.recipes.keys() & self.recipes
            scored_recipes = []
            for recipe in available_recipes:
                off_the_bus_percentage = float(
                    len(recipe.ingredients.keys() & base_ingredients)
                ) / (float(len(recipe.ingredients)) + ETA)
                multi_output_penalty = 1.0 / len(recipe.outputs)

                weight = (multi_output_penalty, off_the_bus_percentage)
                scored_recipes.append((weight, recipe))
            preferred_recipes[item] = max(scored_recipes)[1]

        self.items_to_recipes = preferred_recipes
