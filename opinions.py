from partition import Partition


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
                score = 0
                score -= 100 * len(recipe.outputs)
                score -= len(recipe.ingredients)
                score += recipe.get_yield(item) / recipe.time
                scored_recipes.append((score, recipe))
            preferred_recipes[item] = max(scored_recipes)[1]

        self.items_to_recipes = preferred_recipes
