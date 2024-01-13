from abc import ABC, abstractmethod


class Partition(ABC):
    def __init__(self, base_ingredients):
        available_ingredients = base_ingredients.copy()
        q = list(
            {
                usage
                for ingredient in base_ingredients
                for usage in ingredient.usages
                if usage.ingredients.keys() <= available_ingredients
                and self.allow_recipe(usage)
            }
        )
        known_recipes = set()
        while q:
            cur = q.pop()
            if cur in known_recipes:
                continue
            known_recipes.add(cur)
            available_ingredients.update(cur.outputs.keys())
            for output in cur.outputs:
                for usage in output.usages:
                    if usage.ingredients.keys() <= available_ingredients:
                        q.append(usage)
        self.recipes = known_recipes
        self.items = available_ingredients

    @abstractmethod
    def allow_recipe(self, recipe):
        pass
