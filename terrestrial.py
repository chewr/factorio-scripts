from opinions import OpinionatedPartition
from sxp import FacilityItem, MiningRecipe

"""
from terrestrial import TerrestrialPartition

tp = TerrestrialPartition(factory)
"""


def find_terrestrial(factory):
    terrestrial_recipes = {
        recipe
        for recipe in factory.recipes.values()
        if isinstance(recipe, MiningRecipe) and not recipe.is_space_only()
    }
    terrestrial_items = {
        output for recipe in terrestrial_recipes for output in recipe.outputs.keys()
    }

    q = terrestrial_recipes.copy()
    seen = set()
    while q:
        cur = q.pop()
        if cur in seen:
            continue
        seen.add(cur)
        terrestrial_items.update(cur.outputs.keys())
        for output in cur.outputs.keys():
            for recipe in output.usages.keys():
                if not recipe.ingredients.keys() <= terrestrial_items:
                    # We haven't encountered all the ingredients yet
                    # If this is a fully terrestrial recipe we will
                    # eventually see it again at a point when we have
                    # encountered all ingredients
                    continue
                if all([p.is_space_only for p in recipe.producers]):
                    # This recipe can only be produced in space
                    continue
                # Any recipes at this point can be produced terrestrially
                # from terrestrial ingredients
                terrestrial_recipes.add(recipe)
                q.add(recipe)
    return terrestrial_recipes, terrestrial_items


class TerrestrialPartition(OpinionatedPartition):
    def __init__(self, factory):
        self.recipes, self.items = find_terrestrial(factory)
        self.facilities = {f for f in self.items if isinstance(f, FacilityItem)}
        super().__init__()
