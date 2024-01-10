import factoriolab as fl


class Resource:
    def __init__(self, obj):
        self._id = obj._id
        self.name = obj.name

        self.__data = obj

    def __repr__(self):
        return self._id

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash((self._id, self.name, Resource))

    def __eq__(self, other):
        return self._id == other._id and type(self).__bases__ == type(other).__bases__

    def __lt__(self, other):
        return self._id < other._id

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return self == other or self > other


class BaseItem(Resource):
    def __init__(self, item: fl.Item):
        super().__init__(item)
        self.category = item.category

        self.recipes = {}
        self.usages = {}

    @classmethod
    def of(cls, item):
        bases = [BaseItem]
        if item.stack is not None:
            bases.append(SolidItem)
        if item.fuel is not None:
            bases.append(FuelItem)
        if item.machine is not None:
            bases.append(FacilityItem)
        if item.technology is not None:
            bases.append(TechnologyItem)

        bases.reverse()

        new_class = type("DynamicItem", tuple(bases), {})

        return new_class(item)

    def link_recipe(self, recipe):
        self.link_recipes({recipe._id: recipe})

    def link_recipes(self, recipes):
        for recipe in recipes.values():
            if self in recipe.ingredients:
                self.usages[recipe] = recipe.ingredients[self]
            if self in recipe.outputs:
                self.recipes[recipe] = recipe.outputs[self]

    def __hash__(self):
        return hash((super().__hash__(), BaseItem))


class SolidItem:
    def __init__(self, item: fl.Item):
        super().__init__(item)
        self.stack = item.stack


class FuelItem:
    def __init__(self, item: fl.Item):
        super().__init__(item)
        fuel = item.fuel
        self.fuel_category = fuel.category
        self.fuel_value = fuel.value
        self.fuel_result = fuel.result


class FacilityItem:
    def __init__(self, item: fl.Item):
        super().__init__(item)
        machine = item.machine
        self.speed = machine.speed
        self.allows_productivity = (
            not machine.disallowedEffects
            or "productivity" not in machine.disallowedEffects
        )

        space_machine_ids = {
            "se-space-probe-rocket-silo",
            "se-space-thermodynamics-laboratory",
            "se-space-mechanical-laboratory",
            "se-space-assembling-machine",
            "se-space-manufactory",
            "se-space-biochemical-laboratory",
            "se-space-decontamination-facility",
            "se-space-genetics-laboratory",
            "se-space-growth-facility",
            "se-space-radiation-laboratory",
            "se-space-electromagnetics-laboratory",
            "se-space-laser-laboratory",
            "se-space-material-fabricator",
            "se-space-particle-accelerator",
            "se-space-particle-collider",
            "se-space-plasma-generator",
            "se-space-hypercooler",
            "se-space-radiator",
            "se-space-radiator-2",
            "se-space-astrometrics-laboratory",
            "se-space-gravimetrics-laboratory",
            "se-nexus",
            "se-space-supercomputer-1",
            "se-space-supercomputer-2",
            "se-space-supercomputer-3",
            "se-space-supercomputer-4",
            "se-space-telescope",
            "se-space-telescope-gammaray",
            "se-space-telescope-microwave",
            "se-space-telescope-radio",
            "se-space-telescope-xray",
            "se-space-science-lab",
            "se-fuel-refinery-spaced",
            "se-gate-platform-scaffold",
        }
        self.is_space_only = self._id in space_machine_ids

        terrestrial_machine_ids = {
                "rocket-silo",
                "boiler",
                "burner-mining-drill",
                "offshore-pump",
                "pumpjack",
                "se-core-miner",
                "stone-furnace",
                "steel-furnace",
                "electric-furnace",
                "industrial-furnace",
                "equipment-gantry",
                "equipment-gantry-remover",
                "se-pulveriser",
                "burner-assembling-machine",
                "assembling-machine-1",
                "assembling-machine-2",
                "assembling-machine-3",
                "oil-refinery",
                "chemical-plant",
                "fuel-processor",
                "se-fuel-refinery",
                "burner-lab",
                "lab",
                "se-space-assembling-machine-grounded",
                "se-space-biochemical-laboratory-grounded",
                "se-space-decontamination-facility-grounded",
                "se-space-hypercooler-grounded",
                "se-space-laser-laboratory-grounded",
                "se-space-manufactory-grounded",
                "se-space-mechanical-laboratory-grounded",
                "se-space-particle-accelerator-grounded",
                "se-space-radiation-laboratory-grounded",
                "se-space-radiator-2-grounded",
                "se-space-radiator-grounded",
                "se-space-supercomputer-1-grounded",
                "se-space-supercomputer-2-grounded",
                "se-space-supercomputer-3-grounded",
                "se-space-supercomputer-4-grounded",
                "se-space-thermodynamics-laboratory-grounded",
                "se-energy-transmitter-injector-reactor",
        }

        self.is_terrestrial_only = self._id in terrestrial_machine_ids
        


class TechnologyItem:
    def __init__(self, item: fl.Item):
        super().__init__(item)
        self.prereqs = item.technology.prerequisites or []


class BaseRecipe(Resource):
    def __init__(self, recipe: fl.Recipe):
        super().__init__(recipe)
        self.category = recipe.category
        self.time = recipe.time

        self._proto_ingredients = recipe._in
        self._proto_outputs = recipe.out
        self._proto_producers = set(recipe.producers)

        self.producers = set()
        self.ingredients = {}
        self.outputs = {}

    @classmethod
    def of(cls, recipe: fl.Recipe):
        bases = [BaseRecipe]
        if recipe.catalyst:
            bases.append(CatalyzedRecipe)
        if recipe.isMining:
            bases.append(MiningRecipe)
        if recipe.isTechnology:
            bases.append(TechnologyRecipe)
        bases.reverse()
        new_class = type("DynamicRecipe", tuple(bases), {})
        return new_class(recipe)

    def link_items(self, items):
        self.ingredients.update(
            {items[k]: v for k, v in self._proto_ingredients.items()}
        )
        self.outputs.update({items[k]: v for k, v in self._proto_outputs.items()})

        for item in self.ingredients.keys() | self.outputs.keys():
            item.link_recipe(self)

    def link_facilities(self, facilities):
        self.producers.update(
            [v for f, v in facilities.items() if f in self._proto_producers]
        )

    def is_space_only(self):
        return all([p.is_space_only for p in self.producers])

    def is_terrestrial_only(self):
        return all([p.is_terrestrial_only for p in self.producers])

    def get_yield(self, item, productivity=1.0):
        """
        the amount that comes out minus the amount that came in
        if gross == True, then just the amount that comes out
        """
        # TODO recipes should know if they can be affected by productivity modules
        return productivity * self.outputs.get(item, 0)

    def __hash__(self):
        return hash((super().__hash__(), BaseRecipe))


class CatalyzedRecipe:
    def __init__(self, recipe: fl.Recipe):
        super().__init__(recipe)
        self.category = recipe.category
        self._proto_catalyst = recipe.catalyst
        self.catalyst = {}

    def link_items(self, items):
        super().link_items(items)
        self.catalyst.update({items[k]: v for k, v in self._proto_catalyst.items()})

    def get_yield(self, item, productivity=1.0, gross=False):
        amount_that_comes_out = self.outputs.get(item, 0)
        amount_that_went_in = self.ingredients.get(item, 0)
        eligible_for_bonus = max(0, amount_that_comes_out - self.catalyst.get(item, 0))
        bonus_output = eligible_for_bonus * max(0, productivity - 1)
        result = amount_that_comes_out + bonus_output
        if not gross:
            result -= amount_that_went_in
        return result


class MiningRecipe:
    def is_space_only(self):
        return self._id in {"se-water-ice", "se-methane-ice", "se-naquium-ore"}

    def is_terrestrial_only(self):
        return self._id in {
            "coal",
            "crude-oil",
            "se-cryonite",
            "se-vulcanite",
            "se-vitamelange",
            "se-core-fragment-omni-mining",
            "se-core-fragment-iron-ore-mining",
            "se-core-fragment-copper-ore-mining",
            "se-core-fragment-coal-mining",
            "se-core-fragment-stone-mining",
            "se-core-fragment-uranium-ore-mining",
            "se-core-fragment-crude-oil-mining",
            "se-core-fragment-se-beryllium-ore-mining",
            "se-core-fragment-se-cryonite-mining",
            "se-core-fragment-se-holmium-ore-mining",
            "se-core-fragment-se-iridium-ore-mining",
            "se-core-fragment-se-vulcanite-mining",
            "se-core-fragment-se-vitamelange-mining",
        }


class TechnologyRecipe:
    def __init__(self, item: fl.Item):
        super().__init__(item)


class Factory:
    def __init__(self, data: fl.Data):
        self.items = {item._id: BaseItem.of(item) for item in data.items}
        self.recipes = {recipe._id: BaseRecipe.of(recipe) for recipe in data.recipes}

        self.facilities = {
            k: v for k, v in self.items.items() if isinstance(v, FacilityItem)
        }

        for recipe in self.recipes.values():
            recipe.link_items(self.items)
            recipe.link_facilities(self.facilities)

    @classmethod
    def from_file(cls, fp):
        fl_data = fl.load_factoriolab_data(fp)
        return cls(fl_data)
