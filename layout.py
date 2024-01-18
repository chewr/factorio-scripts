import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from sxp import BaseItem, BaseRecipe, FacilityItem

MACHINES_PER_BANK = 26
SUSHI_BELT_WIDTH = 3
MACHINE_WIDTH = 3
SPACES = 2
LANE_WIDTH = 0.5
MAX_REACHABLE_LANES = 6


@dataclass(frozen=True)
class RecipeDetails:
    recipe_rate_per_machine: float
    productivity: float
    machine: FacilityItem

    @classmethod
    def of(cls, recipe, planner):
        machine, machines_required = planner.get_machine_requirements(recipe)
        productivity = planner.get_productivity(recipe)
        return cls(
            recipe_rate_per_machine=planner.get_recipe_rate(recipe) / machines_required,
            machine=machine,
            productivity=productivity,
        )


@dataclass(frozen=True)
class Aisle:
    # [optimization]TODO this doesn't handle having >6 lanes (separate items reachable for top machiens and bottom mochines
    # [extension]TODO we're also not currently handling per-lane throughput or keeping track of lane exhaustion
    lanes: Set[BaseItem]
    machines: List[Tuple[BaseRecipe, FacilityItem]]

    @classmethod
    def empty(cls):
        return cls(
            lanes=set(),
            machines=[],
        )

    def to_yaml(self):
        return {
            "lanes": [item._id for item in self.lanes],
            "machines": [
                {"recipe": recipe._id, "machine": machine._id}
                for recipe, machine in self.machines
            ],
        }

    def get_space(self):
        banks = 1 + len(self.machines) / MACHINES_PER_BANK
        # [extension]TODO this isn't perfectly reflective of reality as there is some rearrangement that can make some
        # single-bank arrangements a bit narrower, but it's good enough for now
        return (
            banks * (MACHINE_WIDTH + SPACES)
            + math.ceil(len(self.lanes) * LANE_WIDTH)
            + 2 * SUSHI_BELT_WIDTH
        )

    @classmethod
    def from_yaml(cls, data, factory):
        return cls(
            lanes={factory.items[item_id] for item_id in data["lanes"]},
            machines=[
                (entry["recipe"], entry["machine"]) for entry in data["machines"]
            ],
        )

    def get_actions(self, bus_items):
        actions = []
        # [optimization] TODO: include TerminateAisle in fewer cases
        # Confusingly we could choose to handle this at the MachineAction level by deleting
        # those items from the bus
        if self.machines:
            actions.append(TerminateAisle())
        if len(self.lanes) >= MAX_REACHABLE_LANES:
            return actions
        # [optimization] TODO: consider only the lanes which can be used in remaining recipes
        actions.extend([AddLane(item) for item in bus_items - self.lanes])
        return actions


@dataclass(frozen=True)
class Layout:
    aisles: List[Aisle]

    @classmethod
    def from_yaml(cls, data, factory):
        return cls(aisles=[Aisle.from_yaml(aisle, factory) for aisle in data])

    def to_yaml(self):
        return {
            "aisles": [aisle.to_yaml() for aisle in self.aisles],
        }

    def get_space(self):
        return sum([aisle.get_space() for aisle in self.aisles])

    def get_score(self):
        # [optimization]TODO Open to having other metrics than space; e.g. simpler lanes
        return self.get_space()


@dataclass(frozen=True)
class Node:
    layout: Layout
    bus_items: Set[
        BaseItem
    ]  # [optimization]TODO Does it make sense to carry this with us everywhere?
    belt_contents: Dict[BaseItem, float]
    current_production: Dict[BaseRecipe, float]
    remaining_recipes: Dict[
        BaseRecipe, Tuple(RecipeDetails, int)
    ]  # TODO gotta update usages
    current_aisle: Aisle

    def get_layout(self):
        return self._layout

    def has_remaining_recipes(self):
        return len(self.remaining_recipes) > 0

    @classmethod
    def initial(cls, bus_items, planner, recipes):
        return cls(
            layout=Layout(aisles=[]),
            bus_items=bus_items,
            belt_contents={},
            current_production={},
            remaining_recipes={
                recipe: (
                    RecipeDetails.of(recipe, planner),
                    planner.get_machine_requirements(recipe)[1],
                )
                for recipe in recipes
            },
            current_aisle=Aisle.empty(),
        )

    @classmethod
    def from_layout(cls, bus_source, planner, recipes, layout):
        belt_contents = {}
        current_production = {}
        remaining_recipes = {
            recipe: (
                RecipeDetails.of(recipe, planner),
                planner.get_machine_requirements(recipe)[1],
            )
            for recipe in recipes
        }
        for aisle in layout.aisles:
            for recipe, actual_machine in aisle.machines:
                productivity = planner.get_productivity(recipe)
                desired_machine, machines_required = planner.get_machine_requirements(
                    recipe
                )
                if desired_machine != actual_machine:
                    raise ValueError(
                        f"Got unexpected machine type for recipe {recipe}: {actual_machine} != {desired_machine}"
                    )
                recipe_rate_per_machine = (
                    planner.get_recipe_rate(recipe) / machines_required
                )

                # update current_production
                current_production[recipe] = (
                    current_production.get(recipe, 0) + recipe_rate_per_machine
                )

                # Deduct from remaining_recipes
                recipe_details, current_requirements = remaining_recipes[recipe]
                updated_requirements = current_requirements - 1
                if updated_requirements == 0:
                    del remaining_recipes[recipe]
                else:
                    remaining_recipes[recipe] = (recipe_details, updated_requirements)

                # Update belt contents
                for item, consumed in recipe.ingredients.items():
                    if item in bus_source:
                        continue
                    belt_contents[item] -= consumed * recipe_rate_per_machine
                for item in recipe.outputs:
                    belt_contents[item] = (
                        belt_contents.get(item, 0)
                        + recipe.get_yield(item, productivity) * recipe_rate_per_machine
                    )
        return cls(
            layout=layout,
            bus_items=bus_source.keys(),
            belt_contents=belt_contents,
            current_production=current_production,
            remaining_recipes=remaining_recipes,
            current_aisle=Aisle.empty(),
        )

    def get_actions(self):
        # [optimization]TODO Does it make sense to compute next actions here or in the monolith?
        lane_actions = self.current_aisle.get_actions(self.bus_items)

        # [optimization]TODO Does it make sense to precompute/save the recipe options at all?
        machine_actions = []
        available_items = self.current_aisle.lanes | self.belt_contents.keys()
        for recipe, requirements in self.remaining_recipes.items():
            if recipe.ingredients.keys() <= available_items:
                if all(
                    [
                        amount_required * requirements.recipe_rate_per_machine
                        < self.belt_contents[ingredient]
                        for ingredient, amount_required in recipe.ingredients.items()
                        if ingredient in self.belt_contents
                    ]
                ):
                    machine_actions.append(AddMachine(recipe, requirements.machine))
        return lane_actions + machine_actions


class Action(ABC):
    @abstractmethod
    def apply(self, node: Node) -> Node:
        pass


class AddLane(Action):
    def __init__(self, item):
        self.item = item

    def apply(self, node: Node) -> Node:
        current_aisle = node.aisle
        lanes_updated = deepcopy(current_aisle.lanes)
        lanes_updated.add(self.item)
        aisle_updated = Aisle(
            lanes=lanes_updated,
            machines=current_aisle.machines,
        )
        return Node(
            layout=node.layout,
            bus_items=node.bus_items,
            belt_contents=node.belt_contents,
            current_production=node.current_production,
            remaining_recipes=node.remaining_recipes,
            current_aisle=aisle_updated,
        )


class AddMachine(Action):
    def __init__(self, recipe, facility):
        self.recipe = recipe
        self.facility = facility

    def apply(self, node: Node) -> Node:
        recipe_details = node.remaining_recipes[self.recipe]
        productivity = recipe_details.productivity

        belt_contents = deepcopy(node.belt_contents)
        for item, amount in self.recipe.ingredients.items():
            belt_contents[item] -= amount * recipe_details.recipe_rate_per_machine
        for item in self.recipe.outputs:
            belt_contents[item] = (
                belt_contents.get(item, 0)
                + self.recipe.get_yield(item, productivity)
                * recipe_details.recipe_rate_per_machine
            )

        remaining_recipes = deepcopy(node.remaining_recipes)
        recipe_details, requirements = remaining_recipes[self.recipe]
        requirements -= 1
        if requirements == 0:
            del remaining_recipes[self.recipe]
        else:
            remaining_recipes[self.recipe] = (recipe_details, requirements)

        current_production = deepcopy(node.current_production)
        current_production[self.recipe] = (
            current_production.get(self.recipe, 0)
            + recipe_details.recipe_rate_per_machine
        )

        aisle_machines = deepcopy(node.aisle.machines)
        aisle_machines.append((self.recipe, self.facility))
        aisle = Aisle(
            lanes=node.aisle.lanes,
            machines=aisle_machines,
        )
        return Node(
            layout=node.layout,
            bus_items=node.bus_items,
            belt_contents=belt_contents,
            current_production=current_production,
            remaining_recipes=remaining_recipes,
            current_aisle=aisle,
        )


class TerminateAisle(Action):
    def apply(self, node: Node) -> Node:
        current_layout = node.layout
        aisles_updated = deepcopy(current_layout.aisles)
        aisles_updated.append(node.current_aisle)
        layout_updated = Layout(
            aisles=aisles_updated,
        )
        return Node(
            layout=layout_updated,
            bus_items=node.bus_items,
            belt_contents=node.belt_contents,
            current_production=node.current_production,
            remaining_recipes=node.remaining_recipes,
            current_aisle=Aisle.empty(),
        )


class Strategy(ABC):
    @abstractmethod
    def get_next_action(self, node: Node):
        pass


class LayoutPlanner:
    def __init__(self, bus_items, recipes, production_planner, initial_state=None):
        # [functionality] TODO: To make sure that we can terminate, we should avoid recipes with > MAX_REACHABLE_LANES
        # bus ingredients, or seed the belt contents with some ingredients from the bus.
        if not initial_state:
            initial_state = Node.initial(bus_items, recipes, production_planner)
        # Initial state for search
        self._initial_state = initial_state

        # Precompute heuristics
        raise NotImplementedError

    def get_maximal_recipe_significance(self, recipe):
        # Best-case recipe significance: in how many recipes are this recipe's products used?
        # [functionality]TODO: precompute heuristic
        raise NotImplementedError

    def get_actual_recipe_significance(self, recipe, existing_production):
        # Actual recipe significance:
        # - Construct a forest of recipe dependency digraphs
        # - Maintain a lookup table to quickly find any node in the forest
        # - To determine the significance of a recipe, find the size of its child tree, terminating the search at any products that are already fully produced
        # [functionality]TODO: precompute heuristic
        raise NotImplementedError

    def plan_layout(self, strategy: Strategy) -> Layout:
        node = self._initial_state
        while node.has_remaining_recipes():
            action = strategy.get_next_action(node)
            node = action.apply(node)

        return node.get_layout()
