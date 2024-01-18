import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from planner import linearize_recipes
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
    # TODO[optimization] this doesn't handle having >6 lanes (separate items reachable for top machiens and bottom mochines
    # TODO[extension] we're also not currently handling per-lane throughput or keeping track of lane exhaustion
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
        # TODO[extension] this isn't perfectly reflective of reality as there is some rearrangement that can make some
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

    def get_actions(self, bus_items, belt_contents, recipe_requirements):
        if len(self.machines) >= 2 * MACHINES_PER_BANK:
            return [TerminateAisle()]

        machine_actions = []
        lane_actions = []
        for recipe, requirements in recipe_requirements.items():
            details, _ = requirements
            base_ingredients = recipe.ingredients.keys() & bus_items
            if all(
                [
                    belt_contents.get(ingredient, 0)
                    >= details.recipe_rate_per_machine * required
                    for ingredient, required in recipe.ingredients.items()
                    if ingredient not in bus_items
                ]
            ):
                if base_ingredients <= self.lanes:
                    machine_actions.append(AddMachine(recipe, details.machine))
                elif (base_ingredients | self.lanes) <= MAX_REACHABLE_LANES:
                    lane_actions.extend(
                        [AddLane(item) for item in base_ingredients - self.lanes]
                    )

        if self.machines and not machine_actions and not lane_actions:
            # Returning an empty list is meaningful, so in order to ensure
            # convergence we only allow aisle termination if progress has been made
            return [TerminateAisle()]
        return machine_actions + lane_actions


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
        # TODO[optimization] Open to having other metrics than space; e.g. simpler lanes
        return self.get_space()


@dataclass(frozen=True)
class Node:
    layout: Layout
    bus_items: Set[
        BaseItem
    ]  # TODO[optimization] Does it make sense to carry this with us everywhere?
    belt_contents: Dict[BaseItem, float]
    current_production: Dict[BaseRecipe, float]
    remaining_recipes: Dict[BaseRecipe, Tuple(RecipeDetails, int)]
    current_aisle: Aisle

    def get_layout(self):
        return self._layout

    def has_remaining_recipes(self):
        return len(self.remaining_recipes) > 0

    def is_goal_state(self):
        hanging_aisle = len(self.current_aisle.machines) == 0
        return not (self.has_remaining_recipes() or hanging_aisle)

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
        return self.current_aisle.get_actions(
            self.bus_items, self.belt_contents, self.remaining_recipes
        )


class Action(ABC):
    # TODO[quality] Need a better API on the Action for heuristics
    @abstractmethod
    def apply(self, node: Node) -> Node:
        pass

    @abstractmethod
    # TODO[functionality] figure out what this needs to take
    #   If we assume Action is a pretty finite and well defined
    #   interface (i.e. that there are and will ever be only three
    #   action subtypes), I think it makes sense to write a Heuristic
    #   interface which takes the Action as input and implementst a
    #   specific heuristic function for each type of action. This
    #   makes it a bit hard to directly compare across action types
    #   but if we want to commit to preferring machines over lanes
    #   for instance, this could work well.
    #   Another option is to define a Heuristic interface that
    #   computes the heuristic function on a Node. We should avoid
    #   making a heuristic function which looks too far ahead or is
    #   overly complex in time or space as that is more the purview
    #   of the Strategy
    def score(self, TODO_param):
        return 0


class AddLane(Action):
    def __init__(self, item):
        self.item = item

    def score(self, TODO_param):
        # TODO[functionality] Implement this
        # It needs to know about the required recipes, the current items on belt, and the current lanes
        # It should -1 if no recipes require self.item
        # If recipes require self.item it should return a value that is correlated positively with:
        # - The significance of the recipes
        # - The number of recipes
        # - The number of recipes that require self.item as well as items already in the lane
        raise NotImplementedError

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

    def score(self, TODO_param):
        # TODO[functionality] Implement this
        # It should return a positive number that correlates with the actual or dynamic significance of the recipe
        raise NotImplementedError

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
    def score(self, _):
        return 0

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
    def __init__(
        self, bus_items, recipes, production_planner, partition, initial_state=None
    ):
        # TODO[quality] this is kinda jank looking
        # for one, this could theoretically (it won't with current inputs) break the dependencies as written
        # A totally alternative strategy is to seed the belt with some bus items, but that opens a different can of worms
        assignable_recipes = [
            recipe
            for recipe in recipes
            if len(recipe.ingredients.keys() & bus_items) <= MAX_REACHABLE_LANES
        ]
        # bus ingredients, or seed the belt contents with some ingredients from the bus.
        if not initial_state:
            initial_state = Node.initial(
                bus_items, assignable_recipes, production_planner
            )
        # Initial state for search
        self._initial_state = initial_state

        # Precompute heuristics

        # TODO[extension] Subclass this and move heuristics into subclasses to leverage
        # polymorphism for extending heuristics

        # Static significance - how many items depend on a given recipe
        linearized_recipes = linearize_recipes(assignable_recipes, reverse=True)
        static_significance = {}
        for recipe in linearized_recipes:
            # Because the recipes are in reverse toposort, we can be assured
            # that when we see recipe i, we have already seen all recipes which use
            # recipe i. If recipe i has not yet been recorded, it is only relevant
            # to its own product(s)
            static_significance.setdefault(recipe, 1)
            for ingredient in recipe.ingredients:
                source_recipe = partition.items_to_recipes[ingredient]
                static_significance[source_recipe] = (
                    static_significance.get(source_recipe, 1)
                    + static_significance[recipe]
                )
        self._static_significance = static_significance

        # TODO[extension] More heuristic ideas
        # - Dynamic significance, which should discount dependents which can already be achieved with current production
        # - Prioritize recipes which use more of the available lanes in the aisle

    def get_maximal_recipe_significance(self, recipe):
        return self._static_significance[recipe]

    def plan_layout(self, strategy: Strategy) -> Layout:
        initial_node = self._initial_state

        final_node = self._plan_layout_recursive(initial_node, strategy)

        return final_node.get_layout()

    @classmethod
    def _plan_layout_recursive(cls, node: Node, strategy: Strategy):
        for action in strategy.get_actions(node):
            child = action.apply(node)
            if child.is_goal_state():
                return child
            result = cls._plan_layout_recursive(child, strategy)
            if result is not None:
                return result
        return None


class BasicStrategy(Strategy):
    # TODO[optimization] Write more strategies
    def get_actions(self, node: Node) -> List[Action]:
        return node.get_actions()


class HeuristicStrategy(BasicStrategy):
    def __init__(self, heuristic):
        self._heuristic = heuristic

    def get_actions(self, node: Node) -> List[Action]:
        return sorted(super().get_actions(node), key=self._heuristic.score)
