import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from planner import linearize_recipes
from sxp import BaseItem, BaseRecipe, FacilityItem, SolidItem

MACHINES_PER_BANK = 26
MAX_MACHINE_BANKS = 2
SUSHI_BELT_WIDTH = 3
MACHINE_WIDTH = 3
INSERTER_SPACES = 2
LANE_WIDTH = 0.5
MAX_REACHABLE_LANES = 6
ETA = 1e-10
EXPRESS_BELT_LANE_RATE = 22.5
PIPE_RATE = 600
EXPRESS_BELT = "express-belt"
INSERTER_TRAIN = "inserter-train"


class Meter:
    def __init__(self):
        self._value = 0.0
        self._base = 0.0

    def record(self, q):
        self._base += 1.0
        self._value += q

    def mean(self):
        return self._value / self._base

    def total(self):
        return self._value


all_actions_meter = Meter()
nodes_looked_at_meter = Meter()


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
    lanes: List[BaseItem]
    lane_contents: Dict[BaseItem, float]
    machines: List[Tuple[BaseRecipe, FacilityItem]]
    max_allowed_lanes: int
    input_type: str
    max_banks: int

    def get_input_rate(self, item):
        return self._get_input_rate(item, input_type=self.input_type)

    @classmethod
    def _get_input_rate(cls, item, input_type=EXPRESS_BELT):
        allow_fluids, solid_item_rate = {
            EXPRESS_BELT: (True, EXPRESS_BELT_LANE_RATE),
            INSERTER_TRAIN: (False, 1.384),
        }[input_type]
        if not allow_fluids and not isinstance(item, SolidItem):
            return 0

        return solid_item_rate if isinstance(item, SolidItem) else PIPE_RATE / 2

    @classmethod
    def empty(cls):
        return cls(
            lanes=[],
            lane_contents={},
            machines=[],
            max_allowed_lanes=MAX_REACHABLE_LANES,
            input_type=EXPRESS_BELT,
            max_banks=MAX_MACHINE_BANKS,
        )

    def to_yaml(self):
        data = {
            "count": len(self.machines),
            "lanes": sorted([item._id for item in self.lanes]),
            "machines": [
                {"recipe": recipe._id, "machine": machine._id}
                for recipe, machine in self.machines
            ],
            "lane-contents": {
                item._id: amount_remaining
                for item, amount_remaining in self.lane_contents.items()
            },
        }
        if self.max_allowed_lanes != MAX_REACHABLE_LANES:
            data["max-lanes"] = self.max_allowed_lanes
        if self.input_type != EXPRESS_BELT:
            data["input-type"] = self.input_type
        if self.max_banks != MAX_MACHINE_BANKS:
            data["max-banks"] = self.max_banks
        return data

    def get_space(self):
        banks = 1 + len(self.machines) / MACHINES_PER_BANK
        # TODO[extension] this isn't perfectly reflective of reality as there is some rearrangement that can make some
        # single-bank arrangements a bit narrower, but it's good enough for now
        lane_width = math.ceil(len(self.lanes) * LANE_WIDTH)
        if self.input_type == INSERTER_TRAIN:
            lane_width = 2
        return (
            banks * (MACHINE_WIDTH + INSERTER_SPACES)
            + lane_width
            + 2 * SUSHI_BELT_WIDTH
        )

    @classmethod
    def from_yaml(cls, data, factory, planner):
        # TODO[functionality] calculate lanes properly, but we'll need to change the signature first
        # so that we know production statistics
        input_type = data.get("input-type", EXPRESS_BELT)
        lanes = [factory.items[item_id] for item_id in data.get("lanes", [])]
        lane_contents = {}
        for lane in lanes:
            lane_contents[lane] = lane_contents.get(lane, 0) + cls._get_input_rate(
                lane, input_type=input_type
            )
        max_allowed_lanes = data.get("max-lanes", MAX_REACHABLE_LANES)
        machines = [
            (factory.recipes[entry["recipe"]], factory.items[entry["machine"]])
            for entry in data.get("machines", [])
        ]
        for recipe, _ in machines:
            recipe_rate = planner.get_recipe_rate(recipe)
            _, machines_required = planner.get_machine_requirements(recipe)
            recipe_rate_per_machine = recipe_rate / machines_required
            for ingredient, amount in recipe.ingredients.items():
                if ingredient in lane_contents:
                    lane_contents[ingredient] -= recipe_rate_per_machine * amount
                    if lane_contents[ingredient] < 0:
                        raise ValueError(f"Recipe overdraws belt items: {recipe}")
        return cls(
            lanes=lanes,
            lane_contents=lane_contents,
            machines=machines,
            max_allowed_lanes=max_allowed_lanes,
            input_type=input_type,
            max_banks=data.get("max-banks", MAX_MACHINE_BANKS),
        )

    def max_machines(self):
        return self.max_banks * MACHINES_PER_BANK

    def get_actions(self, bus_items, belt_contents, recipe_requirements):
        if len(self.machines) >= self.max_machines():
            return [TerminateAisle()]

        available_items = belt_contents.copy()
        available_items.update(self.lane_contents)
        machine_actions = []
        lanes_to_add = set()
        for recipe, requirements in recipe_requirements.items():
            details, _ = requirements
            item_inputs = {
                ingredient: amount_required * details.recipe_rate_per_machine
                for ingredient, amount_required in recipe.ingredients.items()
            }
            missing_inputs = {
                ingredient
                for ingredient, required_rate in item_inputs.items()
                if required_rate > available_items.get(ingredient, 0) + ETA
            }
            if not missing_inputs:
                machine_actions.append(AddMachine(recipe, details.machine))
            elif (
                missing_inputs <= bus_items
                and len(missing_inputs) + len(self.lanes) <= self.max_allowed_lanes
            ):
                lanes_to_add.update(missing_inputs)

        lane_actions = [AddLane(item) for item in lanes_to_add]

        if self.machines and not machine_actions and not lane_actions:
            # Returning an empty list is meaningful, so in order to ensure
            # convergence we only allow aisle termination if progress has been made
            lanes_to_remove = []
            for item, surplus_rate in self.lane_contents.items():
                unused_lanes = math.floor(surplus_rate / self.get_input_rate(item))
                lanes_to_remove.extend(unused_lanes * [item])
            if lanes_to_remove:
                return [RemoveLanes(lanes_to_remove)]
            return [TerminateAisle()]
        return machine_actions + lane_actions


@dataclass(frozen=True)
class Layout:
    aisles: List[Aisle]

    @classmethod
    def from_yaml(cls, data, factory, planner):
        return cls(
            aisles=[
                Aisle.from_yaml(aisle, factory, planner) for aisle in data["aisles"]
            ]
        )

    def to_yaml(self, planner):
        input_rates = {}
        output_rates = {}
        for aisle in self.aisles:
            for recipe, _ in aisle.machines:
                recipe_rate = planner.get_recipe_rate(recipe)
                _, machines_required = planner.get_machine_requirements(recipe)
                recipe_rate_per_machine = recipe_rate / machines_required
                for ingredient, amount in recipe.ingredients.items():
                    if ingredient in aisle.lanes:
                        continue
                    input_rates[ingredient] = (
                        input_rates.get(ingredient, 0)
                        + recipe_rate_per_machine * amount
                    )
                for output in recipe.outputs:
                    amount = recipe.get_yield(
                        output, productivity=planner.get_productivity(recipe)
                    )
                    output_rates[output] = (
                        output_rates.get(output, 0) + recipe_rate_per_machine * amount
                    )

        return {
            "aisles": [aisle.to_yaml() for aisle in self.aisles],
            "belt-contents": {
                "totals": {
                    "inputs": sum(
                        [
                            q
                            for itm, q in input_rates.items()
                            if isinstance(itm, SolidItem)
                        ]
                    ),
                    "outputs": sum(
                        [
                            q
                            for itm, q in output_rates.items()
                            if isinstance(itm, SolidItem)
                        ]
                    ),
                },
                "input": {item._id: q for item, q in input_rates.items()},
                "output": {item._id: q for item, q in output_rates.items()},
                "productivity": {
                    recipe._id: p
                    for recipe, p in planner.productivity_by_recipe.items()
                },
                "recipe-rates": {
                    recipe._id: p for recipe, p in planner.recipe_rates.items()
                },
            },
        }

    def get_space(self):
        return sum([aisle.get_space() for aisle in self.aisles])

    def get_score(self):
        total_machines = sum([len(aisle.machines) for aisle in self.aisles])
        machine_banks = total_machines / MACHINES_PER_BANK
        full_aisles = machine_banks / 2
        partial_aisles = machine_banks % 2
        best_achievable = full_aisles * (
            2 * (MACHINE_WIDTH + INSERTER_SPACES)
            + math.ceil(MAX_REACHABLE_LANES * LANE_WIDTH)
            + 2 * SUSHI_BELT_WIDTH
        ) + partial_aisles * (
            (MACHINE_WIDTH + INSERTER_SPACES)
            + math.ceil(MAX_REACHABLE_LANES * LANE_WIDTH)
            + 2 * SUSHI_BELT_WIDTH
        )
        # TODO[optimization] Open to having other metrics than space; e.g. simpler lanes
        return round(float(best_achievable) / float(self.get_space()), 3)


@dataclass(frozen=True)
class Node:
    layout: Layout
    bus_items: Set[
        BaseItem
    ]  # TODO[optimization] Does it make sense to carry this with us everywhere?
    belt_contents: Dict[BaseItem, float]
    current_production: Dict[BaseRecipe, float]
    remaining_recipes: Dict[BaseRecipe, Tuple[RecipeDetails, int]]
    current_aisle: Aisle

    def get_layout(self):
        return self.layout

    def has_remaining_recipes(self):
        return len(self.remaining_recipes) > 0

    def is_goal_state(self):
        hanging_aisle = len(self.current_aisle.machines) == 0
        return not (self.has_remaining_recipes() or hanging_aisle)

    @classmethod
    def initial(cls, bus_items, recipes, planner):
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
                    if item in aisle.lanes:
                        continue
                    belt_contents[item] -= consumed * recipe_rate_per_machine
                    if belt_contents[item] < ETA:
                        del belt_contents[item]
                for item in recipe.outputs:
                    belt_contents[item] = (
                        belt_contents.get(item, 0)
                        + recipe.get_yield(item, productivity) * recipe_rate_per_machine
                    )
        starting_layout = Layout(aisles=layout.aisles[:-1])
        current_aisle = layout.aisles[-1]
        return cls(
            layout=starting_layout,
            bus_items=bus_source.keys(),
            belt_contents=belt_contents,
            current_production=current_production,
            remaining_recipes=remaining_recipes,
            current_aisle=current_aisle,
        )

    def get_actions(self):
        actions = self.current_aisle.get_actions(
            self.bus_items, self.belt_contents, self.remaining_recipes
        )
        all_actions_meter.record(len(actions))
        return actions


class Action(ABC):
    @abstractmethod
    def apply(self, node: Node) -> Node:
        pass


class AddLane(Action):
    def __init__(self, item):
        self.item = item

    def apply(self, node: Node) -> Node:
        current_aisle = node.current_aisle
        lanes = current_aisle.lanes.copy()
        lanes.append(self.item)
        lane_contents = current_aisle.lane_contents.copy()
        lane_contents[self.item] = lane_contents.get(
            self.item, 0
        ) + current_aisle.get_input_rate(self.item)
        aisle = Aisle(
            lanes=lanes,
            lane_contents=lane_contents,
            machines=current_aisle.machines,
            max_allowed_lanes=current_aisle.max_allowed_lanes,
            input_type=current_aisle.input_type,
            max_banks=current_aisle.max_banks,
        )
        return Node(
            layout=node.layout,
            bus_items=node.bus_items,
            belt_contents=node.belt_contents,
            current_production=node.current_production,
            remaining_recipes=node.remaining_recipes,
            current_aisle=aisle,
        )


class AddMachine(Action):
    def __init__(self, recipe, facility):
        self.recipe = recipe
        self.facility = facility

    def apply(self, node: Node) -> Node:
        # Update remainig recipes
        remaining_recipes = node.remaining_recipes.copy()
        recipe_details, requirements = remaining_recipes[self.recipe]
        requirements -= 1
        if requirements == 0:
            del remaining_recipes[self.recipe]
        else:
            remaining_recipes[self.recipe] = (recipe_details, requirements)

        # Update belt contents
        productivity = recipe_details.productivity
        belt_contents = node.belt_contents.copy()
        lane_contents = node.current_aisle.lane_contents.copy()
        for item, amount in self.recipe.ingredients.items():
            consumption_rate = amount * recipe_details.recipe_rate_per_machine
            if lane_contents.get(item, 0) > consumption_rate - ETA:
                lane_contents[item] -= consumption_rate
            elif belt_contents.get(item, 0) > consumption_rate - ETA:
                belt_contents[item] -= consumption_rate
                if belt_contents[item] < ETA:
                    del belt_contents[item]
            else:
                raise RuntimeError(f"Unschedulable recipe: {self.recipe}")
        for item in self.recipe.outputs:
            belt_contents[item] = (
                belt_contents.get(item, 0)
                + self.recipe.get_yield(item, productivity)
                * recipe_details.recipe_rate_per_machine
            )

        # Update current production
        current_production = node.current_production.copy()
        current_production[self.recipe] = (
            current_production.get(self.recipe, 0)
            + recipe_details.recipe_rate_per_machine
        )

        aisle_machines = node.current_aisle.machines.copy()
        aisle_machines.append((self.recipe, self.facility))
        aisle = Aisle(
            lanes=node.current_aisle.lanes,
            lane_contents=lane_contents,
            machines=aisle_machines,
            max_allowed_lanes=node.current_aisle.max_allowed_lanes,
            input_type=node.current_aisle.input_type,
            max_banks=node.current_aisle.max_banks,
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
        aisles_updated = current_layout.aisles.copy()
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


class RemoveLanes(Action):
    def __init__(self, lanes):
        self.remove_lanes = lanes

    def apply(self, node):
        lanes = node.current_aisle.lanes.copy()
        lane_contents = node.current_aisle.lane_contents.copy()
        for lane in self.remove_lanes:
            lanes.remove(lane)

            lane_contents[lane] -= node.current_aisle.get_input_rate(lane)
            if lane_contents[lane] < ETA:
                del lane_contents[lane]

        current_aisle = Aisle(
            lanes=lanes,
            lane_contents=lane_contents,
            machines=node.current_aisle.machines,
            max_allowed_lanes=node.current_aisle.max_allowed_lanes,
            input_type=node.current_aisle.input_type,
            max_banks=node.current_aisle.max_banks,
        )
        return Node(
            layout=node.layout,
            bus_items=node.bus_items,
            belt_contents=node.belt_contents,
            current_production=node.current_production,
            remaining_recipes=node.remaining_recipes,
            current_aisle=current_aisle,
        )


class ActionHeuristic(ABC):
    def score_action(self, node: Node, action: Action):
        if isinstance(action, AddLane):
            return self._score_add_lane(node, action)
        elif isinstance(action, AddMachine):
            return self._score_add_machine(node, action)
        elif isinstance(action, TerminateAisle):
            return self._score_terminate(node, action)
        elif isinstance(action, RemoveLanes):
            return self._score_remove(node, action)
        raise TypeError(f"Unrecognized action type: {type(action)}")

    @abstractmethod
    def _score_add_lane(self, node: Node, action: AddLane):
        pass

    @abstractmethod
    def _score_add_machine(self, node: Node, action: AddMachine):
        pass

    @abstractmethod
    def _score_terminate(self, node: Node, action: TerminateAisle):
        pass


class BasicHeuristic(ActionHeuristic):
    def __init__(self, bus_items, recipes, partition):
        # Precompute heuristics

        self._bus_items = set(bus_items)

        # Static significance - how many items depend on a given recipe
        linearized_recipes = linearize_recipes(recipes, self._bus_items, reverse=True)
        static_significance = {}
        for recipe in linearized_recipes:
            # Because the recipes are in reverse toposort, we can be assured
            # that when we see recipe i, we have already seen all recipes which use
            # recipe i. If recipe i has not yet been recorded, it is only relevant
            # to its own product(s)
            static_significance.setdefault(recipe, 1)
            for ingredient in recipe.ingredients:
                if ingredient in bus_items:
                    continue
                source_recipe = partition.items_to_recipes[ingredient]
                static_significance[source_recipe] = (
                    static_significance.get(source_recipe, 1)
                    + static_significance[recipe]
                )
        self._static_significance = static_significance

        # TODO[extension] More heuristic ideas
        # - Dynamic significance, which should discount dependents which can already be achieved with current production
        # - Prioritize recipes which use more of the available lanes in the aisle

    def _score_add_lane(self, node: Node, action: AddLane):
        # TODO[optimization] Improve this rough approximation
        # This current filter is very rough but a filter that looks at the actually achievable recipes based
        # on current state of belt contents and aisle lanes could work. Better yet, it would be nice if the
        # action itself memoized that since we've already computed it
        # It should -1 if no recipes require self.item
        # If recipes require self.item it should return a value that is correlated positively with:
        # - The significance of the recipes
        # - The number of recipes
        # - The number of recipes that require self.item as well as items already in the lane
        achievable_recipes = {}
        immediate_recipes = []
        remaining_recipes_with_this_ingredient = 0
        provisional_ingredients = set(node.current_aisle.lanes) | {action.item}
        for recipe, requirements in node.remaining_recipes.items():
            if action.item not in recipe.ingredients:
                continue
            remaining_recipes_with_this_ingredient += 1
            details, weight = requirements
            base_ingredients = recipe.ingredients.keys() & self._bus_items
            if all(
                [
                    node.belt_contents.get(ingredient, 0)
                    > details.recipe_rate_per_machine * required - ETA
                    for ingredient, required in recipe.ingredients.items()
                    if ingredient not in self._bus_items
                ]
            ):
                if (
                    len(base_ingredients | set(node.current_aisle.lanes))
                    <= node.current_aisle.max_allowed_lanes
                ):
                    achievable_recipes[recipe] = (
                        self._static_significance[recipe],
                        len(base_ingredients & set(node.current_aisle.lanes)),
                        weight,
                    )
                if base_ingredients <= provisional_ingredients:
                    immediate_recipes.append(recipe)
        if not achievable_recipes:
            return ()

        never_see_me_again = (
            1 if len(immediate_recipes) == remaining_recipes_with_this_ingredient else 0
        )
        probably_never_see_me_again = (
            1
            if len(achievable_recipes) == remaining_recipes_with_this_ingredient
            else 0
        )
        most_significance = max(
            [significance for significance, _, __ in achievable_recipes.values()]
        )
        remaining_machine_slots = node.current_aisle.max_machines() - len(
            node.current_aisle.machines
        )
        immediate_payoff = min(
            remaining_machine_slots,
            sum([self._static_significance[recipe] for recipe in immediate_recipes]),
        )
        average_weighted_suitability = sum(
            [suitability**2 for _, suitability, __ in achievable_recipes.values()]
        ) / len(achievable_recipes)
        machine_weight = sum([weight for _, __, weight in achievable_recipes.values()])
        return (
            1,
            1,
            never_see_me_again,
            most_significance,
            immediate_payoff,
            probably_never_see_me_again,
            average_weighted_suitability,
            machine_weight,
        )

    def _score_add_machine(self, node: Node, action: AddMachine):
        significance = self._static_significance[action.recipe]
        uniqueness = len(action.recipe.ingredients.keys() & self._bus_items)
        return (significance, uniqueness)

    def _score_terminate(self, *_):
        return (0,)

    def _score_remove(self, _, action):
        return (1,)


class Strategy(ABC):
    @abstractmethod
    def get_actions(self, node: Node):
        pass


class LayoutPlanner:
    def __init__(self, bus_items, recipes, production_planner, initial_state=None):
        overcomplex_recipes = [
            recipe._id
            for recipe in recipes
            if len(recipe.ingredients.keys() & bus_items) > MAX_REACHABLE_LANES
        ]
        assert (
            len(overcomplex_recipes) == 0
        ), f"All recipes must user no more than {MAX_REACHABLE_LANES} ingredients. The following do not: {overcomplex_recipes}"

        if not initial_state:
            initial_state = Node.initial(bus_items, recipes, production_planner)
        # Initial state for search
        self._initial_state = initial_state

    def plan_layout(self, strategy: Strategy) -> Layout:
        initial_node = self._initial_state

        final_node = self._plan_layout_recursive(initial_node, strategy)

        print(f"Nodes looked at:                    {nodes_looked_at_meter.total()}")
        print(f"Average actions per node:           {all_actions_meter.mean()}")

        return final_node.get_layout()

    @classmethod
    def _plan_layout_recursive(cls, node: Node, strategy: Strategy, branch=False):
        best, best_score = None, 0

        for action in strategy.get_actions(node):
            nodes_looked_at_meter.record(1)
            child = action.apply(node)
            if child.is_goal_state():
                result = child
            else:
                result = cls._plan_layout_recursive(child, strategy)
            if result is not None:
                score = cls._evaluate(result)
                if score > best_score:
                    best = result
                    best_score = score

                if not branch:
                    break

        return best

    @classmethod
    def _evaluate(cls, node):
        return node.layout.get_score()


class BasicStrategy(Strategy):
    # TODO[optimization] Write more strategies
    def get_actions(self, node: Node) -> List[Action]:
        return node.get_actions()


class HeuristicStrategy(BasicStrategy):
    def __init__(self, heuristic):
        super().__init__()
        self._heuristic = heuristic

    def get_actions(self, node: Node) -> List[Action]:
        return sorted(
            super().get_actions(node),
            key=lambda action: self._heuristic.score_action(node, action),
            reverse=True,
        )
