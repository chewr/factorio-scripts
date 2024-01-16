from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from planner import linearize_recipes
from sxp import BaseItem, BaseRecipe, FacilityItem


@dataclass(frozen=True)
class RecipeRequirement:
    recipe_rate_per_machine: float
    machines_required: int
    machine: FacilityItem

    @classmethod
    def of(cls, recipe, planner):
        machine, machines_required = planner.get_machine_requirements(recipe)
        return cls(
            recipe_rate_per_machine=planner.get_recipe_rate(recipe) / machines_required,
            machines_required=int(machines_required),
            machine=machine,
        )

    def to_yaml(self):
        raise NotImplementedError


@dataclass(frozen=True)
class Aisle:
    lanes_out: Dict[BaseItem, float]
    lanes_back: Dict[BaseItem, float]
    machines_out: List[Tuple[BaseRecipe, FacilityItem]]
    machines_back: List[Tuple[BaseRecipe, FacilityItem]]

    def to_yaml(self):
        raise NotImplementedError


@dataclass(frozen=True)
class Layout:
    aisles: List[Aisle]
    belt: Dict[BaseItem, float]
    remaining_recipes: Dict[BaseRecipe, RecipeRequirement]

    def has_remaining_recipes(self):
        return len(self.remaining_recipes) > 0

    @classmethod
    def from_dict(cls, data, factory):
        # TODO also needs production information to rehydrate belt and remaining recipes
        raise NotImplementedError

    def to_yaml(self):
        return {
            "aisles": [aisle.to_yaml() for aisle in self.aisles],
        }

    def weight(self):
        return len(self.aisles)


class Action(ABC):
    @abstractmethod
    def apply(self, layout: Layout) -> Layout:
        pass


class AddLane(Action):
    def apply(self, layout: Layout) -> Layout:
        raise NotImplementedError


class AddMachine(Action):
    def apply(self, layout: Layout) -> Layout:
        raise NotImplementedError


class TerminateAisle(Action):
    def apply(self, layout: Layout) -> Layout:
        raise NotImplementedError


class Strategy(ABC):
    @abstractmethod
    def get_next_action(self, node):
        pass


class LayoutPlanner:
    def __init__(self, production_planner, initial_state=None):
        # TODO: how do I want to take recipes and items_to_recipes as input here?
        if not initial_state:
            initial_state = Layout(
                aisles=[],
                belt_contents={},
                remaining_recipes={
                    recipe: RecipeRequirement.of(recipe, planner=production_planner)
                    for recipe in TODO_recipes
                },
            )
        # Initial state for search
        self._initial_state = initial_state

        # Precompute heuristics
        raise NotImplementedError

    def get_maximal_recipe_significance(self, recipe):
        # Best-case recipe significance: in how many recipes are this recipe's products used?
        raise NotImplementedError

    def get_actual_recipe_significance(self, recipe, existing_production):
        # Actual recipe significance:
        # - Construct a forest of recipe dependency digraphs
        # - Maintain a lookup table to quickly find any node in the forest
        # - To determine the significance of a recipe, find the size of its child tree, terminating the search at any products that are already fully produced
        raise NotImplementedError

    def plan_layout(self, strategy: Strategy) -> Layout:
        layout = self._initial_state
        while layout.has_remaining_recipes():
            action = strategy.get_next_action(layout)
            layout = action.apply(layout)

        return layout
