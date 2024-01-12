from dataclasses import dataclass
from typing import Dict, List, Optional


def load_factoriolab_data(filepath):
    with open(filepath, "r") as f:
        import json

        return Data(json.load(f))


@dataclass
class Category:
    _id: str
    name: str
    icon: Optional[str]

    def __init__(self, data):
        self._id = data.get("id")
        self.name = data.get("name")
        self.icon = data.get("icon")


@dataclass
class Icon:
    _id: str
    position: str
    color: str

    def __init__(self, data):
        self._id = data.get("id")
        self.position = data.get("position")
        self.color = data.get("color")


@dataclass
class Belt:
    speed: int

    def __init__(self, data):
        self.speed = data.get("speed")


@dataclass
class CargoWagon:
    size: int

    def __init__(self, data):
        self.size = data.get("size")


@dataclass
class FluidWagon:
    capacity: int

    def __init__(self, data):
        self.capacity = data.get("capacity")


@dataclass
class Silo:
    parts: int
    launch: int

    def __init__(self, data):
        self.parts = data.get("parts")
        self.launch = data.get("launch")


@dataclass
class Machine:
    speed: float
    _type: Optional[str]
    fuelCategories: Optional[List[str]]
    usage: Optional[float]
    pollution: Optional[float]
    size: List[int]
    modules: Optional[int]
    disallowedEffects: Optional[List[str]]
    drain: Optional[float]
    silo: Optional[Silo]

    def __init__(self, data):
        self.speed = data.get("speed")
        self._type = data.get("type")
        self.fuelCategories = data.get("fuelCategories")
        self.usage = data.get("usage")
        self.pollution = data.get("pollution")
        self.size = data.get("size")
        self.modules = data.get("modules")
        self.disallowedEffects = data.get("disallowedEffects")
        self.drain = data.get("drain")
        self.silo = Silo(data.get("silo")) if "silo" in data else None


@dataclass
class Beacon:
    effectivity: float
    modules: int
    _range: int
    _type: str
    usage: int
    disallowedEffects: List[str]
    size: List[int]

    def __init__(self, data):
        self.effectivity = data.get("effectivity")
        self.modules = data.get("modules")
        self._range = data.get("range")
        self._type = data.get("type")
        self.usage = data.get("usage")
        self.disallowedEffects = data.get("disallowedEffects")
        self.size = data.get("size")


@dataclass
class Module:
    consumption: float
    pollution: float
    productivity: Optional[float]
    speed: Optional[float]
    limitation: Optional[str]

    def __init__(self, data):
        self.consumption = data.get("consumption")
        self.pollution = data.get("pollution")
        self.productivity = data.get("productivity")
        self.speed = data.get("speed")
        self.limitation = data.get("limitation")


@dataclass
class Fuel:
    category: str
    value: float
    result: Optional[str]

    def __init__(self, data):
        self.category = data.get("category")
        self.value = data.get("value")
        self.result = data.get("result")


@dataclass
class Technology:
    prerequisites: Optional[List[str]]

    def __init__(self, data):
        self.prerequisites = data.get("prerequisites")


@dataclass
class Item:
    _id: str
    name: str
    category: str
    stack: Optional[int]
    row: int
    belt: Optional[Belt]
    cargoWagon: Optional[CargoWagon]
    fluidWagon: Optional[FluidWagon]
    icon: Optional[str]
    machine: Optional[Machine]
    beacon: Optional[Beacon]
    module: Optional[Module]
    fuel: Optional[Fuel]
    iconText: Optional[str]
    technology: Optional[Technology]

    def __init__(self, data):
        self._id = data.get("id")
        self.name = data.get("name")
        self.category = data.get("category")
        self.stack = data.get("stack")
        self.row = data.get("row")
        self.belt = Belt(data.get("belt")) if "belt" in data else None
        self.cargoWagon = (
            CargoWagon(data.get("cargoWagon")) if "cargoWagon" in data else None
        )
        self.fluidWagon = (
            FluidWagon(data.get("fluidWagon")) if "fluidWagon" in data else None
        )
        self.icon = data.get("icon")
        self.machine = Machine(data.get("machine")) if "machine" in data else None
        self.beacon = Beacon(data.get("beacon")) if "beacon" in data else None
        self.module = Module(data.get("module")) if "module" in data else None
        self.fuel = Fuel(data.get("fuel")) if "fuel" in data else None
        self.iconText = data.get("iconText")
        self.technology = (
            Technology(data.get("technology")) if "technology" in data else None
        )


@dataclass
class Recipe:
    _id: str
    name: str
    category: str
    row: int
    time: float
    producers: List[str]
    _in: Dict[str, float]
    out: Dict[str, float]
    unlockedBy: Optional[str]
    icon: Optional[str]
    part: Optional[str]
    catalyst: Dict[str, float]
    isBurn: Optional[bool]
    cost: Optional[int]
    isMining: Optional[bool]
    isTechnology: Optional[bool]
    iconText: Optional[str]

    def __init__(self, data):
        self._id = data.get("id")
        self.name = data.get("name")
        self.category = data.get("category")
        self.row = data.get("row")
        self.time = data.get("time")
        self.producers = data.get("producers")
        self._in = data.get("in")
        self.out = data.get("out")
        self.unlockedBy = data.get("unlockedBy")
        self.icon = data.get("icon")
        self.part = data.get("part")
        self.catalyst = data.get("catalyst", {})
        self.isBurn = data.get("isBurn")
        self.cost = data.get("cost")
        self.isMining = data.get("isMining")
        self.isTechnology = data.get("isTechnology")
        self.iconText = data.get("iconText")


@dataclass
class Defaults:
    beacon: str
    minBelt: str
    maxBelt: str
    fuel: str
    cargoWagon: str
    fluidWagon: str
    excludedRecipes: List[str]
    minMachineRank: List[str]
    maxMachineRank: List[str]
    moduleRank: List[str]
    beaconModule: str

    def __init__(self, data):
        self.beacon = data.get("beacon")
        self.minBelt = data.get("minBelt")
        self.maxBelt = data.get("maxBelt")
        self.fuel = data.get("fuel")
        self.cargoWagon = data.get("cargoWagon")
        self.fluidWagon = data.get("fluidWagon")
        self.excludedRecipes = data.get("excludedRecipes")
        self.minMachineRank = data.get("minMachineRank")
        self.maxMachineRank = data.get("maxMachineRank")
        self.moduleRank = data.get("moduleRank")
        self.beaconModule = data.get("beaconModule")


@dataclass
class Data:
    version: Dict[str, str]
    categories: List[Category]
    icons: List[Icon]
    items: List[Item]
    recipes: List[Recipe]
    limitations: Dict[str, str]
    defaults: Defaults

    def __init__(self, data):
        self.version = data.get("version")
        self.categories = [Category(e) for e in data.get("categories")]
        self.icons = [Icon(e) for e in data.get("icons")]
        self.items = [Item(e) for e in data.get("items")]
        self.recipes = [Recipe(e) for e in data.get("recipes")]
        self.limitations = data.get("limitations")
        self.defaults = Defaults(data.get("defaults"))


__all__ = [
    "Category",
    "Icon",
    "Belt",
    "CargoWagon",
    "FluidWagon",
    "Silo",
    "Machine",
    "Beacon",
    "Module",
    "Fuel",
    "Technology",
    "Item",
    "Recipe",
    "Defaults",
    "Data",
    "load_factoriolab_data",
]
