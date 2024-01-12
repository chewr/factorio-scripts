import yaml

"""
from config import Config, load_parameters
from sxp import Factory

factory = Factory.from_file("../factorio/sxp.json")

params = load_parameters("./parameters.yml")
cfg = Config.of(params, factory)
"""


def load_parameters(fp):
    with open(fp, "r") as f:
        return Config.of(yaml.safe_load(f))


class Config:
    def __init__(self, bus_inputs, base_outputs, peak_outputs, modules=None):
        self.bus_inputs = bus_inputs
        self.base_outputs = base_outputs
        self.peak_outputs = peak_outputs
        self.modules = modules

    @classmethod
    def of(cls, data, factory):
        time_unit = data["config"]["time-unit"]
        unit_seconds = {
            "second": 1.0,
            "minute": 60.0,
        }[time_unit]
        return cls(
            bus_inputs={
                factory.items[itid]: q / unit_seconds
                for itid, q in data["bus-inputs"].items()
            },
            base_outputs={
                factory.items[itid]: q / unit_seconds
                for itid, q in data["base-outputs"].items()
            },
            peak_outputs={
                factory.items[itid]: q / unit_seconds
                for itid, q in data["peak-outputs"].items()
            },
            modules={
                module_type: factory.items[itid]
                for module_type, itid in data["config"].get("modules", {})
            },
        )
