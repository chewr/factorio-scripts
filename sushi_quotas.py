import yaml

from sxp import Factory, SolidItem

factory = Factory.from_file("./sxp.json")
with open("generated.yml", "r") as f:
    data = yaml.safe_load(f)

inputs = {factory.items[itid]: r for itid, r in data["belt-contents"]["input"].items()}
outputs = {
    factory.items[itid]: r for itid, r in data["belt-contents"]["output"].items()
}

belted_inputs = {itm: r for itm, r in inputs.items() if isinstance(itm, SolidItem)}
belted_outputs = {itm: r for itm, r in outputs.items() if isinstance(itm, SolidItem)}

belt_rate = 45 * 3
belt_capacity = 55000

proportional_input_rates = {itm: r / belt_rate for itm, r in belted_inputs.items()}
quotas = {itm: max(1, r * belt_capacity) for itm, r in proportional_input_rates.items()}

recalculated_proportional_input_rates = {
    itm: q / belt_capacity for itm, q in quotas.items()
}
recalculated_input_rates = {
    itm: rcr * belt_rate for itm, rcr in recalculated_proportional_input_rates.items()
}

print(f"Target Saturation (items/s):        {sum(belted_inputs.values())}")
print(f"Target Saturation (proportion):     {sum(proportional_input_rates.values())}")
print(f"Actual Saturation (items/s):        {sum(recalculated_input_rates.values())}")
print(
    f"Actual Saturation (proportion):     {sum(recalculated_proportional_input_rates.values())}"
)
print(f"Required free space (spaces/s):     {sum(belted_outputs.values())}")
print(
    f"Available free space (spaces/s):    {belt_rate - sum(recalculated_input_rates.values())}"
)

minimax_inserter_throughput = 3.46

recipe_rates = {
    factory.recipes[rid]: rate
    for rid, rate in data["belt-contents"]["recipe-rates"].items()
}
recipe_productivity = {
    factory.recipes[rid]: prod
    for rid, prod in data["belt-contents"]["productivity"].items()
}

for recipe, rate in recipe_rates.items():
    for ingredient, amount in recipe.ingredients.items():
        if not isinstance(ingredient, SolidItem):
            continue
        if amount * rate > minimax_inserter_throughput:
            print(
                f"Ingredient {ingredient} for recipe {recipe} may be inserter-bottlenecked ({amount * rate})"
            )
    for output in recipe.outputs:
        if not isinstance(output, SolidItem):
            continue
        amount = recipe.get_yield(output, productivity=recipe_productivity[recipe])
        if amount * rate > minimax_inserter_throughput:
            print(
                f"Output {output} for recipe {recipe} may be inserter-bottlenecked ({amount * rate})"
            )
