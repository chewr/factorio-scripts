import yaml

with open("template.yml", "r") as f:
    data = yaml.safe_load(f)

if "belt-contents" in data:
    del data["belt-contents"]

if "aisles" in data:
    aisles = data["aisles"]
    for aisle in aisles:
        if "lane-contents" in aisle:
            del aisle["lane-contents"]
        if "machines" in aisle:
            del aisle["machines"]
    data["staged"] = aisles
    del data["aisles"]

with open("template.yml", "w") as f:
    yaml.safe_dump(data, f)
