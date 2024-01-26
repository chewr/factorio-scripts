import sys

import yaml

filepath = sys.argv[1]

hard = "--hard" in sys.argv

with open(filepath, "r") as f:
    data = yaml.safe_load(f)

if "belt-contents" in data:
    del data["belt-contents"]

if hard and "aisles" in data:
    data["staged"] = data["aisles"]
    del data["aisles"]

if "staged" in data:
    staged = data["staged"].copy()
    for aisle in staged:
        if "count" in aisle:
            del aisle["count"]
        if "lane-contents" in aisle:
            del aisle["lane-contents"]
        if "machines" in aisle:
            del aisle["machines"]
    del data["staged"]
    data["staged"] = staged

with open(filepath, "w") as f:
    yaml.safe_dump(data, f)
