import sys

import yaml

filepath = sys.argv[1]

hard = "--hard" in sys.argv

with open(filepath, "r") as f:
    data = yaml.safe_load(f)

if "belt-contents" in data:
    del data["belt-contents"]

if hard:
    data["staged"] = data["aisles"]
    del data["aisles"]

if "staged" in data:
    staged = data["staged"]
    for aisle in staged:
        if "lane-contents" in aisle:
            del aisle["lane-contents"]
        if "machines" in aisle:
            del aisle["machines"]
    data["staged"] = staged

with open(filepath, "w") as f:
    yaml.safe_dump(data, f)
