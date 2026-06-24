import json

def get_configs():
    with open("./experiments/configs_langgraph.json") as f:
        return json.load(f)["configs"]