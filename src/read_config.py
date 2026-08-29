import json
import os

class Config():
    def __init__(self, configs):
        self.configs = configs
        self._configs = {}
        endpoints = set(x for x in configs)
        y_endpoints = dict()
        for x in configs:
            if not isinstance(configs[x], dict):
                continue
            for y in configs[x]:
                if y in endpoints:
                    continue
                y_endpoints[y] = y_endpoints.get(y, 0) + 1
        for x in configs:
            if not isinstance(configs[x], dict):
                continue
            for y in configs[x]:
                if y_endpoints.get(y, 0) == 1:
                    self._configs[y] = configs[x][y]


    def get(self, x):
        if x in self._configs:
            return self._configs[x]
        return self.configs.get(x)


    def keys(self):
        return [x for x in self._configs] + [x for x in self.configs]


def getConfig():
    config_path = os.path.join(os.path.dirname(__file__), 'configs','config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    print(getConfig())
    config = Config(getConfig())
    print(config)
    print(config.keys())
    print(config.get('network_volume_id'))