from typing import Dict, List
from collections import Counter


class DppJson:
    def __init__(self, data):
        self.new_json = self.convet_json(data)

    def convet_json(self, data):
        new_dict = {}

        def flatten(data):
            for key, value in data.items():
                if isinstance(value, Dict):
                    flatten(value)
                elif isinstance(value, List):
                    if len(value) == 0:
                        if "ok" not in new_dict:
                            new_dict["ok"] = []
                        new_dict["ok"].append(data["filename"])
                    elif isinstance(value[0], int):
                        if key not in new_dict:
                            new_dict[key] = []
                        new_dict[key].append(value)
                    else:
                        for item in value:
                            new_dict["filename"].append(data["filename"])
                            flatten(item)
                else:
                    if key not in new_dict:
                        new_dict[key] = []
                    if key != "filename":
                        new_dict[key].append(value)
        flatten(data)
        return new_dict

    # def __setattr__(self, key, value):
    #     self.__dict__[key] = [value]

    @property
    def keys(self):
        return list(self.new_json.keys())

    @property
    def json_format(self):
        if "size" in self.keys and 'x' in self.keys:
            return "VIA-RECT"
        elif "size" in self.keys and 'all_points_x' in self.keys:
            return "VIA-POLYGON"
        elif "name" in self.keys:
            return "VIA"
        elif "type" in self.keys:
            return "INF"
        elif "all_points_x" in self.keys and 'all_points_y' in self.keys:
            return "JSON"
        elif "size" in self.keys:
            return "VIA"
        else:
            pass

    @property
    def ng_pcs(self):
        return len(set(self.new_json["filename"]))

    @property
    def ok_pcs(self):
        if "ok" in self.keys:
            return len(self.new_json["ok"])
        else:
            return 0

    @property
    def all_pcs(self):
        return self.ng_pcs+self.ok_pcs

    @property
    def instances(self):
        return len(self.new_json["filename"])

    @property
    def labels_dict(self):
        unlabeled_length = len(
            self.new_json["filename"])-len(self.new_json["regions"])
        self.new_json["regions"].extend(["-1"]*unlabeled_length)
        return dict(Counter(self.new_json["regions"]))

    def __str__(self):
        return ""
