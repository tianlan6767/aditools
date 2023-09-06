import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        print(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)
