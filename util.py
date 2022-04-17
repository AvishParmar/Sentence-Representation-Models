'''
author: Sounak Mondal
'''
import os
import json

import torch
import torch.nn as nn

torch.manual_seed(1337)

def load_pretrained_model(serialization_dir: str, device: str = 'cpu') -> nn.Module:
    """
    Given serialization directory, returns: model loaded with the pretrained weights.
    """

    # Load Config
    config_path = os.path.join(serialization_dir, "config.json")
    model_path = os.path.join(serialization_dir, "model.pkg")

    model_files_present = all([os.path.exists(path)
                               for path in [config_path, model_path]])
    if not model_files_present:
        raise Exception(f"Model files in serialization_dir ({serialization_dir}) "
                        f" are missing. Cannot load_the_model.")

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    config['device'] = device
    # Load Model
    model_name = config.pop("type")
    if model_name == "main":
        from main_model import MainClassifier # To prevent circular imports
        model = MainClassifier(**config)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model"])
    else:
        from probing_model import ProbingClassifier # To prevent circular imports
        model = ProbingClassifier(**config)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["model"])

    return model
