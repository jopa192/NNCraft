import numpy as np
import os
import json
from nnmodel import NeuralNetwork
from typing import List, Dict, Type
import layers
import losses
import optimizers

class ModelIO:
    @staticmethod
    def save(model: NeuralNetwork, dir_path: str) -> None:
        os.mkdir(dir_path)
        
        model_dict: Dict["str", List[Dict["str", "str"]] | str] = {
            "architecture": ModelIO.get_layers_and_save_parameters(model.layers, dir_path),
            "loss": {"type": type(model.loss).__name__, "params": model.loss.params},
            "optimizer": {"type": type(model.optimizer).__name__, "params": model.optimizer.params}
        }
        
        with open(os.path.join(dir_path, "model_dict.json"), "w") as outfile: 
            json.dump(model_dict, outfile)
        
        
    @staticmethod
    def get_layers_and_save_parameters(model_layers: list, dir_path) -> List[Dict["str", "str"]]:
        layers_list = []
        for i, layer in enumerate(model_layers):
            if type(layer).__name__ == "Dense":
                weights_path = os.path.join(dir_path, f"weights_{i}.npy")
                biases_path = os.path.join(dir_path, f"biases_{i}.npy")
                layers_list.append({"type": "Dense", 
                                    "weights_path": weights_path,
                                    "biases_path": biases_path,
                                    "params": layer.params})
                np.save(weights_path, layer.weights)
                np.save(biases_path, layer.biases)
            else:
                layers_list.append({"type": type(layer).__name__, "params": layer.params})
        return layers_list
        
    @staticmethod
    def load(dir_path: str) -> NeuralNetwork:
        try:
            with open(os.path.join(dir_path, "model_dict.json"), "r") as file:
                model_dict = json.load(file)
        except FileNotFoundError:       
            print(f"Model dictionary {dir_path}.json does not exist.")
        
        layers_list: List[Type[layers.Layer]] = []
        for i, layer in enumerate(model_dict["architecture"]):
            if layer["type"] == "Dense":
                weights = np.load(os.path.join(dir_path, f"weights_{i}.npy"))
                biases = np.load(os.path.join(dir_path, f"biases_{i}.npy"))
                layers_list.append(layers.Dense(**layer["params"], weights=weights, biases=biases))
            else:
                layers_list.append(getattr(layers, layer["type"])(**layer["params"]))
                
        model = NeuralNetwork(layers_list)
        model.config(
            loss_func=getattr(losses, model_dict["loss"]["type"])(**model_dict["loss"]["params"]),
            optimizer=getattr(optimizers, model_dict["optimizer"]["type"])(trainable=model.trainable, **model_dict["optimizer"]["params"])
        )
        return model