import numpy as np
import os
import json
from nnmodel import NeuralNetwork

class ModelIO:
    @staticmethod
    def save(model: NeuralNetwork, dir_path: str):
        os.mkdir(dir_path)
        
        model_dict = {
            "architecture": ModelIO.get_layers_and_save_parameters(model.layers, dir_path),
            "loss": type(model.loss).__name__,
            "optimizer": type(model.optimizer).__name__
        }
        
        with open(os.path.join(dir_path, f"{dir_path}.json"), "w") as outfile: 
            json.dump(model_dict, outfile)
        
        
    @staticmethod
    def get_layers_and_save_parameters(layers: list, dir_path):
        layers_list = []
        for i, layer in enumerate(layers):
            if type(layer).__name__ == "Dense":
                weights_path = os.path.join(dir_path, f"weights_{i}.npy")
                biases_path = os.path.join(dir_path, f"biases_{i}.npy")
                layers_list.append({"type": "Dense", 
                                    "weights_path": weights_path,
                                    "biases_path": biases_path})
                np.save(weights_path, layer.weights)
                np.save(biases_path, layer.biases)
            elif type(layer).__name__ == "Dropout":
                layers_list.append({"type": "Dropout", "dropout_rate": layer.dropout_rate})
            else:
                layers_list.append({"type": type(layer).__name__})
        return layers_list
        
        