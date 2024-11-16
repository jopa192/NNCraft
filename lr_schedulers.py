import numpy as np
from typing import Type
from optimizers import Optimizer


class LRScheduler:
    
    def __init__(self, optimizer: Type[Optimizer]):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate
        
    def schedule(self, current_epoch: int):
        raise NotImplementedError("Learning rate schedule should be implemented by subclasses.")
        

class StepDecay(LRScheduler):
    
    def __init__(self, optimizer: Type[Optimizer], gamma: float, step_size: int):
        super().__init__(optimizer)
        self.gamma = gamma
        self.step_size = step_size
        
    def schedule(self, current_epoch: int):
        return self.initial_lr * (self.gamma ** (current_epoch // self.step_size))
    

class ExponentialDecay(LRScheduler):
    
    def __init__(self, optimizer: Type[Optimizer], lambda_: float):
        super().__init__(optimizer)
        self.lambda_ = lambda_
        
    def schedule(self, current_epoch: int):
        return self.initial_lr * np.exp(-self.lambda_ * current_epoch)
    
    
class CosineLR(LRScheduler):
    
    def __init__(self, optimizer: Type[Optimizer], total_epochs: int, min_lr: float):
        super().__init__(optimizer)
        self.T = total_epochs
        self.eta_min = min_lr
        
    def schedule(self, current_epoch: int):
        return self.eta_min + ((self.initial_lr - self.eta_min) / 2) * (1 + np.cos((current_epoch * np.pi) / self.T))
        
    