import flwr as fl
import torch
from typing import Callable, Dict, List, Tuple, Union
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes,
    Parameters, FitRes, FitIns, EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from models.simple_model import CustomFashionModel
from torch.utils.data import DataLoader

class CustomClient(fl.client.Client):
    def __init__(
        self,
        cid: str,
        model: CustomFashionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 1,
        lr: float = 0.01,
        attack_type: str = "none"  # none, data, model
    ):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        self.attack_type = attack_type

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties={"cid": self.cid})

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        params = self.model.get_model_parameters()
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(params)
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] Training... (attack_type={self.attack_type})")
        params = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(params)

        for _ in range(self.epochs):
            if self.attack_type == "data":
                loss, accuracy = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device, poison_labels=True
                )
            elif self.attack_type == "model":
                loss, accuracy = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device
                )
            else:
                loss, accuracy = self.model.train_epoch(
                    self.train_loader, self.criterion, self.optimizer, self.device
                )

        updated_params = self.model.get_model_parameters()

        if self.attack_type == "model":
            import numpy as np
            poisoned_params = []
            for p in updated_params:
                noise = np.random.normal(0, 0.01, size=p.shape)
                poisoned = (-p + noise) * 5
                poisoned_params.append(poisoned)
            updated_params = poisoned_params

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=ndarrays_to_parameters(updated_params),
            num_examples=len(self.train_loader.dataset),
            metrics={"loss": loss, "accuracy": accuracy}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] Evaluating...")
        params = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(params)

        loss, accuracy = self.model.test_epoch(self.val_loader, self.criterion, self.device)
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=loss,
            num_examples=len(self.val_loader.dataset),
            metrics={"accuracy": accuracy}
        )

    def to_client(self) -> "CustomClient":
        return self