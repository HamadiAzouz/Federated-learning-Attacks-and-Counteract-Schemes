from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

class KrumStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 10,
        min_evaluate_clients: int = 10,
        min_available_clients: int = 10,
        f: int = 1,  # Number of Byzantine (malicious) clients to tolerate
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.f = f

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        print(f"Waiting for {self.min_available_clients} clients before initializing parameters...")
        client_manager.wait_for(self.min_available_clients, timeout=None)
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size = max(
            int(client_manager.num_available() * self.fraction_fit), self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        print(f"Round {server_round}: {len(clients)} clients sampled for training")
        config = {}
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            print(f"Round {server_round}: No results to aggregate")
            return None, {}
        if failures:
            print(f"Round {server_round}: {len(failures)} clients failed during training")
        # Collect all client model updates as lists of NumPy arrays
        params_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        # Flatten each update into a single vector
        flat_updates = [np.concatenate([p.flatten() for p in params]) for params in params_list]
        n = len(flat_updates)
        f = self.f
        # Compute pairwise Euclidean distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(flat_updates[i] - flat_updates[j])
                dists[i, j] = dists[j, i] = d
        # For each client, sum the distances to its n-f-2 closest neighbors
        scores = []
        for i in range(n):
            sorted_dists = np.sort(dists[i])
            score = np.sum(sorted_dists[1 : n - f - 1])  # skip self (0), sum n-f-2 closest
            scores.append(score)
        # Select the update with the lowest score
        krum_idx = int(np.argmin(scores))
        print(f"Round {server_round}: Krum selected client {krum_idx} as the global model.")
        selected_params = params_list[krum_idx]
        updated_params = ndarrays_to_parameters(selected_params)
        # Aggregate metrics (mean for reporting)
        metrics_aggregated = {}
        for _, fit_res in results:
            for key, value in fit_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = [value]
                else:
                    metrics_aggregated[key].append(value)
        metrics_avg = {key: float(np.mean([float(v) for v in values])) for key, values in metrics_aggregated.items()}
        metrics_avg["round"] = server_round
        print(f"Round {server_round}: Training aggregated from {len(results)} clients (Krum)")
        return updated_params, metrics_avg

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        sample_size = max(
            int(client_manager.num_available() * self.fraction_evaluate),
            self.min_evaluate_clients,
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )
        print(f"Round {server_round}: {len(clients)} clients sampled for evaluation")
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            print(f"Round {server_round}: No evaluation results to aggregate")
            return None, {}
        if failures:
            print(f"Round {server_round}: {len(failures)} clients failed during evaluation")
        loss_results = [
            (evaluate_res.loss, evaluate_res.num_examples)
            for _, evaluate_res in results
        ]
        total_examples = sum([num_examples for _, num_examples in loss_results])
        weighted_loss = sum(
            [loss * num_examples / total_examples for loss, num_examples in loss_results]
        )
        metrics_aggregated = {}
        for _, evaluate_res in results:
            for key, value in evaluate_res.metrics.items():
                if key not in metrics_aggregated:
                    metrics_aggregated[key] = [value]
                else:
                    metrics_aggregated[key].append(value)
        metrics_avg = {key: float(np.mean([float(v) for v in values])) for key, values in metrics_aggregated.items()}
        metrics_avg["round"] = server_round
        print(f"Round {server_round}: Evaluation aggregated from {len(results)} clients (Krum)")
        return weighted_loss, metrics_avg

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print(f"Round {server_round}: Evaluating global model (Krum)")
        return None
