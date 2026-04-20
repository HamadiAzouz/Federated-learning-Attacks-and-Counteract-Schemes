# run_server.py
import argparse
import json
import os
import flwr as fl
from custom_client_manager import CustomClientManager
from krum_strategy import KrumStrategy
from typing import Dict, Any

def main():
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Run FL Server")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of FL rounds (large number simulates unlimited)")
    parser.add_argument("--num_clients", type=int, default=10, help="Total number of clients")
    parser.add_argument("--output", type=str, default="results/history.json", help="Output file for results")
    args = parser.parse_args()

    print("[Server] Starting server...")

    # Initialize manager and strategy
    client_manager = CustomClientManager()
    strategy = KrumStrategy(
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients
    )

    # Prepare server config
    server_config = fl.server.ServerConfig(num_rounds=args.num_rounds)

    # Start Flower server
    history = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=server_config,
        strategy=strategy,
        client_manager=client_manager
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "losses_distributed": history.losses_distributed,
            "metrics_distributed_fit": history.metrics_distributed_fit,
            "metrics_distributed": history.metrics_distributed
        }, f)

    print(f"[Server] Training completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()