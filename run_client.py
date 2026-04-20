# run_client.py
import argparse
import torch
from models.simple_model import CustomFashionModel
from utils import load_client_data
import flwr as fl
from custom_client import CustomClient

def main():
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="Run a FL client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--server", type=str, default="[::]:8080", help="Server address")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "data", "model"], help="Type of attack: none, data, model")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = load_client_data(args.cid, data_dir='data', batch_size=args.batch_size)

    # Initialize model
    model = CustomFashionModel().to(DEVICE)

    # Create client instance with attack_type
    client = CustomClient(str(args.cid), model, train_loader, val_loader, DEVICE, epochs=args.epochs, lr=args.lr, attack_type=args.attack_type)

    # Start client
    fl.client.start_client(server_address=args.server, client=client.to_client())

if __name__ == "__main__":
    main()
