# run_all_clients.py
import argparse
import subprocess
import sys
import time

NUM_CLIENTS = 10  # Change this if you want a different number of clients
CLIENT_SCRIPT = "run_client.py"
SERVER_ADDR = "127.0.0.1:8080"

# New: Add arguments for malicious ratio and attack type
parser = argparse.ArgumentParser(description="Run all clients with poisoning options")
parser.add_argument("--malicious_ratio", type=float, default=0.0, help="Fraction of clients that are malicious (0.0, 0.25, 0.5)")
parser.add_argument("--attack_type", type=str, default="none", choices=["none", "data", "model"], help="Type of attack: none, data, model")
args = parser.parse_args()

processes = []
num_malicious = int(NUM_CLIENTS * args.malicious_ratio)
malicious_cids = set(range(NUM_CLIENTS - num_malicious, NUM_CLIENTS))  # Last N clients are malicious

for cid in range(NUM_CLIENTS):
    attack = args.attack_type if cid in malicious_cids else "none"
    cmd = [sys.executable, CLIENT_SCRIPT, "--cid", str(cid), "--server", SERVER_ADDR, "--attack_type", attack]
    print(f"Starting client {cid} (attack_type={attack})...")
    # Start each client in a new process
    p = subprocess.Popen(cmd)
    processes.append(p)
    time.sleep(0.5)  # Stagger startup to avoid race conditions

print(f"Started {NUM_CLIENTS} clients. Press Ctrl+C to stop all.")

try:
    # Wait for all clients to finish
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("\nStopping all clients...")
    for p in processes:
        p.terminate()
    print("All clients stopped.")
