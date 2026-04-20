from utils import generate_distributed_datasets

if __name__ == "__main__":
    generate_distributed_datasets(
        k=10,             # Number of clients
        alpha=10,        # Dirichlet parameter (controls data heterogeneity)
        save_dir="./data" # Save datasets to the "data" folder
    )

