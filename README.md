# 🛡️ Federated Learning: Attacks & Counteract Schemes – TP3

> Security-focused Federated Learning simulation using **Flower** + **PyTorch**, implementing poisoning attacks and robust aggregation defenses.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red?logo=pytorch)
![Flower](https://img.shields.io/badge/Flower-1.0+-green?logo=flower)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 👥 Authors
Ahmad Dabaja • Caina Figueiredo Pereira • Rachid Elazouzi

---

## 📖 Description
This project explores **security vulnerabilities in Federated Learning (FL)** by implementing and evaluating adversarial attacks alongside robust defense mechanisms. Building on TP1 (FedAvg) and TP2 (FedProx/SCAFFOLD), TP3 focuses on:

🔴 **Attacks**  
- **Data Poisoning**: Label flipping, noise injection to corrupt local training  
- **Model Poisoning**: Gradient scaling, update injection to manipulate global aggregation  

🟢 **Defenses**  
- **FedMedian**: Coordinate-wise median aggregation to resist outliers  
- **Krum**: Distance-based selection of the most "honest" client update  

🔍 **Analysis**  
- Impact of malicious client ratios (0%, 25%, 50%)  
- Trade-offs between robustness and tolerance to data heterogeneity (Dirichlet α ∈ {10, 1, 0.1})  
- Comparative evaluation of FedAvg vs. FedMedian vs. Krum under attack scenarios  

---

## 🎯 Key Objectives
- Implement malicious client behaviors for data/model poisoning  
- Extend Flower's `Strategy` class to integrate FedMedian and Krum  
- Simulate synchronous/asynchronous attack scenarios  
- Analyze convergence, accuracy, and robustness under varying heterogeneity  
- Understand the tension between security and fairness in decentralized learning  

---

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate heterogeneous data (α controls heterogeneity)
python generate_data.py --k 10 --alpha 1.0 --save_dir ./data

# Run simulation with attack + defense config
python run_simulation.py \
  --strategy krum \          # fedavg | fedmedian | krum
  --attack model_poisoning \ # none | data_poisoning | model_poisoning
  --malicious_ratio 0.25 \   # 0.0 | 0.25 | 0.50
  --alpha 1.0 \              # data heterogeneity
  --rounds 30
```

---

## 📁 Core Files
```
├── custom_client.py          # Client with attack modes (none/data/model)
├── strategies/
│   ├── fedavg.py             # Baseline aggregation
│   ├── fedmedian.py          # Median-based robust aggregation
│   └── krum.py               # Distance-based Byzantine-robust aggregation
├── attacks/
│   ├── data_poisoning.py     # Label flipping, noise injection
│   └── model_poisoning.py    # Gradient scaling, update injection
├── run_simulation.py         # Main orchestrator
├── analyze_results.py        # Visualization & comparison tools
└── results/                  # Saved metrics & plots
```

---

## 📊 Experiment Highlights
| Scenario | Metric | FedAvg | FedMedian | Krum |
|----------|--------|--------|-----------|------|
| 0% malicious | Accuracy | ✅ High | ✅ High | ✅ High |
| 25% model poisoning | Accuracy | ❌ Drops | ✅ Stable | ✅ Stable |
| 50% data poisoning + α=0.1 | Convergence | ❌ Fails | ⚠️ Slower | ⚠️ Conservative |

> 💡 **Key Insight**: Robust aggregations protect against attacks but may penalize honest clients under high heterogeneity — a critical design trade-off.

---

## 📄 License
MIT — for educational and research use.

*Built with ❤️ for secure, resilient Federated Learning.* 🌐🔐
