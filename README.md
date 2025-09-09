# Cost-Sensitive Supplier Allocation Model

A machine learning solution for optimizing supplier-task assignments to minimize costs and reduce prediction gaps in supplier selection.

## 🎯 Project Overview

The Acme Corporation faces the daily challenge of assigning tasks to one of 64 available suppliers while minimizing associated costs. Given the specific features of each task, each supplier generates different costs, leading to substantial variability in expenses.

This project develops a machine learning model capable of selecting the most cost-effective supplier for a given task, aiming to reduce the gap between predicted costs and actual costs of the best supplier. The solution enables Acme to streamline supplier selection, enhance cost efficiency, and achieve competitive advantage.

## 📊 Dataset Description

### Tasks Dataset
- **130 unique tasks** (T1, T2, ..., T130)
- **116 task-related features** (TF1, TF2, ..., TF116)
- Features describe various characteristics and attributes of tasks

### Suppliers Dataset  
- **64 suppliers** (S1, S2, ..., S64)
- **18 supplier features** (SF1, SF2, ..., SF18)
- Features capture supplier capabilities and characteristics

### Cost Dataset
- Cost data for tasks performed by suppliers
- Values in millions of dollars (M$)
- Each row represents cost of one task performed by a specific supplier

## 🔬 Methodology

### Performance Metrics

The project uses custom metrics to evaluate cost-effectiveness:

**Selection Error (Eq.1):**
```
ε(t) = C(t, s_selected(t)) - C(t, s_optimal(t))
```

**Root Mean Squared Error (Eq.2):**
```
RMSE = √(1/n × Σ[ε(t)]²)
```

Where:
- `t`: Task identified by unique Task ID
- `s`: Supplier identified by unique Supplier ID  
- `S`: Set of all possible suppliers
- `C(t,s)`: Cost of assigning task t to supplier s
- `s_selected(t)`: Supplier selected by ML model for task t
- `s_optimal(t)`: Supplier with minimum cost for task t
- `n`: Number of tasks

### Key Findings

Analysis revealed that few suppliers generate lower costs compared to others, while nearly half exhibit outlier-specific tasks with disproportionately high costs.

## 🤖 Models

Based on exploratory data analysis and residual analysis, two non-linear models were selected:

### 1. Random Forest Regressor
- **Tree-based ensemble model** providing consistent predictions
- Reduces overfitting and manages variability across tasks
- Handles highly correlated task features effectively
- Provides feature importance insights for interpretability
- Robust to heteroscedasticity and outliers

### 2. Multi-Layer Perceptron (MLP) Regressor
- **Neural network model** designed to capture non-linear patterns
- Handles complex, non-linear relationships between predictors and target variable
- Well-suited for intricate interactions between task and supplier features
- Adapts to complexity in supplier selection dynamics

## 📈 Performance Results

| Model | Training Approach | Median Error (±IQR) |
|-------|------------------|---------------------|
| **Baseline** | Cost Dataset | -0.03507 (±0.01435) |
| **Random Forest** | Train/Test Split | -0.01315 (±0.01410) |
| | LOGOCV | -0.01847 (±0.02185) |
| | Grid Search | -0.01512 (±0.02319) |
| **MLP Regressor** | Train/Test Split | -0.01927 (±0.02242) |
| | LOGOCV | -0.01648 (±0.02121) |
| | Grid Search | -0.01765 (±0.02053) |

## 🛠️ Technology Stack

**Programming Language:** Python 3.12

**Key Libraries:**
- **Pandas**: Efficient manipulation of datasets in tabular form
- **NumPy**: Numerical operations and array handling
- **Scikit-Learn**: Data preprocessing, model building and evaluation
- **Matplotlib**: Data visualizations for analysis and insights
- **Helpers**: Custom modules with project-specific utility functions

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.12+
pandas
numpy
scikit-learn
matplotlib
```

### Installation
```bash
git clone https://github.com/your-username/cost-sensitive-supplier-allocation.git
cd cost-sensitive-supplier-allocation
pip install -r requirements.txt
```

## 📋 Project Structure

```
├── data/
│   ├── tasks.csv
│   ├── suppliers.csv
│   └── costs.csv
├── models/
│   └── ...
├── helpers/
│   └── ...
├── eda/
│   └── ...
└── requirements.txt
```

## 🎯 Business Impact

- **Cost Reduction**: Minimize supplier selection costs through optimized assignments
- **Process Efficiency**: Streamline supplier selection workflow
- **Competitive Advantage**: Data-driven supplier management
- **Risk Mitigation**: Reduce cost prediction gaps and variability

## 📧 Contact

For questions or collaboration opportunities, please reach out to the Business Analytics team.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
