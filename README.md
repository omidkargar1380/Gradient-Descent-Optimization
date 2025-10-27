# 🧠 Gradient Descent Optimization: Comparative Analysis on California Housing Dataset

## 📊 Project Overview

This project presents a comprehensive implementation and comparison of **three fundamental gradient descent optimization algorithms** for linear regression — **Batch Gradient Descent (BGD)**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent (MBGD)** — applied to the **California Housing Dataset**.

The goal is to analyze convergence behavior, computational efficiency, and trade-offs between stability and speed among these methods.

---

## 🏠 Dataset Description

**Source**: `California Housing Dataset` from `scikit-learn`

- **Samples**: 20,640  
- **Features**: 8 numerical and geographical attributes  
  - `MedInc`: Median income in block group  
  - `HouseAge`: Median house age in block group  
  - `AveRooms`: Average number of rooms per household  
  - `AveBedrms`: Average number of bedrooms per household  
  - `Population`: Block group population  
  - `AveOccup`: Average number of household members  
  - `Latitude`: Block group latitude  
  - `Longitude`: Block group longitude  
- **Target**: Median house value for California districts  
- **Preprocessing**: Features scaled using `StandardScaler` for normalization  

---

## ⚙️ Algorithms Implemented

### 1. Batch Gradient Descent (BGD)
- **Update**: Uses the entire dataset for each parameter update  
- **Learning Rate**: 0.01 (fixed)  
- **Iterations**: 1,000  
- **Pros**: Stable, smooth convergence  
- **Cons**: Computationally expensive  

### 2. Stochastic Gradient Descent (SGD)
- **Update**: Single random sample per update  
- **Learning Rate Schedule**: η = 5 / (50 + t)  
- **Epochs**: 50  
- **Pros**: Fast convergence  
- **Cons**: Noisy and less stable  

### 3. Mini-Batch Gradient Descent (MBGD)
- **Update**: Small random batches (size = 20)  
- **Epochs**: 50  
- **Pros**: Balanced between BGD and SGD  
- **Cons**: Requires tuning batch size and learning rate  

---

## 🧩 Technical Implementation

### Model Architecture
- **Type**: Linear Regression  
- **Parameters**: θ₀ (bias) + θ₁–θ₈ (weights)  
- **Loss Function**: Mean Squared Error (MSE)  
- **Optimization**: Gradient-based parameter updates  

### Key Features
- ✅ Complete tracking of parameter history  
- ✅ Adaptive learning rate for SGD and MBGD  
- ✅ Visualization of convergence paths  
- ✅ Fixed random seed (`42`) for reproducibility  
- ✅ Added bias term via `np.c_[np.ones(), X_scaled]`  

---

## 📈 Results & Convergence Analysis

| Algorithm | Convergence Path | Stability | Speed | Use Case |
|------------|------------------|------------|--------|-----------|
| **Batch GD** | Smooth and direct | Very High | Slow | Small datasets |
| **SGD** | Noisy, fluctuating | Low | Fastest | Large/online data |
| **Mini-Batch GD** | Moderate noise | Balanced | Medium | General-purpose |

### 🔹 Example of Convergence Paths

Each algorithm converges toward slightly different parameters due to:
- Update frequency and batch size  
- Learning rate dynamics  
- Random sampling noise  

Mini-Batch GD demonstrates the **best trade-off** between stability and speed, making it the most practical for larger datasets.

---

## 📊 Final Parameter Comparison

| Algorithm | Learning Rate | Epochs | θ₀ (approx.) | Notes |
|------------|----------------|---------|----------------|--------|
| Batch GD | 0.01 | 1000 | ~2.07 | Stable and smooth |
| SGD | 5/(50+t) | 50 | ~2.07 | Fast but noisy |
| Mini-Batch GD | Adaptive | 50 | ~2.06 | Balanced |

---

## 🚀 Installation & Execution

### Prerequisites
Make sure the following dependencies are installed:

```bash
pip install numpy matplotlib scikit-learn jupyter
