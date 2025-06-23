# first_machine_learning_project

# 🌾 Climate and Agriculture Impact on India's Population Growth

This project investigates how various environmental and agricultural factors influence **population growth in India** using historical data from 1961 to 2021. The analysis uses linear regression and includes data cleaning, visualization, descriptive statistics, and model prediction.

---

## 📂 Project Structure

- `popgrowth.csv` – Population growth data
- `rainfall.csv` – Annual rainfall data
- `temp.csv` – Annual temperature data
- `cereal.csv` – Cereal yield per hectare
- `fertiliser.csv` – Fertilizer consumption data

---

## 🧹 Data Preprocessing

- Merged and cleaned all datasets for India from 1961–2021
- Normalized data using **Z-score normalization**
- Melted time-series data into tidy format using `pandas`

---

## 📊 Exploratory Data Analysis

Includes scatter plots showing relationships between:
- Cereal Yield vs Population Growth
- Rainfall vs Population Growth
- Temperature vs Population Growth
- Fertilizer Consumption vs Population Growth

---

## 📈 Modeling

- Implemented a **custom Linear Regression model using Gradient Descent**
- Training and test data split (70/30)
- Evaluated using:
  - R² (coefficient of determination)
  - Mean Squared Error (MSE)

---

## 🔍 Prediction

You can input:
- Cereal Yield
- Annual Rainfall
- Annual Temperature
- Fertilizer Consumption

...and the model will **predict expected population growth**.

---

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy, Matplotlib
- Linear regression from scratch (no sklearn used)

---

## 🚀 How to Run

1. Make sure all CSV files are in the same directory as the Python script.
2. Run the script:
   ```bash
   python3 your_script_name.py

Enter details to predict pop growth:
Cereal yeild (kg per hectare): 3000
Annual Rainfall (mm): 1200
Annual Temperature (°C): 25
Fertilizer Consumption (kg per hectare): 150
Predicted Population growth: 1.7842

