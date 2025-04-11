import pandas as pd
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

file_path = "popgrowth.csv"
file_path_rainfall = "rainfall.csv"
file_path_temperature = "temp.csv"
file_path_cereal_yield = "cereal.csv"
file_path_fertilizer = "fertiliser.csv"

# clean data
df = pd.read_csv(file_path, skiprows=4)
df.columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + [str(year) for year in range(1960, 2024)]

# Filter data for India and select relevant columns
india_data = df[df['Country Name'] == 'India']
india_data = india_data[['Country Name'] + [str(year) for year in range(1961, 2022)]]
india_data_melted = india_data.melt(id_vars=['Country Name'], var_name='Year', value_name='PopGrowth')
india_data_melted = india_data_melted.drop(columns=['Country Name']).reset_index(drop=True)
india_data_melted['Year'] = india_data_melted['Year'].astype(int)

df_rainfall = pd.read_csv(file_path_rainfall)
df_rainfall_filtered = df_rainfall[(df_rainfall['YEAR'] >= 1961) & (df_rainfall['YEAR'] <= 2021)]
df_rainfall_filtered = df_rainfall_filtered[['YEAR', 'ANNUAL']].rename(columns={'ANNUAL': 'Annual Rainfall'})
df_rainfall_filtered['YEAR'] = df_rainfall_filtered['YEAR'].astype(int)
df_rainfall_filtered = df_rainfall_filtered.reset_index(drop=True)

df_temperature = pd.read_csv(file_path_temperature)
df_temperature_filtered = df_temperature[(df_temperature['YEAR'] >= 1961) & (df_temperature['YEAR'] <= 2021)]
df_temperature_filtered = df_temperature_filtered[['YEAR', 'ANNUAL']].rename(columns={'ANNUAL': 'Annual Temperature'})
df_temperature_filtered['YEAR'] = df_temperature_filtered['YEAR'].astype(int)
df_temperature_filtered = df_temperature_filtered.reset_index(drop=True)

df_cereal_yield = pd.read_csv(file_path_cereal_yield, skiprows=4)
df_cereal_yield.columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + [str(year) for year in range(1960, 2024)]
india_cereal_yield = df_cereal_yield[df_cereal_yield['Country Name'] == 'India']
india_cereal_yield = india_cereal_yield[['Country Name'] + [str(year) for year in range(1961, 2022)]]
india_cereal_yield_melted = india_cereal_yield.melt(id_vars=['Country Name'], var_name='YEAR', value_name='Cereal yield (kg per hectare)')
india_cereal_yield_melted = india_cereal_yield_melted.drop(columns=['Country Name']).reset_index(drop=True)
india_cereal_yield_melted['YEAR'] = india_cereal_yield_melted['YEAR'].astype(int)



df_fertilizer = pd.read_csv(file_path_fertilizer, skiprows=5)
df_fertilizer.columns = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'] + [str(year) for year in range(1960, 2025)]
india_fertilizer = df_fertilizer[df_fertilizer['Country Name'] == 'India']
india_fertilizer = india_fertilizer[['Country Name'] + [str(year) for year in range(1961, 2022)]]
india_fertilizer_melted = india_fertilizer.melt(id_vars=['Country Name'], var_name='YEAR', value_name='Fertilizer consumption (kg per hectare)')
india_fertilizer_melted = india_fertilizer_melted.drop(columns=['Country Name']).reset_index(drop=True)
india_fertilizer_melted['YEAR'] = india_fertilizer_melted['YEAR'].astype(int)



# # Descriptive Statistics


# Combine all cleaned data into a single DataFrame
merged_data = pd.DataFrame({
    'Year': india_data_melted['Year'],
    'Population Growth': india_data_melted['PopGrowth'],
    'Annual Rainfall': df_rainfall_filtered['Annual Rainfall'],
    'Annual Temperature': df_temperature_filtered['Annual Temperature'],
    'Cereal Yield': india_cereal_yield_melted['Cereal yield (kg per hectare)'],
    'Fertilizer Consumption': india_fertilizer_melted['Fertilizer consumption (kg per hectare)']
})

print(merged_data)

# Function to compute descriptive statistics
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.describe().T
    stats['median'] = df.median()
    stats['missing_values'] = df.isnull().sum()
    return stats

# Calculate and display descriptive statistics for the dataset
descriptive_stats = descriptive_statistics(merged_data)
print("\nDescriptive Statistics for All Data:")
print(descriptive_stats)

#visual stuff

# Scatter plots
plt.figure(figsize=(15, 12))

# Cereal Yield vs Population Growth
plt.subplot(2, 2, 1)
plt.scatter(merged_data['Cereal Yield'], merged_data['Population Growth'], color='blue', alpha=0.7, edgecolor='k')
plt.title('Cereal Yield vs Population Growth')
plt.xlabel('Cereal Yield (kg/ha)')
plt.ylabel('Population Growth (%)')

# Annual Rainfall vs Population Growth
plt.subplot(2, 2, 2)
plt.scatter(merged_data['Annual Rainfall'], merged_data['Population Growth'], color='green', alpha=0.7, edgecolor='k')
plt.title('Annual Rainfall vs Population Growth')
plt.xlabel('Annual Rainfall (mm)')
plt.ylabel('Population Growth (%)')

# Annual Temperature vs Population Growth
plt.subplot(2, 2, 3)
plt.scatter(merged_data['Annual Temperature'], merged_data['Population Growth'], color='orange', alpha=0.7, edgecolor='k')
plt.title('Annual Temperature vs Population Growth')
plt.xlabel('Annual Temperature (°C)')
plt.ylabel('Population Growth (%)')

# Fertilizer Consumption vs Population Growth
plt.subplot(2, 2, 4)
plt.scatter(merged_data['Fertilizer Consumption'], merged_data['Population Growth'], color='red', alpha=0.7, edgecolor='k')
plt.title('Fertilizer Consumption vs Population Growth')
plt.xlabel('Fertilizer Consumption (kg/ha)')
plt.ylabel('Population Growth (%)')

# Adjust layout
plt.tight_layout()
plt.show()

# put Python code to prepare your features and target

# Ensure column names are consistent across all datasets
india_cereal_yield_melted = india_cereal_yield_melted.rename(columns={'YEAR': 'Year'})
df_rainfall_filtered = df_rainfall_filtered.rename(columns={'YEAR': 'Year'})
df_temperature_filtered = df_temperature_filtered.rename(columns={'YEAR': 'Year'})
india_fertilizer_melted = india_fertilizer_melted.rename(columns={'YEAR': 'Year'})

# Now try merging again
merged_data = pd.merge(india_data_melted, df_rainfall_filtered, how='left', on='Year')
merged_data = pd.merge(merged_data, df_temperature_filtered, how='left', on='Year')
merged_data = pd.merge(merged_data, india_cereal_yield_melted, how='left', on='Year')
merged_data = pd.merge(merged_data, india_fertilizer_melted, how='left', on='Year')

# Drop the 'YEAR' column after merging (if still present)
merged_data = merged_data.drop(columns=['YEAR'], errors='ignore')

features = ['Cereal yield (kg per hectare)', 'Annual Rainfall', 'Annual Temperature', 'Fertilizer consumption (kg per hectare)']
target = ['PopGrowth']



# Function Definitions
def get_features_targets(df: pd.DataFrame, feature_names: list[str], target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Select only numeric columns for features and target
    df_feature = pd.DataFrame(df[feature_names]).apply(pd.to_numeric, errors='coerce')
    df_target = pd.DataFrame(df[target_names]).apply(pd.to_numeric, errors='coerce')
    return df_feature, df_target


# put Python code to build your model

def get_features_targets(df: pd.DataFrame, feature_names: list[str], target_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Select only numeric columns for features and target
    df_feature = pd.DataFrame(df[feature_names]).apply(pd.to_numeric, errors='coerce')
    df_target = pd.DataFrame(df[target_names]).apply(pd.to_numeric, errors='coerce')
    return df_feature, df_target

def normalize_z(array: np.ndarray, columns_means: Optional[np.ndarray] = None, columns_stds: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if columns_means is None:
        columns_means = array.mean(axis=0)
    if columns_stds is None:
        columns_stds = array.std(axis=0)
    normalized = (array - columns_means) / columns_stds
    return normalized, columns_means, columns_stds

def prepare_feature(np_feature: np.ndarray) -> np.ndarray:
    n_rows = np_feature.shape[0]
    return np.concatenate((np.ones((n_rows, 1)), np_feature), axis=1)

def calc_linreg(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.matmul(X, beta)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_seed: Optional[int] = None):
    if random_seed is not None:
        np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int((1 - test_size) * X.shape[0])
    X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
    y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
    return X_train, X_test, y_train, y_test

def gradient_descent(X: np.ndarray, y: np.ndarray, beta_init: np.ndarray, learning_rate: float, num_epochs: int) -> np.ndarray:
    m = X.shape[0]
    beta = beta_init
    for epoch in range(num_epochs):
        predictions = calc_linreg(X, beta)
        gradient = -(1 / m) * X.T @ (y - predictions)
        beta -= learning_rate * gradient
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
            # print(f"Epoch {epoch+1}/{num_epochs}, Cost: {cost:.4f}")
    return beta

# put Python code to test & evaluate the model

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# Prepare the data for model
df_feature, df_target = get_features_targets(merged_data, feature_names=features, target_names=target)
np_feature, means, stds = normalize_z(df_feature.to_numpy())
X = prepare_feature(np_feature)
y = df_target.to_numpy()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_seed=42)

# Train the Model
initial_beta = np.zeros((X_train.shape[1], 1))
learning_rate = 0.1
num_epochs = 1500
beta = gradient_descent(X_train, y_train, initial_beta, learning_rate, num_epochs)

# Evaluate the Model
y_test_pred = calc_linreg(X_test, beta)
test_r2 = r_squared(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"R^2 on Test Set: {test_r2:.4f}")
print(f"Mean Squared Error on Test Set: {test_mse:.4f}")

# User Interaction for Prediction
def predict(pop_growth: float, annual_rainfall: float, annual_temperature: float, fertilizer: float) -> float:
    input_features = np.array([[pop_growth, annual_rainfall, annual_temperature, fertilizer]])
    normalized_input, _, _ = normalize_z(input_features, means, stds)
    input_prepared = prepare_feature(normalized_input)
    prediction = calc_linreg(input_prepared, beta)
    return max(0, prediction[0][0])  # Avoid negative predictions and extract the scalar value

print("Enter details to predict pop growth:")
cereal = float(input("Cereal yeild (kg per hectare): "))
annual_rainfall = float(input("Annual Rainfall (mm): "))
annual_temperature = float(input("Annual Temperature (°C): "))
fertilizer = float(input("Fertilizer Consumption (kg per hectare): "))

predicted_yield = predict(cereal, annual_rainfall, annual_temperature, fertilizer)
print(f"Predicted Population growth: {predicted_yield:.4f}")
