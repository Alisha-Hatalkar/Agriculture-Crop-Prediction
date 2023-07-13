import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(('datafile (3).csv')or('datafile (2).csv')or('datafile (1).csv')or('datafile.csv')or('produce.csv'))
data1 = pd.read_csv('datafile (1).csv')
data2 = pd.read_csv('datafile (2).csv')
data3 = pd.read_csv('datafile (3).csv')
data4 = pd.read_csv('produce.csv')
# Check the dimensions of the dataset
print(data.shape)

# View the first few rows of the dataset
print(data.head())

# Check the data types of columns
print(data.dtypes)

# Get summary statistics of numeric columns
print(data.describe())

# Check for missing values
print(data.isnull().sum())
# Handling missing values (if any)
data = data.dropna()
print(data1.shape)

# View the first few rows of the dataset
print(data1.head())

# Check the data types of columns
print(data1.dtypes)

# Get summary statistics of numeric columns
print(data1.describe())

# Check for missing values
print(data1.isnull().sum())
# Handling missing values (if any)
data1 = data1.dropna()
print(data2.shape)

# View the first few rows of the dataset
print(data2.head())

# Check the data types of columns
print(data2.dtypes)

# Get summary statistics of numeric columns
print(data2.describe())

# Check for missing values
print(data2.isnull().sum())
# Handling missing values (if any)
data2 = data2.dropna()
print(data3.shape)

# View the first few rows of the dataset
print(data3.head())

# Check the data types of columns
print(data3.dtypes)

# Get summary statistics of numeric columns
print(data3.describe())

# Check for missing values
print(data3.isnull().sum())
# Handling missing values (if any)
data3 = data3.dropna()
print(data3.shape)

# View the first few rows of the dataset
print(data4.head())

# Check the data types of columns
print(data4.dtypes)

# Get summary statistics of numeric columns
print(data4.describe())

# Check for missing values
print(data4.isnull().sum())
# Handling missing values (if any)
data4 = data4.dropna()
# Convert data types if needed
# data['Season'] = pd.to_datetime(data['Season'])
# Example: Plotting crop production by state
plt.figure(figsize=(12, 6))
sns.barplot(x='state', y='production', data=data)
plt.xticks(rotation=90)
plt.xlabel('State')
plt.ylabel('Crop Production')
plt.title('Crop Production by State')
plt.show()
# Example: Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Quantity_scaled'] = scaler.fit_transform(data[['Quantity']])

# Example: Encoding categorical variables
data_encoded = pd.get_dummies(data, columns=['Crop', 'Variety', 'state'])

# Example: Splitting the data into train and test sets
from sklearn.model_selection import train_test_split

X = data_encoded.drop(['production'], axis=1)
y = data_encoded['production']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Example: Linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
