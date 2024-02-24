import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

# Read the JSON data
with open('new_transactions_training.json') as f:
    data = json.load(f)

# Initialize lists to store values for each column
types = []
amounts = []
transactionTimes = []
transactionLocations = []
devices = []
paymentMethods = []
recentChanges = []
suspiciousFlags = []

# Iterate over each transaction dictionary
for transaction in data:
    types.append(transaction['type'])
    amounts.append(transaction['amount'])
    transactionTimes.append(transaction['transactionTime'])
    transactionLocations.append(transaction['transactionLocation'])
    devices.append(transaction['device'])
    paymentMethods.append(transaction['paymentMethod'])
    recentChanges.append(transaction['recentChangeInAccountDetails'])
    suspiciousFlags.append(transaction['suspicious'])

# Create DataFrame from the lists of values
df = pd.DataFrame({
    'type': types,
    'amount': amounts,
    'transactionTime': transactionTimes,
    'transactionLocation': transactionLocations,
    'device': devices,
    'paymentMethod': paymentMethods,
    'recentChangeInAccountDetails': recentChanges,
    'suspicious': suspiciousFlags
})

# Extract features and labels
X = df.drop('suspicious', axis=1)  # Features
y = df['suspicious']  # Labels

# Preprocess the features
# Encode categorical variables and scale numerical features
categorical_features = ['type', 'transactionLocation', 'device', 'paymentMethod']
numerical_features = ['amount', 'transactionTime', 'recentChangeInAccountDetails']

# Define transformers for the preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Convert sparse matrices to dense arrays
if isinstance(X_processed, csr_matrix):
    X_processed = X_processed.toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Define the neural network architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.3f}')