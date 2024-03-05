import pandas as pd
import json
import random  # Import the random module
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.sparse import csr_matrix

# Read the JSON data
with open('new_transactions_training.json') as f:
    data = json.load(f)

# Shuffle the JSON data to randomize the order of transactions
random.shuffle(data)  # Shuffle the data in-place

# Print a snippet of the JSON data to inspect its structure
print("JSON data sample after shuffling:", json.dumps(data[:2], indent=4))

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

# Print the first few rows of the DataFrame to check the data
print("DataFrame head after shuffling:", df.head(), "\n")

# Define categorical and numerical features
categorical_features = ['type', 'transactionLocation', 'device', 'paymentMethod']
numerical_features = ['amount', 'transactionTime', 'recentChangeInAccountDetails']

# Define transformers for the preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline with preprocessing and SMOTE for balancing the dataset
#pipeline = make_pipeline(preprocessor, SMOTE(random_state=42))
pipeline = make_pipeline(preprocessor)

# Apply transformations and balance the dataset
#X_processed, y_processed = pipeline.fit_resample(df.drop('suspicious', axis=1), df['suspicious'])
# Apply preprocessing transformations
X_processed = pipeline.fit_transform(df.drop('suspicious', axis=1))

# Since we're not resampling, y remains unchanged and can be directly assigned
y_processed = df['suspicious']



# Convert sparse matrices to dense arrays if necessary
if isinstance(X_processed, csr_matrix):
    X_processed = X_processed.toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

sampled_transactions = df.sample(n=20, random_state=42)['suspicious']
print("Randomly sampled transactions' suspicious status:\n", sampled_transactions)

# Define and compile the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc:.3f}')