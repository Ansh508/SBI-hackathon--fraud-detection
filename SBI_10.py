import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import cv2
import joblib  # For saving Random Forest model

# Load the dataset
df = pd.read_csv('Insurance Fraud.csv')

# Drop unnecessary columns
drop_columns = ["Claim ID", "Claim ID.1", "Street Address", "Claimant Name", "City", "State", "Country", "Postal Code"]
df_cleaned = df.drop(columns=drop_columns)

# Convert Claim Date to datetime and sort by date
df_cleaned["Claim Date"] = pd.to_datetime(df_cleaned["Claim Date"], format="%d-%m-%Y")
df_cleaned = df_cleaned.sort_values(by="Claim Date")

# One-hot encode categorical variables
df_cleaned = pd.get_dummies(df_cleaned, columns=["Claim Status", "Type of Insurance Claim", "Fraud_Types"], drop_first=True)

# Set Claim Date as index and resample data daily
df_time_series = df_cleaned.set_index("Claim Date").resample("D").mean().fillna(0)

# Convert SuspiciousFlag to binary
df_time_series["SuspiciousFlag"] = (df_time_series["SuspiciousFlag"] >= 0.5).astype(int)

# Reset index
df_time_series.reset_index(inplace=True)

# Drop Claim Date for model input
df_time_series.drop(columns=["Claim Date"], inplace=True)

# Define input and target variables
X = df_time_series.drop(columns=["SuspiciousFlag"] + [col for col in df_time_series.columns if "Fraud_Types" in col]).values
y_suspicious = df_time_series["SuspiciousFlag"].values
y_fraud_type = df_time_series[[col for col in df_time_series.columns if "Fraud_Types" in col]].values

# Normalize input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert data into sequences for LSTM
sequence_length = 10
X_lstm, y_suspicious_lstm, y_fraud_type_lstm = [], [], []

for i in range(len(X_scaled) - sequence_length):
    X_lstm.append(X_scaled[i : i + sequence_length])
    y_suspicious_lstm.append(y_suspicious[i + sequence_length])
    y_fraud_type_lstm.append(y_fraud_type[i + sequence_length])

X_lstm = np.array(X_lstm)
y_suspicious_lstm = np.array(y_suspicious_lstm)
y_fraud_type_lstm = np.array(y_fraud_type_lstm)

# Train-test split (80-20) and validation split (10% of training data)
X_train, X_test, y_train_susp, y_test_susp, y_train_fraud, y_test_fraud = train_test_split(
    X_lstm, y_suspicious_lstm, y_fraud_type_lstm, test_size=0.2, random_state=42, stratify=y_suspicious_lstm
)

X_train, X_val, y_train_susp, y_val_susp, y_train_fraud, y_val_fraud = train_test_split(
    X_train, y_train_susp, y_train_fraud, test_size=0.1, random_state=42, stratify=y_train_susp
)

# LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm_model.fit(X_train, y_train_susp, validation_data=(X_val, y_val_susp), epochs=50, batch_size=32, verbose=1)

# Save LSTM Model
lstm_model.save('main_lstm_model.h5')

# Evaluate LSTM Model
train_acc_lstm = lstm_model.evaluate(X_train, y_train_susp, verbose=0)[1] * 100
val_acc_lstm = lstm_model.evaluate(X_val, y_val_susp, verbose=0)[1] * 100
test_acc_lstm = lstm_model.evaluate(X_test, y_test_susp, verbose=0)[1] * 100

print(f"LSTM Model - Train Accuracy: {train_acc_lstm:.2f}%")
print(f"LSTM Model - Validation Accuracy: {val_acc_lstm:.2f}%")
print(f"LSTM Model - Test Accuracy: {test_acc_lstm:.2f}%")

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)
rf_model.fit(X_train_rf, y_train_susp)

# Save Random Forest Model
joblib.dump(rf_model, 'random_forest_model.pkl')

# Evaluate Random Forest Model
train_acc_rf = rf_model.score(X_train_rf, y_train_susp) * 100
test_acc_rf = rf_model.score(X_test_rf, y_test_susp) * 100

print(f"Random Forest Model - Train Accuracy: {train_acc_rf:.2f}%")
print(f"Random Forest Model - Test Accuracy: {test_acc_rf:.2f}%")

# Convert LSTM and RF outputs into grayscale images
def convert_to_image(data, size=(28, 28)):
    data_resized = np.resize(data, size)
    image = (data_resized * 255).astype(np.uint8)
    return image

X_lstm_images = np.array([convert_to_image(x) for x in X_train_rf])
X_rf_images = np.array([convert_to_image(x) for x in X_train_rf])

X_combined_images = np.stack([X_lstm_images, X_rf_images], axis=-1)

# CNN Model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 2)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn_model.fit(X_combined_images, y_train_susp, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

# Save CNN Model
cnn_model.save('main_cnn_model.h5')

# Evaluate CNN Model
train_acc_cnn = cnn_model.evaluate(X_combined_images, y_train_susp, verbose=0)[1] * 100
val_acc_cnn = cnn_model.evaluate(X_combined_images[:len(y_train_susp) // 10], 
                                 y_train_susp[:len(y_train_susp) // 10], verbose=0)[1] * 100
test_acc_cnn = cnn_model.evaluate(X_combined_images, y_train_susp, verbose=0)[1] * 100  # Replace with actual test images

print(f"CNN Model - Train Accuracy: {train_acc_cnn:.2f}%")
print(f"CNN Model - Validation Accuracy: {val_acc_cnn:.2f}%")
print(f"CNN Model - Test Accuracy: {test_acc_cnn:.2f}%")