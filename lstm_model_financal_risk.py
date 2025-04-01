import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pymongo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from backend.financial_risk_model.data_preprocessing import process_data
from sklearn.utils import resample
from pymongo import MongoClient
import datetime
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from database.mongodb_config import collection


# Model directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def build_lstm_model(input_shape):
    """ Builds an optimized LSTM model with dropout, batch normalization, and gradient clipping. """
    model = Sequential([
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(1, activation='sigmoid')  # Binary classification
    ])

    optimizer = Adam(learning_rate=0.0005, clipnorm=1.0)  # Clipping gradients
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def store_risk_data(company, avg_risk_score, test_accuracy):
    # Create a dictionary to store risk data
    risk_entry = {
        "company_name": company,
        "risk_score": float(avg_risk_score),  # Convert numpy.float32 to float
        "model_accuracy": float(test_accuracy),  # Convert numpy.float32 to float
        "timestamp": datetime.datetime.now()
    }

    # Connect to MongoDB (adjust this connection string as per your config)
    client = pymongo.MongoClient("mongodb://localhost:27018/")
    db = client["risk_database"]
    collection = db["risk_data"]

    # Insert the document into the collection
    collection.insert_one(risk_entry)
    print(f"Data for {company} inserted successfully.")


def train_lstm(company_name, epochs=50, batch_size=64):
    """ Trains the LSTM model with early stopping and learning rate reduction. """
    
    # Fetch preprocessed data
    X_train, y_train, X_test, y_test = process_data(company_name)

    # Validate data
    if X_train is None or y_train is None or X_train.shape[0] == 0:
        print(f"❌ Error: No valid data for {company_name}.")
        return None, None, None, None

    # Ensure correct input shape for LSTM
    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)

    # Handle class imbalance with oversampling
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = np.min(class_counts)

    if min_class_count < 0.2 * len(y_train):  # Only apply if imbalance is severe
        print("⚠️ Class imbalance detected. Applying oversampling...")
        df_train = np.hstack((X_train.reshape(X_train.shape[0], -1), y_train.reshape(-1, 1)))
        df_train = pd.DataFrame(df_train)

        df_majority = df_train[df_train.iloc[:, -1] == 0]
        df_minority = df_train[df_train.iloc[:, -1] == 1]

        df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
        df_balanced = pd.concat([df_majority, df_minority_upsampled])

        X_train = df_balanced.iloc[:, :-1].values.reshape(-1, X_train.shape[1], X_train.shape[2])
        y_train = df_balanced.iloc[:, -1].values.astype(int)

    # Define LSTM input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Callbacks
    model_path = os.path.join(MODEL_DIR, f"{company_name}_lstm_model.h5")
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)

    # Train the model
    print(f"🚀 Training LSTM model for {company_name}...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, reduce_lr, checkpoint], verbose=1)

    print(f"✅ Training complete. Best model saved at: {model_path}")
    return model, history, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """ Evaluates the trained model on the test set. """
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Risk", "Risk"], yticklabels=["No Risk", "Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    return acc

if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    model, history, X_test, y_test = train_lstm(company)
    if model:
        test_accuracy = evaluate_model(model, X_test, y_test)
        y_pred_probs=model.predict(X_test)
        avg_risk_score=np.mean(y_pred_probs)

        store_risk_data(company,avg_risk_score,test_accuracy)

