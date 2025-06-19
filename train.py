import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib  # Untuk menyimpan scaler dan label encoder
from model_definition import LSTMModel  # Pastikan file ini ada

# Fungsi untuk memuat dataset gejala
def load_dataset():
    dataset = pd.read_csv('Dataset_Gejala_Ayam.csv')  # Ganti path jika perlu
    return dataset

# Fungsi untuk preprocessing data
def preprocess_data(dataset):
    # Asumsikan ada 36 fitur dan 1 kolom label bernama 'Penyakit'
    X = dataset.drop(columns=['Penyakit'])
    y = dataset['Penyakit']

    # Normalisasi fitur
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode label penyakit menjadi angka
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Konversi ke tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # Simpan scaler dan encoder
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    return X_tensor, y_tensor, len(label_encoder.classes_)

# Fungsi untuk melatih model
def train_model(model, X_train, y_train, num_epochs=200, batch_size=32, learning_rate=0.001):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in data_loader:
            optimizer.zero_grad()

            # LSTM butuh input 3D: (batch, seq_len=1, input_size)
            inputs = inputs.unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")

    print("Training selesai!")

# Fungsi utama
def main():
    # Muat dataset
    dataset = load_dataset()

    # Preprocessing
    X_train, y_train, num_classes = preprocess_data(dataset)

    # Inisialisasi model
    input_size = X_train.shape[1]  # 36 gejala
    hidden_size = 64
    num_layers = 2
    dropout_rate = 0.2

    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)

    # Latih model
    train_model(model, X_train, y_train, num_epochs=200)

    # Simpan model
    torch.save(model.state_dict(), 'lstm_model_penyakit_ayam.pth')
    print("Model disimpan ke lstm_model_penyakit_ayam.pth")

if __name__ == "__main__":
    main()
