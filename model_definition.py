import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.2):
        super(LSTMModel, self).__init__()

        # Simpan atribut untuk jumlah layer dan ukuran hidden state
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Layer LSTM dengan dropout internal (hanya jika num_layers > 1)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Dropout tambahan sebelum Fully Connected layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully connected layer untuk klasifikasi
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x dimensi: (batch_size, seq_len, input_size)

        batch_size = x.size(0)  # Ambil batch size dari input

        # Inisialisasi hidden state (h0) dan cell state (c0)
        # Dimensi harus (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward pass LSTM, memberikan h0 dan c0 eksplisit supaya aman
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_size)

        # Ambil output dari timestep terakhir (seq_len terakhir)
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Dropout dan fully connected layer
        out = self.dropout(out)
        out = self.fc(out)  # shape: (batch_size, num_classes)

        return out

# Fungsi untuk memuat model yang sudah dilatih
def load_model(model_path='model_lstm_penyakit_ayam.pth', input_size=36, hidden_size=128, num_layers=2, num_classes=9, dropout_rate=0.2):
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set model ke mode evaluasi
    return model

# Fungsi untuk melakukan prediksi
def predict_disease(model, input_data):
    # Pastikan input_data adalah list dengan panjang 36
    input_data = np.array(input_data).reshape(1, 1, -1)  # Bentuk (batch_size=1, seq_len=1, fitur=36)
    
    # Normalisasi input data
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data.reshape(-1, 1)).reshape(1, 1, -1)
    
    # Convert ke tensor PyTorch
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    # Prediksi
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()
