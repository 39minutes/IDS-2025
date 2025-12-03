import torch
from scapy.all import sniff
from src.lstm_model import LSTM_IDS
from src.autoencoder import Autoencoder
from src.preprocess import StandardScaler  # Используй сохранённый scaler
from src.utils import extract_features
import numpy as np
import joblib  # Для загрузки scaler (сохрани его в train.py: joblib.dump(scaler, 'models/scaler.pkl'))

# Загрузка моделей
input_size = 78  # Из CIC-IDS (адаптируй)
num_classes = 15  # Из датасета
model_lstm = LSTM_IDS(input_size, 128, 2, num_classes)
model_lstm.load_state_dict(torch.load('models/lstm_ids.pth'))
model_lstm.eval()

model_ae = Autoencoder(input_size)
model_ae.load_state_dict(torch.load('models/autoencoder.pth'))
model_ae.eval()

scaler = joblib.load('models/scaler.pkl')
threshold_ae = 0.01  # MSE threshold для аномалий (подбери на тесте)

def process_packet(packet):
    features = extract_features(packet)
    if len(features) != input_size:
        return  # Пропуск невалидных

    scaled = scaler.transform([features])
    input_tensor = torch.tensor(scaled.reshape(1, 1, -1), dtype=torch.float32)

    # LSTM классификация
    with torch.no_grad():
        pred = model_lstm(input_tensor)
        class_id = torch.argmax(pred, dim=1).item()
        if class_id != 0:  # Не normal
            print(f"ВТОРЖЕНИЕ! Тип: {class_id}")

    # Autoencoder аномалия
    with torch.no_grad():
        recon = model_ae(input_tensor.squeeze(1))
        mse = torch.mean((input_tensor.squeeze(1) - recon)**2).item()
        if mse > threshold_ae:
            print(f"АНОМАЛИЯ! MSE: {mse}")

# Запуск реал-тайм (на интерфейсе, напр. 'eth0' или 'Wi-Fi')
sniff(iface='en0', prn=process_packet, store=0, count=0)  # Бесконечно
