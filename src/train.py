import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from src.preprocess import load_and_preprocess
from src.lstm_model import LSTM_IDS
from src.autoencoder import Autoencoder
from torch.utils.data import DataLoader, TensorDataset

data_path = 'data/merged_cicids2017.csv'  # Объедини CSV сам

# Для LSTM
X_train, X_test, y_train, y_test, scaler, num_classes = load_and_preprocess(data_path)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

input_size = X_train.shape[2]
model_lstm = LSTM_IDS(input_size, hidden_size=128, num_layers=2, num_classes=num_classes)
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Обучение LSTM
epochs = 20
losses = []
for epoch in range(epochs):
    model_lstm.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model_lstm(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Оценка LSTM
model_lstm.eval()
y_pred = []
with torch.no_grad():
    for data, _ in test_loader:
        output = model_lstm(data)
        y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())
print(classification_report(y_test, y_pred))
print(f'ROC-AUC: {roc_auc_score(y_test, y_pred, multi_class="ovr")}')

plt.plot(losses)
plt.title('LSTM Training Loss')
plt.show()

torch.save(model_lstm.state_dict(), 'models/lstm_ids.pth')

# Для Autoencoder (только на нормальном трафике)
normal_X_train = X_train[y_train == 0].reshape(-1, input_size)  # Только normal (класс 0)
model_ae = Autoencoder(input_size)
optimizer_ae = optim.Adam(model_ae.parameters(), lr=0.001)
criterion_ae = nn.MSELoss()

for epoch in range(epochs):
    model_ae.train()
    output = model_ae(torch.tensor(normal_X_train, dtype=torch.float32))
    loss = criterion_ae(output, torch.tensor(normal_X_train, dtype=torch.float32))
    optimizer_ae.zero_grad()
    loss.backward()
    optimizer_ae.step()
    print(f'AE Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model_ae.state_dict(), 'models/autoencoder.pth')
