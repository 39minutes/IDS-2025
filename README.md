# IDS-2025: Система обнаружения вторжений на базе глубокого обучения

## Описание
Курсовая работа по "Методам глубокого машинного обучения". Разработана система IDS для анализа сетевого трафика в реальном времени с использованием LSTM (классификация вторжений) и Autoencoder (обнаружение аномалий). Датасет: CIC-IDS2017. 

## Структура
- `notebooks/`: Jupyter Notebook для обучения и экспериментов (GPU в Colab).
- `src/`: Модульные скрипты (PyTorch).
- `data/`: Датасеты (CIC-IDS2017).
- `models/`: Сохранённые модели (.pth).

## Запуск
1. Открой `notebooks/IDS_Final_Advanced.ipynb` → [Open in Colab](https://colab.research.google.com/github/39minutes/IDS-2025/blob/main/notebooks/IDS_Final_Advanced.ipynb).
2. Выбери GPU → Run All (обучение ~10 мин).
3. Для реал-тайм: `sudo python src/realtime_ids.py` (локально).
