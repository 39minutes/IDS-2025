from scapy.all import *

def extract_features(packet):
    # Пример фич (адаптируй под CIC-IDS)
    features = {
        'Duration': 0,  # Вычисли из timestamp
        'Protocol': packet.proto if hasattr(packet, 'proto') else 0,
        'Src Bytes': len(packet),
        'Dst Bytes': 0,  # Для простоты
        # Добавь больше: flow duration, packet count и т.д. (используй сессии в Scapy)
    }
    return list(features.values())  # Верни как list для scaler
