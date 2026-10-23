import torch

# Dataset Config
NUM_CLASSES = 10
IMAGE_SIZE = (224, 224)
SEQUENCE_LENGTH = 7      # 7-day preceding weather sequence
WEATHER_FEATURES = 7     # Temp, RH, Precip, Wind, LWD, VPD, DTR

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_LABELS = [
    "Bacterial Leaf Blight",
    "Bacterial Leaf Streak",
    "Bacterial Panicle Blight",
    "Blast",
    "Brown Spot",
    "Dead Heart",
    "Downy Mildew",
    "Hispa",
    "Normal",
    "Tungro"
]
