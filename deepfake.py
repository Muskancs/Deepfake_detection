import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

# === Directory Setup ===
os.makedirs("results/model_checkpoints", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/misclassified", exist_ok=True)
os.makedirs("results/training_plots", exist_ok=True)

# === Data Loading ===
base_dir = r"C:\Users\hp\Downloads\Deepfake_Sentinel\celeb_V2"
train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Val")
test_dir = os.path.join(base_dir, "Test")

img_height, img_width = 128, 128
batch_size = 32
datagen = ImageDataGenerator(rescale=1.0/255)

train_data = datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
test_data = datagen.flow_from_directory(test_dir, target_size=(img_height, img_width), batch_size=1, class_mode='binary', shuffle=False)

# === Paths ===
checkpoint_path = "results/model_checkpoints/cnn_model.keras"
model_save_path = "results/models/best_model.keras"

# === Callbacks ===
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

# === Resume or Train ===
# === Resume or Train ===
initial_epoch = 0
past_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

if os.path.exists(checkpoint_path):
    print("🔁 Resuming from checkpoint...")
    model = load_model(checkpoint_path)
    
    history_file = "results/training_plots/history.pkl"
    if os.path.exists(history_file):
        try:
            with open(history_file, "rb") as f:
                past_history = pickle.load(f)
            # Ensure all required keys exist
            for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                if key not in past_history:
                    past_history[key] = []
            initial_epoch = len(past_history['accuracy'])
            print(f"📅 Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"⚠️ Could not load history.pkl, starting fresh history. Error: {e}")
else:
    print("🚀 Starting new training...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    past_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# === Continue Training ===
if initial_epoch < 10:  # Only if not fully trained
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        initial_epoch=initial_epoch,
        callbacks=[early_stopping, checkpoint]
    )

    # Merge histories
    for key in past_history.keys():
        if key in history.history:
            past_history[key].extend(history.history[key])

    # Save updated history
    with open("results/training_plots/history.pkl", "wb") as f:
        pickle.dump(past_history, f)
else:
    print("✅ Model already trained for all epochs. Skipping training.")

# === Plot Curves ===
plt.figure()
plt.plot(past_history['accuracy'], label='Train Accuracy')
plt.plot(past_history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/training_plots/accuracy.png')
plt.show()
plt.close()

plt.figure()
plt.plot(past_history['loss'], label='Train Loss')
plt.plot(past_history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/training_plots/loss.png')
plt.show()
plt.close()

# === Evaluation ===
print("✅ Loading best model for evaluation...")
model = load_model(checkpoint_path)
loss, accuracy = model.evaluate(test_data)
print(f"\n📊 Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
