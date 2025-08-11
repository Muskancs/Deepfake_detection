import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# === Directory Setup ===
os.makedirs("results/model_checkpoints", exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs("results/misclassified", exist_ok=True)
os.makedirs("results/training_plots", exist_ok=True)

# === Data Paths ===
base_dir = r"C:\Users\hp\Downloads\Deepfake_Sentinel\celeb_V2"
train_dir = os.path.join(base_dir, "Train")
val_dir = os.path.join(base_dir, "Val")
test_dir = os.path.join(base_dir, "Test")

img_height, img_width = 128, 128
batch_size = 32

# === Data Generators (with proper preprocessing) ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),
                                               batch_size=batch_size, class_mode='binary')
val_data = val_datagen.flow_from_directory(val_dir, target_size=(img_height, img_width),
                                           batch_size=batch_size, class_mode='binary')
test_data = test_datagen.flow_from_directory(test_dir, target_size=(img_height, img_width),
                                             batch_size=1, class_mode='binary', shuffle=False)

# === Paths ===
checkpoint_path = "results/model_checkpoints/mobilenetv2_model.keras"
history_file = "results/training_plots/mobilenetv2_history.pkl"

# === Callbacks ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)

# === Resume or Train ===
initial_epoch = 0
past_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

if os.path.exists(checkpoint_path):
    print("🔁 Resuming from checkpoint...")
    model = load_model(checkpoint_path)
    if os.path.exists(history_file):
        try:
            with open(history_file, "rb") as f:
                past_history = pickle.load(f)
            for key in past_history.keys():
                if key not in past_history:
                    past_history[key] = []
            initial_epoch = len(past_history['accuracy'])
            print(f"📅 Resuming from epoch {initial_epoch}")
        except Exception as e:
            print(f"⚠️ Could not load history.pkl, starting fresh history. Error: {e}")
else:
    print("🚀 Starting new MobileNetV2 training...")
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(img_height, img_width, 3))
    base_model.trainable = False  # Stage 1: Freeze

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Stage 1 Training ===
if initial_epoch < 10:
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        initial_epoch=initial_epoch,
        callbacks=[early_stopping, checkpoint]
    )

    for key in past_history.keys():
        if key in history.history:
            past_history[key].extend(history.history[key])

    with open(history_file, "wb") as f:
        pickle.dump(past_history, f)

# === Stage 2: Fine-tuning deeper layers ===
print("🔓 Fine-tuning deeper layers...")
model = load_model(checkpoint_path)  # Load best from Stage 1
base_model = model.layers[0]  # Extract the MobileNetV2 base
base_model.trainable = True

# Freeze most layers, unfreeze last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stopping, checkpoint]
)

# Merge fine-tuning history
for key in past_history.keys():
    if key in history_ft.history:
        past_history[key].extend(history_ft.history[key])

with open(history_file, "wb") as f:
    pickle.dump(past_history, f)

# === Plot Curves ===
plt.figure()
plt.plot(past_history['accuracy'], label='Train Accuracy')
plt.plot(past_history['val_accuracy'], label='Val Accuracy')
plt.title('MobileNetV2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('results/training_plots/mobilenetv2_accuracy.png')
plt.show()

plt.figure()
plt.plot(past_history['loss'], label='Train Loss')
plt.plot(past_history['val_loss'], label='Val Loss')
plt.title('MobileNetV2 Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/training_plots/mobilenetv2_loss.png')
plt.show()

# === Evaluation ===
print("✅ Loading best MobileNetV2 model for evaluation...")
model = load_model(checkpoint_path)
loss, accuracy = model.evaluate(test_data)
print(f"\n📊 Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")
