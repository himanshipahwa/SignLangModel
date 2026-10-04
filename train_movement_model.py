import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- CONFIGURATION ---
DATA_PATH = "MP_Data"
SIGNS = [
    "I", "You", "Help", "Please", "Thank You",
    "Sorry", "Yes", "No", "Want", "Go",
    "Eat", "Like", "Need", "Know", "Come",
    "Good", "Bad", "Where", "Who", "What"
]
MIN_CLIPS = 3  # Skip signs with fewer than this many clips
# ---------------------

# 1. Load the data
sequences = []
labels = []
active_signs = []

print("Loading movement data...")
for sign in SIGNS:
    folder = os.path.join(DATA_PATH, sign)
    if not os.path.exists(folder):
        print(f"  [SKIP] '{sign}' — folder missing")
        continue
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(files) < MIN_CLIPS:
        print(f"  [SKIP] '{sign}' — only {len(files)} clip(s) (need at least {MIN_CLIPS})")
        continue
    for file in files:
        res = np.load(os.path.join(folder, file))
        sequences.append(res)
        labels.append(len(active_signs))
    print(f"  [OK]   '{sign}' — {len(files)} clips")
    active_signs.append(sign)

print(f"\nActive signs ({len(active_signs)}): {active_signs}")
num_classes = len(active_signs)

if len(sequences) == 0:
    raise ValueError("No data found! Run 'python collect_sequences.py' first.")
if num_classes < 2:
    raise ValueError(f"Need at least 2 signs with data. Found: {active_signs}")

X = np.array(sequences)
y_raw = np.array(labels)
y = to_categorical(y_raw, num_classes=num_classes).astype('float32')
print(f"\nTotal clips: {len(X)} | Shape: {X.shape}")

# 2. Stratified train/test split
X_train, X_test, y_train, y_test, y_train_raw, _ = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# 3. Class weights (handle imbalance)
weights_arr = compute_class_weight('balanced', classes=np.unique(y_train_raw), y=y_train_raw)
class_weights = dict(enumerate(weights_arr))

# 4. Build a RIGHT-SIZED model for the dataset
#    Rule of thumb: keep params << (num_training_samples * 10)
#    ~670 train clips → target < 6700 params. Use small LSTM.
print("\nBuilding model...")
model = Sequential([
    Input(shape=(30, 42)),
    LSTM(64, return_sequences=True, activation='tanh'),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()
print(f"\nModel parameters: {model.count_params():,} (appropriate for {len(X_train)} training clips)")

# 5. Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_categorical_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'movement_model_best.keras',
        monitor='val_categorical_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    ),
]

# 6. Train
print("\nTraining... this might take a few minutes...")
model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks,
)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {acc * 100:.1f}%")

# 8. Save
model.save("movement_model.h5")
print("Movement brain saved as 'movement_model.h5'!")
print("Best checkpoint saved as 'movement_model_best.keras'!")

if active_signs != SIGNS:
    print(f"\n⚠️  Some signs were skipped. Update SIGNS in app_movement.py to:")
    print(f"   {active_signs}")