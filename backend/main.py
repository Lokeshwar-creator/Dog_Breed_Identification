# dog_breed_full_pipeline.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input

# ---------------------------
# Config / Paths
# ---------------------------
labels_path = r"C:\Users\lokes\OneDrive\Desktop\Documents\Essential Documents\Docs\Documents\FILES\PROJECTS\Dog Breed Identification\dataset\labels.csv"
train_file = r"C:\Users\lokes\OneDrive\Desktop\Documents\Essential Documents\Docs\Documents\FILES\PROJECTS\Dog Breed Identification\dataset\train"
# Where to save/load the trained model:
model_path = r"C:\Users\lokes\OneDrive\Desktop\Documents\Essential Documents\Docs\Documents\FILES\PROJECTS\Dog Breed Identification\dataset\dog_breed_model.h5"

# Prediction input image (change if needed)
pred_img_path = r"C:\Users\lokes\OneDrive\Desktop\Documents\Essential Documents\Docs\Documents\FILES\PROJECTS\Dog Breed Identification\images\2.jpg"

# Hyperparams
num_breeds = 60
im_size = 224
batch_size = 64
epochs = 20
learning_rate = 1e-3

# ---------------------------
# Read labels and prepare breed list
# ---------------------------
df_labels = pd.read_csv(labels_path)
print("Total number of unique Dog Breeds in labels.csv :", len(df_labels['breed'].unique()))

# choose top-most frequent breed names (same method you used before)
breed_order = list(df_labels['breed'].value_counts().keys())
selected_breeds = sorted(breed_order, reverse=True)[:num_breeds*2+1:2]

# keep only selected breeds and add filename column
df = df_labels.query('breed in @selected_breeds').copy()
df['img_file'] = df['id'].astype(str) + ".jpg"
df = df.reset_index(drop=True)
print(f"Dataset reduced to {len(df)} images across {len(df['breed'].unique())} breeds.")

# print the list of breeds (sorted)
all_breeds = sorted(df['breed'].unique())
print("\nList of breeds (dataset):")
for i, b in enumerate(all_breeds, 1):
    print(f"{i:03d}. {b}")

# ---------------------------
# Load & preprocess all images into arrays
# ---------------------------
n = len(df)
X = np.zeros((n, im_size, im_size, 3), dtype='float32')
missing = 0
for i, fname in enumerate(df['img_file']):
    p = os.path.join(train_file, fname)
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        # leave zero array entry and count missing
        missing += 1
        print(f"Missing: {p}")
        continue
    img = cv2.resize(img, (im_size, im_size))
    X[i] = preprocess_input(img.astype(np.float32))

if missing:
    print(f"\nWarning: {missing} images were missing and left as zeros in X array.")

# encode labels
le = LabelEncoder()
y = le.fit_transform(df['breed'].values)

# train-test split (we always perform this so evaluation works regardless of model existence)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# data generators
train_datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
test_gen = test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False)

# ---------------------------
# Build or load model
# ---------------------------
history = None
if os.path.exists(model_path):
    print("\n[OK] Found existing model. Loading:", model_path)
    model = load_model(model_path)
else:
    print("\n[INFO] No model found. Building and training a new model...")
    base = ResNet50V2(input_shape=(im_size, im_size, 3), weights='imagenet', include_top=False)
    for layer in base.layers:
        layer.trainable = False

    x = base.output
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(len(all_breeds), activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)
    optimizer = RMSprop(learning_rate=learning_rate, rho=0.9)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train -- do not force steps_per_epoch; let Keras use generator length
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen
    )

    # Save
    model.save(model_path)
    print("[OK] Model trained and saved at:", os.path.abspath(model_path))

# ---------------------------
# Evaluate on test set
# ---------------------------
print("\n--- Evaluation on test set ---")
# get predictions from model on the test generator
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test[:len(y_pred)]  # align length in case last batch smaller

acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Classification report
target_names = le.inverse_transform(np.arange(len(all_breeds)))
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
print("\nClassification Report:\n")
print(report)

# Per-class support counts (true counts)
unique, counts = np.unique(y_true, return_counts=True)
support_counts = dict(zip(le.inverse_transform(unique), counts))

# Print per-class supports (sorted by name)
print("\nPer-class sample counts in test set:")
for breed in sorted(support_counts.keys()):
    print(f"{breed}: {support_counts[breed]}")

# Confusion matrix (counts)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(all_breeds)))

# Plot confusion matrix (counts)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap="Blues", cbar=True)
plt.title("Confusion Matrix (counts)")
plt.xlabel("Predicted label index")
plt.ylabel("True label index")
plt.tight_layout()
plt.show()

# Plot confusion matrix (normalized by true counts -> recall per-class)
cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_norm, cmap="viridis", vmin=0, vmax=1)
plt.title("Confusion Matrix (normalized by true counts)")
plt.xlabel("Predicted label index")
plt.ylabel("True label index")
plt.tight_layout()
plt.show()

# Bar chart of per-class support (test set)
breed_names = target_names
support = [support_counts.get(b, 0) for b in breed_names]
plt.figure(figsize=(12, 5))
plt.bar(range(len(breed_names)), support)
plt.title("Per-class support (test set counts)")
plt.xlabel("Breed index")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# If training happened, show training graphs
if history is not None:
    print("\nShowing training curves...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("\nNo training performed in this run (model loaded). Skipping training curves.")

# ---------------------------
# Show classification report as a nicer pandas DataFrame (optional)
# ---------------------------
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
df_report = pd.DataFrame({
    'breed': le.inverse_transform(np.arange(len(all_breeds))),
    'precision': np.round(prec, 4),
    'recall': np.round(rec, 4),
    'f1-score': np.round(f1, 4),
    'support': sup
})
df_report = df_report.sort_values(by='support', ascending=False).reset_index(drop=True)
print("\nPer-class metrics (sorted by support):")
print(df_report.head(30).to_string(index=False))

# ---------------------------
# Final: Prediction on input image and display with bounding box
# ---------------------------
print("\n--- Prediction on single input image ---")
img = cv2.imread(pred_img_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f"Prediction image not found: {pred_img_path}")

# keep a copy for drawing (in original size)
orig_img = img.copy()

# prepare for model
img_resized = cv2.resize(orig_img, (im_size, im_size))
img_arr = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))

pred_probs = model.predict(img_arr)
pred_idx = int(np.argmax(pred_probs))
pred_breed = le.inverse_transform([pred_idx])[0]
pred_confidence = float(np.max(pred_probs))

# draw a rectangle around whole image (as requested)
h, w = orig_img.shape[:2]
cv2.rectangle(orig_img, (5, 5), (w-5, h-5), (0, 255, 0), 3)

# Put text (breed + confidence)
text = f"{pred_breed} ({pred_confidence*100:.1f}%)"
# choose font scale relative to image size
font_scale = max(0.6, w / 800)
cv2.putText(orig_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

# Convert BGR->RGB and display using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted: {pred_breed}  |  Confidence: {pred_confidence:.3f}")
plt.axis('off')
plt.show()

print(f"\nPredicted Breed: {pred_breed}")
print(f"Confidence: {pred_confidence:.4f}")
