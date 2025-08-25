import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("yoga_landmarks_segmented.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

num_classes = len(label_encoder.classes_)
input_dim = X_train.shape[1]

model = Sequential([
    Dense(256, input_dim=input_dim, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting model training...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

print("Training finished.")
model.save("yoga_pose_classifier.h5")
print("Model saved as yoga_pose_classifier.h5")

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved as label_encoder.pkl")