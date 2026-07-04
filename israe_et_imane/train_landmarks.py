import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv(
    "data/landmarks.csv"
)


X = df.iloc[:, 1:].values

y = df.iloc[:, 0].values


encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    y
)

with open(
    "models/labels.pkl",
    "wb"
) as f:

    pickle.dump(
        encoder,
        f
    )

y_cat = to_categorical(
    y_encoded
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


model = Sequential([

    Dense(
        256,
        activation="relu",
        input_shape=(42,)
    ),

    Dropout(0.3),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        64,
        activation="relu"
    ),

    Dense(
        y_cat.shape[1],
        activation="softmax"
    )

])


model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=50,
    batch_size=32
)

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    "models/landmark_model.keras"
)

print(
    "\n✅ Model Saved"
)

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"✅ Accuracy : {acc:.4f}"
)