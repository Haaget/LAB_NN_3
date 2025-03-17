import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix

target_size = (300, 300)
batch_size = 32
input_shape = (300, 300, 3)

data_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

train_data = []
for filename in os.listdir(train_dir):
    if filename.lower().startswith("cat"):
        label = "cat"
    elif filename.lower().startswith("dog"):
        label = "dog"
    else:
        continue

    train_data.append({"filename": os.path.join(train_dir, filename), "label": label})

train_df = pd.DataFrame(train_data)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])

test_data = [{"filename": os.path.join(test_dir, filename)} for filename in os.listdir(test_dir)]
test_df = pd.DataFrame(test_data)

train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filename",
    y_col="label",
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filename",
    y_col="label",
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filename",
    y_col=None,
    target_size=target_size,
    batch_size=batch_size,
    class_mode=None,
    shuffle=False
)

model_vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
model_vgg19.trainable = False

flat1 = GlobalAveragePooling2D()(model_vgg19.output)
class1 = Dense(1024, activation="relu")(flat1)
output = Dense(1, activation="sigmoid")(class1)

model = Model(inputs=model_vgg19.inputs, outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.plot(acc, label="Точність навчання")
plt.plot(val_acc, label="Точність перевірки")
plt.legend()
plt.show()

plt.plot(loss, label="Втрата навчання")
plt.plot(val_loss, label="Втрата перевірки")
plt.legend()
plt.show()

y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten()
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("Матриця невідповідностей:")
print(cm)
