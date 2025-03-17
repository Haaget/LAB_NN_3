import os
import zipfile
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

dataset_path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
extract_path = os.path.join(os.getcwd(), "fruits-vegetables")
os.makedirs(extract_path, exist_ok=True)

for file in os.listdir(dataset_path):
    if file.endswith(".zip"):
        with zipfile.ZipFile(os.path.join(dataset_path, file), 'r') as zip_ref:
            zip_ref.extractall(extract_path)

train_dir = os.path.join(extract_path, "train")
test_dir = os.path.join(extract_path, "test")

target_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size,
    class_mode="categorical", subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    train_dir, target_size=target_size, batch_size=batch_size,
    class_mode="categorical", subset="validation"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=target_size, batch_size=batch_size,
    class_mode="categorical", shuffle=False
)

num_classes = len(train_generator.class_indices)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

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

y_pred = model.predict(test_generator).argmax(axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("Матриця невідповідностей:")
print(cm)
