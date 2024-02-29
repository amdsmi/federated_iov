from src.AI.dataset import DataGenerator
import src.config.config as cfg
from tensorflow import keras
from keras import layers
import numpy as np
import json
import tensorflow as tf
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)

def classifier(input_shape, num_classes):
    if isinstance(input_shape, int):
        input_shape = (input_shape, input_shape, 3)
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3)(inputs)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(16, 3)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Dropout(0.25, seed=cfg.seed)(x)

    x = layers.Conv2D(32, 3)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, 3)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25, seed=cfg.seed)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5, seed=cfg.seed)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs)


def get_raw_model_weights():

    model = classifier(cfg.image_size, cfg.class_num)

    model_json = model.to_json()
    weights = model.get_weights()
    json_weights = [np.array(w).tolist() for w in weights]

    return {'model': model_json,
            'weights': json_weights,
            'metrics': {'f1': None, 'acc': None, 'recall': None, 'precision': None, 'mse': None, 'rmse': None, 'mae': None}}


def load_json():

    val_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.val_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.batch_size)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

    with open("weights.json", "r") as json_file:
        json_weights = json.load(json_file)

    weights = [np.array(w) for w in json_weights]
    model.set_weights(weights)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    metrics = model.evaluate(val_dataset)
    print(metrics)


if __name__ == "__main__":
    # load_json()

    print(get_raw_model_weights())