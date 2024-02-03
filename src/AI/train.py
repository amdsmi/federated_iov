import src.config.config as cfg
import tensorflow as tf
from src.utils.utils import keras_callbacks
from tensorflow import keras
from src.AI.dataset import DataGenerator
from src.AI.model import classifier
import numpy as np
import json


def make_result(last_model):

    train_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.train_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.batch_size,
        client_id=cfg.cars.index(cfg.ID),
        client_count=len(cfg.cars)
    )

    model = keras.models.model_from_json(last_model['model'])
    weights = [np.array(w) for w in last_model['weights']]
    model.set_weights(weights)

    if cfg.model_plot:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_dataset,
              epochs=cfg.epoch_num,
              callbacks=keras_callbacks())

    model_json = model.to_json()
    weights = model.get_weights()
    json_weights = [np.array(w).tolist() for w in weights]

    result = model.evaluate(train_dataset)

    data = {'model': model_json,
            'weights': json_weights,
            'metrics': {'loss': result[0], 'acc': result[1]}}
    return data


if __name__ == "__main__":
    make_result()
