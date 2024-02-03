import src.config.config as cfg
import tensorflow as tf
from src.utils.utils import keras_callbacks
from tensorflow import keras
from src.AI.dataset import DataGenerator
import numpy as np
from src.AI.model import classifier
import json


def make_result():
    train_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.train_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.batch_size)

    model = classifier(cfg.image_size, cfg.class_num)

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

    with open('model3.json', 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    make_result()

