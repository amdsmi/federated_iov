import src.config.config as cfg
import tensorflow as tf
from src.utils.utils import keras_callbacks
from tensorflow import keras
from src.AI.dataset import DataGenerator
from src.AI.model import classifier
import numpy as np
import json

train_dataset = DataGenerator(
    images_path=cfg.dataset_path,
    label_csv=cfg.train_labels,
    dim=cfg.image_size,
    pad_resize=cfg.pad_resize,
    batch_size=cfg.batch_size)

val_dataset = DataGenerator(
    images_path=cfg.dataset_path,
    label_csv=cfg.val_labels,
    dim=cfg.image_size,
    pad_resize=cfg.pad_resize,
    batch_size=cfg.batch_size)

model = classifier(cfg.image_size, cfg.class_num)
print(model.summary())
if cfg.model_plot:
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

if __name__ == "__main__":
    model.fit(train_dataset,
              epochs=cfg.epoch_num,
              validation_data=val_dataset,
              callbacks=keras_callbacks())

    result = model.evaluate(val_dataset)

    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    weights = model.get_weights()

    json_weights = [np.array(w).tolist() for w in weights]

    with open("weights.json", "w") as json_file:
        json.dump(json_weights, json_file)

    data = {'model': model_json,
            'weights': json_weights,
            'metrics': {'loss': result[0], 'acc': result[1]}}

    with open("data.json", "w") as json_file:
        json.dump(data, json_file)
