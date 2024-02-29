from src.utils.utils import keras_callbacks
from src.AI.dataset import DataGenerator
from src.AI.model import classifier
from src.AI.metrics import Result
import src.config.config as cfg
from tensorflow import keras
import tensorflow as tf
import numpy as np
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


def make_result(block, inference=False):

    model = keras.models.model_from_json(block['model'])
    weights = [np.array(w) for w in block['weights']]
    model.set_weights(weights)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )

    test_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.test_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.test_batch_size,
        client_id=cfg.stations.index(cfg.ID[:-3] + '000'),
        client_count=len(cfg.stations)
    )
    if not inference:
        train_dataset = DataGenerator(
            images_path=cfg.dataset_path,
            label_csv=cfg.train_labels,
            dim=cfg.image_size,
            pad_resize=cfg.pad_resize,
            batch_size=cfg.batch_size,
            client_id=cfg.cars.index(cfg.ID),
            client_count=len(cfg.cars)
        )

        model.fit(
            train_dataset,
            epochs=cfg.epoch_num
            # callbacks=keras_callbacks(),

        )

        model_json = model.to_json()
        weights = model.get_weights()
        json_weights = [np.array(w).tolist() for w in weights]

        result = Result(model, test_dataset)

        data = {'model': model_json,
                'weights': json_weights,
                'metrics': result.calculate_metrics()}
        return data

    if cfg.model_plot:
        tf.keras.utils.plot_model(
            model, to_file='model.png',
            show_shapes=True,
            show_layer_activations=True,
            show_dtype=True,
        )
    if cfg.summary:
        model.summary()

    result = Result(model, test_dataset)
    result.plot_confusion(cfg.confusion_path)


def test_results():
    test_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.test_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.test_batch_size,
    )

    train_dataset = DataGenerator(
        images_path=cfg.dataset_path,
        label_csv=cfg.train_labels,
        dim=cfg.image_size,
        pad_resize=cfg.pad_resize,
        batch_size=cfg.batch_size,
        client_id=1,
        client_count=4
    )

    model = classifier(cfg.image_size, cfg.class_num)

    if cfg.model_plot:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.MeanSquaredError(name='mse'),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            keras.metrics.MeanAbsoluteError(name="mae")
        ]
    )

    model.fit(train_dataset,
              epochs=20,
              # callbacks=keras_callbacks(),
              verbose=0)
    result = Result(model, test_dataset)
    result.plot_confusion(cfg.confusion_path)
    return result.calculate_metrics()


if __name__ == "__main__":
    print(test_results())
