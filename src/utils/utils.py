import keras
import os


def keras_callbacks():
    callbacks = [
        # keras.callbacks.TensorBoard(
        #     log_dir='logs/',
        #     histogram_freq=0,
        #     write_graph=True,
        #     write_images=True,
        #     update_freq="epoch",
        #     profile_batch=0,
        #     embeddings_freq=0,
        #     embeddings_metadata=None
        # ),
        # keras.callbacks.ModelCheckpoint("checkpoint/", save_best_only=True),
        # keras.callbacks.EarlyStopping(
        #     monitor="val_loss",
        #     patience=5,
        #     restore_best_weights=True
        # ),
        # keras.callbacks.CSVLogger(os.path.join('logs', "result.csv"), separator=",",
        #                           append=False)
    ]
    return callbacks
