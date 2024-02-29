import src.config.config as cfg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import keras
import os


def metrics_to_dataframe(chain):
    dict_ = {'block': []}
    for idx in range(len(chain)):
        if idx != 0:
            dict_['block'].append(idx)
            metrics = chain[idx].data['metrics']
            for metric, value in metrics.items():
                if metric not in dict_.keys():
                    dict_[metric] = []
                dict_[metric].append(value)

    return pd.DataFrame(dict_)


def plot_metrics(chain):
    df = metrics_to_dataframe(chain)
    df.set_index('block', inplace=True)
    df[cfg.losses].plot(title='regression metrics').get_figure().savefig('regression_metrics.png')
    df[cfg.classify].plot(title='classification metrics').get_figure().savefig('classification_metrics.png')


def dataset_statistics(label_path):

    data = pd.read_csv(label_path)
    data['roi_width'] = data['Roi.X2'] - data['Roi.X1']
    data['roi_height'] = data['Roi.Y2'] - data['Roi.Y1']
    numeric = data[['Width', 'Height', 'roi_width', 'roi_height']]
    round(numeric.describe(), 3).T.to_csv(label_path.split('/')[-1].split('.')[0] + '.csv')

    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 15))
    value_counts = data['ClassId'].value_counts()
    sorted_value_counts = data['ClassId'].value_counts().sort_values(ascending=False)
    color_palette = sns.color_palette("Spectral", len(value_counts))
    sns.countplot(
        data=data,
        x='ClassId',
        order=sorted_value_counts.index,
        palette=color_palette
    ).get_figure().savefig(label_path.split('/')[-1].split('.')[0] + '.png')


def keras_callbacks():
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='logs/',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None
        ),
        keras.callbacks.ModelCheckpoint("checkpoint/", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(os.path.join('logs', "result.csv"), separator=",",
                                  append=False)
    ]
    return callbacks


if __name__ == "__main__":
    dataset_statistics(cfg.test_labels)
