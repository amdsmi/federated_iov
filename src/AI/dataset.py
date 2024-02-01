import os
import cv2
import keras.utils
import numpy as np
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    def __init__(self,

                 images_path,
                 label_csv,
                 dim,
                 batch_size=16,
                 channels=3,
                 client_count=None,
                 client_id=None,
                 pad_resize=True,
                 shuffle=True):
        super(DataGenerator, self).__init__()

        self.client_count = client_count
        self.client_id = client_id
        self.indexes = None
        self.images_path = images_path
        self.label_csv = label_csv
        self.label, self.images_list = self.list_ids()
        self.dim = (dim, dim) if isinstance(dim, int) else dim
        self.channels = channels
        self.shuffle = shuffle
        self.pad_resize = pad_resize
        self.batch_size = batch_size
        self.on_epoch_end()

    def list_ids(self):
        data = self.make_client_dataset(self.client_count,
                                        self.client_id,
                                        self.label_csv) if self.client_id else pd.read_csv(self.label_csv)

        images = data['Path'].to_list()
        label = data['ClassId'].to_list()

        return label, images

    @staticmethod
    def make_client_dataset(client_count, client_id, data_path):
        data = pd.read_csv(data_path)
        datasets = []
        for class_id in data['ClassId'].unique():
            dataset = data[data['ClassId'] == class_id]
            sample_num = len(dataset) // client_count
            datasets.append(dataset[client_id * sample_num:(client_id + 1) * sample_num])

        return pd.concat(datasets)

    def __len__(self):
        return int(np.floor(len(self.images_list) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _pad_resize(self, image, color=(114, 114, 114), scaleup=False):

        shape = image.shape[:2]

        r = min(self.dim[0] / shape[0], self.dim[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.dim[1] - new_unpad[0], self.dim[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, r, (dw, dh)

    def __getitem__(self, index):

        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_name = [self.images_list[idx] for idx in indexes]
        images = np.empty((self.batch_size, *self.dim, self.channels))
        labels = np.empty(self.batch_size, dtype=int)

        for i, name in enumerate(batch_image_name):
            image_path = os.path.join(self.images_path, name)
            image = cv2.imread(image_path) / 255
            if self.pad_resize:
                image, r, pad = self._pad_resize(image, self.dim)
            else:
                image = cv2.resize(image, self.dim)
            images[i, ...] = image
            labels[i] = self.label[indexes[i]]
        return images, labels


if __name__ == "__main__":
    data_set = DataGenerator(images_path='dataset',
                             label_csv='dataset/Train.csv',
                             dim=128,
                             client_id=2,
                             client_count=9)

    print("=================number of mini batches in dataset=================\n", len(data_set))

    print("++++++++++++++++++++++++++first bathes data+++++++++++++++++++++++++\n", next(iter(data_set)))
