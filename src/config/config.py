import os

epoch_num = 1
image_size = 64
dataset_path = 'dataset'
val_labels = 'dataset/Test.csv'
train_labels = 'dataset/Train.csv'
model_plot = False
pad_resize = False
batch_size = 32
class_num = 43
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", 'flask_oauth')
Publish_Key = 'pub-c-9175e236-d47d-49c6-a56a-83bebe8f759c'
Subscribe_Key = 'sub-c-9a558ac7-fe2a-4b77-9a43-96c798a5ddb5'
Secret_Key = 'sec-c-ZjFjNjNlYjUtODE5ZS00YTU5LThlOTMtMjg5ZDQ1Yjc5NjQx'

car_per_station = 2

CLIENT_TYPE = os.getenv('CLIENT_TYPE', 'car')

ROUND_PER_BLOCK = 2

WAITING_TIME = 1

SERVER_ADDRESS = 'localhost'
ID = os.environ.get('ID')

cars = [
    '10001',
    '10002',
    # '10003',
    '20001',
    '20002',
    # '20003',
    # '30001',
    # '30002',
    # '30003'
]

stations = [
    '10000',
    '20000',
    # '30000',
]
