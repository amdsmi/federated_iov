import os
seed = 123
#########################
#     model config      #
#########################
epoch_num = 2
image_size = 64

#########################
#     dataset config    #
#########################
dataset_path = 'dataset'
test_labels = 'dataset/Test.csv'
train_labels = 'dataset/Train.csv'
model_plot = True
summary = True
pad_resize = False
batch_size = 32
class_num = 43
test_batch_size = 32

#########################
#     pubnub config     #
#########################

APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", 'flask_oauth')
Publish_Key = 'pub-c-ac8398c2-9376-4615-ae18-0cb202439f39'
Subscribe_Key = 'sub-c-f939a6b2-3899-4d70-8b31-8c8daf8a7dde'
Secret_Key = 'sec-c-OTk0YzAzYWQtNGM3Ni00MTQyLTg0NTktMTg4NDdmMGYxYjBl'

#########################
#     server config     #
#########################

'--------------car-----------'
car_per_station = 2
cars = [
    '10001',
    '10002',
    # '10003',
    # '10004',
    '20001',
    '20002',
    # '20003',
    # '30001',
    # '30002',
    # '30003'
]
#CLIENT_TYPE = os.getenv('CLIENT_TYPE', 'car')

'---------station------------'
stations = [
    '10000',
    '20000',
    # '30000',
]

'---------general------------'
ROUND_PER_BLOCK = 2
WAITING_TIME = 1
SERVER_ADDRESS = 'localhost'
ID = os.environ.get('ID')


'----------plots-------------'
confusion_path = 'confusion_matrix.png'
losses = ['mse', 'rmse', 'mae']
classify = ['f1', 'recall', 'acc', 'precision']

