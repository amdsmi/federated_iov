import os
import requests
import concurrent.futures
import json
import flask
import requests
import time
from flask import Flask, jsonify, request
from flask_cors import CORS
import src.config.config as cfg
from src.AI.train import make_result
from src.block_chain.blockchain import Blockchain
from src.block_chain.block import Block
from src.pubsub import PubSub
from src.AI.fusion_models import fusion_best_models
from threading import Thread


class Miner:
    def __init__(self):
        self.miner = None


class Server:
    def __init__(self, name):
        self.app = Flask(import_name=name)
        CORS(self.app, supports_credentials=True)
        self.app.config['CORS_HEADERS'] = 'Content-Type'
        self.app.config['CORS_SUPPORTS_CREDENTIALS'] = True
        self.app.config['JSON_SORT_KEYS'] = False
        self.app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        self.app.secret_key = cfg.APP_SECRET_KEY

        self._miner = Miner()
        self.blockchain = Blockchain()
        self.pubsub = PubSub(self.blockchain, self._miner)
        self.aggregated_models = list()
        self._other_blocks = list()

        @self.app.route('/')
        def index():
            return self.__index()

        @self.app.route('/replace/chain', methods=['POST'])
        def replace_chain():
            return self.__replace_chain()

        @self.app.route('/make/block', methods=['GET'])
        def make_block():
            return self.__make_block()

        @self.app.route('/get/block', methods=['POST'])
        def get_block():
            return self.__get_block()

        @self.app.route('/car/train', methods=['GET'])
        def car_train():
            return self.__car_train()

        @self.app.route('/station', methods=['GET'])
        def station():
            return self.__station()

        @self.app.route('/blockchain/length', methods=['GET'])
        def route_blockchain_length():
            return self.__route_blockchain_length()

        @self.app.route('/blockchain', methods=['GET'])
        def route_blockchain():
            return self.__route_blockchain()

        @self.app.route('/blockchain/range', methods=['GET'])
        def route_blockchain_range():
            return self.__route_blockchain_range()

        @self.app.route('/last/block', methods=['GET'])
        def last_block():
            return self.__last_block()

        # ======================= home page ==========================

    def __last_block(self):
        return self.blockchain.chain[-1].data['metrics']

    def __index(self):
        return 'federated iov is up'

    def __make_block(self):

        urls = ['http://' + cfg.SERVER_ADDRESS + ':' + port + '/station' for port in cfg.stations]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.make_req, url) for url in urls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        return 'block successfully mined'

    @staticmethod
    def make_req(url):
        result = requests.get(url)
        return result.json()

    def __replace_chain(self):

        block = Block.from_json(request.get_json())

        potential_chain = self.blockchain.chain[:]

        potential_chain.append(block)

        self.blockchain.replace_chain(potential_chain)

        return flask.jsonify({'status': 'success'})

    def __get_block(self):
        mined_block = Block.from_json(request.get_json())
        self._other_blocks.append(mined_block)
        return flask.jsonify({'status': 'success'})

    def __car_train(self):
        last_model = self.blockchain.chain[-1].data
        return jsonify(make_result(last_model))

    '''ToDo Implement when station get the block it should the acc with own acc'''

    def __route_blockchain_length(self):
        return jsonify(len(self.blockchain.chain))

    def __route_blockchain(self):
        return [block.data['metrics'] for block in self.blockchain.chain]

    def __route_blockchain_range(self):
        # http://localhost:5000/blockchain/range?start=2&end=5
        start = int(request.args.get('start'))
        end = int(request.args.get('end'))

        return jsonify(self.blockchain.to_json()[::-1][start:end])


    def __station(self):

        car_addresses = self.generate_car_route(cfg.ID)

        while len(self.aggregated_models) < cfg.ROUND_PER_BLOCK:

            # model_to_aggregate = []  # list of json models
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.make_req, url) for url in car_addresses]
                model_to_aggregate = [future.result() for future in concurrent.futures.as_completed(futures)]
            # for car_address in car_addresses:
            #     result = requests.get(car_address)  # json
            #
            #     model_to_aggregate.append(result.json())

            self.aggregated_models.append(fusion_best_models(model_to_aggregate))  # fill the aggregated_models with
            # list of json models

        best_model = self.find_best_model(self.aggregated_models)

        last_block = self.blockchain.chain[-1]

        mined_block = Block.mine_block(last_block, best_model)

        if self._miner.miner:

            print("=========================", self._miner.miner)
            headers = {'Content-Type': 'application/json'}
            result = requests.post(self._miner.miner, data=json.dumps(Block.to_json(mined_block)), headers=headers)
            self._miner.miner = None
            self._other_blocks = list()
            self.aggregated_models = list()
            print("==========================", self._miner.miner)

        else:

            self.pubsub.broadcast_miner(f'http://localhost:{cfg.ID}/get/block')
            self._other_blocks.append(mined_block)
            while len(self._other_blocks) < len(cfg.stations):
                time.sleep(cfg.WAITING_TIME)
            best_block = self.find_best_block(self._other_blocks)

            self.broad_cast(best_block.to_json(), '/replace/chain')

            self._miner.miner = None
            self._other_blocks = list()
            self.aggregated_models = list()

        return flask.jsonify({"status": 'success'})

    @staticmethod
    def find_best_block(blocks_list):
        best_block = blocks_list[0]

        for idx in range(1, len(blocks_list)):

            block = blocks_list[idx]
            if block.data['metrics']['acc'] > best_block.data['metrics']['acc']:
                best_block = block
        return best_block

    @staticmethod
    def find_best_model(models_list):
        best_model = models_list[0]

        for idx in range(1, len(models_list)):
            model = models_list[idx]
            if model['metrics']['acc'] > best_model['metrics']['acc']:
                best_model = model

        return best_model

    @staticmethod
    def generate_car_route(station_port):
        return ['http://' + cfg.SERVER_ADDRESS + ':' + str(int(station_port) + i) + '/car/train'
                for i in range(1, cfg.car_per_station + 1)]

    @staticmethod
    def broad_cast(data, end_point):

        headers = {'Content-Type': 'application/json'}
        for addr in cfg.cars + cfg.stations:
            url = 'http://' + cfg.SERVER_ADDRESS + ':' + addr + end_point
            result = requests.post(url, data=json.dumps(data), headers=headers)

    def run(self):
        print('Running server...')

        print('Host: {}'.format(cfg.SERVER_ADDRESS))
        print('Port: {}'.format(cfg.ID))

        self.app.run(host=cfg.SERVER_ADDRESS, port=cfg.ID)


if __name__ == '__main__':
    server = Server(__name__)
    server.run()
