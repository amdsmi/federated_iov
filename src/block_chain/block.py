import time
from src.utils.crypto_hash import crypto_hash
from src.AI.model import get_raw_model_weights

GENESIS_DATA = {
    'timestamp': 1,
    'last_hash': 'genesis_last_hash',
    'hash': 'genesis_hash',
    'data': get_raw_model_weights()
}


class Block:
    def __init__(self, timestamp, last_hash, hash, data):
        self.timestamp = timestamp
        self.last_hash = last_hash
        self.hash = hash
        self.data = data

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return (
            'Block('
            f'timestamp: {self.timestamp}, '
            f'last_hash: {self.last_hash}, '
            f'hash: {self.hash}, '
            f'data: {self.data}, '
        )

    def to_json(self):

        return self.__dict__

    @staticmethod
    def mine_block(last_block, data):

        timestamp = time.time_ns()
        last_hash = last_block.hash

        hash = crypto_hash(timestamp, last_hash, data)

        return Block(timestamp, last_hash, hash, data)

    @staticmethod
    def genesis():

        return Block(**GENESIS_DATA)

    @staticmethod
    def from_json(block_json):

        return Block(**block_json)

    @staticmethod
    def is_valid_block(last_block, block):

        if block.last_hash != last_block.hash:
            raise Exception('The block last_hash must be correct')

        reconstructed_hash = crypto_hash(
            block.timestamp,
            block.last_hash,
            block.data,
        )

        if block.hash != reconstructed_hash:
            raise Exception('The block hash must be correct')


if __name__ == "__main__":
    pass
