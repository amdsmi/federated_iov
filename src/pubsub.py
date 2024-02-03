import time
from abc import ABC

from pubnub.pubnub import PubNub
from pubnub.pnconfiguration import PNConfiguration
from pubnub.callbacks import SubscribeCallback

from src.block_chain.block import Block

import src.config.config as cfg

pnconfig = PNConfiguration()
pnconfig.subscribe_key = cfg.Subscribe_Key
pnconfig.publish_key = cfg.Publish_Key

CHANNELS = {
    'TEST': 'TEST',
    'MINER': 'MINER'
}


class Listener(SubscribeCallback, ABC):
    def __init__(self, miner):
        self.miner = miner

    def message(self, pubnub, message_object):
        print(f'\n-- Channel: {message_object.channel} | Message: {message_object.message}')

        if message_object.channel == CHANNELS['MINER']:
            self.miner.miner = message_object.message
            print(f'\n miner {message_object.message} is responsible for mining the block')


class PubSub:
    """
    Handles the publish/subscribe layer of the application.
    Provides communication between the nodes of the blockchain network.
    """

    def __init__(self, blockchain, miner):
        self.pubnub = PubNub(pnconfig)
        self.pubnub.subscribe().channels(CHANNELS.values()).execute()
        self.pubnub.add_listener(Listener(miner))

    def publish(self, channel, message):
        """
        Publish the message object to the channel.
        """
        self.pubnub.unsubscribe().channels([channel]).execute()
        self.pubnub.publish().channel(channel).message(message).sync()
        self.pubnub.subscribe().channels([channel]).execute()

    def broadcast_miner(self, miner_address):
        """
        Broadcast a miner to all nodes.
        """
        self.publish(CHANNELS['MINER'], miner_address)


def main():
    pubsub = PubSub()

    time.sleep(1)

    pubsub.publish(CHANNELS['TEST'], {'foo': 'bar'})


if __name__ == '__main__':
    main()
