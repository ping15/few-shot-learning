from networks.relation import RelationNetwork
from networks.prototypical import PrototypicalNetwork

from dataLoader.omniglotDataLoader import OmniglotDataLoader
from dataLoader.miniDataLoader2 import MiniImageDataLoader

network = RelationNetwork(data_loader=MiniImageDataLoader)

network.train(20, 32)
