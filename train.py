from networks.relation import RelationNetwork
from networks.prototypical import PrototypicalNetwork
from networks.matching import MatchingNetwork

from dataLoader.omniglotDataLoader import OmniglotDataLoader
from dataLoader.miniDataLoader2 import MiniImageDataLoader

network = MatchingNetwork(data_loader=OmniglotDataLoader)

network.train(200, 2)
