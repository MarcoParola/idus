from typing import List, Tuple
import torch
from torch import nn, Tensor
from src.models.embedding import PositionEmbeddingSine
from src.models.transformer import Transformer


class TransitionLayer(nn.Module):
    def __init__(self, inChannels: int, numGroups: int):
        super(TransitionLayer, self).__init__()

        self.outChannels = int(inChannels / 2)

        self.module = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, self.outChannels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, numGroups: int, growthRate: int):
        super(ConvBlock, self).__init__()

        self.outChannels = growthRate

        self.module = nn.Sequential(
            nn.GroupNorm(numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, 4 * growthRate, 1),
            nn.GroupNorm(numGroups, 4 * growthRate),
            nn.ReLU(),
            nn.Conv2d(4 * growthRate, growthRate, 3, padding=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class DenseBlock(nn.Module):
    def __init__(self, inChannels: int, numBlocks: int, numGroups: int, growthRate: int):
        super(DenseBlock, self).__init__()

        self.outChannels = inChannels

        self.layers = nn.ModuleList()
        for _ in range(numBlocks):
            self.layers.append(ConvBlock(self.outChannels, numGroups, growthRate))
            self.outChannels += growthRate

    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self, numGroups: int, growthRate: int, numBlocks: List[int], inChannels: int = 3):
        super(DenseNet, self).__init__()

        self.outChannels = 64

        self.input = nn.Sequential(
            nn.Conv2d(inChannels, self.outChannels, 7, padding=3),
            nn.GroupNorm(numGroups, self.outChannels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layers = [self.input]

        for blocks in numBlocks:
            block = DenseBlock(self.outChannels, blocks, numGroups, growthRate)
            self.outChannels = block.outChannels
            trans = TransitionLayer(self.outChannels, numGroups)
            self.outChannels = trans.outChannels
            layers.append(block)
            layers.append(trans)

        self.module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class Joiner(nn.Module):
    def __init__(self, backbone: nn.Module, positionEmbedding: nn.Module):
        super(Joiner, self).__init__()

        self.backbone = backbone
        self.positionEmbedding = positionEmbedding

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        return features, self.positionEmbedding(features)

def tiny():
    # Transformer parameters are fixed for tiny version
    transformer = Transformer(
        hiddenDims=128, numHead=4, numEncoderLayer=6, numDecoderLayer=6,
        dimFeedForward=512, dropout=0.1
    )
    return transformer, 128


def small():
    # Fixed transformer parameters for small version
    transformer = Transformer(
        hiddenDims=256, numHead=8, numEncoderLayer=8, numDecoderLayer=8,
        dimFeedForward=1024, dropout=0.1
    )
    return transformer, 256


def base():
    # Fixed transformer parameters for base version
    transformer = Transformer(
        hiddenDims=512, numHead=16, numEncoderLayer=12, numDecoderLayer=12,
        dimFeedForward=2048, dropout=0.1
    )
    return transformer, 512

def buildBackbone(args, hidden_dim: int):
    positionEmbedding = PositionEmbeddingSine(hidden_dim // 2)
    denseNet = DenseNet(args.numGroups, args.growthRate, args.detr.numBlocks)
    return Joiner(denseNet, positionEmbedding)
