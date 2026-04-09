import torch
import torch.nn as nn
from torch.nn import init


"""
    CARE-GNN Model (YelpChi Spam Detection)
    Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    Source: https://github.com/YingtongDou/CARE-GNN
"""


class OneLayerCARE(nn.Module):
    """
    The CARE-GNN model in one layer.
    Supports class weighting to handle the imbalanced spam/legit distribution in YelpChi.
    """

    def __init__(self, num_classes, inter1, lambda_1, class_weight=None):
        """
        Initialize the CARE-GNN model.
        :param num_classes: number of classes (2: spam / legit)
        :param inter1: inter-relation aggregator that outputs the final embedding
        :param lambda_1: weight for the similarity (simi) loss term
        :param class_weight: FloatTensor of shape [num_classes] for weighted CE loss;
                             helps deal with the minority spam class in YelpChi (~14.5%)
        """
        super(OneLayerCARE, self).__init__()
        self.inter1 = inter1

        # Weighted CrossEntropyLoss — handles class imbalance (fewer spam than legit)
        self.xent = nn.CrossEntropyLoss(weight=class_weight)

        # Linear layer to transform embedding → class logits
        self.weight = nn.Parameter(torch.FloatTensor(inter1.embed_dim, num_classes))
        init.xavier_uniform_(self.weight)
        self.lambda_1 = lambda_1

    def forward(self, nodes, labels, train_flag=True):
        embeds1, label_scores = self.inter1(nodes, labels, train_flag)
        scores = torch.mm(embeds1, self.weight)
        return scores, label_scores

    def to_prob(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        gnn_prob = nn.functional.softmax(gnn_scores, dim=1)
        label_prob = nn.functional.softmax(label_scores, dim=1)
        return gnn_prob, label_prob

    def get_embeddings(self, nodes, labels, train_flag=False):
        """
        Return the node embeddings BEFORE the classification layer.
        Used for t-SNE visualization after training.
        :param nodes: list of node ids
        :param labels: node labels (needed by InterAgg even in eval mode)
        :return: FloatTensor [len(nodes), embed_dim]
        """
        embeds1, _ = self.inter1(nodes, labels, train_flag)
        return embeds1

    def loss(self, nodes, labels, train_flag=True):
        gnn_scores, label_scores = self.forward(nodes, labels, train_flag)
        # Simi loss, Eq. (4) in the paper
        label_loss = self.xent(label_scores, labels.squeeze())
        # GNN loss, Eq. (10) in the paper
        gnn_loss = self.xent(gnn_scores, labels.squeeze())
        # Final combined loss, Eq. (11) in the paper
        final_loss = gnn_loss + self.lambda_1 * label_loss
        return final_loss
