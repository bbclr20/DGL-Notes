import dgl
import dgl.data
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


def train(g, model, epoch=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(epoch):
        logits = model(g, features)
        pred = logits.argmax(1)

        # Compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
        if best_test_acc < test_acc:
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"Epoch {e}, loss: {loss:.3f}, train acc: {train_acc:.3f}, val acc: {val_acc:.3f}, test acc: {test_acc:.3f}, (best_test {best_test_acc:.3f}))"
            )


if __name__ == "__main__":
    dataset = dgl.data.CoraGraphDataset()
    print(f"Number of categories: {dataset.num_classes}")

    g = dataset[0]  # this dataset has only one graph
    print(f"len(dataset): {len(dataset)}")
    print(f"g.ndata['feat'].shape: {g.ndata['feat'].shape}")
    print(f"g.ndata['label'].shape: {g.ndata['label'].shape}")

    model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
    train(g, model)
