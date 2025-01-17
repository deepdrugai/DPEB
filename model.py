from dgl import DGLGraph
import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import numpy as np

from dgl.data import DGLDataset
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    GraphDataLoader,
    MultiLayerFullNeighborSampler
)

class GAT(th.nn.Module):
  def __init__(self, in_size, hid_size, out_dim, num_heads):
    super(GAT, self).__init__()

    self.layers = nn.ModuleList()

    # four-layer GraphSAGE-mean
    self.layers.append(dglnn.GATConv(in_size                , hid_size[0], num_heads,  allow_zero_in_degree=True))
    self.layers.append(dglnn.GATConv(hid_size[0] * num_heads, hid_size[1], num_heads,  allow_zero_in_degree=True))
    #self.layers.append(dglnn.GATConv(hid_size[1] * num_heads, hid_size[2], num_heads,  allow_zero_in_degree=True))
    self.layers.append(dglnn.GATConv(hid_size[1] * num_heads, hid_size[2], 1        ,  allow_zero_in_degree=True))

    # four-layer GraphSAGE-mean
    #self.layers.append(dglnn.GATConv(in_size                , hid_size[0], num_heads))
    #self.layers.append(dglnn.GATConv(hid_size[0] * num_heads, hid_size[1], num_heads))
    ##self.layers.append(dglnn.GATConv(hid_size[1] * num_heads, hid_size[2], num_heads,  allow_zero_in_degree=True))
    #self.layers.append(dglnn.GATConv(hid_size[1] * num_heads, hid_size[2], 1))

    self.hid_size = hid_size
    self.out_dim = out_dim

    self.predictor = nn.Sequential(
      nn.Linear(hid_size[2], hid_size[2]//2),
      nn.LeakyReLU(negative_slope=0.2),  #nn.LeakyReLU(),
      nn.Linear(hid_size[2]//2, hid_size[2]//4),
      nn.LeakyReLU(negative_slope=0.2),
      nn.Linear(hid_size[2]//4, 1),
    )
    self.dropout = nn.Dropout(0.2)

    self.inference_dim_lst= [hid_size[0] * num_heads,
                             hid_size[1] * num_heads,
                             hid_size[2] ]
                             #hid_size[2] * num_heads]
                             #hid_size[3] ]

  def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
      h = x
      for l, (layer, block) in enumerate(zip(self.layers, blocks)):
          w = block.edata["weight"]
          h = layer(block, h, edge_weight=w)
          #h = layer(block, h)
          h = h.view(-1, h.size(1) * h.size(2))
          if l != len(self.layers) - 1:
            h = F.leaky_relu(h)

      gat_embedding = h

      pos_src, pos_dst = pair_graph.edges()
      neg_src, neg_dst = neg_pair_graph.edges()
      h_pos = self.predictor(h[pos_src] * h[pos_dst])
      h_neg = self.predictor(h[neg_src] * h[neg_dst])
      return h_pos, h_neg, gat_embedding

  def inference(self, g, feature, device, batch_size):
    """Layer-wise inference algorithm to compute GNN node embeddings."""
    feat = g.ndata[feature]
    sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
    #sampler = GATMultiLayerFullNeighborSampler(1)# prefetch_node_feats=["ESM_Embeddings"])
    dataloader = DataLoader(
      g,
      th.arange(g.num_nodes()).to(g.device),
      sampler,
      device=device,
      batch_size=batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=0,
    )
    buffer_device = th.device("cpu")
    pin_memory = buffer_device != device

    for l, layer in enumerate(self.layers):

      y = th.empty(
        g.num_nodes(),
        self.inference_dim_lst[l],
        device=buffer_device,
        pin_memory=pin_memory,
      )
      feat = feat.to(device)
      for input_nodes, output_nodes, blocks in tqdm.tqdm(
        dataloader, desc="GAT Model Inference"
      ):
        x = feat[input_nodes]
        w = blocks[0].edata["weight"]
        h = layer(blocks[0], x, edge_weight=w)
        h = h.view(-1, h.size(1) * h.size(2))
        if l != len(self.layers) - 1:
          h = F.relu(h)
        #h = self.dropout(h)

        y[output_nodes] = h.to(buffer_device)
      feat = y
    return y


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        #self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h = x
        #w = e
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            w = block.edata["weight"]
            h = layer(block, h, edge_weight=w)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h

    def inference(self, g, feature, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata[feature]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
        dataloader = DataLoader(
            g,
            th.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = th.device("cpu")
        pin_memory = buffer_device != device
        for l, layer in enumerate(self.layers):
            y = th.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="GCN Model Inference"
            ):
                x = feat[input_nodes]
                w = blocks[0].edata["weight"]
                h = layer(blocks[0], x, edge_weight=w)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y