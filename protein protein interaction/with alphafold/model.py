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
import torch 
from dgl.data import DGLDataset
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    GraphDataLoader,
    MultiLayerFullNeighborSampler
)
# =============================================================================
# GAT original working well
# =============================================================================
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
                dataloader, desc="SAGE Model Inference"
            ):
                x = feat[input_nodes]
                w = blocks[0].edata["weight"]
                h = layer(blocks[0], x, edge_weight=w)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
    
class GCN(nn.Module):
    """
    A three-layer Graph Convolutional Network (GCN) for link prediction.
    
    This implementation follows the original idea in [Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907) 
    but is adapted to work in the DGL framework and with your link-prediction training loop.
    
    Args:
        in_size (int): Dimensionality of input features.
        hidden_dims (list of int): A list with three integers defining the output dimensions for 
                                   each GCN layer. For example, [512, 256, 160].
    """
    def __init__(self, in_size, hidden_dims):
        super(GCN, self).__init__()
        # Create three GraphConv layers
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_size, hidden_dims[0], allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hidden_dims[0], hidden_dims[1], allow_zero_in_degree=True))
        self.layers.append(dglnn.GraphConv(hidden_dims[1], hidden_dims[2], allow_zero_in_degree=True))
        # Save the output dimensions for each layer for inference
        self.layer_dims = hidden_dims

        # Define the predictor network for link prediction (applied on the element-wise product)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2] // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dims[2] // 2, hidden_dims[2] // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dims[2] // 4, 1),
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training.
        
        Args:
            pair_graph: DGLGraph containing positive edges.
            neg_pair_graph: DGLGraph containing negative edges.
            blocks: List of computation blocks (subgraphs) for multi-layer training.
            x (Tensor): Input node features.
            e: (Unused) edge features; kept for compatibility.
        
        Returns:
            h_pos (Tensor): Logits for positive edges.
            h_neg (Tensor): Logits for negative edges.
            h (Tensor): Final node embeddings.
        """
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # Retrieve the edge weights from the block
            w = block.edata["weight"]
            h = layer(block, h, edge_weight=w)
            # Apply non-linearity for all layers except the last one
            if l != len(self.layers) - 1:
                h = F.relu(h)
        # Retrieve edge indices for positive and negative examples
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # Compute link prediction scores using an element-wise product followed by the MLP predictor
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h

    def inference(self, g, feature, device, batch_size):
        """
        Layer-wise inference to compute node embeddings for the entire graph.
        
        This function mirrors the approach used in your SAGE model inference.
        
        Args:
            g: DGLGraph.
            feature (str): Key for node features in g.ndata.
            device: The target device.
            batch_size (int): Batch size for inference.
        
        Returns:
            Tensor: Node embeddings computed in a layer-wise fashion.
        """
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

        # Perform layer-wise inference
        for l, layer in enumerate(self.layers):
            out_dim = self.layer_dims[l]
            y = th.empty(
                g.num_nodes(),
                out_dim,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"GCN Model Inference (Layer {l})"
            ):
                x = feat[input_nodes]
                w = blocks[0].edata["weight"]
                h = layer(blocks[0], x, edge_weight=w)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y
    
    


# model.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import tqdm

##############################################
# Graph Transformer Layer
##############################################
class GraphTransformerLayer2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A simplified transformer layer for node embeddings.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayer2, self).__init__()
        # PyTorch's MultiheadAttention expects input shape (L, N, D)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, attn_mask=None):
        # x is expected to be of shape (N, embed_dim)
        # Reshape to (L, N, D) with L=1 (treating nodes as a sequence)
        x = x.unsqueeze(1)           # shape: (N, 1, embed_dim)
        x = x.transpose(0, 1)          # shape: (1, N, embed_dim)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.linear1(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        # Convert back to (N, embed_dim)
        x = x.transpose(0, 1).squeeze(1)
        return x


# =============================================================================
# 
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax

class CustomGATv2Layer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout=0.2, bias=True):
        super(CustomGATv2Layer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_fc = nn.Linear(out_feats, 1, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(num_heads * out_feats,)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.xavier_normal_(self.attn_fc.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def edge_attention(self, edges):
        score = edges.data['score']
        eweight = edges.data['weight'].unsqueeze(-1)
        bias = torch.log1p(eweight)  # Edge weight biasing
        return {'e': F.leaky_relu(score + bias)}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'v': edges.data['v']}

    def reduce_func(self, nodes):
        alpha = edge_softmax(nodes.mailbox['e'], nodes.mailbox['e'])
        h = torch.sum(alpha * nodes.mailbox['v'], dim=1)
        return {'h': h}

    def forward(self, g, h, edge_weight):
        with g.local_scope():
            h = self.fc(h).view(-1, self.num_heads, self.out_feats)

            # ✅ Flat assignment for block graphs
            g.ndata['h'] = h
            g.ndata['q'] = h
            g.ndata['k'] = h

            g.apply_edges(lambda edges: {
                'score': self.attn_fc(edges.src['q'] + edges.dst['k'])
            })

            g.edata['weight'] = edge_weight
            g.edata['v'] = g.srcdata['h'][g.edges()[0]]

            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)

            h_out = g.ndata['h']
            h_out = h_out.view(-1, self.num_heads * self.out_feats)

            if self.bias is not None:
                h_out = h_out + self.bias
            return h_out

class GATv2(nn.Module):  # WithEdgeWeights
    def __init__(self, in_size, hid_size, out_dim, num_heads):
        super(GATv2, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(CustomGATv2Layer(in_size, hid_size[0], num_heads))
        self.layers.append(CustomGATv2Layer(hid_size[0] * num_heads, hid_size[1], num_heads))
        self.layers.append(CustomGATv2Layer(hid_size[1] * num_heads, hid_size[2], 1))

        self.hid_size = hid_size
        self.out_dim = out_dim

        self.predictor = nn.Sequential(
            nn.Linear(hid_size[2], hid_size[2] // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_size[2] // 2, hid_size[2] // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_size[2] // 4, 1),
        )

        self.dropout = nn.Dropout(0.2)

        self.inference_dim_lst = [
            hid_size[0] * num_heads,
            hid_size[1] * num_heads,
            hid_size[2]
        ]

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            w = block.edata['weight']
            h = layer(block, h, edge_weight=w)
            if l != len(self.layers) - 1:
                h = F.leaky_relu(h)
        gat_embedding = h

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, gat_embedding

    def inference(self, g, feature, device, batch_size):
        feat = g.ndata[feature]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.inference_dim_lst[l],
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"GATv2 With EdgeWeights Inference (Layer {l})"
            ):
                x = feat[input_nodes]
                w = blocks[0].edata["weight"]
                h = layer(blocks[0], x, edge_weight=w)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y

# =============================================================================
# GIN new trying
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import tqdm

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, mlp_hidden_dim, final_dim):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_feats, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_dim, hidden_feats)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_feats, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden_dim, hidden_feats)
                )
            # GINConv with edge weight-aware aggregator
            self.layers.append(dglnn.GINConv(mlp, 'sum'))

        self.predictor = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Linear(hidden_feats // 2, final_dim)
        )

        self.layer_dims = mlp_hidden_dim

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h = x
        for layer, block in zip(self.layers, blocks):
            # Apply GIN layer with edge weights
            w = block.edata["weight"]
            # Scale messages manually via node features
            h = self.edge_weighted_gin(layer, block, h, w)
            h = F.relu(h)

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()

        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h

    def edge_weighted_gin(self, layer, block, x, edge_weight):
        """
        Modify GIN aggregation to include edge weights.
        We scale the message (source node feature) by edge weight manually.
        """
        with block.local_scope():
            block.srcdata['h'] = x
            block.edata['w'] = edge_weight
            # Message: scale source features by edge weight
            block.update_all(
                message_func=dgl.function.u_mul_e('h', 'w', 'm'),
                reduce_func=dgl.function.sum('m', 'agg')
            )
            h = block.dstdata['agg']
            return layer.apply_func(h) 

    def inference(self, g, feature, device, batch_size):
        """
        Inference with edge weight-aware GIN using full graph.
        """
        feat = g.ndata[feature]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            out_dim = layer.apply_func[-1].out_features

            y = torch.empty(
                g.num_nodes(),
                out_dim,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"GIN Inference (Layer {l})"
            ):
                x = feat[input_nodes]
                w = blocks[0].edata["weight"]
                h = self.edge_weighted_gin(layer, blocks[0], x, w)
                h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


# =============================================================================
# 
# =============================================================================
# gtn_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GTNLayer(nn.Module):
    def __init__(self, num_edge_types, in_dim, out_dim):
        super(GTNLayer, self).__init__()
        self.num_edge_types = num_edge_types
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Learnable edge-type selector
        self.weight = nn.Parameter(torch.Tensor(num_edge_types, 1, 1))
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, A_list, X):
        """
        A_list: list of adjacency matrices (edge_type-wise) [B x N x N]
        X: node feature matrix (N x in_dim)
        """
        # 1. Soft edge-type attention
        softmax_weight = F.softmax(self.weight, dim=0)  # (E, 1, 1)

        # 2. Weighted sum of adjacencies
        A = sum(w * A_i for w, A_i in zip(softmax_weight, A_list))  # (N x N)

        # 3. Propagate
        AX = torch.matmul(A, X)
        out = self.linear(AX)
        return out
# gtn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# from gtn_layer import GTNLayer

class GTN(nn.Module):
    def __init__(self, num_edge_types, in_dim, hidden_dim, num_layers=2):
        super(GTN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GTNLayer(num_edge_types, in_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(GTNLayer(num_edge_types, hidden_dim, hidden_dim))

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x, edge_weight=None):
        """
        pair_graph, neg_pair_graph: used for scoring
        blocks: not used for meta-path GCN, kept for compatibility
        x: node features (N x in_dim)
        edge_weight: not directly used in GTN; adjacency is inferred from multiple views
        """
        # Simulate multiple edge types using multiple edge-weighted views
        A_list = self.build_adjacency_variants(pair_graph, x.device)

        h = x
        for layer in self.layers:
            h = F.relu(layer(A_list, h))

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()

        pos_score = self.predictor(h[pos_src] * h[pos_dst])
        neg_score = self.predictor(h[neg_src] * h[neg_dst])
        return pos_score, neg_score, h

    def inference(self, g, feature, device, batch_size):
        X = g.ndata[feature].to(device)
        A_list = self.build_adjacency_variants(g, device)

        h = X
        for layer in self.layers:
            h = F.relu(layer(A_list, h))
        return h

    def build_adjacency_variants(self, g, device):
        """
        Simulate multiple edge types (adjacency variants) using edge features.
        In the original GTN, this would come from real typed relations.
        Here we construct different versions using edge weights or manipulations.
        """
        N = g.num_nodes()
        edge_index = g.edges()
        edge_weight = g.edata["weight"]

        A_dense = torch.zeros((N, N), device=device)
        A_dense[edge_index[0], edge_index[1]] = edge_weight

        # Build a few variants: original, symmetric, normalized
        A1 = A_dense
        A2 = (A_dense + A_dense.T) / 2
        D_inv = torch.diag(1 / (A_dense.sum(1) + 1e-6))
        A3 = D_inv @ A_dense  # simple normalized

        return [A1, A2, A3]



# =============================================================================
# GTN_g2
# =============================================================================
class GraphTransformerLayer3(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A simplified transformer layer for node embeddings with edge weight support.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayer3, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, edge_weights=None, edge_index=None):
        """
        Forward pass with edge weight incorporation.
        
        Args:
            x (Tensor): Node embeddings of shape (N, embed_dim).
            edge_weights (Tensor, optional): Edge weights from block.edata["weight"].
            edge_index (tuple, optional): Source and destination node indices (src, dst).
        
        Returns:
            Tensor: Updated node embeddings of shape (N, embed_dim).
        """
        # Reshape for MultiheadAttention: (N, embed_dim) -> (1, N, embed_dim)
        x_input = x.unsqueeze(0)  # shape: (1, N, embed_dim)

        # Compute attention
        attn_output, attn_weights = self.self_attn(x_input, x_input, x_input, attn_mask=None)
        
        # If edge weights are provided, adjust the attention output
        if edge_weights is not None and edge_index is not None:
            src, dst = edge_index
            src = src.to(x.device)
            dst = dst.to(x.device)
            edge_weights = edge_weights.to(x.device)
            
            # attn_weights shape: (1, N, N) for single sequence
            # Create a sparse weight adjustment based on edge_index and edge_weights
            N = x.shape[0]
            weight_adjust = torch.zeros(1, N, N, device=x.device)
            weight_adjust[0, src, dst] = edge_weights  # Place weights at corresponding edges
            
            # Apply edge weights to attention output
            # Since attn_output is (1, N, embed_dim), we scale it using the weighted attention
            # attn_output = attn_output * weight_adjust.sum(dim=1, keepdim=True)  # Aggregate weights per node
            weight_sum = weight_adjust.sum(dim=2, keepdim=True)  # shape: (1, N, 1)
            attn_output = attn_output * weight_sum

        
        # Residual connection and normalization
        x = x_input + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ff_output = self.linear1(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        # Convert back to (N, embed_dim)
        x = x.squeeze(0)
        return x
    
    
class GTN_g2(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        super(GTN_g2, self).__init__()
        self.input_proj = nn.Linear(in_feats, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer3(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h_initial = self.input_proj(x)
        h_proj = self.dropout(h_initial)
        
        h = h_proj
        for layer, block in zip(self.layers, blocks):
            w = block.edata["weight"]
            src, dst = block.edges()  # Get edge indices from the block
            h = layer(h, edge_weights=w, edge_index=(src, dst))
        
        h_combined = h + h_proj
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        
        pos_score = self.predictor(h_combined[pos_src] * h_combined[pos_dst])
        neg_score = self.predictor(h_combined[neg_src] * h_combined[neg_dst])
        
        return pos_score, neg_score, h_combined

    def inference(self, g, feature, device, batch_size):
        feat = g.ndata[feature].to(device)
        h = self.input_proj(feat)
        h = self.dropout(h)
        
        w = g.edata.get("weight", None)
        src, dst = g.edges() if w is not None else (None, None)
        for layer in self.layers:
            h = layer(h, edge_weights=w, edge_index=(src, dst))
        
        return h
    
