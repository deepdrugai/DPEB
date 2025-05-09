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
    
class GIN_old(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, mlp_hidden_dim, final_dim):
        super(GIN_old, self).__init__()
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
            # Create a GINConv layer (using the 'sum' aggregator)
            self.layers.append(dglnn.GINConv(mlp, 'sum'))
        
        # Predictor for link prediction
        self.predictor = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Linear(hidden_feats // 2, final_dim)
        )
        self.layer_dims = mlp_hidden_dim

    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for link prediction.
        Args:
            pair_graph: DGLGraph for positive edge examples.
            neg_pair_graph: DGLGraph for negative edge examples.
            blocks: List of DGLBlocks (one per layer) from neighbor sampling.
            x (Tensor): Node features for the block (from blocks[0].srcdata).
            e: (Unused) edge features (kept for interface compatibility).
        Returns:
            h_pos: Logits for positive edges.
            h_neg: Logits for negative edges.
            h: Final node embeddings.
        """
        h = x
        # Iterate over the layers and use the corresponding block
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
            h = F.relu(h)
        # Retrieve edge indices for positive and negative examples
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # Compute link prediction scores using an element-wise product followed by the predictor
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h
    
    
    def inference(self, g, feature, device, batch_size):
        """
        Compute node embeddings for the full graph.
    
        Args:
            g: The full DGLGraph.
            feature (str): Key in g.ndata for the input features.
            device: The target device (e.g., 'cuda:0').
            batch_size: Batch size (unused in this simple implementation).
    
        Returns:
            Tensor: Node embeddings after applying all layers.
        """
        # Move the graph to the same device.
        g = g.to(device)
        # Get the initial node features from the full graph and move them to the device.
        feat = g.ndata[feature].to(device)
        # Apply each GIN layer to the full graph.
        for layer in self.layers:
            feat = F.relu(layer(g, feat))
        return feat
    

# model.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class RGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, out_feats, num_rels=1, num_bases=None):
        """
        RGCN model for link prediction using block-based training.
        
        Args:
            in_feats (int): Dimension of input node features.
            hidden_feats (int): Hidden dimension for intermediate layers.
            num_layers (int): Total number of RGCN layers.
            out_feats (int): Dimension of the final node embeddings.
            num_rels (int): Number of relations. For a homogeneous graph, use 1.
            num_bases (int, optional): Number of bases for basis decomposition (if desired).
        """
        super(RGCN, self).__init__()
        self.layers = nn.ModuleList()
        # First layer: from in_feats to hidden_feats.
        self.layers.append(dglnn.RelGraphConv(in_feats, hidden_feats, num_rels, "bdd", num_bases=num_bases))
        # Intermediate layers.
        for i in range(num_layers - 2):
            self.layers.append(dglnn.RelGraphConv(hidden_feats, hidden_feats, num_rels, "bdd", num_bases=num_bases))
        # Final layer: from hidden_feats to out_feats.
        self.layers.append(dglnn.RelGraphConv(hidden_feats, out_feats, num_rels, "bdd", num_bases=num_bases))
        
        # Predictor MLP for link prediction.
        self.predictor = nn.Sequential(
            nn.Linear(out_feats, out_feats // 2),
            nn.ReLU(),
            nn.Linear(out_feats // 2, 1)
        )
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, rel=None):
        """
        Forward pass using neighbor-sampled blocks.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLGraph): List of sampled blocks (one per RGCN layer).
            x (Tensor): Input node features from blocks[0].srcdata.
            rel (Tensor, optional): Relation types for edges. For homogeneous graphs, if None,
                                    a tensor of zeros is used for each block.
        
        Returns:
            h_pos (Tensor): Logits for positive edges.
            h_neg (Tensor): Logits for negative edges.
            h (Tensor): Final node embeddings (for the source nodes in blocks[0]).
        """
        h = x
        for layer, block in zip(self.layers, blocks):
            # For each block, create an edge type tensor of zeros with the correct shape.
            current_rel = th.zeros(block.num_edges(), dtype=th.long, device=x.device)
            h = layer(block, h, current_rel)
            h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h

    def inference(self, g, feature, device, batch_size):
        """
        Full-graph inference: compute node embeddings for the entire graph.
        
        Args:
            g (DGLGraph): The full graph.
            feature (str): Key in g.ndata holding node features.
            device (str): Target device (e.g., 'cuda:0').
            batch_size (int): (Unused in this simple version.)
        
        Returns:
            Tensor: Node embeddings computed for all nodes in the graph.
        """
        g = g.to(device)
        feat = g.ndata[feature].to(device)
        # For homogeneous graphs, assign all edges relation 0.
        rel = th.zeros(g.num_edges(), dtype=th.long, device=device)
        for layer in self.layers:
            feat = F.relu(layer(g, feat, rel))
        return feat


# model.py
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import tqdm

##############################################
# Graph Transformer Layer
##############################################
class GraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A simplified transformer layer for node embeddings.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayer, self).__init__()
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

##############################################
# GTN Model for Link Prediction
##############################################
class GTN(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        """
        A simplified Graph Transformer Network (GTN) for link prediction.
        
        Args:
            in_feats (int): Input feature dimension.
            embed_dim (int): Embedding dimension (projected from input features).
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GTN, self).__init__()
        # Project input node features to the transformer embedding dimension.
        self.input_proj = nn.Linear(in_feats, embed_dim)
        # Stack transformer layers.
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # A simple predictor for link prediction.
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward2(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training using neighbor-sampled blocks.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLBlocks): Blocks from neighbor sampling.
            x (Tensor): Node features from blocks[0].srcdata.
            e: (Unused) for compatibility.
        
        Returns:
            h_pos (Tensor): Logits for positive edges.
            h_neg (Tensor): Logits for negative edges.
            h (Tensor): Final node embeddings.
        """
        # Project the input features.
        h = self.input_proj(x)
        # (Optional) apply dropout after projection.
        h = self.dropout(h)
        # Although blocks are provided for compatibility, we apply the transformer layers
        # to the entire set of node embeddings from the current block.
        for layer, block in zip(self.layers, blocks):
            h = layer(h)
        # Get positive and negative edge index pairs.
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        # For link prediction, compute element-wise product of node embeddings followed by the predictor.
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, 
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training using neighbor-sampled blocks with a skip connection.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLBlocks): Blocks from neighbor sampling (for compatibility).
            x (Tensor): Input node features from blocks[0].srcdata.
            e: (Unused) for compatibility.
        
        Returns:
            pos_score (Tensor): Logits for positive edges.
            neg_score (Tensor): Logits for negative edges.
            h_combined (Tensor): Final node embeddings (after skip connection).
        """
        # 1. Project the input features.
        h_initial = self.input_proj(x)    # shape: (N, embed_dim)
        h_proj = self.dropout(h_initial)
        
        # 2. Pass through the transformer layers.
        h = h_proj
        for layer, _ in zip(self.layers, blocks):
            h = layer(h)  # Note: We ignore the block structure in this simplified version.
        
        # 3. Add a skip connection: combine the output of the transformer layers with the
        #    original projected (and dropped-out) features.
        h_combined = h + h_proj  # simple elementwise addition as a residual connection.
        
        # 4. Compute link prediction scores using element-wise product of node embeddings.
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        pos_score = self.predictor(h_combined[pos_src] * h_combined[pos_dst])
        neg_score = self.predictor(h_combined[neg_src] * h_combined[neg_dst])
        
        return pos_score, neg_score, h_combined
    
        
    

    def inference(self, g, feature, device, batch_size):
        """
        Full-graph inference: Compute node embeddings for the entire graph.
        
        Args:
            g (DGLGraph): The full graph.
            feature (str): Key in g.ndata for node features.
            device (str): Target device (e.g. 'cuda:0').
            batch_size (int): Batch size (unused in this simple version).
        
        Returns:
            Tensor: Node embeddings.
        """
        feat = g.ndata[feature].to(device)
        h = self.input_proj(feat)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        return h
    

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
    

