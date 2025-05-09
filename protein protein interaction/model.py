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
    
    
# =============================================================================
#     GIN working but with no edgeweights
# =============================================================================
class GIN2(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_layers, mlp_hidden_dim, final_dim):
        super(GIN2, self).__init__()
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

##############################################
# GTN Model for Link Prediction
##############################################
class GTN_old(nn.Module):
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
        super(GTN_old, self).__init__()
        # Project input node features to the transformer embedding dimension.
        self.input_proj = nn.Linear(in_feats, embed_dim)
        # Stack transformer layers.
        self.layers = nn.ModuleList([
            GraphTransformerLayer2(embed_dim, num_heads, dropout)
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

##############################################
# New GTN Model for Link Prediction
##############################################
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import tqdm

##############################################
# Graph Transformer Layer with Edge Weights   _grok
##############################################
class GraphTransformerLayer_grok(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A graph-aware transformer layer that incorporates edge weights.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayer_grok, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward2(self, block, h):
        """
        Forward pass incorporating edge weights from the block.
        
        Args:
            block (DGLBlock): The computation block containing graph structure and edge weights.
            h (Tensor): Node embeddings of shape (num_nodes, embed_dim).
        
        Returns:
            Tensor: Updated node embeddings after attention and feed-forward layers.
        """
        # Get edge weights from the block
        edge_weights = block.edata["weight"].float()  # Shape: (num_edges,)

        # Project to query, key, value
        q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)  # (N, num_heads, head_dim)
        k = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(-1, self.num_heads, self.head_dim)

        # Get source and destination node indices from the block
        src, dst = block.edges()
        num_nodes = h.size(0)

        # Compute attention scores (scaled dot-product attention)
        attn_scores = th.einsum("nhd,mhd->nmh", q, k)  # (N, N, num_heads) if fully connected
        attn_scores = attn_scores / (self.head_dim ** 0.5)  # Scale by sqrt(d_k)

        # Create an adjacency mask based on the graph structure (sparse attention)
        attn_mask = th.zeros(num_nodes, num_nodes, self.num_heads, device=h.device)
        attn_mask[src, dst] = edge_weights.unsqueeze(-1).expand(-1, self.num_heads)  # Apply edge weights
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))  # Non-edges get -inf

        # Apply edge-weighted attention
        attn_scores = attn_scores + attn_mask  # Shape: (N, N, num_heads)
        attn_probs = F.softmax(attn_scores, dim=1)  # Normalize over source nodes
        attn_probs = self.dropout(attn_probs)

        # Compute attention output
        attn_output = th.einsum("nmh,mhd->nhd", attn_probs, v)  # (N, num_heads, head_dim)
        attn_output = attn_output.reshape(-1, self.embed_dim)  # (N, embed_dim)
        h_out = self.out_proj(attn_output)

        # Residual connection and normalization
        h = self.norm1(h + self.dropout(h_out))

        # Feed-forward network
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout(ffn_out))

        return h
    
    def forward(self, block, h):
        edge_weights = block.edata["weight"].float()
        edge_weights = th.clamp(edge_weights, min=0.0)  # Ensure non-negative weights
    
        q = self.q_proj(h).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(-1, self.num_heads, self.head_dim)
    
        src, dst = block.edges()
        num_nodes = h.size(0)
    
        attn_scores = th.einsum("nhd,mhd->nmh", q, k) / (self.head_dim ** 0.5)
        attn_mask = th.zeros(num_nodes, num_nodes, self.num_heads, device=h.device)
        attn_mask[src, dst] = edge_weights.unsqueeze(-1).expand(-1, self.num_heads)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
    
        attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores + 1e-8, dim=1)  # Add epsilon for stability
        attn_probs = self.dropout(attn_probs)
    
        attn_output = th.einsum("nmh,mhd->nhd", attn_probs, v)
        h_out = self.out_proj(attn_output.reshape(-1, self.embed_dim))
        h = self.norm1(h + self.dropout(h_out))
        ffn_out = self.ffn(h)
        h = self.norm2(h + self.dropout(ffn_out))
    
        return h

##############################################
# GTN Model with Edge Weights   ## grok
##############################################
class GTN_grok(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        """
        Graph Transformer Network (GTN) for link prediction with edge weights.
        
        Args:
            in_feats (int): Input feature dimension.
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GTN_grok, self).__init__()
        self.input_proj = nn.Linear(in_feats, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training using neighbor-sampled blocks with edge weights.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLBlocks): Blocks from neighbor sampling.
            x (Tensor): Node features from blocks[0].srcdata.
            e: (Unused) for compatibility.
        
        Returns:
            pos_score (Tensor): Logits for positive edges.
            neg_score (Tensor): Logits for negative edges.
            h (Tensor): Final node embeddings.
        """
        # Project input features
        h = self.input_proj(x)
        h = self.dropout(h)

        # Apply transformer layers with edge weights
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)

        # Compute link prediction scores
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        pos_score = self.predictor(h[pos_src] * h[pos_dst])
        neg_score = self.predictor(h[neg_src] * h[neg_dst])

        return pos_score, neg_score, h
    


    def inference(self, g, feature, device, batch_size):
        """
        Full-graph inference: Compute node embeddings for the entire graph.
        
        Args:
            g (DGLGraph): The full graph.
            feature (str): Key in g.ndata for node features.
            device (str): Target device (e.g., 'cuda:0').
            batch_size (int): Batch size for inference.
        
        Returns:
            Tensor: Node embeddings.
        """
        feat = g.ndata[feature].to(device)
        h = self.input_proj(feat)
        h = self.dropout(h)

        # Use a full-graph sampler for inference
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloader = dgl.dataloading.DataLoader(
            g,
            th.arange(g.num_nodes()).to(device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )

        # Buffer for output embeddings
        output = th.zeros(g.num_nodes(), h.size(1), device=device)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc="GTN Inference"):
            h_batch = h[input_nodes]
            for layer, block in zip(self.layers, blocks):
                h_batch = layer(block, h_batch)
            output[output_nodes] = h_batch

        return output


# =============================================================================
# claude
# =============================================================================


##############################################
# Fixed Graph Transformer Layer with Edge Weights
##############################################
class EnhancedGraphTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        An enhanced transformer layer that incorporates edge weights.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(EnhancedGraphTransformerLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention with manual implementation to incorporate edge weights
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        # Edge weight projection
        self.edge_weight_proj = nn.Linear(1, 1)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, block=None):
        """
        Forward pass incorporating edge weights to modify attention scores.
        
        Args:
            x (Tensor): Node features of shape (N, embed_dim)
            block (DGLBlock): Block containing edge information
            
        Returns:
            Tensor: Updated node features
        """
        # First normalization and self-attention
        residual = x
        x = self.norm1(x)
        
        batch_size = x.size(0)
        head_dim = self.embed_dim // self.num_heads
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, self.num_heads, head_dim)
        k = self.k_proj(x).view(batch_size, self.num_heads, head_dim)
        v = self.v_proj(x).view(batch_size, self.num_heads, head_dim)
        
        # Scaled dot-product attention
        scores = th.bmm(q, k.transpose(1, 2)) / (head_dim ** 0.5)
        
        # Incorporate edge weights if block is provided
        if block is not None and "weight" in block.edata:
            # Create an adjacency matrix from the block
            adj = th.zeros(batch_size, batch_size, device=x.device)
            src, dst = block.edges()
            edge_weights = block.edata["weight"]
            
            # Project edge weights
            edge_weights = self.edge_weight_proj(edge_weights.unsqueeze(1)).squeeze(1)
            
            # Fill adjacency matrix with edge weights
            for i in range(len(src)):
                s, d = src[i].item(), dst[i].item()
                if s < batch_size and d < batch_size:  # Ensure indices are in range
                    adj[s, d] = edge_weights[i]
            
            # Add edge weight information to attention scores
            # We're adding to preserve the original attention mechanism while incorporating edge weights
            adj = adj.unsqueeze(0).expand(self.num_heads, -1, -1)
            adj = adj.reshape(1, self.num_heads, batch_size, batch_size)
            scores = scores.unsqueeze(0) + adj
            scores = scores.squeeze(0)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        out = th.bmm(attn_weights, v)
        out = out.view(batch_size, self.embed_dim)
        out = self.o_proj(out)
        
        # First residual connection
        x = residual + self.dropout(out)
        
        # Feedforward network with residual connection
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        
        return x

##############################################
# Fixed GTN Model for Link Prediction with Edge Weights
##############################################
class GTN_claude(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        """
        An enhanced Graph Transformer Network (GTN) that utilizes edge weights for link prediction.
        
        Args:
            in_feats (int): Input feature dimension.
            embed_dim (int): Embedding dimension (projected from input features).
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GTN_claude, self).__init__()
        # Project input node features to the transformer embedding dimension.
        self.input_proj = nn.Linear(in_feats, embed_dim)
        # Stack enhanced transformer layers.
        self.layers = nn.ModuleList([
            EnhancedGraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # A simple predictor for link prediction.
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training using neighbor-sampled blocks with edge weight incorporation.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLBlocks): Blocks from neighbor sampling.
            x (Tensor): Input node features from blocks[0].srcdata.
            e: (Unused) for compatibility.
        
        Returns:
            pos_score (Tensor): Logits for positive edges.
            neg_score (Tensor): Logits for negative edges.
            h_combined (Tensor): Final node embeddings.
        """
        # 1. Project the input features.
        h_initial = self.input_proj(x)    # shape: (N, embed_dim)
        h_proj = self.dropout(h_initial)
        
        # 2. Pass through the enhanced transformer layers with edge weights
        h = h_proj
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            # Apply the enhanced transformer layer with block containing edge weights
            h = layer(h, block)
        
        # 3. Add a skip connection to preserve important initial features
        h_combined = h + h_proj
        
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
            batch_size (int): Batch size for batched inference.
        
        Returns:
            Tensor: Node embeddings.
        """
        # Move the graph to the device if needed
        if g.device != device:
            g = g.to(device)
            
        feat = g.ndata[feature].to(device)
        
        # Project input features
        h = self.input_proj(feat)
        h = self.dropout(h)
        
        # Apply transformer layers - for inference, we'll process in batches
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
        
        for layer_idx, layer in enumerate(self.layers):
            y = th.empty(
                g.num_nodes(),
                h.size(1),
                device=buffer_device,
                pin_memory=pin_memory,
            )
            
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"GTN Model Inference (Layer {layer_idx})"
            ):
                block = blocks[0]
                x_batch = h[input_nodes].to(device)
                h_batch = layer(x_batch, block)
                y[output_nodes] = h_batch.to(buffer_device)
            
            h = y
            
        return h
   
# =============================================================================
# gpt 4o  its working
# =============================================================================
    
class GraphTransformerLayer4o(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphTransformerLayer4o, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, edge_index=None, edge_weight=None, num_nodes=None):
        """
        Args:
            x: Node embeddings (N, D)
            edge_index: (2, E) - Edge list (src, dst)
            edge_weight: (E,) - Edge weights
            num_nodes: Total number of nodes
        """
        N = x.size(0)
        # Step 1: Build attention bias matrix
        attn_bias = torch.full((N, N), float('-inf'), device=x.device)  # initialize with -inf for masked attention
        attn_bias.fill_diagonal_(0)  # allow self-attention

        if edge_index is not None and edge_weight is not None:
            src, dst = edge_index
            attn_bias[dst, src] = edge_weight  # Attention flows: dst attends to src

        # Normalize weights (optional)
        attn_bias = attn_bias.masked_fill(attn_bias == float('-inf'), -1e9)
        attn_bias = attn_bias.unsqueeze(0).repeat(self.self_attn.num_heads, 1, 1)  # (H, N, N)

        # Step 2: Apply attention (MultiheadAttention expects (N, D), batch_first=True allows (B, N, D))
        x_input = x.unsqueeze(0)  # (1, N, D)
        attn_output, _ = self.self_attn(x_input, x_input, x_input, attn_mask=attn_bias)
        x = x_input + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.linear1(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        return x.squeeze(0)  # Return shape (N, D)
class GTN4o(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        super(GTN4o, self).__init__()
        self.input_proj = nn.Linear(in_feats, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer4o(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h_initial = self.input_proj(x)
        h_proj = self.dropout(h_initial)

        h = h_proj
        for layer, block in zip(self.layers, blocks):
            edge_index = block.edges()
            edge_weight = block.edata["weight"]
            edge_index = torch.stack(edge_index, dim=0)
            h = layer(h, edge_index=edge_index, edge_weight=edge_weight, num_nodes=h.size(0))

        h_combined = h + h_proj  # skip connection

        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        pos_score = self.predictor(h_combined[pos_src] * h_combined[pos_dst])
        neg_score = self.predictor(h_combined[neg_src] * h_combined[neg_dst])
        return pos_score, neg_score, h_combined
    
    def inference(self, g, feature, device, batch_size):
        """
        Compute node embeddings for the full graph using transformer layers.
    
        Args:
            g: DGLGraph (full graph).
            feature: str, the key for node features in g.ndata.
            device: the device to run inference on.
            batch_size: unused in this simplified implementation.
    
        Returns:
            Tensor: Final node embeddings after transformer layers.
        """
        g = g.to(device)
        x = g.ndata[feature].to(device)
        h_initial = self.input_proj(x)
        h_proj = self.dropout(h_initial)
    
        h = h_proj
        edge_index = torch.stack(g.edges(), dim=0)
        edge_weight = g.edata["weight"]
    
        for layer in self.layers:
            h = layer(h, edge_index=edge_index, edge_weight=edge_weight, num_nodes=g.num_nodes())
    
        h_combined = h + h_proj
        return h_combined

# =============================================================================
# gpt 03 mini high
# =============================================================================
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer_high(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A simplified transformer layer for node embeddings.
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayer_high, self).__init__()
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

class GTN_high(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        """
        A simplified Graph Transformer Network (GTN) for link prediction that now utilizes edge weights.
        
        Args:
            in_feats (int): Input feature dimension.
            embed_dim (int): Embedding dimension (projected from input features).
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GTN_high, self).__init__()
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
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training with edge weights incorporated in link prediction.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs with edata["weight"].
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs with edata["weight"].
            blocks (list of DGLBlocks): Blocks from neighbor sampling (used here only to set the number of layers).
            x (Tensor): Input node features from blocks[0].srcdata.
            e: (Unused) Additional edge features.
        
        Returns:
            pos_score (Tensor): Logits for positive edges.
            neg_score (Tensor): Logits for negative edges.
            h_combined (Tensor): Final node embeddings after applying the transformer layers and a skip connection.
        """
        # 1. Project the input features and apply dropout.
        h_initial = self.input_proj(x)    # shape: (N, embed_dim)
        h_proj = self.dropout(h_initial)
        
        # 2. Pass through the transformer layers.
        h = h_proj
        for layer, _ in zip(self.layers, blocks):
            h = layer(h)
        
        # 3. Add a skip connection.
        h_combined = h + h_proj
        
        # 4. Retrieve positive and negative edge indices.
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        
        # 5. Retrieve and utilize edge weights. If edge weights are missing, default to 1.
        if "weight" in pair_graph.edata:
            pos_weight = pair_graph.edata["weight"].unsqueeze(-1)  # shape: (num_pos_edges, 1)
        else:
            pos_weight = 1.0
        
        if "weight" in neg_pair_graph.edata:
            neg_weight = neg_pair_graph.edata["weight"].unsqueeze(-1)  # shape: (num_neg_edges, 1)
        else:
            neg_weight = 1.0
        
        # 6. Compute element-wise product of node embeddings for each edge and scale by the corresponding edge weight.
        pos_edge_repr = (h_combined[pos_src] * h_combined[pos_dst]) * pos_weight
        neg_edge_repr = (h_combined[neg_src] * h_combined[neg_dst]) * neg_weight
        
        # 7. Compute link prediction scores.
        pos_score = self.predictor(pos_edge_repr)
        neg_score = self.predictor(neg_edge_repr)
        
        return pos_score, neg_score, h_combined

    def inference(self, g, feature, device, batch_size):
        """
        Full-graph inference: Compute node embeddings for the entire graph.
        
        Args:
            g (DGLGraph): The full graph.
            feature (str): Key in g.ndata for node features.
            device (str): Target device (e.g., 'cuda:0').
            batch_size (int): Batch size (unused in this simplified version).
        
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
# o3 mini
# =============================================================================
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class GraphTransformerLayermini(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        A simplified transformer layer for node embeddings.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GraphTransformerLayermini, self).__init__()
        # PyTorch's MultiheadAttention expects input shape (L, N, D)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, attn_mask=None):
        # x is expected to be of shape (N, embed_dim)
        # Reshape to (L, N, D) with L=1 (treating nodes as a sequence)
        x = x.unsqueeze(1)            # shape: (N, 1, embed_dim)
        x = x.transpose(0, 1)         # shape: (1, N, embed_dim)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ff_output = self.linear1(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        # Convert back to (N, embed_dim)
        x = x.transpose(0, 1).squeeze(1)
        return x

class GTNmini(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        """
        A Graph Transformer Network (GTN) for link prediction that now incorporates edge weights.
        
        Args:
            in_feats (int): Input feature dimension.
            embed_dim (int): Embedding dimension (projected from input features).
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super(GTNmini, self).__init__()
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
    
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Forward pass for training using neighbor-sampled blocks with edge weights.
        
        Args:
            pair_graph (DGLGraph): Graph containing positive edge pairs.
            neg_pair_graph (DGLGraph): Graph containing negative edge pairs.
            blocks (list of DGLGraph): Blocks from neighbor sampling. Each block is assumed to have 
                                       'edata["weight"]' for its edges.
            x (Tensor): Input node features from blocks[0].srcdata.
            e: (Unused) for compatibility.
        
        Returns:
            pos_score (Tensor): Logits for positive edges.
            neg_score (Tensor): Logits for negative edges.
            h_combined (Tensor): Final node embeddings (after a skip connection).
        """
        # 1. Project the input features.
        h_initial = self.input_proj(x)
        h_proj = self.dropout(h_initial)
        
        # h is computed for all source nodes in the first block.
        h = h_proj
        
        for layer, block in zip(self.layers, blocks):
            # Compute transformer update on all source nodes.
            h_trans_full = layer(h)
            # Destination nodes are conventionally the last block.num_dst_nodes() in the source ordering.
            num_dst = block.num_dst_nodes()
            h_trans = h_trans_full[-num_dst:]
            
            # --- Edge-weighted neighbor aggregation ---
            # Use the source features (h) for message passing.
            block.srcdata['h'] = h
            # Unsqueeze edge weights to match feature dimensions.
            block.edata['w'] = block.edata["weight"].unsqueeze(-1)
            # Multiply source node features with edge weights and average them for each destination.
            block.update_all(message_func=fn.u_mul_e('h', 'w', 'm'),
                             reduce_func=fn.mean('m', 'h_neigh'))
            h_neigh = block.dstdata['h_neigh']
            
            # Combine transformer update and neighbor aggregation.
            h = h_trans + h_neigh
        
        # 3. Apply skip connection. Use the destination nodes from the final block.
        final_num_dst = blocks[-1].num_dst_nodes()
        h_combined = h + h_proj[-final_num_dst:]
        
        # 4. Compute link prediction scores using element-wise product.
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
            batch_size (int): Batch size (unused in this simplified version).
        
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
# deepseek
# =============================================================================
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class GraphTransformerLayerdeepseek(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphTransformerLayerdeepseek, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linear2 = nn.Linear(embed_dim * 2, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_weights=None):
        """
        Args:
            x (Tensor): Node features (N, embed_dim)
            edge_weights (Tensor, optional): Edge weights (N, N) for attention scaling.
        """
        x_in = x.unsqueeze(0)  # (1, N, D)
        
        # Compute attention (QK^T / sqrt(d_k))
        attn_output, attn_weights = self.self_attn(
            x_in, x_in, x_in,
            key_padding_mask=None,
            attn_mask=None,
        )  # attn_output shape: (1, N, D)

        # Apply edge weights (if provided)
        if edge_weights is not None:
            # Scale attention weights (not the output!)
            attn_weights = attn_weights * edge_weights.unsqueeze(0)  # (1, N, N) * (1, N, N)
            # Recompute attention output with scaled weights
            attn_output = th.bmm(attn_weights, x_in)  # (1, N, N) @ (1, N, D) → (1, N, D)

        # Residual + LayerNorm
        x = x + self.dropout(attn_output.squeeze(0))
        x = self.norm1(x)
        
        # FFN
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x
class GTNdeepseek(nn.Module):
    def __init__(self, in_feats, embed_dim, num_layers, num_heads, dropout=0.1):
        super(GTNdeepseek, self).__init__()
        self.input_proj = nn.Linear(in_feats, embed_dim)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(embed_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h = self.input_proj(x)
        h = self.dropout(h)
        
        for layer, block in zip(self.layers, blocks):
            # Get edge weights for the current block
            edge_weights = block.edata.get("weight", None)
            
            if edge_weights is not None:
                # Create a dense adjacency matrix (N x N)
                src, dst = block.edges()
                N = block.num_nodes()
                adj = th.zeros((N, N), device=x.device)
                adj[src, dst] = edge_weights
                h = layer(h, adj)  # Pass edge weights
            else:
                h = layer(h)  # No edge weights
        
        # Link prediction
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        pos_score = self.predictor(h[pos_src] * h[pos_dst])
        neg_score = self.predictor(h[neg_src] * h[neg_dst])
        
        return pos_score, neg_score, h
    
# =============================================================================
#     TAG2 not working
# =============================================================================
class TAG2(nn.Module):
    def __init__(self, in_size, hid_size, k=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.TAGConv(in_size, hid_size, k=k))
        self.layers.append(dglnn.TAGConv(hid_size, hid_size, k=k))
        self.layers.append(dglnn.TAGConv(hid_size, hid_size, k=k))
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
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # Convert block to homogeneous if needed
            if not block.is_homogeneous:
                block = dgl.to_homogeneous(block, ndata=['_N'], edata=['weight'])
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
        feat = g.ndata[feature]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
        dataloader = DataLoader(g, th.arange(g.num_nodes()).to(g.device), sampler, device=device, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        buffer_device = th.device("cpu")
        pin_memory = buffer_device != device
        for l, layer in enumerate(self.layers):
            y = th.empty(g.num_nodes(), self.hid_size, device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, desc="TAG Inference"):
                x = feat[input_nodes]
                block = blocks[0]
                if not block.is_homogeneous:
                    block = dgl.to_homogeneous(block, ndata=['_N'], edata=['weight'])
                w = block.edata["weight"]
                h = layer(block, x, edge_weight=w)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


import math
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv.hgtconv import HGTConv
from dgl import function as fn
from dgl.nn.pytorch.softmax import edge_softmax

class HGTConvWithEdgeWeight(HGTConv):
    """
    A subclass of HGTConv that incorporates external edge weights.
    Assumes that the input graph has an edge feature "w" (set from your external weight),
    which will be multiplied with the computed messages.
    """
    def forward(self, g, x, ntype, etype, *, presorted=False):
        self.presorted = presorted
        if g.is_block:
            x_src = x
            x_dst = x[: g.num_dst_nodes()]
            srcntype = ntype
            dstntype = ntype[: g.num_dst_nodes()]
        else:
            x_src = x
            x_dst = x
            srcntype = ntype
            dstntype = ntype
        with g.local_scope():
            k = self.linear_k(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            q = self.linear_q(x_dst, dstntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            v = self.linear_v(x_src, srcntype, presorted).view(
                -1, self.num_heads, self.head_size
            )
            g.srcdata["k"] = k
            g.dstdata["q"] = q
            g.srcdata["v"] = v
            g.edata["etype"] = etype
            g.apply_edges(self.message)
            # If external edge weights are provided in g.edata["w"],
            # multiply the message tensor by these weights.
            if "w" in g.edata:
                # Make sure the weight tensor has the correct shape.
                # Reshape edge weights to (num_edges, 1, 1) for broadcasting.
                g.edata["m"] = g.edata["m"] * g.edata["w"].view(-1, 1, 1)
                # g.edata["m"] = g.edata["m"] * g.edata["w"].unsqueeze(-1)
            # Continue as in the original HGTConv: apply softmax to the attention scores.
            g.edata["m"] = g.edata["m"] * edge_softmax(g, g.edata["a"]).unsqueeze(-1)
            g.update_all(fn.copy_e("m", "m"), fn.sum("m", "h"))
            h = g.dstdata["h"].view(-1, self.num_heads * self.head_size)
            h = self.drop(self.linear_a(h, dstntype, presorted))
            alpha = torch.sigmoid(self.skip[dstntype]).unsqueeze(-1)
            if x_dst.shape != h.shape:
                h = h * alpha + (x_dst @ self.residual_w) * (1 - alpha)
            else:
                h = h * alpha + x_dst * (1 - alpha)
            if self.use_norm:
                h = self.norm(h)
            return h

    def message(self, edges):
        a, m = [], []
        etype = edges.data["etype"]
        k = torch.unbind(edges.src["k"], dim=1)
        q = torch.unbind(edges.dst["q"], dim=1)
        v = torch.unbind(edges.src["v"], dim=1)
        for i in range(self.num_heads):
            kw = self.relation_att[i](k[i], etype, self.presorted)
            a.append(
                (kw * q[i]).sum(-1) * self.relation_pri[i][etype] / self.sqrt_d
            )
            m.append(
                self.relation_msg[i](v[i], etype, self.presorted)
            )
        return {"a": torch.stack(a, dim=1), "m": torch.stack(m, dim=1)}
class HGT(nn.Module):
    def __init__(self, in_size, hid_size, num_heads, num_layers=3):
        super(HGT, self).__init__()
        
        self.ntypes = ['_N']
        self.etypes = ['_E']
        self.hid_size = hid_size

        # Input projection
        self.input_proj = nn.Linear(in_size, hid_size)

        # Determine head_size: ensure head_size * num_heads == hid_size.
        head_size = hid_size // num_heads

        # HGT layers using the subclass that incorporates edge weights.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HGTConvWithEdgeWeight(
                hid_size,   # in_size
                head_size,  # head_size
                num_heads,  # num_heads
                1,          # num_ntypes (single type)
                1,          # num_etypes (single type)
                0.2,        # dropout
                False       # use_norm (set to True if needed)
            ))

        # Predictor remains as in your code.
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hid_size // 2, hid_size // 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hid_size // 4, 1),
        )
        self.dropout = nn.Dropout(0.2)
    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        # Input projection
        h = F.relu(self.input_proj(x))
        
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # Create dummy type tensors for the block.
            # For DGL blocks, separate source and destination might be needed.
            src_ntype = torch.zeros(block.num_src_nodes(), dtype=torch.int64, device=h.device)
            dst_ntype = torch.zeros(block.num_dst_nodes(), dtype=torch.int64, device=h.device)
            # For edge types, create one for all edges in the block.
            etype_block = torch.zeros(block.num_edges(), dtype=torch.int64, device=h.device)
            
            # Debug assertions to ensure all values are 0.
            assert (src_ntype == 0).all(), "src_ntype contains nonzero values!"
            assert (dst_ntype == 0).all(), "dst_ntype contains nonzero values!"
            assert (etype_block == 0).all(), "etype_block contains nonzero values!"
            
            # Set external edge weights from your graph data.
            block.edata["w"] = block.edata["weight"]
    
            # For blocks (which are DGL blocks), HGTConv expects a single ntype tensor.
            # We mimic the original HGTConv behavior:
            if block.is_block:
                # Use src_ntype for source and dst_ntype for destination.
                ntype = src_ntype  # The layer will slice for destination internally.
            else:
                ntype = src_ntype  # For non-block graphs, use the full tensor.
    
            h = layer(block, h, ntype, etype_block)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        
        # Compute predictions using the predictor.
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        
        return h_pos, h_neg, h
        
    def inference(self, g, feature, device, batch_size):
        # Get raw features and apply input projection to get the correct dimension.
        feat = g.ndata[feature]
        feat = F.relu(self.input_proj(feat))  # Transform raw features to hid_size dimensions
    
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=[feature])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(device),
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
            out_dim = self.hid_size  # Each layer outputs hid_size-dimensional features
            y = torch.empty(
                g.num_nodes(),
                out_dim,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"HGT Inference (Layer {l})"
            ):
                x = feat[input_nodes]
                
                # Create dummy type tensors for the block.
                src_ntype = torch.zeros(blocks[0].num_src_nodes(), dtype=torch.int64, device=device)
                etype_block = torch.zeros(blocks[0].num_edges(), dtype=torch.int64, device=device)
                
                # Set external edge weights from your graph data.
                blocks[0].edata["w"] = blocks[0].edata["weight"]
                
                h_out = layer(blocks[0], x, src_ntype, etype_block)
                if l != len(self.layers) - 1:
                    h_out = F.relu(h_out)
                y[output_nodes] = h_out.to(buffer_device)
            feat = y
        return y
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import tqdm

class GGNN(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, num_timesteps=3):
        """
        Args:
            in_size (int): Dimension of input node features.
            hidden_size (int): Hidden state dimension.
            num_layers (int): Number of layers (if stacking is desired; here we use one GGNN layer).
            num_timesteps (int): Number of message passing steps within the GGNN layer.
        """
        super(GGNN, self).__init__()
        # DGL's GatedGraphConv expects the number of edge types.
        # Here, we assume a homogeneous graph with a single edge type.
        self.ggnn = dglnn.GatedGraphConv(in_feats=in_size, out_feats=hidden_size, n_etypes=1, n_steps=num_timesteps)
        
        # Predictor for link prediction on the element-wise product of node embeddings.
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_size//2, 1)
        )
        self.dropout = nn.Dropout(0.2)
        self.hidden_size = hidden_size

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        """
        Note: GGNN (via GatedGraphConv) in DGL does not directly accept edge weights.
        If edge weights are critical, you might consider incorporating them into the node features
        or adjusting the message function.
        """
        # Here we assume blocks[0] is the graph used for message passing.
        # If you want to work with mini-batches, you might need to adjust this inference.
        block = blocks[0]
        h = self.ggnn(block, x)
        h = self.dropout(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg, h

    def inference(self, g, feature, device, batch_size):
        """
        For GGNN, if the entire graph fits in memory you can compute all node embeddings at once.
        """
        feat = g.ndata[feature].to(device)
        h = self.ggnn(g, feat)
        return h.cpu()

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
# TAG
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import tqdm

class TAG(nn.Module):
    def __init__(self, in_size, hid_size, out_dim, k=3):
        """
        TAGConv-based GNN for link prediction, modeled like GCN/SAGE/GAT.
        
        Args:
            in_size (int): Input feature dimension.
            hid_size (list): List of 3 integers (hidden layer sizes).
            out_dim (int): Final output embedding size (not used directly here).
            k (int): The number of hops for each TAGConv layer.
        """
        super(TAG, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(dglnn.TAGConv(in_size, hid_size[0], k=k))
        self.layers.append(dglnn.TAGConv(hid_size[0], hid_size[1], k=k))
        self.layers.append(dglnn.TAGConv(hid_size[1], hid_size[2], k=k))

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

    def forward(self, pair_graph, neg_pair_graph, blocks, x, e):
        h = x
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
        """
        Layer-wise inference algorithm for full graph embedding generation.
        Matches the structure of GAT, GCN, and SAGE inference.
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
            out_dim = self.hid_size[l]
            y = torch.empty(
                g.num_nodes(),
                out_dim,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc=f"TAGConv Inference (Layer {l})"
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
# GTN_grok2
# =============================================================================

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
    
