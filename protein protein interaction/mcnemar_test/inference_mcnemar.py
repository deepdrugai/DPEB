# Imports
from torch.nn.functional import normalize
import os
# import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["DGLBACKEND"] = "pytorch"
import sys
# sys.stdout = open("console_outputs/inference_output", "w")

# =============================================================================
# 
# =============================================================================
checkpoint_path = "/home/saiful/ePPI_dgl/esm/updated_esm_graph_checkpoints/esm_features_gin_lr_1e-4_updated_esm_checkpoint.pth"
# checkpoint_path = "/home/saiful/ePPI_dgl/mcnemar_tests/sage_predictions/alphafold_feat_sage_lr_1e-3_checkpoint_y_true_y_pred.npz"
# file_prefix_name = "esm_features_sage_lr_1e-3_checkpoint"
file_prefix_name = "esm_features_gin_lr_1e-4_updated_esm_checkpoint"


embedding_type = save_prefix= "esm"  #  "esm" # alphafold, bioembedding, protvec
model_name = "gin"       #sage, gat, gcn, gtn
# features = ["alphafold_feat"]
features = ["esm_features"]
# features = ["bioembed_feat"]
# features = ["protvec_feat"]

# =============================================================================
# 
# =============================================================================
save_path= f"/home/saiful/ePPI_dgl/mcnemar_tests/model_inference_outputs/{file_prefix_name}_y_true_y_pred.txt"
save_npz_path= f"/home/saiful/ePPI_dgl/mcnemar_tests/{model_name}_predictions/{file_prefix_name}_y_true_y_pred.npz"
sys.stdout = open(save_path, "w")

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import numpy as np
import tqdm
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import pickle
import time
import dgl.sparse as dglsp

#from dgl import backend as F
from dgl import EID, NID
from dgl.transforms import to_block
from dgl.utils import get_num_threads
from dgl.dataloading import BlockSampler
import random
import torch


from dgl.data import DGLDataset
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    GraphDataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)

from model import GAT, SAGE, GCN, GIN, RGCN, GTN, TAG, HGT, GGNN, GATv2, GTN_g2
EPOCHS = 10


def split_dataset(g):
  u, v = g.edges()

  np.random.seed(42)
  eids = np.arange(g.number_of_edges())
  eids = np.random.permutation(eids)
  test_size, val_size = int(len(eids) * 0.1), int(len(eids) * 0.2)
  train_size = g.number_of_edges() - (val_size + test_size)

  test_start_index = val_size + train_size
  no_of_val_edges_to_remove = train_size + test_size

  test_ids_to_remove = val_size + train_size
  val_to_remove_from_train = eids[:train_size]
  val_to_remove_from_test = eids[test_ids_to_remove:]
  val_edges_to_remove = np.concatenate((val_to_remove_from_train, val_to_remove_from_test), axis =0)

  # Move the train graph to device
  train_g = dgl.remove_edges(g, eids[train_size:]) #.to('cuda:0')
  test_g = dgl.remove_edges(g, eids[:test_ids_to_remove])
  val_g = dgl.remove_edges(g, eids[val_edges_to_remove])

  print("Total Number of Edges: ", g.num_edges())
  print("Train Number of Edges: ", train_g.num_edges())
  print("Valid Number of Edges: ", val_g.num_edges())
  print("Test Number of Edges: ", test_g.num_edges())

  return train_g, val_g, test_g

def edge_split(g):

  edges = {}

  # Split edge set for training and testing
  u, v = g.edges()
  num_nodes = g.num_nodes()
  num_edges = g.num_edges()
  pos_u, pos_v = u, v

  # Find all negative edges and split them for training and testing
  np.random.seed(42)
  adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(num_nodes, num_nodes))
  adj_neg = 1 - adj.todense() - np.eye(num_nodes)
  neg_u, neg_v = np.where(adj_neg != 0)

  neg_eids = np.random.choice(len(neg_u), num_edges)
  neg_u, neg_v = (
      neg_u[neg_eids],
      neg_v[neg_eids],
  )

  edges["source"] = pos_u
  edges["target"] = pos_v
  edges["neg_source"] = th.from_numpy(neg_u)
  edges["neg_target"] = th.from_numpy(neg_v)

  return edges

class CustomNeighborSampler(BlockSampler):
    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        mask=None,
        replace=False,
        prefetch_node_feats=None,
        prefetch_labels=None,
        prefetch_edge_feats=None,
        output_device=None,
        fused=True,
    ):
        super().__init__(
            prefetch_node_feats=prefetch_node_feats,
            prefetch_labels=prefetch_labels,
            prefetch_edge_feats=prefetch_edge_feats,
            output_device=output_device,
        )
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        self.fused = fused
        self.mapping = {}
        self.g = None

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        # sample_neighbors_fused function requires multithreading to be more efficient
        # than sample_neighbors
        output_nodes = seed_nodes
        blocks = []

        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
            )
            #print("Frontier: ", type(frontier))
            #frontier = dgl.add_self_loop(frontier)
            frontier = dgl.add_edges(frontier, seed_nodes, seed_nodes)
            #eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            #block = dgl.add_self_loop(block)
            #block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True,
        copy_ndata=True, copy_edata=True
    )
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = th.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()

    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = th.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert th.equal(src1, dst2)
    assert th.equal(src2, dst1)
    return g, reverse_mapping # Return the same graph which is being passed to the method.

def create_train_dataloader(graph, device, name, prefetch_nodes=None, prefetch_edges=None):
  g, reverse_eids = to_bidirected_with_reverse_mapping(graph)
  reverse_eids = reverse_eids.to(device)
  #g = graph
  seed_edges = th.arange(g.num_edges()).to(device)
  # if prefetch_nodes is None:
  #   sampler = CustomNeighborSampler([15, 10, 5])
  # elif prefetch_edges is None:
  #   sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=[i for i in prefetch_nodes])
  # else:
  #   sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=[i for i in prefetch_nodes], prefetch_edge_feats=[i for i in prefetch_edges])
  if name == "gat":
    sampler = CustomNeighborSampler([25, 20, 15, 10, 5])
    
  elif name == "rgcn":
    sampler = NeighborSampler([5, 5, 5]) # for R-GCN only

  else:
    sampler = NeighborSampler([25, 20, 15, 10, 5])
    # After: Use lower fan-out values to reduce block size

  sampler = as_edge_prediction_sampler(
      sampler,
      exclude="reverse_id",
      reverse_eids=reverse_eids,
      negative_sampler=negative_sampler.Uniform(1),
  )
  use_uva = True
  dataloader = DataLoader(
      g,
      seed_edges,
      sampler,
      device=device,
       batch_size=512,
      # batch_size=16,  # for R-GCN only

      shuffle=True,
      drop_last=False,
      num_workers=0,  #0
      use_uva=use_uva,
  )
  return dataloader

def save_predictions_from_scores(pos_score, neg_score, save_path, save_prefix):
    """
    Takes raw scores, computes threshold using Youden’s J, saves predictions and labels.
    """
    # Detach and move to CPU if necessary
    pos_score = pos_score.detach().cpu().numpy()
    neg_score = neg_score.detach().cpu().numpy()

    y_scores = np.concatenate([pos_score, neg_score])
    y_true = np.concatenate([
        np.ones(pos_score.shape[0]), 
        np.zeros(neg_score.shape[0])    ])

    # Compute optimal threshold (Youden’s J)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_thresh = thresholds[np.argmax(j_scores)]

    y_pred = (y_scores > best_thresh).astype(int)
    y_true = y_true.astype(int)
    y_pred = y_pred.ravel()

    # Save to file
    # np.savetxt(f"{save_prefix}_y_true.txt", y_true, fmt="%d")
    # np.savetxt(f"{save_prefix}_y_pred.txt", y_pred, fmt="%d")
    np.savez_compressed(save_path, y_true=y_true, y_pred=y_pred)
    print(f"Saved predictions for {save_prefix} using threshold = {best_thresh:.5f}")
          
def evaluate(model, graph, edges, feature, epoch, device, desc):
    # batch_size = 32   # for R-GCN only
    batch_size = 128

    model.eval()
    with th.no_grad():
        node_emb = model.inference(graph, feature, device, batch_size)
        node_emb = node_emb.to(device)
        src = edges["source"].to(node_emb.device)
        dst = edges["target"].to(node_emb.device)
        neg_dst = edges["neg_target"].to(node_emb.device)
        #print(node_emb[src].shape)
        pos_score = model.predictor(node_emb[src] * node_emb[dst]).to(node_emb.device)
        neg_score = model.predictor(node_emb[src] * node_emb[neg_dst]).to(node_emb.device)
        # score = th.cat([pos_score, neg_score])
        # pos_label = th.ones_like(pos_score)
        # neg_label = th.zeros_like(neg_score)
        #return compute_roc(model, node_emb, src, dst, neg_dst, device, batch_size)
        
        if desc == "Testing":
            print("\n#---Saving y_preds and y_labels in a npz file---#")


            # save_path= f"/home/saiful/ePPI_dgl/mcnemar_tests/sage_predictions/{save_prefix}_{model_name}_y_true_y_pred2.npz"
            save_predictions_from_scores(pos_score, neg_score, save_npz_path, save_prefix)
            
        pos_score = pos_score.cpu().detach().numpy()
        neg_score = neg_score.cpu().detach().numpy()

        y_scores = np.concatenate([pos_score, neg_score])
        y_true = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
        return y_true, y_scores
    
        # return calculate_metrics(pos_score, neg_score, epoch, 0, 0, desc)

def calculate_metrics(pos_score, neg_score, epoch, total_loss, it, desc="Training"):
    fltp = th.cat([pos_score, neg_score]).cpu().detach().numpy()# prediction outputs set
    flto = th.cat(
        [th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]
    ).cpu().detach().numpy() # Ground truth set
    if desc == "Testing":
        print("#---Testing Performance Metrics---#")
        running_testing_accuracy, auc = print_performance_metrics(flto, fltp, print_stuff = True)
        print("Epoch: {:02d} | Final Performance: {:05d} | Loss: {:.4f} | {} Accuracy: {:.4f} | AUC Score: {:.4f}".format(epoch, it, total_loss / (it + 1), desc, running_testing_accuracy, auc))
    else:
        running_training_accuracy, auc = print_performance_metrics(flto, fltp, print_stuff = False)
        print("Epoch: {:02d} |  Iteration: {:05d} | Loss: {:.4f} | {} Accuracy: {:.4f} | AUC Score: {:.4f}".format(epoch, it, total_loss / (it + 1), desc, running_training_accuracy, auc))
 

def print_performance_metrics(flto, fltp, print_stuff = True):
    auc = roc_auc_score(flto, fltp)
    fpr, tpr, ths = roc_curve(flto, fltp)
    opt = np.argmax(tpr - fpr)
    th = ths[opt]
    cm = confusion_matrix(flto==1, fltp>th, normalize='all')
    tn, fp, fnn, tp = cm.ravel()
    fnn = fnn.astype("float64")
    mcc = (tp * tn - fp * fnn) / np.sqrt((tp + fp) * (tp + fnn) * (tn + fp) * (tn + fnn))
    acc = (tp + tn)*100 / (tp + tn + fp + fnn)
    j = tp / (tp + fnn) + tn / (tn + fp) - 1
    p = tp / (fp + tp)
    r = tp / (fnn + tp)
    f1 = 2*p*r / (p + r)
    if print_stuff:
        print(f"=== === flag1.11a performance_metrics === ===")
        print(f"Threshold: {th:.5f}")
        print(f"True Positive: {tp}")
        print(f"False Negative: {fnn}")
        print(f"True Negative: {tn}")
        print(f"False Positive: {fp}")
        print(f"confusion_matrix: {cm}")

        print(f"J   = {j}")
        print(f"MCC = {mcc}")
        print(f"AUC = {auc}")
        print(f"Accuracy using deepdrug method:  = {acc:.4f}%")
        print(f"Precision = {p:.4f}")
        print(f"Recall = {r:.4f}")
        print(f"F1 Score = {f1:.4f}")
        print(f"=== === === ===\n\n")
    return acc, auc, mcc, p, r, f1, j

def get_subgraph(g):
    random_seed = 42  
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    print("Original graph:", g)
    
    # Sample a subset of nodes
    num_nodes = g.num_nodes()
    sample_size = 1000  # Adjust this to your desired subgraph size
    sampled_nodes = torch.tensor(random.sample(range(num_nodes), sample_size))
    
    #  Extract the subgraph
    subgraph = dgl.node_subgraph(g, sampled_nodes)
    print("Subgraph:", subgraph)
    print("Number of nodes in subgraph:", subgraph.num_nodes())
    print("Number of edges in subgraph:", subgraph.num_edges())
    return subgraph

def load_trained_checkpoint(device, g, feature, edges, model, name):
    epoch = '20'
    metrics = []
        
    model = model.to(device)
    # checkpoint_path = "/home/saiful/ePPI_dgl/DeepDrug-Protein-Embedding-Database/checkpoints/alphafold_feat_sage_checkpoint.pth"
    print(f"Loading saved checkpoints..")
    model.load_state_dict(torch.load(checkpoint_path))
    
    y_true, y_scores = evaluate(model, g['test'], edges['test'], feature, epoch, device, "Testing")
    result = print_performance_metrics(y_true, y_scores)
    print(f"Run 1: Accuracy={result[0]:.4f}, AUC={result[1]:.4f}, F1={result[5]:.4f}")

    return model


def run_app(device, graph, edges):
  
  
  # features = ["esm_features", "Protvec_embeddings", "norm_esm_features", "norm_protvec_embeddings", "norm_esm_protvec_features"]
  # features = ["alphafold_feat"]
  # features = ["esm_features"]
  # features = ["bioembed_feat"]
  # features = ["protvec_feat"]



  for feature in features:
      
    #feature = "final_feature"
    in_size = g.ndata[feature].shape[1]
    #hid_size2 = [120, 60, 30, 96]
    gat_hid_size2 = [512, 256, 160]
    gat_num_heads=  8
    gat = GAT(
           in_size= in_size,
           hid_size= gat_hid_size2,
           out_dim= 256,
           num_heads= gat_num_heads)

    sage = SAGE(in_size, 160)
    gcn = GCN(in_size, [512, 256, 160])

    # =============================================================================
    # 
    # =============================================================================

    # #pos_enc_size = g.ndata[feature].shape[1]
    # pos_enc_size = 1280
    # out_size = 160
    # gtn = GTModel(in_size=in_size, out_size=out_size, pos_enc_size=pos_enc_size).to(device)
    
    # =============================================================================
          # Run SAGE Model
    # =============================================================================
    # print("For Feature: {0} ; flag 1.10a Sage Model".format(feature))
    # # Run SAGE Model
    # print("=================================")
    # print("flag 1.10b Sage Model:\n", sage)
    # sage_model = load_trained_checkpoint(device, graph, feature, edges, sage, "sage")
    # print("=================================")
    
    
    # # =============================================================================
    # #     # Run GAT Model
    # # =============================================================================
    
    # print("For Feature: {0} ; flag 1.10b GAT Model".format(feature))
    # print("=================================")
    # print("flag 1.10b gat Model:\n", gat)
    # gat_model = load_trained_checkpoint(device, graph, feature, edges, gat, "gat")
    # print("=================================")
    
    # # =============================================================================
    # #     # Run GCN Model
    # # =============================================================================
    ##Here, we choose the same hidden dimensions as for GAT, but you can adjust as needed.
    # print("For Feature: {0} ; flag 1.10c GCN Model".format(feature))
    # print("=================================")
    # print("flag 1.10b gcn Model:\n", gcn)
    # gcn_model = load_trained_checkpoint(device, graph, feature, edges, gcn, "gcn")
    # print("=================================")
    
    
    # # =============================================================================
    #     # Run GIN Model
    # # =============================================================================

    ##Example hyperparameters for GIN: you can adjust 'hidden_feats', 'num_layers', 'mlp_hidden_dim', etc.
    gin_hidden_feats = 160 #160, 256        # Output dimension for each layer's MLP
    num_layers = 4                # Number of GIN layers
    mlp_hidden_dim = 128          # Hidden dimension for the internal MLP
    final_dim = 1                 # Final output dimension from the predictor

    # Instantiate the GIN model. The constructor parameters below must match your GIN implementation.
    gin = GIN(
        in_feats=in_size,
        hidden_feats=gin_hidden_feats,
        num_layers=num_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        final_dim=final_dim
    )
    print("For Feature: {0} ; flag 1.10d GIN Model".format(feature))
    print("=================================")
    print("flag 1.10b gin Model:\n", gin)
    gin_model = load_trained_checkpoint(device, graph, feature, edges, gin, "gin")
    print("=================================")
    
    
    
    
    # # =============================================================================
    #     # Run GTN_g2 Model   ==> i used it last time, worked well
    # # =============================================================================
    # embed_dim = 256 # 256     # Embedding dimension after input projection.
    # num_layers = 3  #5    # Number of transformer layers.
    # num_heads = 8       # Number of attention heads.
    # dropout = 0.1       # Dropout rate.
    
    # gtn = GTN_g2(in_feats=in_size, embed_dim=embed_dim,
    #           num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    # # gtn = nn.DataParallel(gtn)
    # gtn = gtn.to(device)
    
    # print("For Feature: {0} ; flag 1.10f GTN Model".format(feature))
    # print("=================================")
    # print("flag 1.10b gtn Model:\n", gtn)
    # # Assuming your train function accepts (device, graph, feature, edges, model, name)
    # gtn_model = load_trained_checkpoint(device, graph, feature, edges, gtn, "gtn")
    # print("=================================")
    
    

# load esm and protvec embeddings
# with open("/data/saiful/ePPI/prot_esm_ePPI_graph_test.pkl", 'rb') as f:
#   g = pickle.load(f)
 
# load alphafold embeddings 
# with open("/data/saiful/ePPI/alphafold_ePPI_graph.pkl", 'rb') as f:
#   g = pickle.load(f)


# uncomment if only want to run with a subgraph 
# g = get_subgraph(g)

# # Load the DGL graph
# graph_path = "/data/saiful/ePPI/alphafold_ePPI_graph.dgl"
# g_list, _ = dgl.load_graphs(graph_path)  # Returns a list of graphs
# # If there's only one graph in the file, extract it
# g = g_list[0]


  
if embedding_type =="alphafold":    
    # #  ==  == # == #     Alphafold    == #   == #
    # loading the graph with mapping 
    import pickle
    with open("/data/saiful/ePPI/alphafold_ePPI_graph_with_map.pkl", "rb") as f:
        data = pickle.load(f)
    # Extract the graph and mapping
    g = data["graph"]
    protein_to_idx = data["protein_to_idx"]
    # #  ==  == #== #     Alphafold  == #== #  == #
    # # Verify the replacement
    # Convert to homogeneous graph
    print("g.ndata['alphafold_feat']:\n", g.ndata['alphafold_feat'])
    # # =====================================
    # # Normalize and concatenate features
    g.ndata["alphafold_feat"] = normalize(g.ndata["alphafold_feat"], p=2, dim=1)

elif embedding_type =="bioembedding":   
    import pickle
    with open("/data/saiful/ePPI/bioembed_ePPI_graph_with_map.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Extract the graph and mapping
    g = data["graph"]
    protein_to_idx = data["protein_to_idx"]
    # #  ==  == #== #     Alphafold  == #== #  == #
    # # Verify the replacement
    print("g.ndata['bioembed_feat']:\n", g.ndata['bioembed_feat'])
    # # =====================================
    # # Normalize and concatenate features
    g.ndata["bioembed_feat"] = normalize(g.ndata["bioembed_feat"], p=2, dim=1)

elif embedding_type =="esm":
    # load esm and protvec embeddings
    # old esm graph = /data/saiful/ePPI/prot_esm_ePPI_graph_test.pkl
    # with open("/data/saiful/ePPI/prot_esm_ePPI_graph_test", 'rb') as f:
    #   g = pickle.load(f)
      
    with open("/data/saiful/ePPI/esm_ePPI_graph_with_map.pkl", 'rb') as f:
      data = pickle.load(f)
    # Extract the graph and mapping
    g = data["graph"]
    protein_to_idx = data["protein_to_idx"]


elif embedding_type =="protvec":
    with open("/data/saiful/ePPI/protvec_ePPI_graph_with_map.pkl", 'rb') as f:
      data = pickle.load(f)
     
    # Extract the graph and mapping
    g = data["graph"]
    protein_to_idx = data["protein_to_idx"]


print("g", g)

# g = get_subgraph(g)

print("Graph and protein mapping loaded successfully!")
print("Number of nodes in graph:", g.num_nodes())
# print("Example mapping:", list(protein_to_idx.items())[:5])  # Print first 5 mappings  # used of alphafold

#  == #
g = dgl.remove_self_loop(g)

print("Data Loading is completed")
# =====================================

# Splitting the dataset
device = th.device("cuda" if th.cuda.is_available() else "cpu")

graph_edges, graph, edges = {}, {}, {}
graph['train'], graph['val'], graph['test'] = split_dataset(g)

edges['val'] = edge_split(graph['val'])
edges['test'] = edge_split(graph['test'])
edges['train'] = edge_split(graph['train'])

graph_edges['graph'] = graph
graph_edges['edges'] = edges

print("Train graph node types:", graph['train'].ntypes)
print("Train graph edge types:", graph['train'].etypes)

print("Graph Splitting is completed")

# Run App
run_app(device, graph, edges)
print("Execution Finished..")
