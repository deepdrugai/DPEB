# Imports
from torch.nn.functional import normalize
import os

os.environ["DGLBACKEND"] = "pytorch"
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


from dgl.data import DGLDataset
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    GraphDataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)

from model import GAT, SAGE
EPOCH = 20

# gcn_msg1 = fn.copy_u(u="h", out="m")
# gcn_msg = fn.u_mul_e("h", "weight", "m")
# gcn_reduce = fn.sum(msg="m", out="h")

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
  else:
    sampler = NeighborSampler([25, 20, 15, 10, 5])
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
      shuffle=True,
      drop_last=False,
      num_workers=0,
      use_uva=use_uva,
  )
  return dataloader

def evaluate(model, graph, edges, feature, epoch, device, desc):
    batch_size = 1000
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
        return calculate_metrics(pos_score, neg_score, epoch, 0, 0, desc)

def calculate_metrics(pos_score, neg_score, epoch, total_loss, it, desc="Training"):
    fltp = th.cat([pos_score, neg_score]).cpu().detach().numpy()# prediction outputs set
    flto = th.cat(
        [th.ones(pos_score.shape[0]), th.zeros(neg_score.shape[0])]
    ).cpu().detach().numpy() # Ground truth set
    running_training_accuracy, auc = print_performance_metrics(flto, fltp, print_stuff = False)
    print("Epoch: {:02d} | Iteration: {:05d} | Loss: {:.4f} | {} Accuracy: {:.4f} | AUC Score: {:.4f}".format(epoch, it, total_loss / (it + 1), desc, running_training_accuracy, auc))


def print_performance_metrics(flto, fltp, print_stuff = False):
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
        # print(f"=== === === ===")
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
        print(f"\n\n=== === === ===")
    return acc, auc


def train(device, g, feature, edges, model, name):

    model = model.to(device)
    dataloader = create_train_dataloader(g["train"], device, name)
    opt = th.optim.Adam(model.parameters(), lr=1e-3) #1e-2
    for epoch in range(EPOCH):
        model.train()
        total_loss = 0
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader
        ):
            x = blocks[0].srcdata[feature].to(device)
            edge_weight = blocks[0].edata["weight"].to(device)
            #pos_enc = blocks[0].srcdata["PE_PROTVEC"].to(device)
            if name == "gtn":
              pos_score, neg_score, emb = model(
                  pair_graph, neg_pair_graph, blocks, x, pos_enc)
            else :
              pos_score, neg_score, emb = model(
                  pair_graph, neg_pair_graph, blocks, x, edge_weight)
            score = th.cat([pos_score, neg_score])
            pos_label = th.ones_like(pos_score)
            neg_label = th.zeros_like(neg_score)
            labels = th.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if it % 4000 == 0:
                if th.equal(pos_score,neg_score) == True:
                  print("Emb: ", emb)
                  print("Pos Score: ", pos_score)
                  print("Neg Score: ", neg_score)
                calculate_metrics(pos_score, neg_score, epoch, total_loss, it)
                model.train()
            # if (it + 1) == 100:
            #     break
        #acc, roc = evaluate(model, g['val'], edges['val'], feature, epoch, device, "Validation")
        #print(" -- Validation Accuracy -- ")
        #print("Epoch: {:05d} | Accuracy: {:.4f} | AUC Score: {:.4f}".format(epoch, acc, roc
        evaluate(model, g['val'], edges['val'], feature, epoch, device, "Validation")
        #print(" -- Validation Accuracy -- ")
        #print("Epoch: {:05d} | Accuracy: {:.4f} | AUC Score: {:.4f}".format(epoch, acc, roc))
        #print(" -- Validation Ends -- ")
    #test_acc, test_roc = evaluate(model, g['train'], edges['train'], feature, epoch, device, "Testing train")
    #evaluate(model, g['train'], edges['train'], feature, epoch, device, "Complete Train")
    evaluate(model, g['test'], edges['test'], feature, epoch, device, "Testing")
    #print(" -- Testing train graph Scores -- ")
    #print("Accuracy: {:.4f} | AUC Score: {:.4f}".format(test_acc, test_roc))
    #print(" -- Testing train graph Ends -- ")
    return model

def run_app(device, graph, edges):
  
  features = ["esm_features", "Protvec_embeddings", "norm_esm_features", "norm_protvec_embeddings", "norm_esm_protvec_features"]
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

    # #pos_enc_size = g.ndata[feature].shape[1]
    # pos_enc_size = 1280
    # out_size = 160
    # gtn = GTModel(in_size=in_size, out_size=out_size, pos_enc_size=pos_enc_size).to(device)

    print("For Feature: {0} ; Sage Model".format(feature))
    # Run SAGE Model
    sage_model = train(device, graph, feature, edges, sage, "sage")

    print("=================================")
    
    print("For Feature: {0} ; GAT Model".format(feature))
    # Run GAT Model
    gat_model = train(device, graph, feature, edges, gat, "gat")

    print("=================================")


# Read the input data


with open("/home/magesh/prot_esm_ePPI_graph_test.pkl", 'rb') as f:
  g = pickle.load(f)

g = dgl.remove_self_loop(g)

print("Data Loading is completed")

# Normalize and concatenate features

g.ndata["norm_esm_features"] = normalize(g.ndata["esm_features"], p=2, dim=1)
g.ndata["norm_protvec_embeddings"] = normalize(g.ndata["Protvec_embeddings"], p=2, dim=1)
g.ndata["norm_esm_protvec_features"] = th.cat((g.ndata["norm_esm_features"], g.ndata["norm_protvec_embeddings"]), 1)


# Splitting the dataset
device = "cuda:0" # th.cuda.is_available()

graph_edges, graph, edges = {}, {}, {}
graph['train'], graph['val'], graph['test'] = split_dataset(g)

edges['val'] = edge_split(graph['val'])
edges['test'] = edge_split(graph['test'])
edges['train'] = edge_split(graph['train'])

graph_edges['graph'] = graph
graph_edges['edges'] = edges

print("Graph Splitting is completed")

# Run App
run_app(device, graph, edges)