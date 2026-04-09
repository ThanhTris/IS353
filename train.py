import time
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

from utils import *
from model import *
from layers import *
from graphsage import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
    Training CARE-GNN on YelpChi (Spam Review Detection)
    Paper: Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters
    Source: https://github.com/YingtongDou/CARE-GNN
"""

parser = argparse.ArgumentParser()

# model selection
parser.add_argument('--model', type=str, default='CARE',
                    help='Model: [CARE, SAGE]')
parser.add_argument('--inter', type=str, default='GNN',
                    help='Inter-relation aggregator: [Att, Weight, Mean, GNN]')
parser.add_argument('--batch-size', type=int, default=1024,
                    help='Mini-batch size.')

# hyper-parameters
parser.add_argument('--lr',          type=float, default=0.01,  help='Learning rate.')
parser.add_argument('--lambda_1',    type=float, default=2,     help='Simi loss weight.')
parser.add_argument('--lambda_2',    type=float, default=1e-3,  help='L2 regularization weight.')
parser.add_argument('--emb-size',    type=int,   default=64,    help='Node embedding dimension.')
parser.add_argument('--num-epochs',  type=int,   default=31,    help='Number of training epochs.')
parser.add_argument('--test-epochs', type=int,   default=3,     help='Epoch interval for evaluation.')
parser.add_argument('--under-sample',type=int,   default=1,     help='Under-sampling scale.')
parser.add_argument('--step-size',   type=float, default=2e-2,  help='RL threshold step size.')

# class imbalance
parser.add_argument('--no-class-weight', action='store_true', default=False,
                    help='Disable class weighting (enabled by default to handle imbalance).')

# other
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable GPU training.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print("=" * 60)
print("  CARE-GNN — YelpChi Spam Review Detection")
print("=" * 60)
print(f"  Model: {args.model}  |  Inter-AGG: {args.inter}")
print(f"  Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  Embed: {args.emb_size}")
print(f"  CUDA: {args.cuda}")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────────────
print("\n[1] Loading YelpChi data ...")
[homo, relation1, relation2, relation3], feat_data, labels = load_data()

spam_count  = int(labels.sum())
legit_count = len(labels) - spam_count
print(f"  Nodes (reviews)  : {len(labels):,}")
print(f"  Spam  (label=1)  : {spam_count:,}  ({spam_count/len(labels)*100:.1f}%)")
print(f"  Legit (label=0)  : {legit_count:,}  ({legit_count/len(labels)*100:.1f}%)")
print(f"  Feature dimension: {feat_data.shape[1]}")

# ── Train / Test split ────────────────────────────────────────────────────────
print("\n[2] Train/Test split (40% train, 60% test, stratified) ...")
np.random.seed(args.seed)
random.seed(args.seed)

index = list(range(len(labels)))
idx_train, idx_test, y_train, y_test = train_test_split(
    index, labels, stratify=labels, test_size=0.60,
    random_state=2, shuffle=True
)
print(f"  Train: {len(idx_train):,}  |  Test: {len(idx_test):,}")

train_pos, train_neg = pos_neg_split(idx_train, y_train)
print(f"  Train spam: {len(train_pos):,}  |  Train legit: {len(train_neg):,}")

# ── Node features ─────────────────────────────────────────────────────────────
print("\n[3] Initializing node features ...")
features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
feat_data = normalize(feat_data)
features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
if args.cuda:
    features.cuda()

# ── Class weights (handle imbalance) ─────────────────────────────────────────
if args.no_class_weight:
    class_weight = None
    print("  Class weighting: DISABLED")
else:
    weight_ratio = legit_count / spam_count
    class_weight = torch.FloatTensor([1.0, weight_ratio])
    if args.cuda:
        class_weight = class_weight.cuda()
    print(f"  Class weighting: ENABLED  (spam weight = {weight_ratio:.2f}x)")

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\n[4] Building model ...")
adj_lists = homo if args.model == 'SAGE' else [relation1, relation2, relation3]

if args.model == 'CARE':
    intra1 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
    intra2 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
    intra3 = IntraAgg(features, feat_data.shape[1], cuda=args.cuda)
    inter1 = InterAgg(features, feat_data.shape[1], args.emb_size,
                      adj_lists, [intra1, intra2, intra3],
                      inter=args.inter, step_size=args.step_size, cuda=args.cuda)
    gnn_model = OneLayerCARE(2, inter1, args.lambda_1,
                             class_weight=class_weight)
elif args.model == 'SAGE':
    agg1 = MeanAggregator(features, cuda=args.cuda)
    enc1 = Encoder(features, feat_data.shape[1], args.emb_size,
                   adj_lists, agg1, gcn=True, cuda=args.cuda)
    enc1.num_samples = 5
    gnn_model = GraphSage(2, enc1)

if args.cuda:
    gnn_model.cuda()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, gnn_model.parameters()),
    lr=args.lr, weight_decay=args.lambda_2
)

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n[5] Training for {args.num_epochs} epochs ...")
performance_log = []

for epoch in range(args.num_epochs):
    # Under-sample legit nodes each epoch to balance training
    sampled_idx_train = undersample(train_pos, train_neg, scale=1)
    random.shuffle(sampled_idx_train)

    num_batches = int(len(sampled_idx_train) / args.batch_size) + 1
    if args.model == 'CARE':
        inter1.batch_num = num_batches

    epoch_loss  = 0.0
    epoch_time  = 0.0

    for batch in range(num_batches):
        i_start = batch * args.batch_size
        i_end   = min((batch + 1) * args.batch_size, len(sampled_idx_train))
        batch_nodes = sampled_idx_train[i_start:i_end]
        batch_label = labels[np.array(batch_nodes)]

        optimizer.zero_grad()
        t0 = time.time()
        if args.cuda:
            loss = gnn_model.loss(batch_nodes,
                                  Variable(torch.cuda.LongTensor(batch_label)))
        else:
            loss = gnn_model.loss(batch_nodes,
                                  Variable(torch.LongTensor(batch_label)))
        loss.backward()
        optimizer.step()
        epoch_time += time.time() - t0
        epoch_loss += loss.item()

    avg_loss = epoch_loss / num_batches
    print(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | time={epoch_time:.1f}s")

    if epoch % args.test_epochs == 0:
        print(f"  --- Test @ epoch {epoch} ---")
        if args.model == 'SAGE':
            test_sage(idx_test, y_test, gnn_model, args.batch_size)
        else:
            result = test_care(idx_test, y_test, gnn_model, args.batch_size)
            gnn_auc, label_auc, gnn_recall, label_recall = result[:4]
            performance_log.append([gnn_auc, label_auc, gnn_recall, label_recall])

# ── Final evaluation + Visualization ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL EVALUATION & VISUALIZATION")
print("=" * 60)

os.makedirs('results', exist_ok=True)

# Collect all test predictions and embeddings
print("\n[6] Collecting predictions and embeddings for visualization ...")
gnn_model.eval()
all_gnn_scores    = []
all_label_scores  = []
all_embeddings    = []

test_batch_num = int(len(idx_test) / args.batch_size) + 1

with torch.no_grad():
    for it in range(test_batch_num):
        i_s = it * args.batch_size
        i_e = min((it + 1) * args.batch_size, len(idx_test))
        if i_s >= i_e:
            break
        b_nodes = idx_test[i_s:i_e]
        b_label = y_test[i_s:i_e]

        if args.model == 'CARE':
            gnn_prob, label_prob = gnn_model.to_prob(b_nodes, b_label, train_flag=False)
            all_label_scores.extend(label_prob.data.cpu().numpy()[:, 1].tolist())
            emb = gnn_model.get_embeddings(b_nodes, b_label, train_flag=False)
            all_embeddings.append(emb.data.cpu().numpy())
        else:
            gnn_prob = gnn_model.to_prob(b_nodes)
            all_embeddings.append(gnn_prob.data.cpu().numpy())

        all_gnn_scores.extend(gnn_prob.data.cpu().numpy()[:, 1].tolist())

all_gnn_scores   = np.array(all_gnn_scores)
all_label_scores = np.array(all_label_scores) if all_label_scores else all_gnn_scores
all_embeddings   = np.vstack(all_embeddings)
y_test_arr       = np.array(y_test)

# Save raw scores for reproducibility
np.savez('results/final_predictions.npz',
         y_true=y_test_arr,
         gnn_scores=all_gnn_scores,
         label_scores=all_label_scores,
         embeddings=all_embeddings)
print("  [OK] Predictions saved: results/final_predictions.npz")

# Generate visualizations
print("\n[7] Generating visualizations ...")
from visualize import (plot_class_distribution, plot_precision_recall,
                       plot_tsne, plot_training_curve, print_case_study)

plot_class_distribution(labels,
                        save_path='results/class_distribution.png')

if args.model == 'CARE':
    plot_precision_recall(y_test_arr, all_gnn_scores, all_label_scores,
                          save_path='results/precision_recall_curve.png')
else:
    plot_precision_recall(y_test_arr, all_gnn_scores, all_gnn_scores,
                          save_path='results/precision_recall_curve.png')

plot_tsne(all_embeddings, y_test_arr,
          save_path='results/tsne_embeddings.png')

if performance_log:
    plot_training_curve(performance_log, args.test_epochs,
                        save_path='results/training_curve.png')

if args.model == 'CARE':
    print_case_study(feat_data, labels,
                     [homo, relation1, relation2, relation3],
                     all_gnn_scores, idx_test, y_test,
                     save_path='results/case_study.txt')

# Final summary
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
final_preds = (all_gnn_scores >= 0.5).astype(int)
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"  AUC   : {roc_auc_score(y_test_arr, all_gnn_scores):.4f}")
print(f"  AP    : {average_precision_score(y_test_arr, all_gnn_scores):.4f}")
print(f"  F1    : {f1_score(y_test_arr, final_preds, average='macro'):.4f}")
print(f"  F1-Spam: {f1_score(y_test_arr, final_preds, average=None)[1]:.4f}")
print("=" * 60)
print("\n  All results saved in results/")
