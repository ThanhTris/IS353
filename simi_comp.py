"""
simi_comp.py
============
Tính Feature Similarity và Label Similarity cho YelpChi.

Hai metric này thể hiện mức độ "ngụy trang" của spammer:
  - Feature Similarity cao nhưng Label Similarity thấp (trong R-T-R, R-S-R)
    → Spammer có đặc trưng giống review thật nhưng lại bị gán nhãn khác nhau
  - Label Similarity cao trong R-U-R
    → Review cùng user thường cùng là spam hoặc cùng là legit

Chạy: python simi_comp.py
"""

from scipy.io import loadmat
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


# ── Load YelpChi ─────────────────────────────────────────────────────────────
print("Loading data/YelpChi.mat ...")
data = loadmat('data/YelpChi.mat')

net_list = [
    data['net_rur'].nonzero(),   # Review-User-Review
    data['net_rtr'].nonzero(),   # Review-Text-Review
    data['net_rsr'].nonzero(),   # Review-Star-Review
    data['homo'].nonzero(),      # Homogeneous (all relations)
]
rel_names = ['R-U-R', 'R-T-R', 'R-S-R', 'Homo']

feature = normalize(data['features']).toarray()
label   = data['label'][0]

# ── Chỉ tính cho positive (spam) nodes ───────────────────────────────────────
mode = 'pos'
pos_nodes    = set(label.nonzero()[0].tolist())
node_list    = [set(net[0].tolist()) for net in net_list]
pos_node_list = [list(net_nodes & pos_nodes) for net_nodes in node_list]
pos_idx_list  = []
for net, pos_node in zip(net_list, pos_node_list):
    pos_idx_list.append(np.in1d(net[0], np.array(pos_node)).nonzero()[0])

# ── Tính Feature Sim và Label Sim ────────────────────────────────────────────
print("\nComputing similarity scores for SPAM nodes ...\n")
print(f"{'Relation':<8} {'Feature Sim':>14} {'Label Sim':>12}")
print("-" * 38)

for name, net, pos_idx in zip(rel_names, net_list, pos_idx_list):
    feat_sim  = 0.0
    label_sim = 0.0

    for idx in pos_idx:
        u, v = net[0][idx], net[1][idx]
        feat_sim  += np.exp(-1 * np.square(np.linalg.norm(feature[u] - feature[v])))
        label_sim += int(label[u] == label[v])

    n = max(pos_idx.size, 1)
    print(f"{name:<8} {feat_sim/n:>14.4f} {label_sim/n:>12.4f}")

print("\nInterpretation:")
print("  R-U-R Label Sim cao (~0.91) → cùng user thường cùng spam/legit")
print("  R-T-R/R-S-R Label Sim thap → spammer nguy trang giong review that")
