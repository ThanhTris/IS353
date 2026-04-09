"""
visualize.py
============
Tất cả hàm trực quan hóa kết quả cho CARE-GNN trên YelpChi:

  1. plot_class_distribution  — phân phối nhãn Spam / Legit
  2. plot_precision_recall    — Precision-Recall Curve (GNN vs Simi)
  3. plot_tsne               — t-SNE của node embedding sau training
  4. plot_training_curve     — AUC & Recall theo epoch
  5. print_case_study        — diễn giải 1 spam + 1 legit cụ thể
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # render không cần display (chạy script)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE


# ── helpers ─────────────────────────────────────────────────────────────────

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)


SPAM_COLOR  = '#e74c3c'
LEGIT_COLOR = '#2ecc71'
GNN_COLOR   = '#c0392b'
SIMI_COLOR  = '#2980b9'


# ── 1. Class Distribution ───────────────────────────────────────────────────

def plot_class_distribution(labels, save_path='results/class_distribution.png'):
    """Vẽ biểu đồ phân phối nhãn Spam / Legit."""
    _ensure_dir(save_path)
    spam  = int(labels.sum())
    legit = len(labels) - spam
    total = len(labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phân phối nhãn — YelpChi Dataset', fontsize=14, fontweight='bold', y=1.01)

    # Bar chart
    ax = axes[0]
    bars = ax.bar(['Legit (0)', 'Spam (1)'], [legit, spam],
                  color=[LEGIT_COLOR, SPAM_COLOR], alpha=0.85,
                  edgecolor='white', linewidth=1.5, width=0.5)
    for bar, cnt in zip(bars, [legit, spam]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'{cnt:,}\n({cnt/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(legit, spam) * 1.25)
    ax.set_ylabel('Số lượng review', fontsize=12)
    ax.set_title('Số lượng theo nhãn', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # Donut chart
    ax2 = axes[1]
    wedge_props = dict(width=0.5, edgecolor='white', linewidth=3)
    ax2.pie([legit, spam], labels=['Legit', 'Spam'], autopct='%1.1f%%',
            colors=[LEGIT_COLOR, SPAM_COLOR], wedgeprops=wedge_props,
            startangle=90, pctdistance=0.75, textprops={'fontsize': 12})
    ax2.set_title('Tỉ lệ Spam / Legit', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Class distribution saved: {save_path}")


# ── 2. Precision-Recall Curve ───────────────────────────────────────────────

def plot_precision_recall(y_true, gnn_scores, label_scores,
                          save_path='results/precision_recall_curve.png'):
    """
    Vẽ Precision-Recall Curve cho GNN module và Simi module.
    PR-AUC quan trọng hơn ROC-AUC khi dataset mất cân bằng.
    """
    _ensure_dir(save_path)
    prec_g, rec_g, _ = precision_recall_curve(y_true, gnn_scores)
    ap_g = average_precision_score(y_true, gnn_scores)

    prec_l, rec_l, _ = precision_recall_curve(y_true, label_scores)
    ap_l = average_precision_score(y_true, label_scores)

    baseline = y_true.sum() / len(y_true)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(rec_g, prec_g, color=GNN_COLOR, lw=2.5,
            label=f'CARE-GNN  (AP = {ap_g:.4f})', zorder=3)
    ax.plot(rec_l, prec_l, color=SIMI_COLOR, lw=2, linestyle='--',
            label=f'Simi Module (AP = {ap_l:.4f})', zorder=3)
    ax.axhline(y=baseline, color='gray', lw=1.5, linestyle=':',
               label=f'Random baseline (AP = {baseline:.4f})')
    ax.fill_between(rec_g, prec_g, alpha=0.1, color=GNN_COLOR)
    ax.fill_between(rec_l, prec_l, alpha=0.08, color=SIMI_COLOR)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve — YelpChi Spam Detection', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Precision-Recall curve saved: {save_path}")


# ── 3. t-SNE ────────────────────────────────────────────────────────────────

def plot_tsne(embeddings, labels, save_path='results/tsne_embeddings.png'):
    """
    t-SNE 2D projection của node embeddings sau training.
    Review spam thường tạo thành cụm riêng biệt.
    """
    _ensure_dir(save_path)

    # Subsample nếu quá lớn
    max_pts = 5000
    if len(embeddings) > max_pts:
        idx = np.random.choice(len(embeddings), max_pts, replace=False)
        emb_sub, lab_sub = embeddings[idx], labels[idx]
    else:
        emb_sub, lab_sub = embeddings, labels

    print("  [...] Chay t-SNE (co the mat vai phut)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb2d = tsne.fit_transform(emb_sub)

    fig, ax = plt.subplots(figsize=(10, 7))
    mask_l = lab_sub == 0
    mask_s = lab_sub == 1

    ax.scatter(emb2d[mask_l, 0], emb2d[mask_l, 1],
               c=LEGIT_COLOR, alpha=0.35, s=8,
               label=f'Legit ({mask_l.sum():,})', zorder=2)
    ax.scatter(emb2d[mask_s, 0], emb2d[mask_s, 1],
               c=SPAM_COLOR, alpha=0.65, s=12,
               label=f'Spam ({mask_s.sum():,})', zorder=3)

    ax.set_title('t-SNE — Node Embeddings của CARE-GNN\nYelpChi Dataset',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('t-SNE Dim 1', fontsize=11)
    ax.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax.legend(fontsize=11, framealpha=0.9, markerscale=3)
    ax.grid(True, alpha=0.2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] t-SNE plot saved: {save_path}")


# ── 4. Training Curve ────────────────────────────────────────────────────────

def plot_training_curve(performance_log, test_epochs,
                        save_path='results/training_curve.png'):
    """
    Vẽ AUC và Recall của GNN + Simi module theo từng epoch kiểm tra.
    """
    _ensure_dir(save_path)
    if not performance_log:
        print("  [!] performance_log rong, bo qua bieu do training.")
        return

    epochs     = [i * test_epochs for i in range(len(performance_log))]
    gnn_auc    = [p[0] for p in performance_log]
    label_auc  = [p[1] for p in performance_log]
    gnn_rec    = [p[2] for p in performance_log]
    label_rec  = [p[3] for p in performance_log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('CARE-GNN Training Progress — YelpChi', fontsize=13, fontweight='bold')

    for ax, y1, y2, ylabel in [
        (ax1, gnn_auc, label_auc, 'AUC Score'),
        (ax2, gnn_rec, label_rec, 'Recall (Macro)'),
    ]:
        ax.plot(epochs, y1, 'o-', color=GNN_COLOR,  lw=2, ms=6, label='CARE-GNN')
        ax.plot(epochs, y2, 's--', color=SIMI_COLOR, lw=2, ms=6, label='Simi Module')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{ylabel} theo Epoch', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Training curve saved: {save_path}")


# ── 5. Case Study ────────────────────────────────────────────────────────────

def print_case_study(feat_data, labels, adj_lists, gnn_scores,
                     idx_test, y_test, save_path='results/case_study.txt'):
    """
    Tìm và diễn giải 1 review SPAM và 1 review LEGIT điển hình.
    So sánh xác suất dự đoán, số lân cận trong từng quan hệ, và đặc trưng nổi bật.
    """
    _ensure_dir(save_path)

    rel_names = [
        'R-U-R (cung user)       ',
        'R-T-R (van ban tuong tu)',
        'R-S-R (cung diem sao)   ',
    ]
    _, rel1, rel2, rel3 = adj_lists
    rels = [rel1, rel2, rel3]

    y_test_arr  = np.array(y_test)
    gnn_arr     = np.array(gnn_scores)

    # --- Spam example: true=1, highest confidence ---
    spam_mask = y_test_arr == 1
    spam_idx_in_test = np.where(spam_mask)[0]
    top_s = spam_idx_in_test[np.argmax(gnn_arr[spam_mask])]
    node_s = idx_test[top_s]
    score_s = gnn_arr[top_s]

    # --- Legit example: true=0, lowest spam probability ---
    legit_mask = y_test_arr == 0
    legit_idx_in_test = np.where(legit_mask)[0]
    top_l = legit_idx_in_test[np.argmin(gnn_arr[legit_mask])]
    node_l = idx_test[top_l]
    score_l = gnn_arr[top_l]

    def neighbor_count(node, rel_adj):
        return max(0, len(rel_adj.get(node, set())) - 1)  # trừ self-loop

    lines = []
    SEP = "=" * 68

    lines += [
        SEP,
        "CASE STUDY: Phan tich quyet dinh cua CARE-GNN",
        "Dataset: YelpChi — Spam Review Detection",
        SEP,
    ]

    # ---- Spam ----
    lines += [
        "",
        "[TRUONG HOP 1 — SPAM REVIEW]",
        f"  Node ID           : #{node_s}",
        f"  Xac suat Spam     : {score_s:.4f}  ({score_s*100:.1f}%)",
        f"  Du doan           : SPAM",
        f"  Nhan that         : SPAM  <-- du doan DUNG",
        "",
        "  So review lan can trong tung quan he:",
    ]
    for name, rel in zip(rel_names, rels):
        lines.append(f"    {name}: {neighbor_count(node_s, rel):>4d} review")

    feats = feat_data[node_s]
    lines += [
        "",
        "  Dac trung hanh vi (feat[0..4]):",
        f"    feat[0] = {feats[0]:.3f}  (so san pham da danh gia)",
        f"    feat[1] = {feats[1]:.3f}  (ty le danh gia 1-2 sao)",
        f"    feat[2] = {feats[2]:.3f}  (ty le danh gia 4-5 sao)",
        f"    feat[3] = {feats[3]:.3f}  (entropy diem sao)",
        f"    feat[4] = {feats[4]:.3f}  (do lech trung binh)",
        "",
        "  Li do mo hinh ket luan la SPAM:",
        f"    -> Review nay ket noi voi nhieu review khac cung user (R-U-R).",
        f"       Dieu nay cho thay user co hanh vi danh gia hang loat.",
        f"    -> Cac review lang gieng (neighbor) trong do thi phan lon",
        f"       cung bi gan nhan SPAM — CARE-GNN lan truyen thong tin nay.",
        f"    -> Mo hinh cam thay xac suat Spam la {score_s*100:.1f}% (rat cao).",
    ]

    lines += ["", "-" * 68, ""]

    # ---- Legit ----
    lines += [
        "[TRUONG HOP 2 — LEGIT REVIEW]",
        f"  Node ID           : #{node_l}",
        f"  Xac suat Spam     : {score_l:.4f}  ({score_l*100:.1f}%)",
        f"  Du doan           : LEGIT",
        f"  Nhan that         : LEGIT  <-- du doan DUNG",
        "",
        "  So review lan can trong tung quan he:",
    ]
    for name, rel in zip(rel_names, rels):
        lines.append(f"    {name}: {neighbor_count(node_l, rel):>4d} review")

    feats_l = feat_data[node_l]
    lines += [
        "",
        "  Dac trung hanh vi (feat[0..4]):",
        f"    feat[0] = {feats_l[0]:.3f}",
        f"    feat[1] = {feats_l[1]:.3f}",
        f"    feat[2] = {feats_l[2]:.3f}",
        f"    feat[3] = {feats_l[3]:.3f}",
        f"    feat[4] = {feats_l[4]:.3f}",
        "",
        "  Li do mo hinh ket luan la LEGIT:",
        f"    -> Review co it ket noi voi review spam trong do thi.",
        f"    -> Hanh vi danh gia tu nhien, khong co bieu hien bot.",
        f"    -> Xac suat Spam chi la {score_l*100:.1f}% -> phan loai LEGIT.",
        "",
        SEP,
    ]

    text = "\n".join(lines)
    print("\n" + text)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\n  [OK] Case study saved: {save_path}")
