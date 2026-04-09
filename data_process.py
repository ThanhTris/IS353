from utils import sparse_to_adjlist
from scipy.io import loadmat

"""
Read YelpChi data and save the adjacency matrices to adjacency lists.

YelpChi Graph Relations:
  - net_rur : Review-User-Review  (hai review cùng một user viết)
  - net_rtr : Review-Text-Review  (hai review có nội dung văn bản tương tự)
  - net_rsr : Review-Star-Review  (hai review cùng điểm sao cho cùng sản phẩm)
  - homo    : Homogeneous graph   (gộp tất cả các quan hệ trên)
"""


if __name__ == "__main__":
    prefix = 'data/'

    print("Loading YelpChi.mat ...")
    yelp = loadmat(prefix + 'YelpChi.mat')

    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']

    n_nodes = net_rur.shape[0]
    print(f"  Số node (review)  : {n_nodes:,}")
    print(f"  R-U-R edges       : {net_rur.nnz:,}")
    print(f"  R-T-R edges       : {net_rtr.nnz:,}")
    print(f"  R-S-R edges       : {net_rsr.nnz:,}")
    print(f"  Homo edges        : {yelp_homo.nnz:,}")

    print("\nBuilding adjacency lists ...")
    sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
    print("  [OK] yelp_rur_adjlists.pickle  (Review-User-Review)")
    sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
    print("  [OK] yelp_rtr_adjlists.pickle  (Review-Text-Review)")
    sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
    print("  [OK] yelp_rsr_adjlists.pickle  (Review-Star-Review)")
    sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')
    print("  [OK] yelp_homo_adjlists.pickle (Homogeneous graph)")

    print("\nTat ca adjacency lists da duoc luu vao data/")
