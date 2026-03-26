# Import libraries
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pyucell as puc
import os
import warnings
from scipy.stats import ttest_ind
import harmonypy as hm

# Suppress warnings
warnings.filterwarnings('ignore')
plt.close('all')

# Set scanpy parameters
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=300, facecolor="white", frameon=False)

# Set matplotlib font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load and preprocess dataset
adata = sc.read_h5ad('cell.h5ad')

# Store raw counts
adata.layers['counts'] = adata.X.copy()

# Normalization and log transformation
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat', batch_key='donor_id')

# Scaling and PCA
sc.pp.scale(adata, max_value=10)
sc.pp.pca(adata, n_comps=50, use_highly_variable=True)

# Run Harmony batch correction
ho = hm.run_harmony(
    adata.obsm['X_pca'],
    adata.obs,
    'donor_id',
    max_iter_harmony=30,
    random_state=42
)
adata.obsm['X_pca_harmony'] = ho.Z_corr.T

# Compute neighbors and UMAP
sc.pp.neighbors(adata, use_rep="X_pca_harmony", random_state=42)
sc.tl.umap(adata, random_state=42)

# Compute UCell scores
gene_signatures = {
    'list': ['BCAT1','BCAT2','BCKDHA','BCKDHB','DBT','DLD','IVD','ACADM','ECHS1',
              'EHHADH','MCCC1','MCCC2','AUH','HMGCL','HMGCLL1','OXCT1','OXCT2',
              'AACS','HMGCS1','HMGCS2','ACAT1','ACAT2'],
}

puc.compute_ucell_scores(adata, gene_signatures, max_rank=800, chunk_size=500)
score_cols = ['list_UCell']
puc.smooth_knn_scores(adata, obs_columns=score_cols, graph_key='connectivities')
knn_cols = ['list_UCell_kNN']
all_score_cols_h = score_cols + knn_cols

# UMAP plot by cell type
cell_type_colors = {
    'T':     '#3A86FF',
    'B':     '#06D6A0',
    'Tumor': '#FB5607',
    'Mye':   '#8338EC',
    'Fibro': '#FF006E',
    'Endo':  '#FFBE0B',
    'Peri':  '#3D405B',
}

name = "cell_type"
save_dir = "singlecell"
os.makedirs(save_dir, exist_ok=True)

umap_coords = adata.obsm['X_umap']
cell_types = adata.obs['author_cell_type'].astype(str).values
ct_order = sorted(adata.obs['author_cell_type'].unique())

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
ax.set_facecolor('white')

for ct in ct_order:
    mask = cell_types == ct
    ax.scatter(
        umap_coords[mask, 0], umap_coords[mask, 1],
        c=cell_type_colors[ct],
        s=1.5, alpha=0.6, linewidths=0, rasterized=True
    )

patches = [mpatches.Patch(color=cell_type_colors[ct], label=ct) for ct in ct_order]
ax.legend(
    handles=patches, fontsize=9, loc='upper right',
    frameon=False, markerscale=8, handlelength=2, handleheight=2
)

ax.spines[['top','right','bottom','left']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

png_path = os.path.join(save_dir, f"umap_celltype_{name}.png")
svg_path = os.path.join(save_dir, f"umap_celltype_{name}.svg")
plt.tight_layout()
fig.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
plt.close()