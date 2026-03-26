import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import anndata as ad
import scipy.sparse
import time
import os
import harmonypy as hm
import pyucell as puc
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
plt.close('all')

# Load raw data and filter cells
INPUT_H5AD = "rectum.h5ad"
OUTPUT_FIG_DIR = "figures/"
adata = sc.read_h5ad(INPUT_H5AD)

# Keep only Rectum_T and BL timepoint cells
mask = (adata.obs['Tissue'] == 'Rectum_T') & (adata.obs['SampleTimePoint'] == 'BL')
adata_filtered = adata[mask, :].copy()
output_path = "rectum_t_bl_filtered.h5ad"
adata_filtered.write_h5ad(output_path, compression="gzip")

# Load filtered data for downstream analysis
INPUT_H5AD = "rectum_t_bl_filtered.h5ad"
adata = sc.read_h5ad(INPUT_H5AD)

# Normalization and log transformation
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
t0 = time.time()

# Identify highly variable genes
sc.pp.highly_variable_genes(
    adata,
    flavor='seurat',
    n_top_genes=3000,
    batch_key='sample',
    subset=False
)
n_hvg = adata.var['highly_variable'].sum()
adata_hvg = adata[:, adata.var['highly_variable']].copy()

# Scaling and PCA
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50, svd_solver='arpack')
adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']

# Set multi-threading parameters
os.environ['OPENBLAS_NUM_THREADS'] = '100'
os.environ['OMP_NUM_THREADS'] = '100'
os.environ['MKL_NUM_THREADS'] = '100'

# Run Harmony batch correction
batch_key = 'donor_id' if 'donor_id' in adata.obs.columns else 'sample'
ho = hm.run_harmony(
    adata.obsm['X_pca'],
    adata.obs,
    batch_key,
    max_iter_harmony=30,
    random_state=42,
    verbose=False
)
adata.obsm['X_pca_harmony'] = ho.Z_corr

# UMAP visualization based on corrected PCA
sc.pp.neighbors(adata, use_rep='X_pca_harmony', random_state=42)
sc.tl.umap(adata, random_state=42)

# Calculate UCell scores for gene signature
gene_signatures = {
    'list': ['BCAT1','BCAT2','BCKDHA','BCKDHB','DBT','DLD','IVD','ACADM','ECHS1',
             'EHHADH','MCCC1','MCCC2','AUH','HMGCL','HMGCLL1','OXCT1','OXCT2',
             'AACS','HMGCS1','HMGCS2','ACAT1','ACAT2'],
}
puc.compute_ucell_scores(adata, gene_signatures, max_rank=1500, chunk_size=500)

# Smooth UCell scores using KNN
score_cols = ['list_UCell']
puc.smooth_knn_scores(adata, k=10, use_rep="X_pca_harmony", obs_columns=score_cols, suffix="_kNN")
knn_cols = ['list_UCell_kNN']
all_score_cols_h = score_cols + knn_cols

# Plot settings
plt.rcParams['font.sans-serif'] = ['DejaSans']
plt.rcParams['figure.dpi'] = 600
save_dir = "/home/shuquan/software/singlecell/1/"
os.makedirs(save_dir, exist_ok=True)

# Define cell types to plot and target score
selected_cells = [
    'CD4T', 'plasma', 'CD8T', 'Epi', 'Cancer',
    'B', 'Macrophage', 'ILC', 'Mast', 'Fibroblast',
    'DC', 'Neutrophil', 'Endo', 'Pericyte'
]
target_score = 'list_UCell_kNN'

# Subset by C1_group and cell types
c1_data = adata[adata.obs['C1_group'].isin(['C1_R', 'C1_NR'])].copy()
cell_type_col = 'group'
c1_selected_data = c1_data[c1_data.obs[cell_type_col].isin(selected_cells)].copy()

# Extract UMAP coordinates and cell type labels
umap_coords = c1_selected_data.obsm['X_umap']
cell_types = c1_selected_data.obs[cell_type_col].values
ct_unique = sorted(np.unique(cell_types))

# Define color palette
cell_palette = sc.pl.palettes.default_20[:len(ct_unique)]
cell_color_map = dict(zip(ct_unique, cell_palette))
cell_color_map['Epi'] = '#007F00'
cell_color_map['Cancer'] = '#FF0000'
score_cmap = 'RdYlBu_r'

# Plot UMAP colored by cell type
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
ax.set_facecolor('white')

for ct in ct_unique:
    mask = cell_types == ct
    ax.scatter(
        umap_coords[mask, 0], umap_coords[mask, 1],
        c=cell_color_map[ct],
        s=1.0, alpha=0.6, linewidths=0, rasterized=True
    )

# Add legend
patches = [mpatches.Patch(color=cell_color_map[ct], label=ct) for ct in ct_unique]
ax.legend(
    handles=patches,
    fontsize=9,
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=False,
    handlelength=1.5,
    handleheight=1.5,
    borderpad=0.5
)

# Clean axes
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Save cell type UMAP
celltype_file = "umap_celltype.png"
celltype_path = os.path.join(save_dir, celltype_file)
plt.tight_layout()
fig.savefig(celltype_path, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(celltype_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
plt.close(fig)

# Plot UMAP colored by signature score
score_vals = c1_selected_data.obs[target_score].values
vmin_score = 0.0
vmax_score = 0.11

fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
ax.set_facecolor('white')

scatter = ax.scatter(
    umap_coords[:, 0], umap_coords[:, 1],
    c=score_vals,
    cmap=score_cmap,
    s=0.8, alpha=0.7, linewidths=0, rasterized=True,
    vmin=vmin_score, vmax_score=vmax_score
)

# Add color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_ticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
cbar.ax.tick_params(labelsize=10)
cbar.set_label('')

# Clean axes
ax.set_title('')
ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Save score UMAP
score_file = f"umap_C1_{target_score}.png"
score_path = os.path.join(save_dir, score_file)
plt.tight_layout()
fig.savefig(score_path, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(score_path.replace('.png', '.svg'), bbox_inches='tight', facecolor='white')
plt.close(fig)