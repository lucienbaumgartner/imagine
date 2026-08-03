# corpus_cleaning.py
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
import matplotlib.patches as mpatches

# Ensure output directories exist
os.makedirs("../output/plots", exist_ok=True)
os.makedirs("../output/stats", exist_ok=True)

# Load the CSV
file_path = "../output/data/corpus_v3.csv"
df = pd.read_csv(file_path)


# Define cleaning function
def clean_sentence(text):
    if pd.isna(text):
        return ""

    # Remove speaker tags (e.g., SP:PS0HM, can be letters+colon+letters/numbers)
    text = re.sub(r'\b[A-Z]{1,3}:[A-Z0-9]+\b', '', text)

    # Remove discourse annotations like (pause), (unclear), (reading)
    text = re.sub(r'\([a-zA-Z0-9_\- ]+\)', '', text)

    # Remove extra whitespace (leading/trailing and multiple spaces)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Apply cleaning
df['sentence_clean'] = df['sentence'].apply(clean_sentence)

# Inspect results
print(df[['sentence', 'sentence_clean']].head(10))

##############################################################
########## Sense overlap based on centroid distance ##########
##############################################################

# Choose model
MODEL_NAME = "bert-base-uncased"  # can swap to other models later
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Use Apple Silicon's Metal GPU backend when available, falling back to CPU elsewhere (e.g. on
# a machine without MPS) so this still runs unchanged on other hardware.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


# Helper function to get token embeddings
# NINA, PLEASE ADAPT THIS WITH YOUR FUNCTION
def get_token_embeddings(text):
    """
    Returns token strings, a dict of {variant_name: [seq_len, hidden_dim] tensor} giving three
    different ways of reading a token's embedding off BERT's hidden states, and each token's
    (start, end) character offset into `text` (offsets are needed to disambiguate sentences
    that contain "imagine" more than once).

    The three variants exist because the very last hidden layer is somewhat specialized
    toward BERT's masked-language-modeling pretraining objective; middle-to-late layers, or an
    average across the last several layers, sometimes carry semantic content at least as well
    (see e.g. Ethayarajh 2019). "last" is what the rest of this script's plots and stats use;
    the other two are only used in the layer-sensitivity robustness check at the end.
    """
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop("offset_mapping").squeeze(0).tolist()
    input_ids = inputs["input_ids"].squeeze(0)  # keep a CPU copy for token-string lookup below
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (n_layers + 1) tensors [1, seq_len, hidden_dim]; index 0 is
    # the input embedding layer, indices 1..12 are the outputs of BERT-base's 12 transformer
    # layers. hidden_states[-1] is identical to outputs.last_hidden_state.
    hidden_states = outputs.hidden_states
    # Move back to CPU here so every downstream .numpy() call keeps working unchanged.
    variant_embeds = {
        "last": hidden_states[-1].squeeze(0).cpu(),
        "second_to_last": hidden_states[-2].squeeze(0).cpu(),
        "avg_last4": torch.stack(hidden_states[-4:]).mean(dim=0).squeeze(0).cpu(),
    }
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return tokens, variant_embeds, offsets


# Extract token embeddings
records = []
n_ambiguous = 0
for idx, row in df.iterrows():
    sentence = row['sentence_clean']
    sense = row['sense']
    intentionality = row['intentionality']
    pictoriality = row['pictoriality']
    factivity = row['factivity']
    tokens, variant_embeds, offsets = get_token_embeddings(sentence)

    # find all "imagine" tokens (BERT tokenization may split, e.g., 'imagine' -> ['im', '##agine'])
    token_positions = [i for i, t in enumerate(tokens) if "imagine" in t]
    if not token_positions:
        continue  # skip sentences without "imagine"

    # Most sentences contain exactly one "imagine" token, but a small minority (~2% of the
    # corpus) contain it more than once (e.g. "I could imagine X, but I couldn't imagine Y").
    # There is no positional marker in the exported data indicating which occurrence was the
    # one hand-annotated for sense/intentionality/factivity/pictoriality, so we fall back on
    # the corpus-linguistics convention that concordance extracts center the keyword in its
    # context window: pick whichever occurrence's character span is closest to the sentence's
    # character midpoint. This is more robust than comparing raw token indices, since cleaning
    # and sub-word splitting can otherwise skew a token-index midpoint away from the true
    # center of the sentence.
    if len(token_positions) > 1:
        n_ambiguous += 1
    char_midpoint = len(sentence) / 2
    selected_pos = min(
        token_positions,
        key=lambda i: abs((offsets[i][0] + offsets[i][1]) / 2 - char_midpoint)
    )
    records.append({
        "idx": idx,
        "sentence": sentence,
        "sense": sense,
        "intentionality": intentionality,
        "pictoriality": pictoriality,
        "factivity": factivity,
        "token_position": selected_pos,
        "embedding": variant_embeds["last"][selected_pos].numpy(),
        "embedding_second_to_last": variant_embeds["second_to_last"][selected_pos].numpy(),
        "embedding_avg_last4": variant_embeds["avg_last4"][selected_pos].numpy(),
    })

    # Option 2: explode all tokens (uncomment if desired)
    # for pos in token_positions:
    #     records.append({
    #         "idx": idx,
    #         "sentence": sentence,
    #         "sense": sense,
    #         "token_position": pos,
    #         "embedding": variant_embeds["last"][pos].numpy()
    #     })

print(f"{n_ambiguous} of {len(df)} sentences contained more than one candidate 'imagine' "
      f"token; the occurrence closest to the sentence's character midpoint was used for each.")

imagine_df = pd.DataFrame(records)

# Global color mapping so every sense keeps the same color across every figure in this
# script (previously each plotting section built its own color assignment independently,
# some sorted and some in pandas' arbitrary first-seen order, so the same sense could end up
# a different color in different figures).
SENSE_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3"]
ALL_SENSES = sorted(imagine_df['sense'].unique())
SENSE_COLOR = dict(zip(ALL_SENSES, SENSE_PALETTE))

# Compute centroids for each sense
sense_groups = imagine_df.groupby('sense')
centroids = {}
for sense, group in sense_groups:
    embeddings = np.stack(group['embedding'].values)
    centroids[sense] = embeddings.mean(axis=0)

# Measure distances between centroids
senses = list(centroids.keys())
dist_matrix = np.zeros((len(senses), len(senses)))
for i, s1 in enumerate(senses):
    for j, s2 in enumerate(senses):
        dist_matrix[i, j] = 1 - cosine_similarity(
            centroids[s1].reshape(1, -1),
            centroids[s2].reshape(1, -1)
        )[0, 0]

dist_df = pd.DataFrame(dist_matrix, index=senses, columns=senses)
print("Cosine distance between centroids:\n", dist_df)

# Save embeddings for downstream visualization / clustering
imagine_df.to_pickle("../output/data/imagine_token_embeddings_v3.pkl")

# Compute distance distributions
distance_records = []

for idx, row in imagine_df.iterrows():
    token_embed = row['embedding'].reshape(1, -1)
    own_sense = row['sense']

    # distance to own centroid
    dist_to_own = 1 - cosine_similarity(token_embed, centroids[own_sense].reshape(1, -1))[0, 0]
    distance_records.append({
        "idx": row['idx'],
        "sense": own_sense,
        "token_position": row['token_position'],
        "distance_to": own_sense,
        "cosine_distance": dist_to_own,
        "type": "own_centroid"
    })

    # distances to other centroids
    for other_sense, centroid in centroids.items():
        if other_sense == own_sense:
            continue
        dist_to_other = 1 - cosine_similarity(token_embed, centroid.reshape(1, -1))[0, 0]
        distance_records.append({
            "idx": row['idx'],
            "sense": own_sense,
            "token_position": row['token_position'],
            "distance_to": other_sense,
            "cosine_distance": dist_to_other,
            "type": "other_centroid"
        })

distance_df = pd.DataFrame(distance_records)

# Summarize distributions
summary_df = distance_df.groupby(['type', 'sense', 'distance_to'])['cosine_distance'].describe()
print(summary_df)

# Save for plotting at later stage if necessary
distance_df.to_pickle("../output/data/imagine_token_distance_distributions_v3.pkl")

# Plot overlap
overlap_scores = {}

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
axes = axes.flatten()

for i, sense in enumerate(ALL_SENSES):
    ax = axes[i]

    # Own centroid
    own_dist = distance_df[(distance_df['sense'] == sense) &
                           (distance_df['type'] == "own_centroid")]['cosine_distance'].values
    kde_own = gaussian_kde(own_dist)
    xs = np.linspace(0, 2, 200)
    ax.plot(xs, kde_own(xs), color=SENSE_COLOR[sense], linestyle='-', linewidth=1.5, label='Own')

    # Each other centroid separately
    other_senses = [s for s in ALL_SENSES if s != sense]
    for other_sense in other_senses:
        other_dist = distance_df[(distance_df['sense'] == sense) &
                                 (distance_df['distance_to'] == other_sense)]['cosine_distance'].values
        if len(other_dist) > 1:
            kde_other = gaussian_kde(other_dist)
            ax.plot(xs, kde_other(xs), color=SENSE_COLOR[other_sense],
                    linestyle='--', linewidth=1, label=f'→ {other_sense}')

    ax.set_title(f'Sense {sense}', fontsize=9)
    ax.legend(fontsize=6)
    ax.set_xlabel("Cosine distance", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)

plt.tight_layout()
plt.savefig("../output/plots/overlap_density_v3.png", dpi=300)
plt.close()

overlap_scores = {}
xs = np.linspace(0, 2, 200)

for sense in ALL_SENSES:
    own_dist = distance_df[(distance_df['sense'] == sense) &
                           (distance_df['type'] == "own_centroid")]['cosine_distance'].values
    kde_own = gaussian_kde(own_dist)

    for other_sense in ALL_SENSES:
        if other_sense == sense:
            continue
        other_dist = distance_df[(distance_df['sense'] == sense) &
                                 (distance_df['distance_to'] == other_sense)]['cosine_distance'].values
        if len(other_dist) > 1:
            kde_other = gaussian_kde(other_dist)
            overlap = np.trapezoid(np.minimum(kde_own(xs), kde_other(xs)), xs)
            overlap_scores[f"{sense} vs {other_sense}"] = overlap

with open("../output/stats/overlap_density_estimates_v3.txt", "w") as f:
    for pair, score in overlap_scores.items():
        f.write(f"{pair}\t{score:.4f}\n")

print("Overlap scores:", overlap_scores)
print("Plot saved to ../output/plots/overlap_density_v3.png")
print("Overlap estimates saved to ../output/stats/overlap_density_estimates_v3.txt")


#######################################
########## Geometric overlap ##########
#######################################
senses = ALL_SENSES
dims_to_run = [1, 2]  # 1D and 2D KDEs

for n_dim in dims_to_run:
    # PCA reduction
    emb_matrix = np.stack(imagine_df['embedding'].values)
    if n_dim < emb_matrix.shape[1]:
        pca = PCA(n_components=n_dim)
        emb_reduced = pca.fit_transform(emb_matrix)
    else:
        emb_reduced = emb_matrix

    # Add reduced dimensions to dataframe
    dim_cols = [f"dim{i+1}" for i in range(n_dim)]
    emb_df = imagine_df.copy()
    for i, col in enumerate(dim_cols):
        emb_df[col] = emb_reduced[:, i]

    # Prepare grid for KDE evaluation
    if n_dim == 1:
        xs = np.linspace(emb_df['dim1'].min(), emb_df['dim1'].max(), 200)
    elif n_dim == 2:
        x_min, x_max = emb_df['dim1'].min(), emb_df['dim1'].max()
        y_min, y_max = emb_df['dim2'].min(), emb_df['dim2'].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

    # Compute sense-specific overlap fractions
    overlap_scores = {}
    for sense1, sense2 in combinations(senses, 2):
        data1 = emb_df[emb_df['sense'] == sense1][dim_cols].values.T
        data2 = emb_df[emb_df['sense'] == sense2][dim_cols].values.T
        kde1 = gaussian_kde(data1)
        kde2 = gaussian_kde(data2)

        if n_dim == 1:
            density1 = kde1(xs)
            density2 = kde2(xs)
            min_density = np.minimum(density1, density2)
            # Fraction of sense1 surface overlapping sense2
            overlap_1 = np.trapezoid(min_density, xs) / np.trapezoid(density1, xs)
            # Fraction of sense2 surface overlapping sense1
            overlap_2 = np.trapezoid(min_density, xs) / np.trapezoid(density2, xs)
        else:  # 2D
            density1 = kde1(grid_points).reshape(xx.shape)
            density2 = kde2(grid_points).reshape(xx.shape)
            min_density = np.minimum(density1, density2)
            dx = (x_max - x_min) / (xx.shape[1] - 1)
            dy = (y_max - y_min) / (xx.shape[0] - 1)
            area = dx * dy
            total_area1 = np.sum(density1) * area
            total_area2 = np.sum(density2) * area
            overlap_area = np.sum(min_density) * area
            overlap_1 = overlap_area / total_area1
            overlap_2 = overlap_area / total_area2

        overlap_scores[f"{sense1} over {sense2}"] = overlap_1
        overlap_scores[f"{sense2} over {sense1}"] = overlap_2

    # Plot all senses with shaded overlap
    plt.figure(figsize=(8, 4) if n_dim == 1 else (6, 6))

    if n_dim == 1:
        densities = []
        for sense in senses:
            kde = gaussian_kde(emb_df[emb_df['sense'] == sense][dim_cols].values.T)
            density = kde(xs)
            densities.append(density)
            plt.plot(xs, density, color=SENSE_COLOR[sense], label=f'Sense {sense}')
        plt.xlabel("Dim 1")
        plt.ylabel("Density")
        plt.legend()
    else:
        for sense in senses:
            Z = gaussian_kde(emb_df[emb_df['sense'] == sense][dim_cols].values.T)(grid_points).reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=5, colors=[SENSE_COLOR[sense]], alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"../output/plots/kde{n_dim}D_embeddings_overlap_v3.png", dpi=300)
    plt.close()

    # Save overlap scores
    with open(f"../output/stats/kde{n_dim}D_sense_specific_overlap_embeddings_v3.txt", "w") as f:
        for pair, score in overlap_scores.items():
            f.write(f"{pair}\t{score:.4f}\n")

    print(f"{n_dim}D sense-specific KDE overlaps:", overlap_scores)

##############################################################
########## Embedding similarity vs sense similarity ##########
##############################################################
# Stack embeddings into a matrix
emb_matrix = np.stack(imagine_df['embedding'].values)  # shape: (n_samples, emb_dim)

# Pairwise cosine similarity
emb_similarity = cosine_similarity(emb_matrix)  # shape: (n_samples, n_samples)

labels = imagine_df['sense'].to_numpy(dtype=str)
sense_similarity = (labels[:, None] == labels[None, :]).astype(int)

# Extract upper triangle indices
triu_idx = np.triu_indices_from(emb_similarity, k=1)

emb_vals = emb_similarity[triu_idx]
sense_vals = sense_similarity[triu_idx]

# Spearman correlation
rho, pval = spearmanr(emb_vals, sense_vals)

print(f"Spearman correlation between embedding similarity and sense similarity: rho={rho:.3f}, p={pval:.3e}")
with open("../output/stats/embedding_vs_sense_similarity_v3.txt", "w") as f:
    f.write(f"Spearman correlation (rho): {rho:.4f}\n")
    f.write(f"P-value: {pval:.4e}\n")

plt.figure(figsize=(6, 4))

# Separate embedding similarities by sense similarity
same_sense = emb_vals[sense_vals == 1]
diff_sense = emb_vals[sense_vals == 0]

# KDE plots
sns.kdeplot(same_sense, fill=True, color="red", alpha=0.5, label="Same sense")
sns.kdeplot(diff_sense, fill=True, color="blue", alpha=0.5, label="Different sense")

plt.xlabel("Cosine similarity between embeddings")
plt.ylabel("Density")
# plt.title("Embedding similarity vs sense similarity")
plt.legend()
plt.tight_layout()
plt.savefig("../output/plots/embedding_vs_sense_similarity_v3.png", dpi=300)
plt.close()


#######################################################
########## Senses under embedding projection ##########
#######################################################
# Stack embeddings into matrix
emb_matrix = np.stack(imagine_df['embedding'].values)

# Dimensions to predict
dims = ['intentionality', 'factivity', 'pictoriality']

# Dictionary to hold projections
proj_dict = {}
r2_records = []

for dim in dims:
    # Train a ridge-regularized regression to predict this dimension from embeddings. The
    # embedding dimension (768 for bert-base-uncased) is close to the sample size here, which
    # is a regime where plain, unregularized OLS is essentially guaranteed to overfit and
    # yield an unstable, non-generalizable "direction" -- ridge (with its penalty chosen via
    # cross-validation) is the same regularization strategy used throughout the rest of this
    # project's models.
    y = imagine_df[dim].values
    lr = RidgeCV(alphas=np.logspace(-3, 3, 13))
    lr.fit(emb_matrix, y)
    print(f"{dim}: selected ridge alpha = {lr.alpha_:.3g}")

    # In-sample R^2 is optimistic (the model has seen every row it's scored on); 5-fold
    # cross-validated R^2 is the honest, held-out estimate of how well the embeddings actually
    # predict this dimension, and the one the report interprets.
    r2_in_sample = lr.score(emb_matrix, y)
    cv = KFold(n_splits=5, shuffle=True, random_state=1847)
    cv_r2 = cross_val_score(RidgeCV(alphas=np.logspace(-3, 3, 13)), emb_matrix, y, cv=cv, scoring="r2")
    print(f"{dim}: in-sample R2={r2_in_sample:.3f}  5-fold CV R2={cv_r2.mean():.3f} (+/- {cv_r2.std():.3f})")
    r2_records.append({
        "dimension": dim,
        "alpha": lr.alpha_,
        "r2_in_sample": r2_in_sample,
        "r2_cv_mean": cv_r2.mean(),
        "r2_cv_std": cv_r2.std(),
    })

    # Project embeddings onto learned direction
    proj = emb_matrix @ lr.coef_

    # Rescale projection to match original z-score range for interpretability
    min_val, max_val = y.min(), y.max()
    proj_scaled = (proj - proj.min()) / (proj.max() - proj.min())  # 0-1
    proj_scaled = proj_scaled * (max_val - min_val) + min_val       # rescale to original z-score range

    # Store
    proj_dict[dim] = proj_scaled

r2_df = pd.DataFrame(r2_records)
r2_df.to_csv("../output/stats/embedding_projection_r2_v3.txt", sep="\t", index=False)
print("Projection R^2 saved to ../output/stats/embedding_projection_r2_v3.txt")

# Create DataFrame with projections and sense
proj_df = pd.DataFrame(proj_dict)
proj_df['sense'] = imagine_df['sense'].values

# Plot 3D scatter colored by sense
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

senses = ALL_SENSES

for sense in senses:
    subset = proj_df[proj_df['sense'] == sense]
    ax.scatter(
        subset['intentionality'],
        subset['factivity'],
        subset['pictoriality'],
        label=f"Sense {sense}",
        color=SENSE_COLOR[sense],
        alpha=0.7,
        s=40
    )

ax.set_xlabel("Intentionality (z-score)")
ax.set_ylabel("Factivity (z-score)")
ax.set_zlabel("Pictoriality (z-score)")
# ax.set_title("Embedding projections onto dimension directions")
ax.legend()
plt.tight_layout()
plt.savefig("../output/plots/embedding_projection_3D_v3.png", dpi=300)
plt.close()

# 2D Plots
dim_pairs = list(combinations(['intentionality', 'factivity', 'pictoriality'], 2))
senses = ALL_SENSES

for dim_x, dim_y in dim_pairs:
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    legend_patches = []

    for sense in senses:
        subset = proj_df[proj_df['sense'] == sense]

        # Only plot core density (remove outermost shape)
        sns.kdeplot(
            x=subset[dim_x],
            y=subset[dim_y],
            fill=True,
            thresh=0.05,  # ignore densities below 5% of max
            levels=50,    # smoother inner shapes
            color=SENSE_COLOR[sense],
            alpha=0.5,
            ax=ax
        )

        legend_patches.append(mpatches.Patch(color=SENSE_COLOR[sense], label=f"Sense {sense}"))

    plt.xlabel(f"{dim_x.replace('','').capitalize()} (z-score)")
    plt.ylabel(f"{dim_y.replace('','').capitalize()} (z-score)")
    # plt.title(f"2D KDE projections: {dim_x} vs {dim_y}")
    plt.legend(handles=legend_patches)
    plt.tight_layout()
    plt.savefig(f"../output/plots/embedding_projection_{dim_x}_{dim_y}_2D_core_v3.png", dpi=300)
    plt.close()

##############################################################################
########## Robustness check: sensitivity to embedding-layer choice #########
##############################################################################
# Every analysis above uses "last": the raw final BERT hidden layer. We check whether that
# choice matters by repeating the centroid-based analysis with two alternative ways of
# reading a token's embedding off BERT's hidden states -- "second_to_last" and "avg_last4"
# (the average of the last four layers) -- and comparing all three pairwise with two metrics
# suited to what is actually being compared:
#   - Cohen's kappa on each token's nearest-centroid sense (by cosine distance): a categorical
#     "predicted sense" per variant, so kappa (chance-corrected agreement) is the natural fit.
#   - Spearman's rho on the per-sense-pair centroid-overlap fractions (as in "Sense overlap
#     based on centroid distance" above): those are continuous scores, so a rank correlation
#     between the score vectors shows directly whether the variants agree on which sense pairs
#     overlap most/least.
EMBEDDING_VARIANT_COLUMNS = {
    "last": "embedding",
    "second_to_last": "embedding_second_to_last",
    "avg_last4": "embedding_avg_last4",
}

variant_nearest_sense = {}
variant_overlap_scores = {}
xs_variant = np.linspace(0, 2, 200)

for variant, col in EMBEDDING_VARIANT_COLUMNS.items():
    # Centroids for this variant
    centroids_v = {}
    for sense, group in imagine_df.groupby('sense'):
        embeddings = np.stack(group[col].values)
        centroids_v[sense] = embeddings.mean(axis=0)

    # Nearest-centroid classification (categorical) for this variant
    emb_matrix_v = np.stack(imagine_df[col].values)
    centroid_matrix_v = np.stack([centroids_v[s] for s in ALL_SENSES])
    sim_to_centroids = cosine_similarity(emb_matrix_v, centroid_matrix_v)
    nearest_idx = sim_to_centroids.argmax(axis=1)
    variant_nearest_sense[variant] = np.array(ALL_SENSES)[nearest_idx]

    # Centroid-overlap fractions (continuous) for this variant, mirroring the "Sense overlap
    # based on centroid distance" computation above
    dist_records_v = []
    for _, row in imagine_df.iterrows():
        token_embed = row[col].reshape(1, -1)
        own_sense = row['sense']
        dist_to_own = 1 - cosine_similarity(token_embed, centroids_v[own_sense].reshape(1, -1))[0, 0]
        dist_records_v.append({"sense": own_sense, "distance_to": own_sense, "cosine_distance": dist_to_own, "type": "own_centroid"})
        for other_sense, centroid in centroids_v.items():
            if other_sense == own_sense:
                continue
            dist_to_other = 1 - cosine_similarity(token_embed, centroid.reshape(1, -1))[0, 0]
            dist_records_v.append({"sense": own_sense, "distance_to": other_sense, "cosine_distance": dist_to_other, "type": "other_centroid"})
    distance_df_v = pd.DataFrame(dist_records_v)

    overlap_v = {}
    for sense in ALL_SENSES:
        own_dist = distance_df_v[(distance_df_v['sense'] == sense) & (distance_df_v['type'] == "own_centroid")]['cosine_distance'].values
        kde_own = gaussian_kde(own_dist)
        for other_sense in ALL_SENSES:
            if other_sense == sense:
                continue
            other_dist = distance_df_v[(distance_df_v['sense'] == sense) & (distance_df_v['distance_to'] == other_sense)]['cosine_distance'].values
            if len(other_dist) > 1:
                kde_other = gaussian_kde(other_dist)
                overlap_v[f"{sense} vs {other_sense}"] = np.trapezoid(np.minimum(kde_own(xs_variant), kde_other(xs_variant)), xs_variant)
    variant_overlap_scores[variant] = overlap_v

# Pairwise Cohen's kappa on nearest-centroid classifications
kappa_results = {}
for v1, v2 in combinations(EMBEDDING_VARIANT_COLUMNS.keys(), 2):
    kappa_results[f"{v1} vs {v2}"] = cohen_kappa_score(variant_nearest_sense[v1], variant_nearest_sense[v2])

# Pairwise Spearman correlation on centroid-overlap fraction vectors
corr_results = {}
for v1, v2 in combinations(EMBEDDING_VARIANT_COLUMNS.keys(), 2):
    common_pairs = sorted(set(variant_overlap_scores[v1]) & set(variant_overlap_scores[v2]))
    x = [variant_overlap_scores[v1][p] for p in common_pairs]
    y = [variant_overlap_scores[v2][p] for p in common_pairs]
    rho, _ = spearmanr(x, y)
    corr_results[f"{v1} vs {v2}"] = rho

print("Pairwise Cohen's kappa (nearest-centroid classification agreement):", kappa_results)
print("Pairwise Spearman rho (centroid-overlap fraction agreement):", corr_results)

with open("../output/stats/embedding_variant_agreement_v3.txt", "w") as f:
    f.write("Pairwise agreement between embedding-layer variants\n")
    f.write("last = raw last hidden layer; second_to_last = second-to-last hidden layer; "
            "avg_last4 = average of the last four hidden layers\n")
    f.write("=" * 70 + "\n\n")
    f.write("Cohen's kappa (nearest-centroid sense classification agreement):\n")
    for pair, kappa in kappa_results.items():
        f.write(f"  {pair}\t{kappa:.4f}\n")
    f.write("\nSpearman rho (centroid-overlap fraction agreement):\n")
    for pair, rho in corr_results.items():
        f.write(f"  {pair}\t{rho:.4f}\n")

print("Embedding-variant agreement saved to ../output/stats/embedding_variant_agreement_v3.txt")

# Extend the layer-sensitivity check to "Geometric overlap" (1D/2D PCA + KDE overlap) above.
# That method has no per-token classification step (it only ever reports overlap fractions
# between pairs of KDE-estimated densities), so there is no natural categorical output to
# compare with Cohen's kappa here -- only Spearman's rho on the overlap-fraction vectors,
# computed separately for 1D and 2D since the report presents both.
geometric_rho_records = []

for n_dim in dims_to_run:
    variant_geo_overlap = {}
    for variant, col in EMBEDDING_VARIANT_COLUMNS.items():
        emb_matrix_v = np.stack(imagine_df[col].values)
        pca = PCA(n_components=n_dim)
        emb_reduced_v = pca.fit_transform(emb_matrix_v)
        dim_cols_v = [f"dim{i+1}" for i in range(n_dim)]
        emb_df_v = pd.DataFrame(emb_reduced_v, columns=dim_cols_v)
        emb_df_v['sense'] = imagine_df['sense'].values

        if n_dim == 1:
            xs_geo = np.linspace(emb_df_v['dim1'].min(), emb_df_v['dim1'].max(), 200)
        else:
            x_min, x_max = emb_df_v['dim1'].min(), emb_df_v['dim1'].max()
            y_min, y_max = emb_df_v['dim2'].min(), emb_df_v['dim2'].max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            grid_points_v = np.vstack([xx.ravel(), yy.ravel()])

        overlap_geo = {}
        for sense1, sense2 in combinations(ALL_SENSES, 2):
            data1 = emb_df_v[emb_df_v['sense'] == sense1][dim_cols_v].values.T
            data2 = emb_df_v[emb_df_v['sense'] == sense2][dim_cols_v].values.T
            kde1, kde2 = gaussian_kde(data1), gaussian_kde(data2)
            if n_dim == 1:
                density1, density2 = kde1(xs_geo), kde2(xs_geo)
                min_density = np.minimum(density1, density2)
                overlap_geo[f"{sense1} over {sense2}"] = np.trapezoid(min_density, xs_geo) / np.trapezoid(density1, xs_geo)
                overlap_geo[f"{sense2} over {sense1}"] = np.trapezoid(min_density, xs_geo) / np.trapezoid(density2, xs_geo)
            else:
                density1 = kde1(grid_points_v).reshape(xx.shape)
                density2 = kde2(grid_points_v).reshape(xx.shape)
                min_density = np.minimum(density1, density2)
                area = ((x_max - x_min) / (xx.shape[1] - 1)) * ((y_max - y_min) / (xx.shape[0] - 1))
                overlap_geo[f"{sense1} over {sense2}"] = (np.sum(min_density) * area) / (np.sum(density1) * area)
                overlap_geo[f"{sense2} over {sense1}"] = (np.sum(min_density) * area) / (np.sum(density2) * area)
        variant_geo_overlap[variant] = overlap_geo

    for v1, v2 in combinations(EMBEDDING_VARIANT_COLUMNS.keys(), 2):
        common_pairs = sorted(set(variant_geo_overlap[v1]) & set(variant_geo_overlap[v2]))
        x = [variant_geo_overlap[v1][p] for p in common_pairs]
        y = [variant_geo_overlap[v2][p] for p in common_pairs]
        rho, _ = spearmanr(x, y)
        geometric_rho_records.append({"dimensionality": f"{n_dim}D", "pair": f"{v1} vs {v2}", "rho": rho})

geometric_rho_df = pd.DataFrame(geometric_rho_records)
geometric_rho_df.to_csv("../output/stats/embedding_variant_agreement_geometric_v3.txt", sep="\t", index=False)
print("Pairwise Spearman rho (geometric-overlap fraction agreement):")
print(geometric_rho_df)
print("Geometric-overlap variant agreement saved to ../output/stats/embedding_variant_agreement_geometric_v3.txt")