# corpus_cleaning.py
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from itertools import combinations


# Ensure output directories exist
os.makedirs("../output/plots", exist_ok=True)
os.makedirs("../output/stats", exist_ok=True)

# --- 1. Load the CSV ---
file_path = "../output/data/corpus.csv"
df = pd.read_csv(file_path)


# --- 2. Define cleaning function ---
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


# --- 3. Apply cleaning ---
df['sentence_clean'] = df['sentence'].apply(clean_sentence)

# --- 4. Inspect results ---
print(df[['sentence', 'sentence_clean']].head(10))

# embeddings_analysis

# --- 2. Choose model ---
MODEL_NAME = "bert-base-uncased"  # can swap to other models later
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


# --- 3. Helper function to get token embeddings ---
def get_token_embeddings(text):
    """
    Returns token embeddings and token positions.
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the last hidden state (tokens x embedding_dim)
    token_embeds = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
    return tokens, token_embeds


# --- 4. Extract "imagine" token embeddings ---
records = []
for idx, row in df.iterrows():
    sentence = row['sentence_clean']
    sense = row['sense']
    intentionality_z = row['intentionality_z']
    pictoriality_z = row['pictoriality_z']
    factivity_z = row['factivity_z']
    tokens, token_embeds = get_token_embeddings(sentence)

    # find all "imagine" tokens (BERT tokenization may split, e.g., 'imagine' -> ['im', '##agine'])
    token_positions = [i for i, t in enumerate(tokens) if "imagine" in t]
    if not token_positions:
        continue  # skip sentences without "imagine"

    # Option 1: pick token closest to sentence center
    midpoint = len(tokens) // 2
    selected_pos = min(token_positions, key=lambda x: abs(x - midpoint))
    records.append({
        "idx": idx,
        "sentence": sentence,
        "sense": sense,
        "intentionality_z": intentionality_z,
        "pictoriality_z": pictoriality_z,
        "factivity_z": factivity_z,
        "token_position": selected_pos,
        "embedding": token_embeds[selected_pos].numpy()
    })

    # Option 2: explode all tokens (uncomment if desired)
    # for pos in token_positions:
    #     records.append({
    #         "idx": idx,
    #         "sentence": sentence,
    #         "sense": sense,
    #         "token_position": pos,
    #         "embedding": token_embeds[pos].numpy()
    #     })

imagine_df = pd.DataFrame(records)

# --- 5. Compute centroids for each sense ---
sense_groups = imagine_df.groupby('sense')
centroids = {}
for sense, group in sense_groups:
    embeddings = np.stack(group['embedding'].values)
    centroids[sense] = embeddings.mean(axis=0)

# --- 6. Measure distances between centroids ---
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

# --- 7. Save embeddings for downstream visualization / clustering ---
imagine_df.to_pickle("../output/data/imagine_token_embeddings.pkl")

# --- 8. Compute distance distributions ---
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

# --- 9. Optional: summarize distributions ---
summary_df = distance_df.groupby(['type', 'sense', 'distance_to'])['cosine_distance'].describe()
print(summary_df)

# --- 10. Save for plotting ---
distance_df.to_pickle("../output/data/imagine_token_distance_distributions.pkl")


# Overlap scores
overlap_scores = {}

plt.figure(figsize=(8, 6))
colors = ["#F8766D", "#00BFC4"]

for i, sense in enumerate(sorted(distance_df['sense'].unique())):
    own_dist = distance_df[(distance_df['sense'] == sense) & (distance_df['type'] == "own_centroid")][
        'cosine_distance'].values
    other_dist = distance_df[(distance_df['sense'] == sense) & (distance_df['type'] == "other_centroid")][
        'cosine_distance'].values

    # KDEs
    kde_own = gaussian_kde(own_dist)
    kde_other = gaussian_kde(other_dist)

    xs = np.linspace(0, 2, 200)  # adjust if your cosine distance range differs
    density_own = kde_own(xs)
    density_other = kde_other(xs)

    # Overlap area
    overlap_area = np.trapezoid(np.minimum(density_own, density_other), xs)
    overlap_scores[sense] = overlap_area

    # Plot KDEs
    plt.plot(xs, density_own, color=colors[i], linestyle='-', label=f'Sense {sense} own')
    plt.plot(xs, density_other, color=colors[i], linestyle='--', label=f'Sense {sense} other')

plt.xlabel("Cosine distance")
plt.ylabel("Density")
plt.title("Distance distributions to own vs other centroids")
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig("../output/plots/overlap_density.png", dpi=300)
plt.close()

# Save overlap scores
with open("../output/stats/overlap_density_estimates.txt", "w") as f:
    for sense, score in overlap_scores.items():
        f.write(f"Sense {sense}\t{score:.4f}\n")

print("Overlap scores:", overlap_scores)
print("Plot saved to ../output/plots/overlap_density.png")
print("Overlap estimates saved to ../output/stats/overlap_density_estimates.txt")

# embeddings: your dataframe with columns ['sentence_idx', 'token_idx', 'sense', 'embedding']

senses = imagine_df['sense'].unique()
dims_to_run = [1, 2]  # 1D and 2D KDEs
colors = ["#F8766D", "#00BFC4"]  # Distinct colors for each sense

for n_dim in dims_to_run:
    # --- PCA reduction if needed ---
    emb_matrix = np.stack(imagine_df['embedding'].values)
    if n_dim < emb_matrix.shape[1]:
        pca = PCA(n_components=n_dim)
        emb_reduced = pca.fit_transform(emb_matrix)
    else:
        emb_reduced = emb_matrix

    # --- Add reduced dimensions to dataframe ---
    dim_cols = [f"dim{i+1}" for i in range(n_dim)]
    emb_df = imagine_df.copy()
    for i, col in enumerate(dim_cols):
        emb_df[col] = emb_reduced[:, i]

    # --- Prepare grid for KDE evaluation ---
    if n_dim == 1:
        xs = np.linspace(emb_df['dim1'].min(), emb_df['dim1'].max(), 200)
    elif n_dim == 2:
        x_min, x_max = emb_df['dim1'].min(), emb_df['dim1'].max()
        y_min, y_max = emb_df['dim2'].min(), emb_df['dim2'].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_points = np.vstack([xx.ravel(), yy.ravel()])

    # --- Compute sense-specific overlap fractions ---
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

    # --- Plot all senses with shaded overlap ---
    plt.figure(figsize=(8, 4) if n_dim == 1 else (6, 6))

    if n_dim == 1:
        # Plot KDEs
        for i, sense in enumerate(senses):
            kde = gaussian_kde(emb_df[emb_df['sense'] == sense][dim_cols].values.T)
            density = kde(xs)
            plt.plot(xs, density, color=colors[i], label=f'Sense {sense}')
        # Shaded overlap
        plt.fill_between(xs, 0, min_density, color='grey', alpha=0.4, label='Overlap')
        plt.xlabel("Dim 1")
        plt.ylabel("Density")
        plt.legend()
    else:
        # 2D contours for senses
        Z1 = gaussian_kde(emb_df[emb_df['sense'] == senses[0]][dim_cols].values.T)(grid_points).reshape(xx.shape)
        Z2 = gaussian_kde(emb_df[emb_df['sense'] == senses[1]][dim_cols].values.T)(grid_points).reshape(xx.shape)
        # Individual contour lines
        plt.contour(xx, yy, Z1, levels=5, colors=[colors[0]], alpha=0.7)
        plt.contour(xx, yy, Z2, levels=5, colors=[colors[1]], alpha=0.7)
        # Shaded overlap
        plt.contourf(xx, yy, min_density, levels=20, cmap='Greys', alpha=0.4)
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"../output/plots/kde{n_dim}D_embeddings_overlap.png", dpi=300)
    plt.close()

    # --- Save overlap scores ---
    with open(f"../output/stats/kde{n_dim}D_sense_specific_overlap_embeddings.txt", "w") as f:
        for pair, score in overlap_scores.items():
            f.write(f"{pair}\t{score:.4f}\n")

    print(f"{n_dim}D sense-specific KDE overlaps:", overlap_scores)

#### Experiment 2
# Stack embeddings into a matrix
emb_matrix = np.stack(imagine_df['embedding'].values)  # shape: (n_samples, emb_dim)

# Pairwise cosine similarity
emb_similarity = cosine_similarity(emb_matrix)  # shape: (n_samples, n_samples)

labels = imagine_df['sense'].values
sense_similarity = (labels[:, None] == labels[None, :]).astype(int)

from scipy.stats import spearmanr

# Extract upper triangle indices
triu_idx = np.triu_indices_from(emb_similarity, k=1)

emb_vals = emb_similarity[triu_idx]
sense_vals = sense_similarity[triu_idx]

# Spearman correlation
rho, pval = spearmanr(emb_vals, sense_vals)

print(f"Spearman correlation between embedding similarity and sense similarity: rho={rho:.3f}, p={pval:.3e}")
with open("../output/stats/embedding_vs_sense_similarity.txt", "w") as f:
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
plt.savefig("../output/plots/embedding_vs_sense_similarity.png", dpi=300)
plt.close()


####### Experiment 3
from sklearn.linear_model import LinearRegression


# assume imagine_df is already loaded and contains:
# - "embedding" column (list/array per row)
# - z-scored dimensions: "intentionality_z", "factivity_z", "pictoriality_z"
# - "sense" column (categorical labels)

# Stack embeddings into matrix
emb_matrix = np.stack(imagine_df['embedding'].values)

# Dimensions to predict
dims = ['intentionality_z', 'factivity_z', 'pictoriality_z']

# Dictionary to hold projections
proj_dict = {}

for dim in dims:
    # Train linear regression to predict this dimension from embeddings
    y = imagine_df[dim].values
    lr = LinearRegression()
    lr.fit(emb_matrix, y)

    # Project embeddings onto learned direction
    proj = emb_matrix @ lr.coef_

    # Rescale projection to match original z-score range for interpretability
    min_val, max_val = y.min(), y.max()
    proj_scaled = (proj - proj.min()) / (proj.max() - proj.min())  # 0-1
    proj_scaled = proj_scaled * (max_val - min_val) + min_val       # rescale to original z-score range

    # Store
    proj_dict[dim] = proj_scaled

# Create DataFrame with projections and sense
proj_df = pd.DataFrame(proj_dict)
proj_df['sense'] = imagine_df['sense'].values

# Plot 3D scatter colored by sense
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

senses = proj_df['sense'].unique()
colors = ["#F8766D", "#00BFC4"]  # Distinct colors for each sense

for i, sense in enumerate(senses):
    subset = proj_df[proj_df['sense'] == sense]
    ax.scatter(
        subset['intentionality_z'],
        subset['factivity_z'],
        subset['pictoriality_z'],
        label=f"Sense {sense}",
        color=colors[i],
        alpha=0.7,
        s=40
    )

ax.set_xlabel("Intentionality (z-score)")
ax.set_ylabel("Factivity (z-score)")
ax.set_zlabel("Pictoriality (z-score)")
# ax.set_title("Embedding projections onto dimension directions")
ax.legend()
plt.tight_layout()
plt.savefig("../output/plots/embedding_projection_3D.png", dpi=300)
plt.close()


import matplotlib.patches as mpatches

dim_pairs = list(combinations(['intentionality_z', 'factivity_z', 'pictoriality_z'], 2))
senses = proj_df['sense'].unique()
colors = ["#F8766D", "#00BFC4"]  # Distinct colors for each sense

for dim_x, dim_y in dim_pairs:
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    legend_patches = []

    for i, sense in enumerate(senses):
        subset = proj_df[proj_df['sense'] == sense]

        # Only plot core density (remove outermost shape)
        sns.kdeplot(
            x=subset[dim_x],
            y=subset[dim_y],
            fill=True,
            thresh=0.05,  # ignore densities below 5% of max
            levels=50,    # smoother inner shapes
            color=colors[i],
            alpha=0.5,
            ax=ax
        )

        legend_patches.append(mpatches.Patch(color=colors[i], label=f"Sense {sense}"))

    plt.xlabel(f"{dim_x.replace('_z','').capitalize()} (z-score)")
    plt.ylabel(f"{dim_y.replace('_z','').capitalize()} (z-score)")
    # plt.title(f"2D KDE projections: {dim_x} vs {dim_y}")
    plt.legend(handles=legend_patches)
    plt.tight_layout()
    plt.savefig(f"../output/plots/embedding_projection_{dim_x}_{dim_y}_2D_core.png", dpi=300)
    plt.close()