# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # Data
# data = {
#     "x": [1, 2, 4, 8, 16, 32, 64],
#     "chunks_only": [0.16494845360824742, 0.17783505154639176, 0.18041237113402062, 0.23582474226804123, 0.32989690721649484, 0.602770618556701, 0.7870489690721649],
#     "method2": [0.18556701030927836, 0.18298969072164947, 0.18298969072164947, 0.2184278350515464, 0.31862113402061853, 0.598743556701031, 0.7858408505154639],
#     "method3": [0.15979381443298968, 0.18814432989690721, 0.19716494845360824, 0.22551546391752578, 0.32023195876288657, 0.5955219072164949, 0.7848743556701031],
#     "method4": [0.15979381443298968, 0.17783505154639176, 0.18685567010309279, 0.23840206185567012, 0.33021907216494845, 0.6063144329896907, 0.786243556701031]
# }

# df = pd.DataFrame(data)

# # labels for the methods
# labels = {
#     "chunks_only": "Chunks Only",
#     "method2": "Method 2: Chunk w Window Level (4 Chunk) Metadata",
#     "method3": "Method 3: Chunks w Doc Level Metadata",
#     "method4": "Method 4: Chunks w Keyword Metadata"
# }

# plt.figure(figsize=(10, 6))

# for key, label in labels.items():
#     plt.plot(df['x'], df[key], marker='o', label=label)

# plt.xscale('log', base=2)
# plt.xticks(df['x'], df['x'])  # Ensure all x points are shown
# plt.xlabel('K')
# plt.ylabel('DRM Score')
# plt.title('Comparison of DRM Scores for RAG-enhanced Pipeline')
# plt.legend()
# plt.grid(True, which="both", ls="-", alpha=0.5)

# plt.tight_layout()
# plt.savefig('/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_plots/drm_macro_plot.png')

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Populated from your input)
# ---------------------------
topk = [1, 2, 4, 8, 16, 32, 64]

# Converting decimal probabilities to Percentages (%) by multiplying by 100
maud = np.array([
    0.34659685863874345, 0.36596858638743457, 0.48769633507853405, 
    0.65, 0.762303664921466, 0.8331479057591623, 0.8786485602094241
]) * 100

maud_enhance = np.array([
    0.09214659685863874, 0.09528795811518324, 0.10890052356020942, 
    0.11910994764397906, 0.12395287958115184, 0.12905759162303665, 
    0.13838350785340314
]) * 100

pq = np.array([
    0.16494845360824742, 0.17783505154639176, 0.18041237113402062, 
    0.23582474226804123, 0.32989690721649484, 0.602770618556701, 
    0.7870489690721649
]) * 100

pq_enhanced = np.array([
    0.10309278350515463, 0.10309278350515463, 0.10309278350515463, 
    0.12693298969072164, 0.21746134020618557, 0.5605670103092784, 
    0.7802835051546392
]) * 100

# Optional: standard deviations (using a constant placeholder as none provided)
# You can update this if you have real std dev data
std_small = np.array([1.5] * 7)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 6))

def plot_with_shade(x, y, label, color):
    plt.plot(x, y, marker='o', label=label, color=color)
    plt.fill_between(x, y - std_small, y + std_small, color=color, alpha=0.15)

# Colors
colors = {
    "maud": "#2ca02c",          # Green
    "maud_enhance": "#98df8a",  # Light Green
    "pq": "#1f77b4",            # Blue
    "pq_enhanced": "#aec7e8"    # Light Blue
}

# Plot lines
plot_with_shade(topk, maud, "MAUD", colors["maud"])
plot_with_shade(topk, maud_enhance, "MAUD Enhanced", colors["maud_enhance"])
plot_with_shade(topk, pq, "PQ", colors["pq"])
plot_with_shade(topk, pq_enhanced, "PQ Enhanced", colors["pq_enhanced"])

# ---------------------------
# Labels & Styling
# ---------------------------
# Using log scale for x-axis to evenly space 1, 2, 4, 8...
plt.xscale('log', base=2)
plt.xticks(topk, topk) 

plt.xlabel("Top-K", fontsize=14)
plt.ylabel("DRM(%)", fontsize=14)
plt.ylim(0, 105)
plt.legend(fontsize=12, loc='upper left')

plt.title("DRM Plot: MAUD vs PQ", fontsize=16, pad=20)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_plots/drm_maud_vs_pq_plot.png')