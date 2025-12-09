import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Populated from your input)
# ---------------------------
topk = [1, 2, 4, 8, 16, 32, 64]

# Converting decimal probabilities to Percentages (%) by multiplying by 100
# ------------------------------------------------------------------------------
# DRM Data
# ------------------------------------------------------------------------------
maud_drm = np.array([
    0.34659685863874345, 0.36596858638743457, 0.48769633507853405,
    0.65, 0.762303664921466, 0.8331479057591623, 0.8786485602094241
]) * 100

maud_enhanced_drm = np.array([
    0.09214659685863874, 0.09528795811518324, 0.10890052356020942,
    0.11910994764397906, 0.12395287958115184, 0.12905759162303665,
    0.13838350785340314
]) * 100

pq_drm = np.array([
    0.16494845360824742, 0.17783505154639176, 0.18041237113402062,
    0.23582474226804123, 0.32989690721649484, 0.602770618556701,
    0.7870489690721649
]) * 100

pq_enhanced_drm = np.array([
    0.10309278350515463, 0.10309278350515463, 0.10309278350515463,
    0.12693298969072164, 0.21746134020618557, 0.5605670103092784,
    0.7802835051546392
]) * 100

# ------------------------------------------------------------------------------
# Alt Recall Data
# ------------------------------------------------------------------------------
maud_alt = np.array([
    0.013795811518324609, 0.023944153577661432, 0.036303914235851414,
    0.06629020194465221, 0.11032120003324192, 0.15759868694423668,
    0.22849414111194213
]) * 100

maud_enhanced_alt = np.array([
    0.10403681542424999, 0.14881284800132968, 0.21464431147677218,
    0.3002052688440123, 0.4142616138951218, 0.5550070639075875,
    0.7111817501869858
]) * 100

pq_alt = np.array([
    0.21467231222385863, 0.3562101129111438, 0.48808296514482086,
    0.6289089347079038, 0.7628375061364752, 0.8187101129111438,
    0.8329774177712322
]) * 100

pq_enhanced_alt = np.array([
    0.2190721649484536, 0.3245704467353952, 0.48607020127638684,
    0.6483308787432499, 0.7825601374570447, 0.8329774177712322,
    0.8329774177712322
]) * 100

# ------------------------------------------------------------------------------
# Span Recall Data
# ------------------------------------------------------------------------------
maud_span = np.array([
    0.01036649214659686, 0.020837696335078534, 0.03756669159810521,
    0.06698329593617552, 0.11349871187567523, 0.16680711377046456,
    0.24926452256295187
]) * 100

maud_enhanced_span = np.array([
    0.1309739881991191, 0.19077329011883984, 0.28364539183910914,
    0.40650004155239755, 0.5765137538435968, 0.7806020942408377,
    1.0443650793650794
]) * 100

pq_span = np.array([
    0.1980915562101129, 0.36557437407952875, 0.5637088856161021,
    0.8103767795778105, 1.035640648011782, 1.1190905743740795,
    1.1395434462444773
]) * 100

pq_enhanced_span = np.array([
    0.1814432989690722, 0.3124570446735395, 0.5482695139911635,
    0.8184462444771724, 1.0801055473735885, 1.144698085419735,
    1.144698085419735
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
plot_with_shade(topk, maud_span, "MAUD", colors["maud"])
plot_with_shade(topk, maud_enhanced_span, "MAUD Enhanced", colors["maud_enhance"])
plot_with_shade(topk, pq_span, "PQ", colors["pq"])
plot_with_shade(topk, pq_enhanced_span, "PQ Enhanced", colors["pq_enhanced"])

# ---------------------------
# Labels & Styling
# ---------------------------
# Using log scale for x-axis to evenly space 1, 2, 4, 8...
plt.xscale('log', base=2)
plt.xticks(topk, topk)

plt.xlabel("Top-K", fontsize=14)
plt.ylabel("Span Recall@K (%)", fontsize=14)
plt.ylim(0, 120)
plt.legend(fontsize=12, loc='upper left')

plt.title("Span Recall Plot", fontsize=16, pad=20)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/plots/span_recall_maud_vs_pq_plot.png')