import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data
data = {
    "x": [1, 2, 4, 8, 16, 32, 64],
    "chunks_only": [0.16494845360824742, 0.17783505154639176, 0.18041237113402062, 0.23582474226804123, 0.32989690721649484, 0.602770618556701, 0.7870489690721649],
    "method2": [0.18556701030927836, 0.18298969072164947, 0.18298969072164947, 0.2184278350515464, 0.31862113402061853, 0.598743556701031, 0.7858408505154639],
    "method3": [0.15979381443298968, 0.18814432989690721, 0.19716494845360824, 0.22551546391752578, 0.32023195876288657, 0.5955219072164949, 0.7848743556701031],
    "method4": [0.15979381443298968, 0.17783505154639176, 0.18685567010309279, 0.23840206185567012, 0.33021907216494845, 0.6063144329896907, 0.786243556701031]
}

df = pd.DataFrame(data)

# labels for the methods
labels = {
    "chunks_only": "Chunks Only",
    "method2": "Method 2: Chunk w Window Level (4 Chunk) Metadata",
    "method3": "Method 3: Chunks w Doc Level Metadata",
    "method4": "Method 4: Chunks w Keyword Metadata"
}

plt.figure(figsize=(10, 6))

for key, label in labels.items():
    plt.plot(df['x'], df[key], marker='o', label=label)

plt.xscale('log', base=2)
plt.xticks(df['x'], df['x'])  # Ensure all x points are shown
plt.xlabel('K')
plt.ylabel('DRM Score')
plt.title('Comparison of DRM Scores for RAG-enhanced Pipeline')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig('/home/sunjaekwon_umass_edu/UMASS/deepali/cs685/project/RAG_data/privacy_qa_plots/drm_macro_plot.png')