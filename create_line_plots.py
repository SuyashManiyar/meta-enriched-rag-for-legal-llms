#!/usr/bin/env python3
"""
Create line plots showing performance across different k values
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def load_evaluation_results():
    """Load evaluation results from all configurations."""
    
    result_files = {
        'Recursive Dense-only': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_only_alternate_results.json',
        'Recursive Dense+Sparse': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_alternate_results.json',
        'Meta-Recursive Dense-only': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_only_alternate_results.json',
        'Meta-Recursive Dense+Sparse': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_alternate_results.json'
    }
    
    results = {}
    
    for config_name, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results[config_name] = data['macro']
        except FileNotFoundError:
            print(f"Warning: {file_path} not found, skipping {config_name}")
            continue
    
    return results

def extract_metrics_across_k(results):
    """Extract metrics across all k values."""
    
    k_values = [1, 2, 4, 8, 16, 32, 64]
    
    metrics_data = {
        'k_values': k_values,
        'configurations': list(results.keys()),
        'doc_retrieval': {},
        'span_recall': {},
        'alt_recall': {},
        'drm': {}
    }
    
    for config_name, metrics in results.items():
        # Extract metrics for all k values
        doc_ret = [metrics.get('doc_retrieved_macro', {}).get(str(k), 0.0) * 100 for k in k_values]
        span_rec = [metrics.get('span_recall_macro', {}).get(str(k), 0.0) * 100 for k in k_values]
        alt_rec = [metrics.get('alt_recall_macro', {}).get(str(k), 0.0) * 100 for k in k_values]
        drm_vals = [metrics.get('drm_macro', {}).get(str(k), 0.0) * 100 for k in k_values]
        
        metrics_data['doc_retrieval'][config_name] = doc_ret
        metrics_data['span_recall'][config_name] = span_rec
        metrics_data['alt_recall'][config_name] = alt_rec
        metrics_data['drm'][config_name] = drm_vals
    
    return metrics_data

def create_line_plots(metrics_data):
    """Create line plots for all metrics across k values."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    k_values = metrics_data['k_values']
    
    # Clean color and style mapping for alternate results only
    style_map = {
        'Recursive Dense-only': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o'},
        'Recursive Dense+Sparse': {'color': '#1f77b4', 'linestyle': '--', 'marker': 's'},
        'Meta-Recursive Dense-only': {'color': '#d62728', 'linestyle': '-', 'marker': '^'},
        'Meta-Recursive Dense+Sparse': {'color': '#d62728', 'linestyle': '--', 'marker': 'v'}
    }
    
    # Plot 1: Span Recall
    for config in metrics_data['configurations']:
        style = style_map.get(config, {'color': 'black', 'linestyle': '-', 'marker': 'o'})
        ax1.plot(k_values, metrics_data['span_recall'][config], 
                label=config, **style, linewidth=2, markersize=6)
    
    ax1.set_title('Span Recall', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Top-K')
    ax1.set_ylabel('Span Recall (%)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(k_values)
    ax1.set_xticklabels(k_values)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Document Retrieval Mismatch (DRM)
    for config in metrics_data['configurations']:
        style = style_map.get(config, {'color': 'black', 'linestyle': '-', 'marker': 'o'})
        ax2.plot(k_values, metrics_data['drm'][config], 
                label=config, **style, linewidth=2, markersize=6)
    
    ax2.set_title('Document Retrieval Mismatch (DRM)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Top-K')
    ax2.set_ylabel('DRM (%)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(k_values)
    ax2.set_xticklabels(k_values)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_line_plots_alternate_only.png', dpi=300, bbox_inches='tight')
    plt.savefig('performance_line_plots_alternate_only.pdf', bbox_inches='tight')
    
    print("Line plots (Span Recall + DRM, Alternate Results) saved as:")
    print("- performance_line_plots_alternate_only.png")
    print("- performance_line_plots_alternate_only.pdf")
    
    plt.show()

def main():
    """Main function to create line plots."""
    
    print("Loading evaluation results...")
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found. Please run the evaluation first.")
        return
    
    print(f"Found results for {len(results)} configurations")
    
    print("Extracting metrics across k values...")
    metrics_data = extract_metrics_across_k(results)
    
    print("Creating line plots...")
    create_line_plots(metrics_data)

if __name__ == "__main__":
    main()