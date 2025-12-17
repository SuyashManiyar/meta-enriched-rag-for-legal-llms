#!/usr/bin/env python3
"""
Create clean bar graphs for DRM, Span Recall, and Alt Recall metrics at k=16
Based on Australian Legal QA evaluation results
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def load_evaluation_results():
    """Load evaluation results from all configurations."""
    
    # File paths for all 8 configurations
    result_files = {
        'Recursive Dense-only': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_only_results.json',
        'Recursive Dense+Sparse': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_results.json',
        'Meta-Recursive Dense-only': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_only_results.json',
        'Meta-Recursive Dense+Sparse': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_results.json',
        'Recursive Dense-only (Alt)': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_only_alternate_results.json',
        'Recursive Dense+Sparse (Alt)': 'australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_alternate_results.json',
        'Meta-Recursive Dense-only (Alt)': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_only_alternate_results.json',
        'Meta-Recursive Dense+Sparse (Alt)': 'australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_alternate_results.json'
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

def extract_metrics_at_k16(results):
    """Extract DRM, Span Recall, and Alt Recall metrics at k=16."""
    
    metrics_data = {
        'configurations': [],
        'drm': [],
        'span_recall': [],
        'alt_recall': []
    }
    
    for config_name, metrics in results.items():
        metrics_data['configurations'].append(config_name)
        
        # Extract metrics at k=16
        drm_16 = metrics.get('drm_macro', {}).get('16', 0.0)
        span_recall_16 = metrics.get('span_recall_macro', {}).get('16', 0.0)
        alt_recall_16 = metrics.get('alt_recall_macro', {}).get('16', 0.0)
        
        # Convert to percentages
        metrics_data['drm'].append(drm_16 * 100)
        metrics_data['span_recall'].append(span_recall_16 * 100)
        metrics_data['alt_recall'].append(alt_recall_16 * 100)
    
    return metrics_data

def create_clean_bargraph(metrics_data):
    """Create clean bar graphs for all three metrics."""
    
    # Set up the figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    configurations = metrics_data['configurations']
    x_pos = np.arange(len(configurations))
    
    # Colors for different configuration types
    colors = []
    for config in configurations:
        if 'Meta-Recursive' in config:
            if 'Alt' in config:
                colors.append('#ff7f0e')  # Orange for Meta-Recursive Alt
            else:
                colors.append('#d62728')  # Red for Meta-Recursive
        else:
            if 'Alt' in config:
                colors.append('#2ca02c')  # Green for Recursive Alt
            else:
                colors.append('#1f77b4')  # Blue for Recursive
    
    # Plot 1: DRM (lower is better)
    bars1 = ax1.bar(x_pos, metrics_data['drm'], color=colors, alpha=0.8)
    ax1.set_title('Document Retrieval Mismatch (DRM) at k=16', fontsize=14, fontweight='bold')
    ax1.set_ylabel('DRM (%)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configurations, rotation=45, ha='right', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Span Recall (higher is better)
    bars2 = ax2.bar(x_pos, metrics_data['span_recall'], color=colors, alpha=0.8)
    ax2.set_title('Span Recall at k=16', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Span Recall (%)', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configurations, rotation=45, ha='right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Alt Recall (higher is better)
    bars3 = ax3.bar(x_pos, metrics_data['alt_recall'], color=colors, alpha=0.8)
    ax3.set_title('Alternative Recall at k=16', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Alt Recall (%)', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(configurations, rotation=45, ha='right', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Create legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#1f77b4', alpha=0.8, label='Recursive'),
        plt.Rectangle((0,0),1,1, facecolor='#d62728', alpha=0.8, label='Meta-Recursive'),
        plt.Rectangle((0,0),1,1, facecolor='#2ca02c', alpha=0.8, label='Recursive (Alt)'),
        plt.Rectangle((0,0),1,1, facecolor='#ff7f0e', alpha=0.8, label='Meta-Recursive (Alt)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
               ncol=4, fontsize=12)
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the plot
    plt.savefig('clean_metrics_bargraph.png', dpi=300, bbox_inches='tight')
    plt.savefig('clean_metrics_bargraph.pdf', bbox_inches='tight')
    
    print("Clean bar graphs saved as:")
    print("- clean_metrics_bargraph.png")
    print("- clean_metrics_bargraph.pdf")
    
    plt.show()

def main():
    """Main function to create clean bar graphs."""
    
    print("Loading evaluation results...")
    results = load_evaluation_results()
    
    if not results:
        print("No evaluation results found. Please run the evaluation first.")
        return
    
    print(f"Found results for {len(results)} configurations")
    
    print("Extracting metrics at k=16...")
    metrics_data = extract_metrics_at_k16(results)
    
    print("Creating clean bar graphs...")
    create_clean_bargraph(metrics_data)
    
    # Print summary statistics
    print("\nSummary Statistics at k=16:")
    print("=" * 50)
    for i, config in enumerate(metrics_data['configurations']):
        print(f"{config}:")
        print(f"  DRM: {metrics_data['drm'][i]:.2f}%")
        print(f"  Span Recall: {metrics_data['span_recall'][i]:.2f}%")
        print(f"  Alt Recall: {metrics_data['alt_recall'][i]:.2f}%")
        print()

if __name__ == "__main__":
    main()