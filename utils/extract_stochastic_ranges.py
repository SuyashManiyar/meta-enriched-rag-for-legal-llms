#!/usr/bin/env python3
"""
Extract DRM and Span Recall ranges from stochastic evaluation results
"""

import json
import numpy as np

def load_stochastic_results():
    """Load both baseline and enhanced stochastic evaluation results"""
    
    # Load baseline (recursive) results
    with open('australian_legal_data/results_recursive/australian_legal_evaluation_recursive_dense_sparse_stochastic_results.json', 'r') as f:
        baseline_results = json.load(f)
    
    # Load enhanced (meta-recursive) results  
    with open('australian_legal_data/results_meta_recursive/australian_legal_evaluation_meta_recursive_dense_sparse_stochastic_results.json', 'r') as f:
        enhanced_results = json.load(f)
    
    return baseline_results, enhanced_results

def calculate_confidence_intervals(bootstrap_samples, confidence_level=0.95):
    """Calculate confidence intervals from bootstrap samples"""
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_samples, lower_percentile)
    ci_upper = np.percentile(bootstrap_samples, upper_percentile)
    mean_value = np.mean(bootstrap_samples)
    
    return mean_value, ci_lower, ci_upper

def extract_metric_ranges(baseline_results, enhanced_results, metric_name):
    """Extract ranges for a specific metric across all k values"""
    
    k_values = baseline_results['k_values']
    results = {}
    
    print(f"\n{metric_name.upper()} PERFORMANCE RANGES")
    print("=" * 60)
    
    for k in k_values:
        k_str = str(k)
        
        # Get bootstrap samples for both approaches
        baseline_samples = baseline_results['bootstrap_macros'][metric_name][k_str]
        enhanced_samples = enhanced_results['bootstrap_macros'][metric_name][k_str]
        
        # Calculate confidence intervals
        baseline_mean, baseline_ci_lower, baseline_ci_upper = calculate_confidence_intervals(baseline_samples)
        enhanced_mean, enhanced_ci_lower, enhanced_ci_upper = calculate_confidence_intervals(enhanced_samples)
        
        # Calculate improvement
        absolute_improvement = enhanced_mean - baseline_mean
        relative_improvement = (absolute_improvement / baseline_mean) * 100 if baseline_mean != 0 else 0
        
        if metric_name == 'drm':
            # DRM is shown as percentage
            print(f"k={k}:")
            print(f"  Baseline: {baseline_mean*100:.1f}% (95% CI: {baseline_ci_lower*100:.1f}%-{baseline_ci_upper*100:.1f}%)")
            print(f"  Enhanced: {enhanced_mean*100:.1f}% (95% CI: {enhanced_ci_lower*100:.1f}%-{enhanced_ci_upper*100:.1f}%)")
            print(f"  Improvement: {absolute_improvement*100:+.1f} percentage points ({relative_improvement:+.1f}%)")
        else:
            # Span recall shown as decimal
            print(f"k={k}:")
            print(f"  Baseline: {baseline_mean:.3f} (95% CI: {baseline_ci_lower:.3f}-{baseline_ci_upper:.3f})")
            print(f"  Enhanced: {enhanced_mean:.3f} (95% CI: {enhanced_ci_lower:.3f}-{enhanced_ci_upper:.3f})")
            print(f"  Improvement: {absolute_improvement:+.3f} ({relative_improvement:+.1f}%)")
        
        print()
        
        results[k] = {
            'baseline': {'mean': baseline_mean, 'ci_lower': baseline_ci_lower, 'ci_upper': baseline_ci_upper},
            'enhanced': {'mean': enhanced_mean, 'ci_lower': enhanced_ci_lower, 'ci_upper': enhanced_ci_upper},
            'improvement': {'absolute': absolute_improvement, 'relative': relative_improvement}
        }
    
    return results

def generate_latex_table(drm_results, span_recall_results):
    """Generate LaTeX table showing ranges for both metrics"""
    
    latex = """
\\begin{table}[h!]
\\centering
\\caption{Performance Ranges with 95\\% Confidence Intervals}
\\label{tab:performance-ranges}
\\begin{tabular}{l|c|c|c|c|c|c|c}
\\hline
\\textbf{Metric} & \\textbf{k=1} & \\textbf{k=2} & \\textbf{k=4} & \\textbf{k=8} & \\textbf{k=16} & \\textbf{k=32} & \\textbf{k=64} \\\\
\\hline
\\multicolumn{8}{c}{\\textbf{Document Retrieval Mismatch Rate (\\%)}} \\\\
\\hline
Baseline & """
    
    # DRM Baseline row
    drm_baseline_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        mean = drm_results[k]['baseline']['mean'] * 100
        ci_lower = drm_results[k]['baseline']['ci_lower'] * 100
        ci_upper = drm_results[k]['baseline']['ci_upper'] * 100
        drm_baseline_values.append(f"{mean:.1f} ({ci_lower:.1f}-{ci_upper:.1f})")
    
    latex += " & ".join(drm_baseline_values) + " \\\\\n"
    
    # DRM Enhanced row
    latex += "Enhanced & "
    drm_enhanced_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        mean = drm_results[k]['enhanced']['mean'] * 100
        ci_lower = drm_results[k]['enhanced']['ci_lower'] * 100
        ci_upper = drm_results[k]['enhanced']['ci_upper'] * 100
        drm_enhanced_values.append(f"{mean:.1f} ({ci_lower:.1f}-{ci_upper:.1f})")
    
    latex += " & ".join(drm_enhanced_values) + " \\\\\n"
    
    # DRM Improvement row
    latex += "\\textbf{Improvement} & "
    drm_improvement_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        abs_imp = drm_results[k]['improvement']['absolute'] * 100
        drm_improvement_values.append(f"\\textbf{{{abs_imp:+.1f}}}")
    
    latex += " & ".join(drm_improvement_values) + " \\\\\n"
    
    # Span Recall section
    latex += """\\hline
\\multicolumn{8}{c}{\\textbf{Span Recall}} \\\\
\\hline
Baseline & """
    
    # Span Recall Baseline row
    span_baseline_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        mean = span_recall_results[k]['baseline']['mean']
        ci_lower = span_recall_results[k]['baseline']['ci_lower']
        ci_upper = span_recall_results[k]['baseline']['ci_upper']
        span_baseline_values.append(f"{mean:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
    
    latex += " & ".join(span_baseline_values) + " \\\\\n"
    
    # Span Recall Enhanced row
    latex += "Enhanced & "
    span_enhanced_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        mean = span_recall_results[k]['enhanced']['mean']
        ci_lower = span_recall_results[k]['enhanced']['ci_lower']
        ci_upper = span_recall_results[k]['enhanced']['ci_upper']
        span_enhanced_values.append(f"{mean:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
    
    latex += " & ".join(span_enhanced_values) + " \\\\\n"
    
    # Span Recall Improvement row
    latex += "\\textbf{Improvement} & "
    span_improvement_values = []
    for k in [1, 2, 4, 8, 16, 32, 64]:
        abs_imp = span_recall_results[k]['improvement']['absolute']
        span_improvement_values.append(f"\\textbf{{{abs_imp:+.3f}}}")
    
    latex += " & ".join(span_improvement_values) + " \\\\\n"
    
    latex += """\\hline
\\end{tabular}
\\end{table}
"""
    
    return latex

def main():
    """Main function"""
    
    print("Loading stochastic evaluation results...")
    baseline_results, enhanced_results = load_stochastic_results()
    
    # Extract DRM ranges
    drm_results = extract_metric_ranges(baseline_results, enhanced_results, 'drm')
    
    # Extract Span Recall ranges
    span_recall_results = extract_metric_ranges(baseline_results, enhanced_results, 'span_recall')
    
    # Generate LaTeX table
    latex_table = generate_latex_table(drm_results, span_recall_results)
    
    # Save results
    with open('stochastic_ranges_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("\n" + "="*60)
    print("SUMMARY OF KEY IMPROVEMENTS")
    print("="*60)
    
    # Summary statistics
    print("\nDRM Improvements (percentage point reductions):")
    for k in [1, 2, 4, 8, 16, 32, 64]:
        improvement = drm_results[k]['improvement']['absolute'] * 100
        print(f"  k={k}: {improvement:+.1f} pp")
    
    print("\nSpan Recall Improvements:")
    for k in [1, 2, 4, 8, 16, 32, 64]:
        improvement = span_recall_results[k]['improvement']['absolute']
        relative = span_recall_results[k]['improvement']['relative']
        print(f"  k={k}: {improvement:+.3f} ({relative:+.1f}%)")
    
    print(f"\nLaTeX table saved to: stochastic_ranges_table.tex")

if __name__ == "__main__":
    main()