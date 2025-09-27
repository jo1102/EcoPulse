import sys
import os
sys.path.append(r'c:\Users\jonhe\Downloads\Ecopulse\predictive-maintenance')

import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gen_summary(meter_id="A", outdir=None):
    """
    Generate a summary visualization for UFD data.
    This function mimics the gen_summary() functionality found in other dataset modules.
    
    Args:
        meter_id (str): Meter ID to load ("A", "B", "C", or "D")
        outdir (str): Output directory to save the summary PDF
    
    Returns:
        tuple: (data, figure) - The loaded data and the generated figure
    """
    
    # Load the data
    data = datasets.ufd.load_data(meter_id=meter_id)
    
    # Basic statistics
    print(f"=== UFD Meter {meter_id} Summary ===")
    print(f"Data shape: {data.shape}")
    print(f"Number of features: {data.shape[1] - 1}")  # excluding health_state
    print(f"\nHealth state distribution:")
    health_counts = data['health_state'].value_counts()
    for state, count in health_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  State {state}: {count} ({percentage:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'UFD Meter {meter_id} Data Summary', fontsize=16)
    
    # 1. Health state distribution
    health_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Health State Distribution')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_xlabel('Health State')
    
    # 2. Feature correlation heatmap (sample of features)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != 'health_state']
    
    # Select a subset of features for correlation matrix (to keep it readable)
    sample_features = feature_cols[:min(10, len(feature_cols))]
    if sample_features:
        corr_matrix = data[sample_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,1], 
                    fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('Feature Correlation Matrix (Sample)')
    
    # 3. Box plot for health state vs first feature
    if feature_cols:
        first_feature = feature_cols[0]
        data.boxplot(column=first_feature, by='health_state', ax=axes[1,0])
        axes[1,0].set_title(f'{first_feature} by Health State')
        axes[1,0].set_xlabel('Health State')
    
    # 4. Feature distribution histogram
    if feature_cols:
        first_feature = feature_cols[0]
        for state in sorted(data['health_state'].unique()):
            subset = data[data['health_state'] == state][first_feature]
            axes[1,1].hist(subset, alpha=0.7, label=f'State {state}', bins=20)
        axes[1,1].set_title(f'{first_feature} Distribution by Health State')
        axes[1,1].set_xlabel(first_feature)
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
    
    plt.tight_layout()
    
    # Save if output directory specified
    if outdir:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        output_file = os.path.join(outdir, f'ufd_meter_{meter_id}_summary.pdf')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        print(f"\nSummary visualization saved to: {output_file}")
    else:
        print("\nShowing visualization (close the plot window to continue)")
    
    plt.show()
    
    return data, fig

# Usage example (matching the format you mentioned):
if __name__ == "__main__":
    # Dataset-specific values will be returned
    data = datasets.ufd.load_data()
    print("Data loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # A visualization pdf will be generated
    data, fig = gen_summary(meter_id="A", outdir="output")
    
    # You can also test other meters:
    # data_b, fig_b = gen_summary(meter_id="B", outdir="output")
    # data_c, fig_c = gen_summary(meter_id="C", outdir="output")
    # data_d, fig_d = gen_summary(meter_id="D", outdir="output")
