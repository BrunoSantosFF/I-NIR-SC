"""
Run statistical tests on experimental results.

Usage:
    python statistical_tests.py

Author: Matheus Becali Rocha
Email: matheusbecali@gmail.com
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autorank import autorank, create_report
from autorank._util import cd_diagram


def load_data(path, map_rep, metric='balanced_accuracy', backbone_mode=False):
    """
    Load CSV files from the specified path and consolidate them into a single DataFrame.

    Args:
        path: Directory path containing CSV files
        map_rep: Dictionary mapping representation names to their labels
        metric: Column name of the metric to convert (default: 'balanced_accuracy')
        backbone_mode: If True, skip NIR files (default: False)

    Returns:
        DataFrame with consolidated data including representation and attention columns
    """
    frames = []
    # Iterate through all CSV files in the path, sorted alphabetically
    for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
        # Skip NIR files when backbone_mode is True
        if 'nir' in f and backbone_mode:
            pass
        else:
            # Extract the file stem and check if attention mechanism is used
            stem = os.path.basename(f).replace('_backbones.csv', '')
            has_attention = stem.endswith('_att')
            if has_attention:
                stem = stem[:-4]

            # Load CSV file and convert metric to float (handles both "0,858" and "0.858" formats)
            data = pd.read_csv(f)
            data[metric] = pd.to_numeric(
                data[metric]
                .astype(str)
                .str.replace(',', '.', regex=False),
                errors='raise'
            )
            # Add representation and attention columns
            data['representation'] = map_rep[stem]
            data['attention'] = has_attention
            frames.append(data)

    # Concatenate all dataframes
    consolidated_data = pd.concat(frames, ignore_index=True)
    # Create a configuration column combining representation, backbone, and attention
    consolidated_data['config'] = (consolidated_data['representation'] + '-' +
                                   consolidated_data['backbone'] + '-'
                                   + np.where(consolidated_data['attention'], 'ATT', 'noATT'))
    return consolidated_data

def create_pivot_table(long_df, columns, metric='balanced_accuracy', index=('fold',)):
    """
    Create a pivot table from long-format data.

    Args:
        long_df: DataFrame in long format
        columns: Column name(s) to pivot into new columns
        metric: Metric column to aggregate (default: 'balanced_accuracy')
        index: Column(s) to use as index (default: 'fold')

    Returns:
        Pivot table with mean aggregation as float
    """
    # Create pivot table with mean aggregation
    pivot = long_df.pivot_table(index=list(index), columns=columns, values=metric, aggfunc='mean')
    pivot.columns = [str(c) for c in pivot.columns]
    # Validate that there are no missing values (balanced design)
    assert not pivot.isna().any().any(), 'unbalanced design'
    return pivot.astype(float)

def run_modality_analysis(path, metric, map_rep):
    """
    Analyze and compare different modality representations.
    Performs statistical ranking analysis and generates a critical difference diagram.

    Args:
        path: Directory path containing CSV files
        metric: Metric column to use for analysis
        map_rep: Dictionary mapping representation names to their labels
    """
    # Load data for modality analysis (exclude backbone-only modes)
    df_modality = load_data(path, map_rep, metric, backbone_mode=False)

    # Create backbone with attention label
    df_modality['bb_att'] = df_modality.backbone + ' (' + np.where(df_modality.attention, 'ATT', 'noATT') + ")"

    # Create representation with attention label
    df_modality['repre_att'] = np.where(
        df_modality['representation'].eq('Visual'),
        'Visual only',
        df_modality['representation']
        + ' ('
            + np.where(df_modality['attention'], 'ATT', 'noATT')
            + ')'
    )

    # Handle NIR-specific labeling
    df_modality['repre_att'] = np.where(
        df_modality['representation'].eq('NIR'),
        'NIR only',
        df_modality['repre_att']
    )

    # Create pivot table with representation as columns
    pivot_modality = create_pivot_table(
        df_modality,
        columns='repre_att',
        index=('fold',)
    )

    # Perform statistical ranking analysis using autorank (non-parametric test)
    result_modality = autorank(
        pivot_modality,
        alpha=0.05,
        order='descending',
        force_mode='nonparametric',
        verbose=False,
        random_state=42
    )

    # Generate statistical report
    create_report(result_modality)

    # Prepare results for plotting with mean rank annotations
    result_plot = result_modality._replace(rankdf=result_modality.rankdf.rename(
        index={i: f"{i} [{r:.3f}]" for i, r in result_modality.rankdf.meanrank.items()}))

    # Create and save critical difference diagram
    _, ax = plt.subplots(figsize=(10, 2.6))
    cd_diagram(result_plot, False, ax, 8)  # reverse=False: better results on the left
    ax.set_title('Critical difference diagram of average score ranks')
    plt.savefig('./baseline_and_fusion.pdf')

def run_backbone_analysis(path, metric, map_rep):
    """
    Analyze and compare different backbone architectures.
    Performs statistical ranking analysis excluding visual-only representations.

    Args:
        path: Directory path containing CSV files
        metric: Metric column to use for analysis
        map_rep: Dictionary mapping representation names to their labels
    """
    # Load data for backbone analysis (include backbone mode)
    df_backbones = load_data(path, map_rep, metric, backbone_mode=True)

    # Define column order for backbone results
    backbone_columns = ["DenseNet-121 (ATT)", "DenseNet-121 (noATT)", "FocalNet-Tiny (ATT)",
                        "FocalNet-Tiny (noATT)", "RegNetY-32 (ATT)", "RegNetY-32 (noATT)",
                        "ResNet50 (ATT)", "ResNet50 (noATT)"]

    # Create backbone with attention label
    df_backbones['bb_att'] = df_backbones.backbone + ' (' + np.where(df_backbones.attention, 'ATT', 'noATT') + ")"
    # Filter out visual-only representations (keep multimodal and NIR representations)
    df_backbones = df_backbones[df_backbones['representation'] != 'Visual']

    # Create pivot table with backbone architectures as columns
    pivot_backbone = df_backbones.pivot_table(
        index='fold',
        columns='bb_att',
        values=metric,
        aggfunc='mean'
    )
    # Reorder columns according to the predefined order
    pivot_backbone.columns = backbone_columns

    # Perform statistical ranking analysis
    result_backbone = autorank(
        pivot_backbone,
        alpha=0.05,
        order='descending',
        force_mode='nonparametric',
        verbose=False
    )

    # Generate statistical report
    create_report(result_backbone)

    # Prepare results for plotting with mean rank annotations
    result_plot = result_backbone._replace(rankdf=result_backbone.rankdf.rename(
        index={i: f"{i} [{r:.3f}]" for i, r in result_backbone.rankdf.meanrank.items()}))

    # Create and save critical difference diagram
    _, ax = plt.subplots(figsize=(10, 2.6))
    cd_diagram(result_plot, False, ax, 6.5)
    ax.set_title('Critical difference diagram of average score ranks')
    plt.savefig('./backbones.pdf')   

def main():
    """
    Main execution function that runs statistical analysis on experimental results.
    Performs both modality and backbone architecture comparison analyses.
    """
    # Path to directory containing experiment results
    PATH = './results'

    # Metric to use for statistical comparison
    METRIC = 'balanced_accuracy'

    # Mapping between experiment folder names and human-readable representation labels
    REPRESENTATION_MAP = {
        'full_spectra': 'Spectra',
        'VL': 'Latent',
        '50_shap': 'SHAP',
        'retrain_all': 'Visual',
        'nir': 'NIR'
    }

    # Run modality comparison analysis (different input representations)
    run_modality_analysis(PATH, METRIC, REPRESENTATION_MAP)

    # Run backbone architecture comparison analysis
    run_backbone_analysis(PATH, METRIC, REPRESENTATION_MAP)


if __name__ == "__main__":
    main()