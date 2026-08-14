import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autorank import autorank, create_report, latex_table, plot_stats
from autorank._util import cd_diagram

def carregar(path=PATH):
    frames = []
    for f in sorted(glob.glob(os.path.join(path, '*.csv'))):
        stem = os.path.basename(f).replace('_backbones.csv', '')
        att = stem.endswith('_att')
        if att:
            stem = stem[:-4]
        d = pd.read_csv(f)
        # Converte tanto "0,858" quanto "0.858" para float
        d[METRIC] = pd.to_numeric(
            d[METRIC]
            .astype(str)
            .str.replace(',', '.', regex=False),
            errors='raise'
        )
        d['representacao'] = MAP_REP[stem]
        d['attention'] = att
        frames.append(d)
    L = pd.concat(frames, ignore_index=True)
    L['config'] = (L['representacao'] + '-' + L['backbone'] + '-'
                   + np.where(L['attention'], 'ATT', 'noATT'))
    return L

def blocos(long_df, pop, index=('fold',), metric=METRIC):
    w = long_df.pivot_table(index=list(index), columns=pop, values=metric, aggfunc='mean')
    w.columns = [str(c) for c in w.columns]
    assert not w.isna().any().any(), 'desenho desbalanceado'
    return w.astype(float)



def main():
    PATH = './results'
    METRIC = 'balanced_accuracy'
    MAP_REP = {'full_spectra': 'Spectra', 'VL': 'Latent', '50_shap': 'SHAP', 'retrain_all': 'Visual', 'nir': 'NIR'}

    L = carregar()
    print(L.shape, L.config.nunique(), sorted(L.fold.unique()))   # (120, 11) 24 [1,2,3,4,5]

    L['bb_att'] = L.backbone + ' (' + np.where(L.attention, 'ATT', 'noATT') + ")"
    # L['repre_att'] = L.representacao + ' (' + np.where(L.attention, 'ATT', 'noATT') + ")"

    L['repre_att'] = np.where(
        L['representacao'].eq('Visual'),
        'Visual only',
        L['representacao']
        + ' ('
        + np.where(L['attention'], 'ATT', 'noATT')
        + ')'
    )

    L['repre_att'] = np.where(
        L['representacao'].eq('NIR'),
        'NIR only',
        L['repre_att']
    )
    
    w_com_visual = blocos(
    L,
    pop='repre_att',
    index=('fold',)
    )   

    print(w_com_visual.shape)  # (5, 7)

    res_visual = autorank(
        w_com_visual,
        alpha=0.05,
        order='descending',
        force_mode='nonparametric',
        verbose=False,
        random_state=42
    )

    create_report(res_visual)

if __name__ == "__main__":
    main()