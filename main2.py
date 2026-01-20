from bciflow.datasets.cbcic import cbcic
from bciflow.datasets.bciciv2b import bciciv2b
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.convolution import bandpass_conv
from bciflow.modules.analysis.metric_functions import accuracy
from eegnet_mi import eegnet_mi
import pandas as pd
import numpy as np # Import necessário para a função customizada
# Reaproveitando sua função customizada para evitar o erro de float do bciflow
from scipy.interpolate import CubicSpline

def z_score_standardization(eegdata):
    """
    Normaliza os dados para que tenham média 0 e desvio padrão 1 por canal.
    """
    X = eegdata['X']
    # Calcula média e desvio ao longo do eixo do tempo (último eixo)
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    
    # Aplica a fórmula: (x - μ) / σ
    eegdata['X'] = (X - mean) / (std + 1e-8) # 1e-8 evita divisão por zero
    return eegdata

def cubic_resample_custom(eegdata, new_sfreq):
    X = eegdata['X']
    old_sfreq = eegdata['sfreq']
    n_samples = X.shape[-1]
    duration = n_samples / old_sfreq
    old_time = np.linspace(0, duration, n_samples)
    new_n_samples = int(n_samples * new_sfreq / old_sfreq)
    new_time = np.linspace(0, duration, new_n_samples)
    cs = CubicSpline(old_time, X, axis=-1)
    eegdata['X'] = cs(new_time).astype(np.float32)
    eegdata['sfreq'] = new_sfreq
    return eegdata
for i in range(1,10):
    print("Sujeito ", i, ":\n\n")
    dataset = bciciv2b(subject=i, path='C:/Users/Vitor/OneDrive/Documentos/EEGNet/temp_dataset/bciciv2b', labels=['left-hand', 'right-hand'])


    rs = (cubic_resample_custom, {'new_sfreq': 128}) 
    tf = (bandpass_conv, {'low_cut': 4, 'high_cut': 40})

    # Se window_size=2.0 e sfreq=128, Samples deve ser 256
    modelo_eeg = eegnet_mi(
        n_classes=2, 
        Chans=3, 
        Samples=256, 
        kernLength=64,   # Metade da sfreq (128/2)
        F1=8,           # Mais filtros para capturar padrões temporais
        D=2,             # tentando pegar mais eletrodos
        dropoutRate=0.5  # Mantém dropout alto para evitar overfitting
    )
    clf = (modelo_eeg, {'epochs': 100, 'batch_size': 16})

    pre_folding = { 
        'rs': (cubic_resample_custom, {'new_sfreq': 128}),
        'tf': (bandpass_conv, {'low_cut': 4, 'high_cut': 40}),
        'norm': (z_score_standardization, {}) 
    }
    pos_folding = { 'clf': clf }

    results = kfold(
        target=dataset,
        start_window=dataset['events']['cue'][0] + 0.5,
        pre_folding=pre_folding,
        pos_folding=pos_folding,
        window_size=2.0
    )

    df = pd.DataFrame(results)
    print(df)
    acc = accuracy(df)
    print(f"Accuracy: {acc:.4f}")
    print("\n\n\n")