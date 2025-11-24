from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.resample_fft import fft_resample
from bciflow.modules.analysis.metric_functions import accuracy

import pandas as pd
from eegnet_mi import eegnet_mi

dataset = cbcic(subject=1, path='C:/Users/Vitor/OneDrive/Documentos/EEGNet/temp_dataset/') #endereço pro meu pc

pre_folding = {}
clf = eegnet_mi(n_classes=2, Chans=12, Samples=250)

pos_folding = {
    'clf': (clf, {})
}

results = kfold(
    target=dataset,
    start_window=dataset['events']['cue'][0] + 0.5,
    pre_folding=pre_folding,
    pos_folding=pos_folding
)

df = pd.DataFrame(results)
acc = accuracy(df)
print(f"Accuracy: {acc:.4f}")