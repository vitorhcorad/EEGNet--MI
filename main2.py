from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.resample_cubic import cubic_resample
from bciflow.modules.tf.bandpass.convolution import bandpass_conv
from bciflow.modules.analysis.metric_functions import accuracy

from eegnet_mi import eegnet_mi

import pandas as pd

dataset = cbcic(subject=1, path='../../data/cbcic/')

rs = (cubic_resample, {'new_sfreq': 128})
tf = (bandpass_conv, {})
clf = (eegnet_mi(), {})

pre_folding = { 'rs': rs,
                'tf': tf 
              }
pos_folding = { 'clf': clf 
              }

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