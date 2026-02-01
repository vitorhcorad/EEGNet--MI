from bciflow.datasets.cbcic import cbcic

from bciflow.modules.core.kfold import kfold

from bciflow.modules.tf.resample_fft import fft_resample

from bciflow.modules.analysis.metric_functions import accuracy

from bciflow.modules.tf.resample_cubic import cubic_resample

from bciflow.modules.tf.bandpass.convolution import bandpass_conv



import numpy as np

from scipy.interpolate import CubicSpline

from bciflow.modules.tf.bandpass.convolution import bandpass_conv

import pandas as pd

from eegnet_mi import eegnet_mi

def cubic_resample_custom(eegdata, new_sfreq):

    """

    Implementação local de resample cúbico para corrigir bug de tipagem da biblioteca.

    """

    X = eegdata['X']

    old_sfreq = eegdata['sfreq']

    n_samples = X.shape[-1]

   

    # Define o eixo de tempo original

    duration = n_samples / old_sfreq

    old_time = np.linspace(0, duration, n_samples)

   

    # Calcula novo número de amostras (garantindo que seja INT)

    new_n_samples = int(n_samples * new_sfreq / old_sfreq)

    new_time = np.linspace(0, duration, new_n_samples)

   

    # Aplica Cubic Spline no último eixo (tempo)

    cs = CubicSpline(old_time, X, axis=-1)

    X_new = cs(new_time)

   

    # Retorna o dicionário atualizado

    # Importante: copiamos para evitar mutação indesejada, se necessário,

    # mas aqui seguimos o padrão da lib modificando in-place ou retornando ref.

    eegdata['X'] = X_new.astype(np.float32) # Garante float32 para o TensorFlow

    eegdata['sfreq'] = new_sfreq

   

    return eegdata



dataset = cbcic(subject=1, path='C:/Users/Vitor/OneDrive/Documentos/EEGNet/temp_dataset/') #endereço pro meu pc



#cubic resamplknt com 125hz

#bandpass->convolution 4:40

pre_folding = {

    'resample': (cubic_resample_custom, {'new_sfreq': 125}),

    'filter': (bandpass_conv, {'low_cut': 4, 'high_cut': 40})

}

clf = eegnet_mi(n_classes=2, Chans=12, Samples=250)



pos_folding = {

    'clf': (clf, {'epochs': 100, 'batch_size': 16})

}



results = kfold(

    target=dataset,

    start_window=dataset['events']['cue'][0] + 0.5,

    pre_folding=pre_folding,

    pos_folding=pos_folding

)

#refatoração do código

df = pd.DataFrame(results)

acc = accuracy(df)

print(f"Accuracy: {acc:.4f}")