

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Activation, Dropout,
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, DepthwiseConv2D
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import numpy as np


class eegnet_mi:
    """
    Classe principal do EEGNet para classificação de Imaginação Motora
    """
    
    def __init__(self, n_classes=4, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25):
    
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate
        
    def construir_modelo(self):

        # entradano formato (canais, amostras, 1 canal de profundidade)
        entrada = Input(shape=(self.Chans, self.Samples, 1))
        
        #extração das caracteristicas temporais
        bloco1 = Conv2D(self.F1, (1, self.kernLength), 
                       padding='same',
                       use_bias=False)(entrada)
        
        
        bloco1 = BatchNormalization()(bloco1)
        
        #depthwise
        bloco1 = DepthwiseConv2D((self.Chans, 1), 
                                depth_multiplier=self.D,
                                depthwise_constraint=max_norm(1.),
                                use_bias=False)(bloco1)
        
        bloco1 = BatchNormalization()(bloco1)
        bloco1 = Activation('elu')(bloco1)
        
        #pooling(reduz a dimensão temporal)
        bloco1 = MaxPooling2D((1, 4))(bloco1)
        
        #dropout
        bloco1 = Dropout(self.dropoutRate)(bloco1)
        
        #convolucao separavel
        bloco2 = Conv2D(self.F2, (1, 16),
                       padding='same',
                       use_bias=False)(bloco1)
        bloco2 = BatchNormalization()(bloco2)
        bloco2 = Activation('elu')(bloco2)
        bloco2 = MaxPooling2D((1, 8))(bloco2)
        bloco2 = Dropout(self.dropoutRate)(bloco2)
        

        # achata a saída para camadas densas
        achatado = Flatten(name='flatten')(bloco2)
        
        
        densa = Dense(self.n_classes, name='dense', 
                     kernel_constraint=max_norm(self.norm_rate))(achatado)
        
        #softmax para probabilidades das classes
        softmax = Activation('softmax', name='softmax')(densa)
        
        
        modelo = Model(inputs=entrada, outputs=softmax)
        
        return modelo
    
    def compilar_modelo(self, taxa_aprendizado=0.001):
        """
        Compila o modelo com parâmetros otimizados para dados de EEG
        """
        modelo = self.construir_modelo()
        
        #cCompila com crossentropy categórica (para múltiplas classes)
        modelo.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado),
            metrics=['accuracy']
        )
        
        return modelo

# teste da classe
if __name__ == "__main__":
    print("testando EEGNet pro MI")
    
    try:
        # criando modelo
        eegnet = eegnet_mi(
            n_classes=4,    # 4 classes de imaginação motora
            Chans=64,       # 64 canais EEG
            Samples=512     # 512 amostras por trial
        )
        
        modelo = eegnet.compilar_modelo()
        
        # Mostrar resumo
        print("\n resumo:")
        modelo.summary()
        
        # teste aleatório
        print("\nteste com dados ficticios: ")
        X_teste = np.random.randn(5, 64, 512, 1).astype(np.float32)
        y_teste = tf.keras.utils.to_categorical([0, 1, 2, 3, 0], 4)
        
        # predição do teste
        predicoes = modelo.predict(X_teste)
        print(f"✅ Predições realizadas: {predicoes.shape}")
        print("🎉 EEGNet funcionando perfeitamente!")
        
    except Exception as e:
        print(f"Erro: {e}")