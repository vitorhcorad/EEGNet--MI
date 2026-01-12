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

                 dropoutRate=0.5, kernLength=64, F1=16,

                 D=8, F2=32, norm_rate=0.25):

   

        self.n_classes = n_classes

        self.Chans = Chans

        self.Samples = Samples

        self.dropoutRate = dropoutRate

        self.kernLength = kernLength

        self.F1 = F1

        self.D = D

        self.F2 = F2

        self.norm_rate = norm_rate

        self.model = self.compilar_modelo()

       

    def construir_modelo(self):



        # entradano formato (canais, amostras, 1 canal de profundidade)

        entrada = Input(shape=(1, self.Chans, self.Samples))



        #extração das caracteristicas temporais

        bloco1 = Conv2D(self.F1, (1, self.kernLength),

                       padding='same',

                       use_bias=False)(entrada)

       

       

        bloco1 = BatchNormalization()(bloco1)

       

        #depthwise

        bloco1 = DepthwiseConv2D((1, self.Chans),

                                depth_multiplier=self.D,

                                depthwise_constraint=max_norm(1.),

                                use_bias=False)(bloco1)

       

        bloco1 = BatchNormalization()(bloco1)

        bloco1 = Activation('elu')(bloco1)

       

        #pooling(reduz a dimensão temporal)

        bloco1 = MaxPooling2D((1, 2), padding='same')(bloco1)

       

        #dropout

        bloco1 = Dropout(self.dropoutRate)(bloco1)

       

        #convolucao separavel

        bloco2 = Conv2D(self.F2, (1, 8),

                       padding='same',

                       use_bias=False)(bloco1)

        bloco2 = BatchNormalization()(bloco2)

        bloco2 = MaxPooling2D((1, 4), padding='same')(bloco2)

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

   

    def fit(self, X, y, epochs=50, batch_size=16, validation_split=0.2, verbose=0):
            # Garante o shape (batch, 1, channels, samples)
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            
            # Redimensiona temporalmente se necessário
            if X.shape[3] != self.Samples:
                X = self._redimensionar_dados(X, self.Samples)
            
            # Converte labels para One-Hot (essencial para categorical_crossentropy)
            if len(y.shape) == 1 or y.shape[1] == 1:
                y = tf.keras.utils.to_categorical(y, self.n_classes)
            
            self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose
            )
            return self



    def _redimensionar_dados(self, X, target_samples):

        """Redimensiona os dados para o número de amostras esperado"""

        import tensorflow as tf

       

        # redimensionar apenas a dimensão temporal (amostra)

        X_reshaped = tf.transpose(X, [0, 2, 3, 1])  #(batch, canais, amostras, 1)

       

       

        X_resized = tf.image.resize(

            X_reshaped,

            [self.Chans, target_samples],  # (canais, amostras_alvo)

            method='bilinear'

        )

       

        # rolta pro formato original (batch, 1, canais, amostras)

        X_final = tf.transpose(X_resized, [0, 3, 1, 2])

        return X_final.numpy()



    def predict(self, X):
            """Retorna as classes (0, 1, 2...)"""
            # Garante o shape (Batch, 1, Chans, Samples)
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            
            preds = self.model.predict(X, verbose=0)
            return np.argmax(preds, axis=1)

    def predict_proba(self, X):
        """Retorna as probabilidades (0.85, 0.15...)"""
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        return self.model.predict(X, verbose=0)


# teste da classe

if __name__ == "__main__":

    print("testando EEGNet pro MI")

   

    try:

        # criando modelo

        eegnet = eegnet_mi(

            n_classes=2,    # 4 classes de imaginação motora

            Chans=12,       # 64 canais EEG

            Samples=250     # 512 amostras por trial

        )

       

        modelo = eegnet.compilar_modelo()

       

        # Mostrar resumo

        print("\n resumo:")

        modelo.summary()

       

        # teste aleatório

        print("\nteste com dados ficticios: ")

        X_teste = np.random.randn(5, 1, 12, 250).astype(np.float32)

        y_teste = tf.keras.utils.to_categorical([0, 1, 2, 3, 0], 4)

       

        # predição do teste

        predicoes = modelo.predict(X_teste)

        print(f"✅ Predições realizadas: {predicoes.shape}")


       

    except Exception as e:

        print(f"Erro: {e}")