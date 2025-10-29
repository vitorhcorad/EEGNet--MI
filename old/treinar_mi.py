"""
Script principal para treinar o EEGNet em dados de Imaginação Motora
Execute este arquivo para treinar a rede com seus dados
"""

import numpy as np
import tensorflow as tf
from eegnet_mi import EEGNet_MI, preprocessar_dados_mi, criar_callbacks_mi
import matplotlib.pyplot as plt

class TreinadorMI:
    """
    Classe para gerenciar o treinamento do EEGNet para Imaginação Motora
    """
    
    def __init__(self, n_classes=4, Chans=64, Samples=512):
        """
        Inicializa o treinador
        
        Parâmetros:
        - n_classes: Número de classes MI
        - Chans: Número de canais EEG
        - Samples: Número de amostras por trial
        """
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.modelo = None
        self.historico = None
        
    def carregar_dados_simulados(self):
        """
        Gera dados simulados para teste
        SUBSTITUA ESTA FUNÇÃO PELA LEITURA DOS SEUS DADOS REAIS
        """
        print("Gerando dados simulados para demonstração...")
        n_tentativas = 1000
        
        # Simula dados EEG com ruído e padrões
        # Dados reais de EEG teriam estrutura mais complexa
        X = np.random.randn(n_tentativas, self.Chans, self.Samples)
        
        # Simula labels (classes de imaginação motora)
        y = np.random.randint(0, self.n_classes, n_tentativas)
        y = tf.keras.utils.to_categorical(y, self.n_classes)
        
        print(f"Dados simulados: {X.shape}")
        print(f"Labels: {y.shape}")
        
        return X, y
    
    def treinar(self, X=None, y=None, epocas=300, batch_size=16):
        """
        Treina o modelo EEGNet com dados de Imaginação Motora
        
        Parâmetros:
        - X: Dados de EEG (se None, usa dados simulados)
        - y: Labels (se None, usa dados simulados)
        - epocas: Número máximo de épocas de treinamento
        - batch_size: Tamanho do lote para treinamento
        """
        
        # Carrega dados se não fornecidos
        if X is None or y is None:
            X, y = self.carregar_dados_simulados()
            print("AVISO: Usando dados simulados. Substitua por dados reais de MI!")
        
        # Pré-processamento dos dados
        X_treino, X_validacao, y_treino, y_validacao = preprocessar_dados_mi(X, y)
        
        # Cria o modelo EEGNet
        eegnet_mi = EEGNet_MI(
            n_classes=self.n_classes,
            Chans=self.Chans,
            Samples=self.Samples
        )
        
        self.modelo = eegnet_mi.compilar_modelo(taxa_aprendizado=0.001)
        
        # Mostra resumo da arquitetura
        print("\n" + "="*50)
        print("ARQUITETURA DO EEGNet-MI")
        print("="*50)
        self.modelo.summary()
        
        print("\nIniciando treinamento...")
        print(f"Épocas: {epocas}")
        print(f"Tamanho do lote: {batch_size}")
        print(f"Dados de treino: {X_treino.shape[0]} amostras")
        print(f"Dados de validação: {X_validacao.shape[0]} amostras")
        
        # Treina o modelo
        self.historico = self.modelo.fit(
            X_treino, y_treino,
            batch_size=batch_size,
            epochs=epocas,
            verbose=1,  # Mostra progresso
            validation_data=(X_validacao, y_validacao),
            callbacks=criar_callbacks_mi()  # Callbacks para melhor treino
        )
        
        print("\nTreinamento concluído!")
        return self.historico
    
    def avaliar(self, X_teste, y_teste):
        """
        Avalia o modelo treinado em dados de teste
        
        Parâmetros:
        - X_teste: Dados de teste
        - y_teste: Labels de teste
        """
        if self.modelo is None:
            print("ERRO: Modelo não foi treinado ainda!")
            return None
        
        # Pré-processa dados de teste
        if X_teste.ndim == 3:
            X_teste = X_teste.reshape(X_teste.shape[0], X_teste.shape[1], X_teste.shape[2], 1)
        
        # Avalia o modelo
        loss, acuracia = self.modelo.evaluate(X_teste, y_teste, verbose=0)
        print(f"\nResultados no conjunto de teste:")
        print(f"Loss: {loss:.4f}")
        print(f"Acurácia: {acuracia:.4f} ({acuracia*100:.2f}%)")
        
        return acuracia
    
    def plotar_treinamento(self):
        """
        Gera gráficos do histórico de treinamento
        """
        if self.historico is None:
            print("Nenhum histórico de treinamento disponível!")
            return
        
        # Cria figura com dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Gráfico 1: Acurácia
        ax1.plot(self.historico.history['accuracy'], label='Treino', linewidth=2)
        ax1.plot(self.historico.history['val_accuracy'], label='Validação', linewidth=2)
        ax1.set_title('Acurácia durante o Treinamento', fontsize=14)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Loss
        ax2.plot(self.historico.history['loss'], label='Treino', linewidth=2)
        ax2.plot(self.historico.history['val_loss'], label='Validação', linewidth=2)
        ax2.set_title('Loss durante o Treinamento', fontsize=14)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def salvar_modelo(self, caminho='modelo_eegnet_mi.h5'):
        """
        Salva o modelo treinado para uso futuro
        """
        if self.modelo is None:
            print("Nenhum modelo para salvar!")
            return
        
        self.modelo.save(caminho)
        print(f"Modelo salvo em: {caminho}")


def main():
    """
    Função principal - execute esta função para treinar o modelo
    """
    print("EEGNet para Imaginação Motora - Sistema de Treinamento")
    print("="*60)
    
    # Configurações para dados típicos de MI
    treinador = TreinadorMI(
        n_classes=4,      # Exemplo: mão esquerda, direita, pés, repouso
        Chans=64,         # Número de canais EEG (ajuste conforme seus dados)
        Samples=512       # Amostras por trial (depende da taxa de amostragem)
    )
    
    # Treina o modelo
    print("\nIniciando treinamento do EEGNet...")
    historico = treinador.treinar(epocas=100, batch_size=16)
    
    # Mostra resultados
    print("\nAnalisando resultados do treinamento...")
    
    # Plota gráficos de evolução
    treinador.plotar_treinamento()
    
    # Salva o modelo treinado
    treinador.salvar_modelo('eegnet_mi_treinado.h5')
    
    # Mensagem final
    print("\n" + "="*60)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("Próximos passos:")
    print("1. Substitua os dados simulados por seus dados reais de EEG")
    print("2. Ajuste os parâmetros conforme sua configuração experimental")
    print("3. Use o modelo salvo para fazer previsões em novos dados")
    print("="*60)


if __name__ == "__main__":
    # Executa o treinamento quando o arquivo é rodado diretamente
    main()