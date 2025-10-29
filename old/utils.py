"""
Funções auxiliares para processamento de dados EEG de Imaginação Motora
"""

import numpy as np
import mne
from scipy import signal

def carregar_dados_bci(caminho_arquivo):
    """
    Função para carregar dados de competições BCI
    ADAPTE ESTA FUNÇÃO PARA SEUS DADOS ESPECÍFICOS
    """
    print(f"Carregando dados BCI de: {caminho_arquivo}")
    # Implemente aqui a leitura dos seus arquivos de dados
    # Exemplo: .mat, .csv, .edf, etc.
    pass

def filtrar_dados_mi(dados, freq_amostragem=256, freq_baixa=8, freq_alta=30):
    """
    Aplica filtro passa-banda para ritmos sensório-motores (8-30 Hz)
    
    Os ritmos mu (8-12 Hz) e beta (13-30 Hz) são importantes para MI
    """
    print(f"Filtrando dados: {freq_baixa}-{freq_alta} Hz")
    
    # Calcula frequência de Nyquist
    nyquist = freq_amostragem / 2
    low = freq_baixa / nyquist
    high = freq_alta / nyquist
    
    # Cria filtro Butterworth
    b, a = signal.butter(4, [low, high], btype='band')
    
    # Aplica filtro (filtfilt evita deslocamento de fase)
    dados_filtrados = signal.filtfilt(b, a, dados, axis=-1)
    
    return dados_filtrados

def extrair_epocas_mi(dados_crus, eventos, id_eventos, tmin=-1, tmax=4):
    """
    Extrai épocas (trials) de dados contínuos para análise MI
    
    Parâmetros:
    - dados_crus: Dados EEG brutos
    - eventos: Marcadores de eventos
    - id_eventos: IDs dos eventos de interesse
    - tmin, tmax: Janela temporal em relação ao evento (segundos)
    """
    print("Extraindo épocas para análise MI...")
    
    epocas = mne.Epochs(dados_crus, eventos, id_eventos, tmin, tmax, 
                       baseline=(-1, 0),  # Linha de base antes do evento
                       preload=True)
    
    print(f"Épocas extraídas: {len(epocas)}")
    return epocas

def calcular_metricas_mi(y_verdadeiro, y_predito):
    """
    Calcula métricas específicas para avaliação de sistemas MI
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nMétricas de Desempenho MI:")
    print("="*40)
    
    # Relatório de classificação detalhado
    print(classification_report(y_verdadeiro, y_predito))
    
    # Matriz de confusão
    matriz_confusao = confusion_matrix(y_verdadeiro, y_predito)
    print("\nMatriz de Confusão:")
    print(matriz_confusao)
    
    return matriz_confusao