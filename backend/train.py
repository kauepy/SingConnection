"""
Treina o modelo LSTM de reconhecimento de Libras a partir dos dados
coletados pelo coletar_dados.py, e/ou convertidos pelo converter_letras_png.py
a partir das fotos em backend/letras/ (sequências de 30 frames x 126
landmarks salvas em backend/data/<SINAL>/*.npy).

Salva o modelo treinado e o LabelEncoder (que mapeia índice -> nome do
sinal) em backend/model/, prontos para o main.py carregar.
"""

import numpy as np
import os
import pickle
import random
import shutil
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminhos relativos à posição deste arquivo, não ao diretório de onde o
# script é executado. Antes eram "backend/data" e "backend/model"
# (caminhos relativos ao cwd), o que criava uma pasta "backend/backend/..."
# duplicada quando o script era rodado de dentro da própria pasta backend/.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")   # sequências filmadas + convertidas de letras/ (converter_letras_png.py)
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "libras_model.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Checkpoint da rodada atual (protege contra queda no meio de UMA rodada)
# e registro da melhor acurácia já alcançada entre TODAS as rodadas do
# loop infinito - é o que decide se o MODEL_PATH final deve ser
# substituído ou não ao fim de cada rodada.
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_rodada_atual.keras")
MELHOR_ACURACIA_PATH = os.path.join(MODEL_DIR, "melhor_acuracia.txt")

FRAMES_POR_SEQUENCIA = 30
FEATURES = 126  # 63 mão esquerda + 63 mão direita

# Quantidade de sequências de TREINO que queremos ter depois do data
# augmentation. Hoje a coleta faz ciclos de 100 sequências reais por
# sinal (26 letras = ~2600 no total) - augmentation multiplica isso
# por ~7-8x até chegar perto de 20000 sequências de treino.
TOTAL_TREINO_DESEJADO = 20000

def carregar_melhor_acuracia():
    """Lê a melhor acurácia já alcançada em rodadas anteriores (ou -1 se nunca treinou)."""
    if os.path.exists(MELHOR_ACURACIA_PATH):
        with open(MELHOR_ACURACIA_PATH) as f:
            return float(f.read().strip())
    return -1.0


def salvar_melhor_acuracia(acc):
    with open(MELHOR_ACURACIA_PATH, "w") as f:
        f.write(str(acc))


def carregar_dados():
    """Lê todos os .npy de DATA_DIR e monta X (sequências) e y (rótulos = nome da pasta)."""
    X, y = [], []

    if not os.path.isdir(DATA_DIR):
        print(f"[ERRO] Pasta de dados não encontrada: {DATA_DIR}")
        return np.array(X), np.array(y)

    for sinal in sorted(os.listdir(DATA_DIR)):
        pasta = os.path.join(DATA_DIR, sinal)
        if not os.path.isdir(pasta):
            continue

        arquivos = [f for f in os.listdir(pasta) if f.endswith(".npy")]
        print(f"  {sinal}: {len(arquivos)} sequências")

        for arquivo in arquivos:
            sequencia = np.load(os.path.join(pasta, arquivo))
            X.append(sequencia)
            y.append(sinal)

    return np.array(X), np.array(y)


# ===================== Data augmentation =====================
# Cada sequência é (30 frames, 126) = 63 coords da mão esquerda + 63 da
# direita por frame. Um bloco de 63 todo zerado significa "mão não
# detectada nesse frame" - as funções abaixo preservam esses zeros em
# vez de adicionar ruído/deslocamento neles, senão estaríamos ensinando
# o modelo a tratar "sem mão" como "mão tremendo bem devagarinho".

def _mascara_maos(seq):
    """Retorna, por frame, se a mão esquerda/direita foi detectada."""
    mask_esq = ~np.all(seq[:, :63] == 0, axis=1)
    mask_dir = ~np.all(seq[:, 63:] == 0, axis=1)
    return mask_esq, mask_dir


def adicionar_ruido(seq, escala=0.01):
    """Ruído gaussiano leve nas coordenadas - simula tremor/imprecisão de captura."""
    seq = seq.copy()
    mask_esq, mask_dir = _mascara_maos(seq)
    seq[mask_esq, :63] += np.random.normal(0, escala, seq[mask_esq, :63].shape)
    seq[mask_dir, 63:] += np.random.normal(0, escala, seq[mask_dir, 63:].shape)
    return seq


def escalar_transladar(seq, escala_range=(0.9, 1.1), deslocamento_max=0.03):
    """Simula a mão um pouco mais perto/longe da câmera ou deslocada no quadro."""
    seq = seq.copy()
    fator = np.random.uniform(*escala_range)
    dx = np.random.uniform(-deslocamento_max, deslocamento_max)
    dy = np.random.uniform(-deslocamento_max, deslocamento_max)
    mask_esq, mask_dir = _mascara_maos(seq)

    for mask, offset in ((mask_esq, 0), (mask_dir, 63)):
        bloco = seq[mask, offset:offset + 63].reshape(-1, 21, 3)
        bloco *= fator
        bloco[:, :, 0] += dx
        bloco[:, :, 1] += dy
        seq[mask, offset:offset + 63] = bloco.reshape(-1, 63)

    return seq


def variar_velocidade(seq, fator_range=(0.85, 1.15)):
    """
    Reamostra a sequência simulando sinalização mais rápida ou mais lenta,
    depois interpola de volta pra sempre manter FRAMES_POR_SEQUENCIA frames
    (o modelo exige shape fixo).
    """
    n = seq.shape[0]
    fator = np.random.uniform(*fator_range)
    n_intermediario = max(2, int(round(n * fator)))

    eixo_original = np.linspace(0, 1, n)
    eixo_intermediario = np.linspace(0, 1, n_intermediario)
    intermediario = np.array([
        np.interp(eixo_intermediario, eixo_original, seq[:, i])
        for i in range(seq.shape[1])
    ]).T

    eixo_final = np.linspace(0, 1, n)
    eixo_atual = np.linspace(0, 1, n_intermediario)
    final = np.array([
        np.interp(eixo_final, eixo_atual, intermediario[:, i])
        for i in range(intermediario.shape[1])
    ]).T

    return final


def espelhar_maos(seq):
    """
    Espelha horizontalmente (x -> 1 - x) e troca esquerda<->direita -
    simula a mesma configuração de mão feita por um sinalizante canhoto.
    """
    seq = seq.copy()
    esquerda_orig = seq[:, :63].reshape(-1, 21, 3).copy()
    direita_orig = seq[:, 63:].reshape(-1, 21, 3).copy()

    esquerda_orig[:, :, 0] = 1.0 - esquerda_orig[:, :, 0]
    direita_orig[:, :, 0] = 1.0 - direita_orig[:, :, 0]

    seq[:, :63] = direita_orig.reshape(-1, 63)
    seq[:, 63:] = esquerda_orig.reshape(-1, 63)
    return seq


TECNICAS_AUGMENTATION = [adicionar_ruido, escalar_transladar, variar_velocidade, espelhar_maos]


def aumentar_dataset(X, y, total_desejado):
    """
    Gera variações sintéticas das sequências originais até bater
    total_desejado no total (originais + geradas). Cada sequência nova
    aplica 1 ou 2 técnicas escolhidas aleatoriamente sobre uma sequência
    original sorteada, mantendo o rótulo dela.
    """
    n_original = len(X)
    faltam = total_desejado - n_original

    if faltam <= 0:
        print(f"Já há {n_original} sequências, sem necessidade de augmentation.")
        return X, y

    print(f"\nGerando {faltam} sequências sintéticas via augmentation "
        f"({n_original} reais -> {total_desejado} no total)...")

    X_geradas = []
    y_geradas = []

    for _ in range(faltam):
        idx = random.randrange(n_original)
        seq = X[idx]
        rotulo = y[idx]

        n_tecnicas = random.choice([1, 1, 2])  # maioria com só 1 técnica, algumas com 2 combinadas
        for tecnica in random.sample(TECNICAS_AUGMENTATION, n_tecnicas):
            seq = tecnica(seq)

        X_geradas.append(seq)
        y_geradas.append(rotulo)

    X_final = np.concatenate([X, np.array(X_geradas)], axis=0)
    y_final = np.concatenate([y, np.array(y_geradas)], axis=0)
    return X_final, y_final

def treinar():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Carregando dados...")
    X, y = carregar_dados()

    if len(X) == 0:
        print(f"Nenhum dado encontrado em '{DATA_DIR}'! Execute coletar_dados.py primeiro.")
        return

    print(f"\nTotal de sequências: {len(X)}")
    print(f"Shape dos dados: {X.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    print(f"Classes: {list(le.classes_)}")
    print(f"Total de classes: {n_classes}")

    # Salva o encoder já aqui, antes do treino começar. Ele não muda mais
    # depois disso (é definido pelas classes existentes nos dados), então
    # já fica disponível mesmo se o treino cair no meio da madrugada.
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=0.2,
        random_state=42,
        stratify=y_enc
    )

    print(f"\nTreino (reais): {len(X_train)} sequências")
    print(f"Teste (reais, sem augmentation): {len(X_test)} sequências")

    # Augmentation só no treino - o teste fica só com dados reais, senão
    # a precisão final reportada fica inflada por variações da mesma
    # sequência original aparecendo tanto no treino quanto no teste.
    X_train, y_train = aumentar_dataset(X_train, y_train, TOTAL_TREINO_DESEJADO)
    print(f"Treino (após augmentation): {len(X_train)} sequências")

    # LSTM de 2 camadas: a primeira devolve a sequência inteira pra segunda
    # processar, a segunda resume tudo num vetor só. Dropout entre as camadas
    # reduz overfitting - importante aqui já que há poucas amostras por classe.
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(FRAMES_POR_SEQUENCIA, FEATURES)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("\nTreinando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=20000,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(
                patience=15,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=CHECKPOINT_PATH,
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecisão final desta rodada: {acc*100:.1f}%")

    melhor_acc_global = carregar_melhor_acuracia()

    if acc > melhor_acc_global:
        # Essa rodada superou o melhor resultado já visto - promove o
        # checkpoint dela a modelo oficial (o que o main.py carrega).
        shutil.copyfile(CHECKPOINT_PATH, MODEL_PATH)
        salvar_melhor_acuracia(acc)
        print(f"Novo recorde! ({melhor_acc_global*100:.1f}% -> {acc*100:.1f}%) "
            f"Modelo atualizado em: {MODEL_PATH}")
    else:
        print(f"Não superou o melhor resultado registrado ({melhor_acc_global*100:.1f}%) "
            f"- mantendo o modelo anterior em {MODEL_PATH}")

    print(f"Encoder salvo em: {ENCODER_PATH}")
    return acc


if __name__ == "__main__":
    rodada = 1
    try:
        while True:
            print(f"\n{'=' * 60}")
            print(f"RODADA {rodada} (melhor acurácia registrada: "
                  f"{max(carregar_melhor_acuracia(), 0) * 100:.1f}%)")
            print(f"{'=' * 60}")

            try:
                treinar()
            except Exception as e:
                # Uma rodada falhando (ex: erro pontual de memória/GPU) não
                # deve derrubar o loop inteiro - loga e parte pra próxima.
                print(f"Rodada {rodada} falhou com erro: {e}")

            rodada += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário (Ctrl+C). Encerrando.")