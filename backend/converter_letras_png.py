"""
Converte as fotos estáticas (.png) coletadas em backend/letras/<LETRA>/
para o mesmo formato .npy que o coletar_dados.py gera (sequência de 30
frames x 126 landmarks), e salva direto em backend/data/<LETRA>/.

Por que repetir o frame 30 vezes: quase todo o alfabeto de Libras (exceto
J e Z, que têm movimento) é uma pose estática - a mão não muda de posição
enquanto o sinal é feito. Uma sequência com o mesmo frame repetido 30
vezes representa bem "a mão parada nessa pose" e é o formato que o
train.py já sabe ler, então não precisa mudar a arquitetura do modelo
nem o pipeline de treino - só complementa os dados que já existem.

Depois de rodar este script, train.py (com DATA_DIR = backend/data) vai
enxergar tanto as sequências filmadas quanto as convertidas das fotos.
"""

import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LETRAS_DIR = os.path.join(BASE_DIR, "letras")   # onde estão as fotos .png
DATA_DIR = os.path.join(BASE_DIR, "data")       # onde train.py lê os .npy

FRAMES_POR_SEQUENCIA = 30
EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg")


def extrair_landmarks(resultado):
    """Mesma extração usada no coletar_dados.py/main.py: 63 coords por mão, zeradas se não detectada."""
    esquerda = np.zeros(63)
    direita = np.zeros(63)

    if resultado.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(resultado.multi_hand_landmarks):
            handedness = resultado.multi_handedness[i].classification[0].label
            coords = []
            for ponto in hand_landmarks.landmark:
                coords.extend([ponto.x, ponto.y, ponto.z])
            if handedness == "Left":
                esquerda = np.array(coords)
            else:
                direita = np.array(coords)

    return np.concatenate([esquerda, direita])


def converter_letra(sinal, hands):
    pasta_entrada = os.path.join(LETRAS_DIR, sinal)
    pasta_saida = os.path.join(DATA_DIR, sinal)

    if not os.path.isdir(pasta_entrada):
        return 0, 0

    os.makedirs(pasta_saida, exist_ok=True)

    imagens = sorted(
        f for f in os.listdir(pasta_entrada)
        if f.lower().endswith(EXTENSOES_VALIDAS)
    )

    convertidas = 0
    puladas = 0

    for nome_arquivo in imagens:
        caminho = os.path.join(pasta_entrada, nome_arquivo)
        frame = cv2.imread(caminho)

        if frame is None:
            print(f"  [AVISO] Não consegui abrir {nome_arquivo}, pulando.")
            puladas += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(rgb)

        if not resultado.multi_hand_landmarks:
            print(f"  [AVISO] Nenhuma mão detectada em {nome_arquivo}, pulando.")
            puladas += 1
            continue

        landmarks = extrair_landmarks(resultado)
        sequencia = np.tile(landmarks, (FRAMES_POR_SEQUENCIA, 1))  # repete o frame 30x -> shape (30, 126)

        nome_saida = f"{sinal}_png_{convertidas:04d}.npy"
        np.save(os.path.join(pasta_saida, nome_saida), sequencia)
        convertidas += 1

    return convertidas, puladas


def converter_tudo():
    if not os.path.isdir(LETRAS_DIR):
        print(f"[ERRO] Pasta não encontrada: {LETRAS_DIR}")
        return

    sinais = sorted(
        d for d in os.listdir(LETRAS_DIR)
        if os.path.isdir(os.path.join(LETRAS_DIR, d))
    )

    if not sinais:
        print(f"[AVISO] Nenhuma subpasta de letra encontrada em {LETRAS_DIR}")
        return

    total_convertidas = 0
    total_puladas = 0

    with mp_hands.Hands(
        static_image_mode=True,   # imagem parada, não vídeo - sem tracking entre frames
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:

        for sinal in sinais:
            print(f"\nConvertendo '{sinal}'...")
            convertidas, puladas = converter_letra(sinal, hands)
            print(f"  -> {convertidas} convertidas, {puladas} puladas (sem mão detectada)")
            total_convertidas += convertidas
            total_puladas += puladas

    print(f"\n{'=' * 50}")
    print(f"Total: {total_convertidas} sequências novas em '{DATA_DIR}'")
    if total_puladas:
        print(f"({total_puladas} imagens puladas por não terem mão detectada)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    converter_tudo()
