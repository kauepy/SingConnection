"""
Script de coleta de dados para treinar o modelo de reconhecimento de Libras.

Pra cada sinal da lista SINAIS, grava sequências de 30 frames pela webcam,
extrai os landmarks das mãos com MediaPipe e salva cada sequência como um
arquivo .npy em backend/data/<SINAL>/. Esses arquivos são o dataset que o
train.py usa depois pra treinar a LSTM.
"""

import cv2
import os
import time
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

SINAIS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",

    # "OI", "OBRIGADO", "SIM", "NAO", "EU", "VOCE", "BOM",
    # "POR_FAVOR", "NOITE", "BOA_TARDE", "OLA", "MEU_NOME_É",
    # "AJUDA", "EU_TE_AMO", "COM_LICENCA", "DESCULPA", "PARE",
    # "DOR", "DOR_INTENSA", "COMER", "BEBER", "MAIS", "BANHEIRO",
    # "COMO", "BOM_DIA", "BOA_NOITE",
    # "TUDO_BEM", "TCHAU", 
    
    # "HOJE", "AMANHA","ONTEM", "ANTEONTEM", "JA", "PASSADO", "FUTURO", "CALENDARIO",
    # "SEMANA", "SEGUNDA_FEIRA", "TERCA_FEIRA", "QUARTA_FEIRA",
    # "QUINTA_FEIRA", "SEXTA_FEIRA", "SABADO", "DOMINGO",
    #"IDADE"

    # "MES", "JANEIRO", "FEVEREIRO", "MARCO", "ABRIL", "MAIO",
    # "JUNHO", "JULHO", "AGOSTO", "SETEMBRO", "OUTUBRO",
    # "NOVEMBRO", "DEZEMBRO", "ANO",

    # "FAMILIA", "PAPAI", "MAMAE", "FILHO", "FILHA", "TIO", "TIA",
    # "IRMAO", "IRMAO_(1)", "IRMAO_(2)", "AVO_(1)", "AVO_(2)",
    # "SOLTEIRA", "CASADO(A)", "SEPARADO(A)", "VIUVO(A)",
    # "DIVORCIADO(A)", "NAMORADO(A)", "NOIVO(A)",
    # "PESSOA_(1)", "PESSOA_(2)", "HOMEM", "MULHER", "VELHO",
    # "IDOSO", "ADULTO", "JOVEM", "CRIANCA", "MENINO", "MENINA",
    # "MAE_ADOTIVA", "PAI_ADOTIVO", "FILHO_ADOTIVO", "MADRASTA",
    # "PADRASTO", "ENTEADO", "NETO(A)", "PRIMO(A)", "TITIO(A)",
    # "SOBRINHO(A)", "GENRO", "CUNHADO(A)", "SOGRO", "SOGRA",
    # "VIZINHO(A)", "EMPREGADO(A)", "SOLTEIRO(A)_(1)",
    # "SOLTEIRO(A)_(2)",

    # "ESCOLA", "MATERIAIS", "SALA_DE_AULA", "PROFESSOR", "ALUNO",
    # "COMPUTADOR", "LAPIS", "CANETA", "PAPEL", "CADERNO", "LIVRO",
    # "CURSO", "ADMINISTRACAO_PUBLICA", "BIBLIOTECONOMIA",
    # "DOCUMENTACAO", "CIENCIA_BIOLOGIA", "FILOSOFIA", "FISICA",
    # "HISTORIA", "GEOGRAFIA", "LETRAS_ESPANHOL", "LETRAS_INGLES",
    # "LETRAS_PORTUGUES", "MATEMATICA", "QUIMICA",

    # "TUDO_BOM", "QUAL_SEU_NOME", "MEU_NOME",
    # "PRAZER_EM_CONHECER", "SURDO", "OUVINTE", "FALAR",
    # "SINALIZAR", "REPETIR", "COMUNICAR", "ME_AJUDA", "AJUDAR",
    # "ENSINAR", "APRENDER", "RAPIDO", "CALMA", "DEVAGAR",
    # "ENTENDER", "NAO_ENTENDI", "QUER", "NAO_QUER",
    # "CONHECER", "NAO_CONHECER", "SABER", "NAO_SABER",
    # "PRESENTE", "MANHA", "TARDE", "HORA_DURACAO",
    # "HORA_RELOGIO", "TODO_DIA", "DIA_TODO", "SEMANA_(1)",
    # "SEMANA_(2)", "ANO_FUTURO", "ANO_PASSADO",
    # "DIAS_DA_SEMANA", "FERIADO", "FERIAS",
    # "MESES_DO_ANO",

    # "VERBOS", "ACONTECER", "AMAR", "APRESENTAR",
    # "BEIJAR_(1)", "BEIJAR_(2)", "BEIJAR_(3)",
    # "BRINCAR", "CAMINHAR", "CASAR", "CHEGAR",
    # "COMBINAR", "COMPRAR", "CONSEGUIR", "CONVERSAR",
    # "CORRER", "DESCOBRIR", "EMPRESTAR", "ENCONTRAR",
    # "ESCOLHER", "FAZER", "FORMAR", "GOSTAR",
    # "MENTIR", "NAMORAR", "OBEDECER", "PAGAR",
    # "PASSEAR", "PARAR", "PERGUNTAR", "PODER",
    # "PROCURAR", "RESPONDER", "REUNIR", "TER",
    # "TRABALHAR", "VER", "VIAJAR", "VIVER", "VOLTAR",

    # "CORES", "AMARELO", "AZUL", "ROXO", "VERDE",
    # "VERMELHO", "ROSA_(1)", "ROSA_(2)", "BEGE",
    # "BRANCO", "CINZA", "DOURADO", "LILAS",
    # "MARROM", "MORENO", "NEGRO", "PRADO",
    # "PRETO_(1)", "PRETO_(2)",

    # "ASUSTADO(A)", "CANSADO(A)_(1)", "CANSADO(A)_(2)",
    # "CIUMENTO(A)", "CORAJOSO(A)", "ESPERTO(A)",
    # "FRACO(A)", "INTELIGENTE", "INVEJOSO(A)",
    # "LEGAL", "MEDROSO(A)", "NERVOSO(A)",
    # "RAIVOSO(A)", "TIMIDO(A)",

    # "CIDADE", "BRASILIA", "NORTE", "NORDESTE",
    # "SUL", "SUDESTE", "LESTE", "NOROESTE",
    # "OESTE", "SUDOESTE",

    # "TRABALHO", "PUBLICO", "PRIVADO", "DOCUMENTO",
    # "COPIA", "ARQUIVO", "COLA", "CHAVE",
    # "RESTAURANTE",

    # "TRANSPORTES", "AVIAO", "BICICLETA", "CAMINHAO",
    # "CARRO", "CARROCA", "DISCO_VOADOR", "FOGUETE",
    # "HELICOPTERO", "BARCO_A_VELA", "LANCHA", "METRO",
    # "MOTO", "NAVIO", "ONIBUS", "SUBMARINO",
    # "TAXI", "TREM", "UBER",

    # "JORNAL", "TELEVISAO", "RADIO", "INTERNET",
    # "E_MAIL", "WHATSAPP", "FACEBOOK", "INSTAGRAM",

    # "PESSOAS", "AUXILIAR_DE_COZINHA",
    # "AUXILIAR_DE_LIMPEZA", "BOMBEIRO", "CHEFE",
    # "COZINHEIRO", "DIRETOR(A)", "MEDICO",
    # "POLICIAL", "PROFESSOR(A)",
    # "SECRETARIO(A)", "VIGIA"
    ]

SEQUENCIAS_POR_SINAL = 30   # quantas sequências (repetições) coletar de cada sinal
FRAMES_POR_SEQUENCIA = 30   # tamanho da janela temporal - tem que bater com main.py e train.py

# Caminho relativo à posição deste arquivo, não ao diretório de onde o
# script é executado. Antes era "backend/data" (caminho relativo ao cwd),
# o que criava uma pasta "backend/backend/data" duplicada quando o script
# era rodado de dentro da própria pasta backend/. Agora aponta direto pra
# backend/letras, que é onde os dados já coletados estão.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def criar_pastas():
    """Cria uma subpasta em DATA_DIR para cada sinal, se ainda não existir."""
    for sinal in SINAIS:
        pasta = os.path.join(DATA_DIR, sinal)
        os.makedirs(pasta, exist_ok=True)
    print(f"Pastas criadas em '{DATA_DIR}'")


def contar_sequencias(sinal):
    """Conta quantos .npy já foram salvos para esse sinal (retoma coleta de onde parou)."""
    pasta = os.path.join(DATA_DIR, sinal)
    if not os.path.exists(pasta):
        return 0
    return len([f for f in os.listdir(pasta) if f.endswith(".npy")])


def extrair_landmarks(resultado):
    """
    Mesma extração usada no main.py: 63 coordenadas da mão esquerda + 63 da
    direita, zeradas quando a mão não é detectada. Precisa ficar idêntica
    à do backend, senão o modelo treina com um formato de dado e recebe
    outro na inferência.
    """
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


def coletar(sinal):
    """
    Coleta as sequências que faltam para um sinal. Para cada sequência:
    mostra contagem regressiva de 3s, grava 30 frames com tracking de mão
    ativo (static_image_mode=False), e salva o resultado em .npy.
    """
    pasta = os.path.join(DATA_DIR, sinal)
    ja_tem = contar_sequencias(sinal)
    faltam = SEQUENCIAS_POR_SINAL - ja_tem

    if faltam <= 0:
        print(f"Sinal '{sinal}' já tem {ja_tem} sequências. Pulando...")
        return

    print(f"\nSinal: {sinal} | Faltam: {faltam} sequências")
    print("Feche a janela ou aperte Q para sair\n")

    cap = cv2.VideoCapture(0)
    contador = ja_tem

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while contador < SEQUENCIAS_POR_SINAL:

            # ── Contagem regressiva ──────────────────────────
            for i in range(3, 0, -1):
                inicio = time.time()
                while time.time() - inicio < 1.0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    cv2.putText(frame, f"{sinal}: {contador}/{SEQUENCIAS_POR_SINAL}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)
                    cv2.putText(frame, "Prepare-se...",
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                    cv2.putText(frame, str(i),
                                (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 10)

                    cv2.imshow(f"Coletando: {sinal}", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

            # ── Grava os 30 frames ───────────────────────────
            sequencia = []

            for f_idx in range(FRAMES_POR_SEQUENCIA):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = hands.process(rgb)
                landmarks = extrair_landmarks(resultado)
                sequencia.append(landmarks)

                if resultado.multi_hand_landmarks:
                    for hand_landmarks in resultado.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

                # Indicador vermelho de gravação
                cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, f"GRAVANDO {f_idx+1}/{FRAMES_POR_SEQUENCIA}",
                            (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"{sinal}: {contador}/{SEQUENCIAS_POR_SINAL}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

                cv2.imshow(f"Coletando: {sinal}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # ── Salva sequência ──────────────────────────────
            if len(sequencia) == FRAMES_POR_SEQUENCIA:
                nome = os.path.join(pasta, f"{sinal}_{contador:04d}.npy")
                np.save(nome, np.array(sequencia))
                contador += 1
                print(f"  Sequência {contador}/{SEQUENCIAS_POR_SINAL} salva")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Coleta finalizada. Total: {contador} sequências para '{sinal}'")


def menu():
    criar_pastas()
    print("\n=== COLETA DE DADOS - LIBRAS ===")

    for i, sinal in enumerate(SINAIS):
        seqs = contar_sequencias(sinal)
        status = "✓" if seqs >= SEQUENCIAS_POR_SINAL else f"{seqs}/{SEQUENCIAS_POR_SINAL}"
        print(f"  {i+1}. {sinal} [{status}]")

    print("  0. Todos os sinais")
    escolha = input("\nEscolha o número do sinal: ").strip()

    if escolha == "0":
        for sinal in SINAIS:
            coletar(sinal)
    elif escolha.isdigit() and 1 <= int(escolha) <= len(SINAIS):
        coletar(SINAIS[int(escolha) - 1])
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    menu()