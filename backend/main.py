#.venv\Scripts\Activate.ps1
#uvicorn main:app --host 0.0.0.0 --port 8000

"""
Backend de reconhecimento de Libras.

Recebe uma sequência de 30 frames (imagens em base64) capturados pelo
frontend, extrai os landmarks das mãos com MediaPipe e usa o modelo LSTM
treinado (train.py) para prever qual sinal foi feito.

Endurecido pra rodar exposto numa rede local por várias horas seguidas
(ex: demonstração/feira): CORS restrito, limite de tamanho de payload,
rate limiting básico por IP, e erros nunca vazam detalhes internos pro
cliente (só ficam logados no servi dor).
"""

import logging
import os
import time
from collections import defaultdict, deque

import numpy as np
import cv2
import pickle
import base64
import mediapipe as mp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from traducao import traduzir_frase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("libras-backend")

app = FastAPI()

# CORS restrito às origens do frontend. Em desenvolvimento local, o front
# geralmente roda em uma dessas portas (Live Server, http.server, etc).
# Pra rodar em outro endereço (ex: IP da rede local no dia da demonstração),
# defina a variável de ambiente ALLOWED_ORIGINS com as origens separadas
# por vírgula, ex: "http://192.168.0.10:5500,http://localhost:5500".
origens_padrao = (
    "http://localhost:5500,http://127.0.0.1:5500,"
    "http://localhost:5173,http://127.0.0.1:5173,"
    "http://localhost:8000,http://127.0.0.1:8000,"
    "http://localhost:8080,http://127.0.0.1:8080"
)
allowed_origins = os.environ.get("ALLOWED_ORIGINS", origens_padrao).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

mp_hands = mp.solutions.hands

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(BASE_DIR, "model", "libras_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

model = None
le = None
_modelo_tentou_carregar = False


def obter_modelo():
    """Carrega o LSTM só na primeira previsão, pra o servidor subir rápido."""
    global model, le, _modelo_tentou_carregar
    if model is not None and le is not None:
        return model, le
    if _modelo_tentou_carregar:
        return None, None

    _modelo_tentou_carregar = True
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        logger.error(
            "Modelo não encontrado em %s / %s. "
            "Rode train.py para gerar libras_model.keras e label_encoder.pkl.",
            MODEL_PATH,
            ENCODER_PATH,
        )
        return None, None

    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    logger.info("Modelo carregado!")
    return model, le

FRAMES_ESPERADOS = 30

# Limite defensivo de tamanho por frame em base64 (~2MB dá folga de sobra
# pra um frame JPEG de webcam, que normalmente fica bem abaixo disso).
# Protege contra alguém mandar payloads gigantes de propósito.
MAX_TAMANHO_FRAME_B64 = 2_000_000


class FrameRequest(BaseModel):
    frames: list[str]


class TraducaoRequest(BaseModel):
    sinais: list[str]


# ===================== Rate limiting simples por IP =====================
# Sem dependência externa: guarda os timestamps das últimas requisições de
# cada IP numa janela deslizante. Suficiente pra impedir que uma única
# origem (com ou sem má intenção) sobrecarregue o servidor de pedidos e
# derrube a demonstração. Não substitui um rate limiter de produção real,
# mas cobre bem o caso de uso de uma demonstração local/pequena.
JANELA_SEGUNDOS = 10
LIMITE_REQUISICOES_POR_JANELA = 30
historico_requisicoes = defaultdict(deque)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method != "POST":
        return await call_next(request)

    ip = request.client.host if request.client else "desconhecido"
    agora = time.monotonic()
    fila = historico_requisicoes[ip]

    while fila and agora - fila[0] > JANELA_SEGUNDOS:
        fila.popleft()

    if len(fila) >= LIMITE_REQUISICOES_POR_JANELA:
        logger.warning(f"Rate limit atingido para {ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Muitas requisições. Aguarde um instante e tente de novo."}
        )

    fila.append(agora)
    return await call_next(request)


# Captura qualquer exceção não tratada: loga o detalhe completo no servidor
# (pra você conseguir debugar), mas devolve uma mensagem genérica pro
# cliente, sem vazar stack trace nem detalhes internos. Isso também
# garante que um erro inesperado numa requisição não derruba o processo -
# o FastAPI/Starlette já isola isso por requisição, mas deixamos explícito.
@app.exception_handler(Exception)
async def excecao_nao_tratada(request: Request, exc: Exception):
    logger.exception(f"Erro não tratado em {request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Erro interno do servidor."})


def extrair_landmarks(resultado):
    """
    Converte o resultado do MediaPipe Hands em um vetor fixo de 126 números:
    63 coordenadas (x, y, z x 21 pontos) da mão esquerda + 63 da direita.
    Mão não detectada = vetor de zeros, garantindo shape constante mesmo
    quando só uma mão (ou nenhuma) aparece no frame.
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


@app.get("/health")
async def health():
    """Endpoint simples pro frontend checar se o backend está no ar."""
    modelo_ok = os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)
    return {"status": "ok", "modelo_presente": modelo_ok}


@app.post("/reconhecer")
async def reconhecer(request: FrameRequest):
    """
    Recebe 30 frames em base64, extrai os landmarks de cada um (modo vídeo,
    com tracking entre frames - mesmo modo usado na coleta de dados) e
    manda a sequência pro modelo prever o sinal.
    """
    modelo_atual, encoder_atual = obter_modelo()
    if modelo_atual is None or encoder_atual is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de reconhecimento indisponível. Treine com train.py e coloque os arquivos em backend/model/."
        )

    if len(request.frames) != FRAMES_ESPERADOS:
        raise HTTPException(
            status_code=400,
            detail=f"Esperado {FRAMES_ESPERADOS} frames, recebido {len(request.frames)}"
        )

    sequencia = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        for frame_b64 in request.frames:
            if len(frame_b64) > MAX_TAMANHO_FRAME_B64:
                raise HTTPException(status_code=413, detail="Frame excede o tamanho máximo permitido")

            try:
                # Aceita tanto base64 puro quanto data URL completa
                # ("data:image/jpeg;base64,...").
                if "," in frame_b64:
                    frame_b64 = frame_b64.split(",", 1)[1]

                img_bytes = base64.b64decode(frame_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("imdecode retornou None")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resultado = hands.process(rgb)
                landmarks = extrair_landmarks(resultado)
                sequencia.append(landmarks)
            except Exception as e:
                # Detalhe completo só no log do servidor - o cliente recebe
                # uma mensagem genérica, sem informação interna.
                logger.warning(f"Frame inválido descartado: {e}")
                raise HTTPException(status_code=400, detail="Um ou mais frames enviados são inválidos")

    X = np.array(sequencia).reshape(1, FRAMES_ESPERADOS, 126)
    pred = modelo_atual.predict(X, verbose=0)[0]
    idx = np.argmax(pred)
    sinal = encoder_atual.inverse_transform([idx])[0]
    confianca = float(pred[idx])

    return {
        "sinal": sinal,
        "confianca": confianca
    }


@app.post("/traduzir")
async def traduzir(request: TraducaoRequest):
    """
    Recebe a lista de sinais brutos reconhecidos (glosas, na ordem em que
    foram sinalizados) e devolve uma frase em português corrigida pela IA.
    """
    if not request.sinais:
        raise HTTPException(status_code=400, detail="Lista de sinais vazia")

    if len(request.sinais) > 200:
        raise HTTPException(status_code=400, detail="Lista de sinais excede o tamanho máximo permitido")

    try:
        frase = traduzir_frase(request.sinais)
    except Exception as e:
        logger.error(f"Erro ao chamar a IA de tradução: {e}")
        raise HTTPException(status_code=502, detail="Não foi possível traduzir agora. Tente novamente.")

    return {"frase": frase}


@app.get("/")
async def pagina_inicial():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/index.html")
async def pagina_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/como_usar.html")
async def pagina_como_usar():
    return FileResponse(os.path.join(FRONTEND_DIR, "como_usar.html"))


for pasta_static, nome in (("css", "css"), ("js", "js"), ("img", "img")):
    caminho = os.path.join(FRONTEND_DIR, pasta_static)
    if os.path.isdir(caminho):
        app.mount(f"/{nome}", StaticFiles(directory=caminho), name=nome)
