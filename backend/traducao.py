"""
Camada de tradução Libras -> português fluente, usando a API do OpenRouter
(modelo gratuito, sem cartão de crédito).

Libras tem gramática própria (ordem dos sinais, ausência de artigos e
flexões verbais, estrutura tópico-comentário) diferente do português
escrito. Então concatenar os sinais reconhecidos direto ("EU VOCE AMAR")
não forma uma frase correta. Esse módulo pega a lista de sinais brutos, na
ordem em que foram reconhecidos, e pede pra IA reescrever como uma frase
fluente em português, preservando o sentido.

Requer a variável de ambiente OPENROUTER_API_KEY (chave gratuita gerada em
https://openrouter.ai/keys). Nunca coloque a chave direto no código.
"""

import os
from openai import OpenAI

# Llama 3.3 70B no tier gratuito do OpenRouter - suficiente pra reescrever
# frases curtas. O sufixo ":free" é o que garante que a chamada não
# consome crédito pago. Se um dia quiser trocar de modelo, é só mudar essa
# string por outro ID gratuito listado em https://openrouter.ai/models.
MODEL = "meta-llama/llama-3.3-70b-instruct:free"

BASE_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "Você recebe uma sequência de sinais de Libras reconhecidos por um "
    "modelo de visão computacional, na ordem em que foram sinalizados "
    "(pode incluir letras soltas formando palavras por soletração, e "
    "palavras completas). Sua tarefa é reescrever isso como uma frase "
    "fluente e gramaticalmente correta em português do Brasil, preservando "
    "o significado. Corrija ordem, adicione artigos e conectivos quando "
    "fizer sentido. Responda APENAS com a frase final, sem aspas, sem "
    "explicações."
)

_client = None
    

def _get_client() -> OpenAI:
    """Cria o client de forma preguiçosa (lazy), só na primeira chamada."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY não configurada. "
                "Defina a variável de ambiente antes de rodar o backend."
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def traduzir_frase(sinais: list[str]) -> str:
    """Recebe uma lista de sinais (ex: ["EU", "VOCE", "AMAR"]) e devolve a frase traduzida."""
    if not sinais:
        return ""

    glosas = " ".join(sinais)
    client = _get_client()

    resposta = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sinais reconhecidos: {glosas}"},
        ],
        max_tokens=200,
    )

    return resposta.choices[0].message.content.strip()