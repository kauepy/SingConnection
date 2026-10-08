/* ===========================================================
  Frontend do tradutor de Libras.

  Fluxo: captura frames da webcam continuamente -> junta em lotes
  de 30 -> manda cada lote pro backend (/reconhecer) -> recebe o
  sinal previsto + confiança -> atualiza a UI e o texto traduzido.

  Contrato da API (main.py):
    POST { frames: ["<base64 sem prefixo>", ... 30 itens] }
    -> { sinal: "A", confianca: 0.93 }

  Feito pra rodar por longos períodos sem travar (ex: demonstração
  de várias horas): reconecta a câmera sozinho se o navegador
  suspender o stream, evita a tela dormir enquanto ativo, e limita
  o tamanho do texto acumulado pra não crescer sem limite.
  =========================================================== */

const API_PADRAO = "http://127.0.0.1:8000";
const origem = window.location.origin;
const BASE_URL = (origem === API_PADRAO || origem === "http://localhost:8000") ? "" : API_PADRAO;
const ROTA_RECONHECER = "/reconhecer";
const ROTA_TRADUZIR = "/traduzir";

// Tem que bater com FRAMES_ESPERADOS no backend e o reshape(1, 30, 126).
const FRAMES_POR_LOTE = 30;
const INTERVALO_CAPTURA_MS = 60;

// Abaixo desse valor, tratamos a predição como ruído (mão fora de quadro,
// gesto ambíguo, etc.) e não adicionamos nada ao texto.
const CONFIANCA_MINIMA = 0.6;

// Limites de memória pra uma sessão longa (ex: 6h de demonstração) não
// deixar o texto/lista crescerem sem parar.
const MAX_CARACTERES_TEXTO = 2000;
const MAX_SINAIS_TRADUCAO = 80;

const video = document.getElementById("video");
const btnCamera = document.getElementById("btn-camera");
const btnLimpar = document.getElementById("btn-limpar");
const btnEspaco = document.getElementById("btn-espaco");
const btnApagar = document.getElementById("btn-apagar");
const btnCopiar = document.getElementById("btn-copiar");
const btnTraduzirIA = document.getElementById("btn-traduzir-ia");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const confidenceBarFill = document.getElementById("confidence-bar-fill");
const confidenceValue = document.getElementById("confidence-value");
const captureProgress = document.getElementById("capture-progress");
const handGuide = document.getElementById("hand-guide");
const lastSign = document.getElementById("last-sign");
const translationOutput = document.getElementById("translation-output");
const charCounter = document.getElementById("char-counter");
const aiOutput = document.getElementById("ai-output");

let stream = null;
let capturaAtiva = false;
let bufferFrames = [];
let enviandoLote = false;
// Se um novo lote de 30 frames ficar pronto enquanto o anterior ainda
// está sendo processado, guardamos só o mais recente aqui em vez de
// descartar - assim não perdemos sinalizações feitas durante o envio.
let loteEmEspera = null;
let ultimoSinalAdicionado = null;
let texto = "";
// Lista dos sinais brutos, na ordem reconhecida - é o que mandamos pra IA
// traduzir. Fica separada de "texto" porque "texto" mistura letras soltas
// (soletração) com palavras completas, tudo concatenado.
let sinaisReconhecidos = [];
let wakeLock = null;
let falhasSeguidas = 0;

// Canvas auxiliar (fora da tela) só pra converter frames do vídeo em base64.
const canvas = document.createElement("canvas");
const ctx = canvas.getContext("2d");

if (btnCamera && video) {
  btnCamera.addEventListener("click", () => {
    if (stream) {
      pararCamera();
    } else {
      iniciarCamera();
    }
  });

  btnLimpar.addEventListener("click", limparTudo);
  btnEspaco.addEventListener("click", () => adicionarAoTexto(" "));
  btnApagar.addEventListener("click", apagarUltimaLetra);
  btnCopiar.addEventListener("click", copiarTexto);
  btnTraduzirIA.addEventListener("click", traduzirComIA);
}

async function iniciarCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    await video.play();

    // Se o navegador suspender a faixa de vídeo sozinho (economia de
    // energia, troca de aba por muito tempo, etc.), detecta e tenta
    // reiniciar a câmera automaticamente - importante numa demonstração
    // longa sem alguém acompanhando o tempo todo.
    stream.getVideoTracks()[0].addEventListener("ended", () => {
      if (capturaAtiva) {
        console.warn("Faixa de vídeo encerrada inesperadamente, tentando reconectar...");
        statusText.textContent = "Reconectando câmera...";
        pararCamera({ mantendoIntencao: true });
        setTimeout(iniciarCamera, 1000);
      }
    });

    await solicitarWakeLock();

    btnCamera.textContent = "Parar câmera";
    btnLimpar.disabled = false;
    btnEspaco.disabled = false;
    btnApagar.disabled = false;

    statusPill.classList.add("on");
    statusText.textContent = "Posicione a mão no guia";
    handGuide.classList.remove("hidden");

    capturaAtiva = true;
    bufferFrames = [];
    falhasSeguidas = 0;
    loopDeCaptura();
  } catch (err) {
    statusText.textContent = "Sem acesso à câmera";
    alert("Não foi possível acessar a câmera. Verifique as permissões do navegador.");
  }
}

function pararCamera({ mantendoIntencao = false } = {}) {
  capturaAtiva = false;
  bufferFrames = [];
  atualizarProgresso(0);

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  liberarWakeLock();

  if (!mantendoIntencao) {
    btnCamera.textContent = "Iniciar câmera";
    statusPill.classList.remove("on");
    statusText.textContent = "Câmera desligada";
    lastSign.textContent = "Aguardando";
    lastSign.classList.add("empty");
  }
  handGuide.classList.add("hidden");
  atualizarConfianca(0);
}

// Wake Lock evita que a tela do dispositivo apague/durma enquanto a
// câmera está ativa - relevante numa demonstração de várias horas onde
// ninguém fica tocando na tela o tempo todo. Falha silenciosamente em
// navegadores que não suportam (a app continua funcionando normalmente).
async function solicitarWakeLock() {
  try {
    if ("wakeLock" in navigator) {
      wakeLock = await navigator.wakeLock.request("screen");
    }
  } catch (err) {
    console.warn("Wake Lock não disponível:", err.message);
  }
}

function liberarWakeLock() {
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
}

// Se a aba voltar a ficar visível (usuário trocou de app e voltou) e a
// câmera devia estar ativa mas o wake lock foi perdido, readquire.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && capturaAtiva && !wakeLock) {
    solicitarWakeLock();
  }
});

function limparTudo() {
  texto = "";
  ultimoSinalAdicionado = null;
  sinaisReconhecidos = [];
  btnTraduzirIA.disabled = true;
  aiOutput.textContent = 'Sinalize algo e clique em "Traduzir com IA".';
  aiOutput.classList.add("empty");
  renderTexto();
}

function copiarTexto() {
  if (!texto) return;
  navigator.clipboard.writeText(texto).then(() => {
    const original = btnCopiar.textContent;
    btnCopiar.textContent = "Copiado!";
    setTimeout(() => { btnCopiar.textContent = original; }, 1500);
  }).catch(() => {
    console.warn("Não foi possível copiar automaticamente.");
  });
}

// Captura um frame a cada INTERVALO_CAPTURA_MS. Quando o buffer completa
// 30 frames, dispara o envio (sem travar a captura do próximo lote) e
// reinicia o buffer do zero.
function loopDeCaptura() {
  if (!capturaAtiva) return;

  capturarFrame();
  atualizarProgresso(bufferFrames.length / FRAMES_POR_LOTE);

  if (bufferFrames.length >= FRAMES_POR_LOTE) {
    const lote = bufferFrames;
    bufferFrames = [];
    enviarLote(lote);
  }

  setTimeout(loopDeCaptura, INTERVALO_CAPTURA_MS);
}

function atualizarProgresso(fracao) {
  captureProgress.style.width = `${Math.min(100, Math.max(0, fracao * 100))}%`;
}

function capturarFrame() {
  if (video.videoWidth === 0) return; // vídeo ainda não carregou

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // toDataURL retorna "data:image/jpeg;base64,XXXX" - guardamos só a
  // parte depois da vírgula, que é o base64 puro que o backend espera.
  const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
  const base64Puro = dataUrl.split(",")[1];
  bufferFrames.push(base64Puro);
}

async function enviarLote(frames) {
  if (enviandoLote) {
    // Backend ainda ocupado com o lote anterior - guarda só o mais
    // recente pra processar em seguida, em vez de perder os frames.
    loteEmEspera = frames;
    return;
  }
  enviandoLote = true;
  statusText.textContent = "Reconhecendo...";

  try {
    const resposta = await fetch(BASE_URL + ROTA_RECONHECER, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames }),
    });

    if (!resposta.ok) throw new Error("Erro do servidor: " + resposta.status);
    const dados = await resposta.json(); // { sinal, confianca }
    processarResultado(dados);
    falhasSeguidas = 0;
    if (capturaAtiva) statusText.textContent = "Posicione a mão no guia";
  } catch (err) {
    console.warn("Falha no reconhecimento:", err.message);
    falhasSeguidas += 1;
    // Se o backend cair no meio da demonstração, avisa claramente mas
    // continua tentando nos próximos lotes - não trava a captura.
    statusText.textContent = falhasSeguidas >= 3
      ? "Servidor indisponível - verifique o backend"
      : "Servidor indisponível";
  } finally {
    enviandoLote = false;
    if (loteEmEspera) {
      const proximo = loteEmEspera;
      loteEmEspera = null;
      enviarLote(proximo);
    }
  }
}

function processarResultado({ sinal, confianca }) {
  atualizarConfianca(confianca || 0);

  if (!sinal || confianca < CONFIANCA_MINIMA) {
    lastSign.textContent = "Nenhum sinal";
    lastSign.classList.add("empty");
    ultimoSinalAdicionado = null; // permite repetir o mesmo gesto após uma pausa
    return;
  }

  lastSign.textContent = sinal;
  lastSign.classList.remove("empty");

  // Evita adicionar o mesmo sinal duas vezes seguidas (ex: usuário
  // segurando o mesmo gesto entre lotes consecutivos).
  if (sinal !== ultimoSinalAdicionado) {
    adicionarAoTexto(sinal);
    sinaisReconhecidos.push(sinal);
    if (sinaisReconhecidos.length > MAX_SINAIS_TRADUCAO) {
      sinaisReconhecidos.shift(); // descarta o mais antigo, mantém uso de memória estável
    }
    btnTraduzirIA.disabled = false;
    ultimoSinalAdicionado = sinal;
  }
}

function atualizarConfianca(confianca) {
  const pct = Math.round((confianca || 0) * 100);
  confidenceBarFill.style.width = `${pct}%`;
  confidenceBarFill.classList.toggle("low", confianca > 0 && confianca < CONFIANCA_MINIMA);
  confidenceValue.textContent = confianca > 0 ? `${pct}%` : "—";
}

function adicionarAoTexto(caractere) {
  texto += caractere;
  // Limite de segurança pra sessões muito longas não deixarem o texto
  // (e o DOM) crescendo pra sempre.
  if (texto.length > MAX_CARACTERES_TEXTO) {
    texto = texto.slice(texto.length - MAX_CARACTERES_TEXTO);
  }
  renderTexto();
}

function apagarUltimaLetra() {
  texto = texto.slice(0, -1);
  if (sinaisReconhecidos.length > 0) {
    sinaisReconhecidos.pop();
    btnTraduzirIA.disabled = sinaisReconhecidos.length === 0;
  }
  renderTexto();
}

async function traduzirComIA() {
  if (sinaisReconhecidos.length === 0) return;

  btnTraduzirIA.disabled = true;
  btnTraduzirIA.textContent = "Traduzindo...";

  try {
    const resposta = await fetch(BASE_URL + ROTA_TRADUZIR, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sinais: sinaisReconhecidos }),
    });

    if (!resposta.ok) throw new Error("Erro do servidor: " + resposta.status);
    const dados = await resposta.json(); // { frase }

    aiOutput.textContent = dados.frase;
    aiOutput.classList.remove("empty");
  } catch (err) {
    console.warn("Falha na tradução por IA:", err.message);
    aiOutput.textContent = "Não foi possível traduzir agora. Tente de novo.";
    aiOutput.classList.remove("empty");
  } finally {
    btnTraduzirIA.disabled = sinaisReconhecidos.length === 0;
    btnTraduzirIA.textContent = "Traduzir com IA";
  }
}

function renderTexto() {
  btnCopiar.disabled = texto.length === 0;
  charCounter.textContent = `${texto.length} caracteres`;

  if (texto.length === 0) {
    translationOutput.textContent = "O texto reconhecido vai aparecer aqui conforme você for fazendo os sinais.";
    translationOutput.classList.add("empty");
    return;
  }
  translationOutput.classList.remove("empty");

  // Usa nós de texto em vez de innerHTML - o conteúdo aqui só vem de
  // sinais reconhecidos pelo modelo (nunca de input livre do usuário),
  // mas evitar innerHTML com texto dinâmico é boa prática de segurança
  // por padrão (evita qualquer risco de injeção de HTML/script).
  translationOutput.replaceChildren(
    document.createTextNode(texto),
    Object.assign(document.createElement("span"), { className: "cursor-blink" })
  );
}

window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach((t) => t.stop());
});