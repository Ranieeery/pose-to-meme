# meme-pose

App em Python que rastreia a pose do usuário pela webcam, classifica a posição dos braços/mãos e exibe um meme centralizado sobre o vídeo quando uma pose conhecida é detectada. Os memes vêm de uma pasta local e são associados às poses automaticamente via embeddings (CLIP, zero-shot).

Projeto de portfólio para vaga de AI Engineer: a qualidade das decisões de ML (dados, avaliação, latência) importa tanto quanto o app funcionar. Toda decisão relevante deve ficar explicável no README.

## Ambiente

- **Python 3.11 ou 3.12** (confirmar compatibilidade da versão do `mediapipe` antes de subir de versão).
- **Rodar no Windows nativo, não no WSL2.** O WSL2 não acessa a webcam por padrão. Use venv no Windows (`python -m venv .venv`, `.venv\Scripts\activate`).
- Sem GPU. Tudo precisa rodar em CPU comum em tempo real.
- Dependências principais: `mediapipe`, `opencv-contrib-python` (exigido pelo `mediapipe`; não adicionar `opencv-python`, que instala o mesmo pacote `cv2`), `numpy`, `scikit-learn`, `joblib`, `sentence-transformers` (CLIP), `pyyaml`, `pytest`.

## Arquitetura

```
câmera → tracker → features → classifier → stabilizer → meme_index → renderer → janela
```

```
meme-pose/
├── main.py                 # loop principal; só orquestra, sem lógica de negócio
├── config.yaml             # câmera, thresholds, caminhos, prompts das classes
├── pose_meme/
│   ├── config.py           # carrega e valida o config.yaml em dataclasses tipadas
│   ├── capture.py          # wrapper da webcam (OpenCV)
│   ├── tracker.py          # MediaPipe PoseLandmarker → lista de landmarks
│   ├── features.py         # landmarks → vetor de features (ÚNICA fonte de verdade)
│   ├── classifier.py       # interface PoseClassifier + RuleClassifier + MLClassifier
│   ├── stabilizer.py       # debounce por janela/maioria
│   ├── meme_index.py       # embeddings CLIP, cache, buckets classe → memes, sorteio
│   ├── renderer.py         # esqueleto, overlay com alpha, HUD (FPS, pose, confiança)
│   └── timing.py           # medição de tempo por etapa
├── scripts/
│   ├── collect.py          # grava amostras rotuladas por sessão
│   ├── train.py            # treina, avalia, salva modelo + relatório
│   └── index_memes.py      # gera/atualiza cache de embeddings e buckets
├── data/
│   ├── samples/            # sessao_XX.csv (features + label + session_id)
│   └── cache/              # embeddings .npy + buckets.json (gerados, não versionar)
├── models/                 # pose_landmarker_{lite,full,heavy}.task, pose_clf.joblib
├── memes/                  # imagens (PNG com alpha preferível)
├── tests/
└── docs/
    └── roadmap.md
```

## Decisões tomadas (não mudar sem discutir)

### Tracking
- Usar a **API de Tasks do MediaPipe** (`mediapipe.tasks.python.vision.PoseLandmarker`, arquivo `.task`). **Não usar `mp.solutions`**, que é legado; tutoriais antigos usam essa API.
- Modelo `lite`/`full`/`heavy` configurável no `config.yaml`; padrão `full`.
- `num_poses=1`, captura em **640x480**, frame espelhado (`cv2.flip(frame, 1)`).
- Ignorar landmarks com `visibility` abaixo do threshold configurado.

### Features
- `features.py` é **compartilhado entre coleta, treino e inferência**. Nunca recalcular features de outro jeito em outro lugar (evita training-serving skew).
- Pontos do tronco superior (nariz, ombros, cotovelos, punhos, mãos), coordenadas x/y.
- Normalização: transladar para o centro dos ombros e dividir pela distância entre ombros. As features precisam ser invariantes a posição no quadro e distância da câmera.
- Lembrar que no MediaPipe **y cresce para baixo**.

### Classificação
- Interface única `PoseClassifier.predict(features) -> (classe, confiança)`.
- `RuleClassifier`: regras geométricas, serve como baseline e fallback quando não há modelo treinado.
- `MLClassifier`: pipeline scikit-learn (`StandardScaler` + `MLPClassifier`), comparado com KNN como baseline.
- Existe uma **classe `neutro`** obrigatória. Pose só é aceita se `classe != "neutro"` e `confiança >= threshold` (padrão 0.8).

### Dados e avaliação
- Cada amostra carrega `session_id`. **Split de treino/teste sempre por sessão, nunca aleatório**: frames consecutivos são quase idênticos e o split aleatório vaza informação, inflando a acurácia.
- Gravar sessões em condições diferentes (iluminação, roupa, distância).
- `train.py` gera `classification_report`, matriz de confusão e comparação MLP vs KNN, salvos em `reports/`.

### Estabilização
- Janela deslizante (padrão 8 frames); a pose muda só quando uma classe atinge a proporção mínima (padrão 0.7).
- O meme é sorteado **quando a pose muda** e se mantém enquanto ela durar (sem trocar de imagem a cada frame).

### Memes e embeddings
- Formatos aceitos em `memes/`: `.png`, `.webp`, `.jpg`/`.jpeg` (extensão case-insensitive). PNG e WebP podem ter alpha. WebP animado usa só o primeiro frame. Outros arquivos da pasta são ignorados com aviso no log.
- CLIP `clip-ViT-B-32` via `sentence-transformers`. **O CLIP nunca roda no loop de frames**: só na indexação.
- Cada classe tem um prompt textual em inglês no `config.yaml`; memes são atribuídos por similaridade cosseno (zero-shot). Cada meme vai para a classe de maior similaridade, desde que acima de um threshold mínimo; abaixo disso, é ignorado.
- Embeddings ficam em cache em `data/cache/`, invalidados por hash do conteúdo da pasta `memes/` + nome do modelo.
- O módulo recebe o encoder por **injeção de dependência**, para que testes usem um encoder falso sem baixar o CLIP.

### Renderização
- Meme centralizado, altura ≈ 40% do frame (configurável), alpha blending usando o canal alpha (carregar com `cv2.IMREAD_UNCHANGED`). Imagens sem alpha são coladas opacas.
- Imagens pré-carregadas e redimensionadas uma vez na inicialização.
- HUD com FPS, pose atual e confiança; esqueleto desenhável on/off por tecla.

### Desempenho
- Orçamento por frame: tracking é o gargalo; classificação e render devem ser desprezíveis.
- `timing.py` mede tempo por etapa; os números vão para uma tabela no README.

## Testes

- `pytest`. Rodar com `pytest -q` antes de considerar qualquer etapa concluída.
- **Testes unitários não dependem de webcam, de arquivos `.task` nem do download do CLIP.** Use landmarks sintéticos (objetos com `x`, `y`, `visibility`) e encoders falsos.
- Testes que precisam de modelo real ou câmera ficam marcados com `@pytest.mark.integration` e são pulados por padrão.
- O que deve estar coberto:
  - `features`: invariância a translação e escala; tamanho fixo do vetor; tratamento de landmarks com baixa visibilidade.
  - `classifier`: regras do `RuleClassifier` em poses sintéticas; `MLClassifier` com modelo minúsculo treinado no próprio teste; threshold de confiança e classe `neutro`.
  - `stabilizer`: não troca de pose por ruído de 1–2 frames; troca quando a maioria se sustenta.
  - Split por sessão: nenhuma `session_id` aparece em treino e teste ao mesmo tempo.
  - `meme_index`: atribuição correta com embeddings falsos; threshold mínimo; invalidação de cache quando a pasta muda.
  - `renderer`: overlay centralizado, sem estourar bordas, frame de saída com mesmo shape, alpha aplicado corretamente.
  - Carregamento do `config.yaml` com valores padrão.

## Convenções

- Código e identificadores em inglês; nomes de classes de pose em português snake_case (`maos_na_cabeca`).
- Type hints em tudo; funções puras sempre que possível (facilita teste).
- Nada de números mágicos no código: thresholds e caminhos vêm do `config.yaml`.
- `main.py` só orquestra. Lógica vai para `pose_meme/`.
- Não versionar `data/cache/`, `data/samples/`, `models/*.joblib`, `models/*.task`, `.venv/`.
- Ideias fora do escopo atual vão para `docs/roadmap.md`, não para o código.