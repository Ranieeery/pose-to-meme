# meme-pose

Rastreia a pose pela webcam e mostra um meme quando reconhece uma pose conhecida.

## Setup

Requer Windows nativo (o WSL2 não acessa a webcam por padrão) e Python 3.11 ou 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Baixar os modelos

Os arquivos `.task` não são versionados. Baixe o modelo de pose `full`, que é o padrão, e o de mãos:

```powershell
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models\pose_landmarker_full.task
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task -OutFile models\hand_landmarker.task
```

Para usar a pose `lite` ou `heavy`, troque `full` nos três pontos do primeiro comando e ajuste `tracker.model_variant` no `config.yaml`.

## Uso

```powershell
python main.py
```

| Tecla | Ação |
|---|---|
| `q` | fecha o app |
| `s` | liga/desliga o esqueleto |

## Treinar o classificador

O app usa o modelo treinado em `models/pose_clf.joblib` se ele existir, e as regras geométricas se não existir.

**1. Grave pelo menos duas sessões**, em condições diferentes (luz, roupa, distância):

```powershell
python -m scripts.collect
```

| Tecla | Ação |
|---|---|
| `0` a `9`, `a` a `c` | escolhe a classe (`0` é `neutro`); a tela mostra a tecla de cada uma |
| espaço | inicia a contagem e grava; de novo, para |
| `s` | liga/desliga o esqueleto |
| `q` | fecha e encerra a sessão |

A gravação para sozinha ao atingir `collect.samples_per_class` por classe. Cada execução cria uma nova sessão em `data/samples/`. O passo a passo, com as poses de cada tecla, está em [docs/gravacao.md](docs/gravacao.md).

**2. Avalie e treine:**

```powershell
python -m scripts.train
```

O relatório sai em `reports/report.md`. Cada sessão é testada por um modelo que não a viu, e o modelo salvo é treinado depois com todas elas.

## Testes

```powershell
pytest -q                 # testes unitários
pytest -m integration -q  # precisa do modelo baixado
```
