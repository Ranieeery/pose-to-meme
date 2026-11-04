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

## Testes

```powershell
pytest -q                 # testes unitários
pytest -m integration -q  # precisa do modelo baixado
```
