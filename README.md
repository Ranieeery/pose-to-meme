# meme-pose

Rastreia a pose pela webcam e mostra um meme quando reconhece uma pose conhecida.

## Setup

Requer Windows nativo (o WSL2 não acessa a webcam por padrão) e Python 3.11 ou 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Baixar o modelo de pose

O arquivo `.task` não é versionado. Baixe o modelo `full`, que é o padrão:

```powershell
Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task -OutFile models\pose_landmarker_full.task
```

Para usar `lite` ou `heavy`, troque `full` nos três pontos do comando e ajuste `tracker.model_variant` no `config.yaml`.

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
