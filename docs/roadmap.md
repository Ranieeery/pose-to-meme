# Roadmap

Itens fora do escopo do MVP. Marque `[x]` quando implementado e mova para "Concluído" com a data.

## Prioridade alta

- [ ] **Tabela de desempenho no README**: FPS e tempo por etapa (tracking, features, classificação, render) medidos na máquina de desenvolvimento, com modelo `lite` e `full`.
- [ ] **Skill `/nova-pose`** (Claude): checklist reproduzível para adicionar uma pose:
  1. adicionar classe e prompt textual no `config.yaml`;
  2. coletar amostras em pelo menos duas sessões diferentes;
  3. retreinar com split por sessão;
  4. comparar métricas com o modelo anterior e rejeitar se alguma classe existente piorar além do tolerado;
  5. reindexar memes;
  6. atualizar a tabela de métricas no README.
- [ ] **Skill `/ml-cards`** (Claude): versão irmã da `/dsa-cards`, gerando flashcards de Anki em deck próprio sobre conceitos de ML/IA que surgirem no projeto (data leakage, training-serving skew, zero-shot, similaridade cosseno, embeddings multimodais, precision/recall, orçamento de latência), priorizando os pontos de maior dúvida.

## Prioridade média

- [ ] **Pose extraída dos próprios memes**: rodar o PoseLandmarker nas imagens da pasta; quando houver pessoa detectável, usar o vetor de pose do meme para matching por vizinho mais próximo com a pose do usuário, com fallback para o CLIP.
- [ ] **Rotulagem automática com LLM multimodal**: na indexação, um LLM descreve cada meme e sugere a classe/prompt; resultado em cache. Documentar custo e latência.
- [ ] **Gestos de mão** com Hand Landmarker (joinha, apontar, dedo no queixo) combinados às features de pose.
- [ ] **CI no GitHub Actions** rodando `pytest` (sem os testes `integration`).
- [ ] **GIF de demonstração** no README.

## Ideias

- [ ] Suporte a GIFs/vídeos curtos como meme.
- [ ] Som associado a cada meme.
- [ ] Versão web com MediaPipe tasks-vision (WASM) rodando no navegador.
- [ ] Recarregar `config.yaml` com uma tecla, sem reiniciar.

## Concluído

_(vazio)_
