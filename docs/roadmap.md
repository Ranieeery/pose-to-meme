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

## Prioridade média

- [ ] **Pose extraída dos próprios memes**: rodar o PoseLandmarker nas imagens da pasta; quando houver pessoa detectável, usar o vetor de pose do meme para matching por vizinho mais próximo com a pose do usuário, com fallback para o CLIP.
- [ ] **Gestos de mão** com Hand Landmarker (joinha, apontar, dedo no queixo) combinados às features de pose. _Antecipado para o M2._
- [ ] **Aumento de dados por espelhamento**: inverter x e trocar esquerda/direita nas features de treino, para poses de uma mão generalizarem para a outra sem gravar as duas. Só no treino, nunca nas sessões de teste.
- [ ] **Classe nula no índice de memes**: imagens sem pessoa pontuam 0,145–0,197 e memes reais 0,178–0,240, então o `min_similarity` sozinho não rejeita imagens fora do tema. Um prompt de rejeição ("a picture with no person") que concorre com as classes resolveria sem depender do valor absoluto do cosseno.
- [ ] **CI no GitHub Actions** rodando `pytest` (sem os testes `integration`).
- [ ] **GIF de demonstração** no README.

## Ideias

- [ ] Som associado a cada meme.

## Concluído

_(vazio)_
