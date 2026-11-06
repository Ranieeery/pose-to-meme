# Roadmap

Itens fora do escopo do MVP. Marque `[x]` quando implementado e mova para "Concluído" com a data.

## Prioridade alta

_(vazio)_

## Prioridade média

- [ ] **Pose extraída dos próprios memes**: rodar o PoseLandmarker nas imagens da pasta; quando houver pessoa detectável, usar o vetor de pose do meme para matching por vizinho mais próximo com a pose do usuário, com fallback para o CLIP.
- [ ] **Gestos de mão** com Hand Landmarker (joinha, apontar, dedo no queixo) combinados às features de pose. _Antecipado para o M2._
- [ ] **Classe nula no índice de memes**: imagens sem pessoa pontuam 0,145–0,197 e memes reais 0,178–0,240, então o `min_similarity` sozinho não rejeita imagens fora do tema. Um prompt de rejeição ("a picture with no person") que concorre com as classes resolveria sem depender do valor absoluto do cosseno.
- [ ] **CI no GitHub Actions** rodando `pytest` (sem os testes `integration`).
- [ ] **GIF de demonstração** no README.

## Ideias

- [ ] Som associado a cada meme.

## Concluído

- [x] **Aumento de dados por espelhamento** (2026-11-06): `training.mirror_augmentation`, só nas sessões de treino. O MLP foi de 0,874 para 0,922 de acurácia, e a pior sessão de 0,791 para 0,872.
- [x] **Skill `/nova-pose`** (2026-11-06): checklist em `.claude/skills/nova-pose/SKILL.md`. A comparação com o modelo anterior é feita pelo `scripts/compare_metrics.py`, que rejeita o modelo se alguma classe existente perder mais F1 que `training.max_class_f1_drop`.
- [x] **Tabela de desempenho no README** (2026-11-06): tempo por etapa com `full` e `lite`; o tracking é o gargalo, e o `lite` não compensa porque as mãos rodam em paralelo com a pose.
