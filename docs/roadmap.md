# Roadmap

Itens fora do escopo do MVP. Marque `[x]` quando implementado e mova para "Concluído" com a data.

## Prioridade alta

_(vazio)_

## Prioridade média

- [ ] **Pose extraída dos próprios memes**: rodar o PoseLandmarker nas imagens da pasta; quando houver pessoa detectável, usar o vetor de pose do meme para matching por vizinho mais próximo com a pose do usuário, com fallback para o CLIP.
- [ ] **GIF de demonstração** no README.

## Ideias

- [ ] Som associado a cada meme.

## Concluído

- [x] **CI no GitHub Actions** (2026-11-06): `.github/workflows/tests.yml` roda `pytest` (sem os testes `integration`) a cada push e pull request, no Windows com Python 3.12.
- [x] **Gestos de mão** (M2): o `HandLandmarker` roda em paralelo com a pose, e as features carregam os 21 pontos de cada mão. Várias classes dependem dos dedos (`dedo_levantado`, `sinal_de_rock`, `dedo_na_boca`, `apontando_para_si`). Joinha e dedo no queixo podem entrar como poses novas pela `/nova-pose`.
- [x] **Aumento de dados por espelhamento** (2026-11-06): `training.mirror_augmentation`, só nas sessões de treino. O MLP foi de 0,874 para 0,922 de acurácia, e a pior sessão de 0,791 para 0,872.
- [x] **Skill `/nova-pose`** (2026-11-06): checklist em `.claude/skills/nova-pose/SKILL.md`. A comparação com o modelo anterior é feita pelo `scripts/compare_metrics.py`, que rejeita o modelo se alguma classe existente perder mais F1 que `training.max_class_f1_drop`.
- [x] **Tabela de desempenho no README** (2026-11-06): tempo por etapa com `full` e `lite`; o tracking é o gargalo, e o `lite` não compensa porque as mãos rodam em paralelo com a pose.
