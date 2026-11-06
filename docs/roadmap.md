# Roadmap

Itens fora do escopo do MVP. Marque `[x]` quando implementado e mova para "Concluído" com a data.

## Prioridade alta

_(vazio)_

## Prioridade média

- [ ] **GIF de demonstração** no README.

## Ideias

- [ ] Som associado a cada meme.

## Descartado

- **Classificar a pose do usuário pelo meme mais próximo, sem treino** (testado em 2026-11-06):
  - **Acurácia:** nos 5600 frames de pose das 3 sessões, o meme mais próximo (com a versão espelhada) acertou 0,42, contra 0,92 do MLP.
  - **`neutro`:** não dá para separar. A distância mediana ao meme mais próximo é 8,2 nos frames de pose e 8,5 nos de `neutro`, então o app mostraria meme o tempo todo.
  - **Por que falha:** um único exemplo por classe, tirado de uma foto e não da câmera do usuário, não cobre a variação das sessões gravadas.

## Concluído

- [x] **Som ao aparecer um meme** (2026-11-06): o mesmo plim curto para todos os memes, gerado por código. Toca uma vez por aparição, com intervalo mínimo entre toques, sem travar o loop, e liga e desliga pela tecla `m`. Um som por meme continua em "Ideias".
- [x] **CI no GitHub Actions** (2026-11-06): `.github/workflows/tests.yml` roda `pytest` (sem os testes `integration`) a cada push e pull request, no Windows com Python 3.12.
- [x] **Gestos de mão** (M2): o `HandLandmarker` roda em paralelo com a pose, e as features carregam os 21 pontos de cada mão. Várias classes dependem dos dedos (`dedo_levantado`, `sinal_de_rock`, `dedo_na_boca`, `apontando_para_si`). Joinha e dedo no queixo podem entrar como poses novas pela `/nova-pose`.
- [x] **Aumento de dados por espelhamento** (2026-11-06): `training.mirror_augmentation`, só nas sessões de treino. O MLP foi de 0,874 para 0,922 de acurácia, e a pior sessão de 0,791 para 0,872.
- [x] **Skill `/nova-pose`** (2026-11-06): checklist em `.claude/skills/nova-pose/SKILL.md`. A comparação com o modelo anterior é feita pelo `scripts/compare_metrics.py`, que rejeita o modelo se alguma classe existente perder mais F1 que `training.max_class_f1_drop`.
- [x] **Tabela de desempenho no README** (2026-11-06): tempo por etapa com `full` e `lite`; o tracking é o gargalo, e o `lite` não compensa porque as mãos rodam em paralelo com a pose.
