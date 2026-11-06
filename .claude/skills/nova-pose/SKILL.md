---
name: nova-pose
description: Adiciona uma pose nova ao meme-pose de forma reproduzível, cobrindo classe e prompt no config, meme, gravação, retreino, comparação com o modelo anterior e documentação. Use quando o usuário pedir para criar, adicionar ou incluir uma pose, classe ou gesto novo.
argument-hint: "[nome_da_classe] [descrição do gesto]"
---

# /nova-pose

Checklist para adicionar uma pose sem piorar as que já existem. Siga na ordem e pare onde está indicado. Fale com o usuário em português.

Argumentos: `$ARGUMENTS`

## 0. Entender a pose

Antes de mexer em qualquer arquivo, confirme com o usuário:

- **Nome da classe:** em português, snake_case, sem acento (`maos_na_cintura`). Não pode existir em `poses.prompts` nem ser `neutro`.
- **O gesto:** como é feito e com quantas mãos.
- **O meme:** qual imagem vai para a pasta `memes/`, se houver.

Compare o gesto com as classes existentes em `config.yaml`. Se ele se parecer com alguma (mesma região do corpo, mesma forma de mão), avise antes de continuar. Duplas parecidas são as que mais confundem o modelo, como `dedo_na_boca` e `mao_fechada_na_boca`, ou `mao_no_ombro` e `mao_na_gravata`.

## 1. Guardar a base de comparação

O passo 7 compara o modelo novo com este. Rode a partir da raiz do projeto, com o venv ativo:

```powershell
python -c "import json; m = json.load(open('reports/metrics.json', encoding='utf-8')); print('ok' if 'per_class_f1' in m['classifiers']['mlp'] else 'sem per_class_f1')"
```

Se aparecer `sem per_class_f1`, rode `python -m scripts.train` antes, com os dados atuais. Depois:

```powershell
Copy-Item reports\metrics.json data\cache\metrics_base.json
Copy-Item models\pose_clf.joblib models\pose_clf_base.joblib
```

As duas pastas não são versionadas.

## 2. Registrar a classe

Siga o TDD do CLAUDE.md: atualize primeiro o teste que fixa o conjunto de classes.

1. **`config.yaml` → `poses.prompts`:** acrescente a classe **no fim**, porque a ordem define a tecla na coleta. O prompt deve estar em inglês e **sem** "a person", que os templates de `memes.prompt_templates` já adicionam. Exemplo: `"with both hands on their hips, elbows out"`.
2. **`config.yaml` → `collect.class_keys`:** acrescente uma tecla no fim (depois de `c` vem `d`). Ela não pode ser `q`, `s` nem espaço, que já são usadas pela interface.
3. **`pose_meme/classifier.py` → `POSE_CLASSES`:** acrescente a classe.
4. **`tests/test_config.py`:** atualize o conjunto de classes de `test_repo_config_has_the_documented_defaults`.
5. **`docs/gravacao.md`:** acrescente uma linha na tabela de poses (tecla, classe, meme e como fazer). Se o gesto se parecer com outra classe, acrescente também a dupla em "Duplas que se parecem".
6. **`README.md`:** atualize a contagem de poses na abertura ("São N poses mais o `neutro`").

O `RuleClassifier` não reconhece a classe nova, e isso é esperado: as regras são só baseline. Uma regra nova só deve ser criada se o usuário pedir, com teste em pose sintética.

Rode `pytest -q`. Tudo deve passar.

## 3. Meme

1. Coloque a imagem em `memes/` (`.png`, `.webp`, `.jpg` ou `.jpeg`).
2. Verifique se ela tem metadados com nomes ou dados pessoais, porque figurinhas de WhatsApp costumam trazer o nome do pacote e de quem o criou:

   ```powershell
   python -c "from PIL import Image; import sys; im = Image.open(sys.argv[1]); print(dict(im.getexif()), list(im.info))" memes\ARQUIVO
   ```

   Se tiver, avise o usuário e proponha remover os metadados sem reprocessar a imagem.
3. Rode `python -m scripts.index_memes` e confira:
   - **O meme novo caiu na classe nova.** Se não caiu, ajuste o prompt ou use `memes.overrides`. Uma margem abaixo de 0,01 indica que o CLIP ficou em dúvida.
   - **Nenhum meme antigo mudou de classe.** Um prompt novo pode atrair memes de outras poses. Compare com a tabela anterior, ou com o `data/cache/buckets.json` de antes. Se algum mudou, deixe o prompt mais específico ou use um override.
   - **Nenhuma linha "classes sem meme"**, a menos que o usuário aceite a pose sem meme.

## 4. Gravar (o usuário faz)

**Pare aqui.** Gravar exige a câmera, então quem faz é o usuário. Peça:

- **pelo menos duas sessões completas novas**, com todas as classes, incluindo a nova, em dias ou condições diferentes, seguindo o `docs/gravacao.md`;
- `python -m scripts.collect`, usando a tecla da classe nova.

As sessões antigas não têm a classe nova, e o `train` vai avisar que estão incompletas. Pergunte se o usuário quer mantê-las, porque somam dados, ou apagá-las, porque distorcem as métricas. A decisão é dele.

Uma sessão só com a classe nova não serve. Na avaliação por sessão, ela seria testada contra uma classe só, e as métricas perderiam o sentido.

## 5. Treinar

```powershell
python -m scripts.train
```

Confira o aviso de sessões incompletas e a seção da classe nova em `reports/report.md`: o F1 e a linha e a coluna dela na matriz de confusão do MLP.

## 6. Comparar com a base

```powershell
python -m scripts.compare_metrics data\cache\metrics_base.json
```

- **APROVADO:** nenhuma classe existente perdeu mais F1 que `training.max_class_f1_drop`. Siga para o passo 7.
- **REJEITADO:** restaure o modelo anterior e mostre ao usuário a classe que piorou e com quem ela passou a se confundir, pela matriz de confusão:

  ```powershell
  Copy-Item models\pose_clf_base.joblib models\pose_clf.joblib
  ```

  Sugira as saídas possíveis e deixe o usuário decidir. Não desfaça as mudanças de config sem perguntar.
  - Gravar de novo exagerando a diferença entre os gestos.
  - Gravar mais sessões.
  - Rever se a pose nova é distinguível com as features atuais.

Limitação da comparação: a avaliação nova inclui sessões novas, então os números não são medidos sobre exatamente os mesmos dados. Uma queda pequena pode vir das sessões, e não da classe nova. Diga isso ao usuário quando a decisão estiver no limite.

## 7. Documentar e fechar

1. **`README.md` → "Resultados":** atualize a tabela do classificador (sessões, amostras, métricas do MLP, KNN e regras) com os números de `reports/report.md`. Se as classes mais fracas mudaram, atualize também essa frase. Se um override foi necessário, atualize a contagem de acertos do CLIP em "Memes".
2. **Testes:** rode `pytest -q` e mostre a saída.
3. **Fechamento:** apresente ao usuário, em português:
   - o que mudou e em quais arquivos;
   - a tabela do `compare_metrics`;
   - como verificar no app, com `python main.py` e a pose nova;
   - sugestões de commit em Conventional Commits, separadas por natureza. Por exemplo:
     - `feat(classifier): add <classe> pose` para config, classificador e testes;
     - `docs: ...` para a documentação.

**Nunca faça commit nem push sem autorização explícita naquele momento.** As mensagens de commit não podem ter trailer `Co-Authored-By` nem qualquer menção de coautoria ou geração por IA.
