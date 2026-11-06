# Como gravar as sessões

O classificador aprende com estas gravações. Uma sessão é uma execução completa do `collect`, com todas as poses, e leva uns 6 minutos.

## Quantas sessões

Grave **pelo menos duas sessões completas**, de preferência em dias diferentes. Entre uma e outra, mude a luz, a roupa e a distância da câmera.

A `sessao_02` não tem `maos_na_cabeca` nem `mao_na_gravata`, e o `neutro` dela não tem mãos visíveis. Ela pode ficar, mas o `train` vai avisar que está incompleta. Se você gravar três sessões completas, apague-a.

## Passo a passo

1. Ative o ambiente e abra a coleta:

   ```powershell
   .venv\Scripts\activate
   python -m scripts.collect
   ```

2. Confira se os ombros aparecem. Sem eles, nada é gravado, e a tela avisa.
3. Aperte a tecla da classe. A seta na tela mostra qual está escolhida.
4. Aperte **espaço**, entre na pose durante os 3 segundos de contagem e mexa um pouco enquanto grava.
5. Faça **3 ou 4 trechos curtos** por classe. Aperte espaço para parar, ajuste a posição e aperte espaço de novo: a contagem continua de onde parou.
6. Repita até todas as classes chegarem a 200/200. A gravação de cada classe para sozinha nesse ponto.
7. Aperte `q` só quando terminar tudo. Depois disso, a próxima execução cria outra sessão, sem retomar esta.

## Poses

| Tecla | Classe | Meme | Como fazer |
|---|---|---|---|
| `0` | `neutro` | nenhum | sem pose, com as mãos visíveis na maior parte do tempo (veja abaixo) |
| `1` | `mao_fechada_na_boca` | `bebe-pensativo` | dedos fechados e juntos encostados na boca |
| `2` | `mao_aberta_na_boca` | `maria-joaquina-unha` | mão aberta na frente da boca, mostrando as unhas |
| `3` | `maos_unidas` | `bjork-pose` | palmas unidas na frente da boca, como em um "amém" |
| `4` | `dedo_levantado` | `lily-nerd` | indicador levantado ao lado da cabeça |
| `5` | `apontando_para_si` | `ryujin-eu` | indicador apontando para o próprio rosto, sem encostar |
| `6` | `sinal_de_rock` | `licurgo-rock` | indicador e mindinho esticados, os outros dedos dobrados |
| `7` | `maos_abertas_para_frente` | `numetal-pose` | as duas mãos para a câmera, bem abertas e com os dedos separados |
| `8` | `garras_para_frente` | `ozzy-osbourne` | as duas mãos para a câmera, com os dedos curvados como garras |
| `9` | `mao_no_ombro` | `valverde-pose` | a mão **cruza o corpo** até o ombro ou o braço oposto |
| `a` | `dedo_na_boca` | `macaco-dedo-boca` | só o indicador esticado nos lábios, com o mindinho bem dobrado |
| `b` | `maos_na_cabeca` | `aaaaaaaaaaaaaa` | as duas mãos na cabeça, com as pontas dos dedos acima da testa e a boca aberta |
| `c` | `mao_na_gravata` | `trajado` | uma mão no peito, **sem passar do centro do corpo**, com os dedos dobrados como quem segura a gravata; se a outra mão aparecer, deixe-a na cintura |

## Como gravar o `neutro`

É ele que evita meme aparecendo sem você fazer pose. Inclua:

- braços em movimento e mãos paradas na frente do corpo
- mão no queixo, coçando o rosto, ajustando o cabelo ou o fone
- poucos frames com as mãos abaixadas

## Cuidados

- **Varie dentro da pose:** mude um pouco o ângulo da mão e a distância entre os trechos. Pose congelada ensina pouco.
- **Alterne as mãos** nas poses de uma mão só, dentro da mesma classe.
- **Pare antes de sair da pose:** se você baixar o braço com a gravação ligada, esses frames entram com o rótulo errado.
- **Mantenha o gesto:** faça cada pose do mesmo jeito em todas as sessões. Senão, o modelo aprende dois gestos com o mesmo nome.
- **Duplas que se parecem:** exagere a diferença em `1` e `a`, `a` e `6`, e `9` e `c`.
- **`maos_unidas` com uma mão só:** é normal. Palmas unidas costumam ser detectadas como uma mão.

## Depois de gravar

```powershell
python -m scripts.train
python main.py
```

O terminal deve mostrar `classifier: MLClassifier`. O resultado sai em `reports/report.md`.

Se o `train` avisar de uma sessão incompleta, as métricas ficam distorcidas. Se gravar algo errado, só dá para apagar a sessão inteira, removendo `data/samples/sessao_XX.csv`.
