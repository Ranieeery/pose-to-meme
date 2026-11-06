# Relatório de treino

Sessões (3): sessao_02, sessao_03, sessao_04. Amostras: 7400.

Todas as métricas vêm de sessões que o modelo avaliado não viu (leave-one-session-out). As regras não são treinadas, então todas as sessões são inéditas para elas.

MLP e KNN treinaram também com cópias espelhadas (esquerda ↔ direita) das sessões de treino. As sessões de teste são avaliadas como foram gravadas.

## Aviso: sessões incompletas

Estas sessões distorcem as métricas. Classes ausentes:

- sessao_02: mao_na_gravata, maos_na_cabeca

## Comparação

| Classificador | Acurácia | F1 macro | Pior sessão | Acurácia após threshold | Meme indevido em `neutro` | Latência (ms) |
|---|---|---|---|---|---|---|
| MLP | 0.922 | 0.921 | 0.872 | 0.906 | 0.165 | 0.302 |
| KNN | 0.813 | 0.813 | 0.764 | 0.786 | 0.198 | 2.296 |
| Regras | 0.720 | 0.688 | 0.673 | 0.720 | 0.255 | 0.026 |

## Amostras por classe

| Classe | Amostras |
|---|---|
| apontando_para_si | 600 |
| dedo_levantado | 600 |
| dedo_na_boca | 600 |
| garras_para_frente | 600 |
| mao_aberta_na_boca | 600 |
| mao_fechada_na_boca | 600 |
| mao_na_gravata | 400 |
| mao_no_ombro | 600 |
| maos_abertas_para_frente | 600 |
| maos_na_cabeca | 400 |
| maos_unidas | 600 |
| neutro | 600 |
| sinal_de_rock | 600 |

## MLP

```
                          precision    recall  f1-score   support

       apontando_para_si      0.843     0.852     0.847       600
          dedo_levantado      0.839     0.940     0.887       600
            dedo_na_boca      0.956     0.985     0.970       600
      garras_para_frente      0.919     0.715     0.804       600
      mao_aberta_na_boca      0.981     0.972     0.977       600
     mao_fechada_na_boca      0.930     0.977     0.953       600
          mao_na_gravata      0.961     0.853     0.903       400
            mao_no_ombro      0.992     0.987     0.989       600
maos_abertas_para_frente      0.820     1.000     0.901       600
          maos_na_cabeca      0.941     0.958     0.949       400
             maos_unidas      0.998     1.000     0.999       600
                  neutro      0.887     0.747     0.811       600
           sinal_de_rock      0.964     0.990     0.977       600

                accuracy                          0.922      7400
               macro avg      0.926     0.921     0.921      7400
            weighted avg      0.924     0.922     0.920      7400
```

### Matriz de confusão

| real \ previsto | apontando_para_si | dedo_levantado | dedo_na_boca | garras_para_frente | mao_aberta_na_boca | mao_fechada_na_boca | mao_na_gravata | mao_no_ombro | maos_abertas_para_frente | maos_na_cabeca | maos_unidas | neutro | sinal_de_rock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apontando_para_si | 511 | 57 | 4 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 1 | 3 | 0 |
| dedo_levantado | 24 | 564 | 3 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| dedo_na_boca | 0 | 0 | 591 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 |
| garras_para_frente | 31 | 0 | 0 | 429 | 0 | 0 | 0 | 0 | 102 | 0 | 0 | 38 | 0 |
| mao_aberta_na_boca | 0 | 0 | 6 | 0 | 583 | 9 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_fechada_na_boca | 0 | 0 | 14 | 0 | 0 | 586 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_na_gravata | 0 | 36 | 0 | 0 | 0 | 0 | 341 | 5 | 0 | 0 | 0 | 0 | 18 |
| mao_no_ombro | 1 | 0 | 0 | 0 | 0 | 0 | 7 | 592 | 0 | 0 | 0 | 0 | 0 |
| maos_abertas_para_frente | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 | 0 | 0 |
| maos_na_cabeca | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 383 | 0 | 9 | 0 |
| maos_unidas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 |
| neutro | 39 | 15 | 0 | 38 | 1 | 1 | 1 | 0 | 30 | 24 | 0 | 448 | 3 |
| sinal_de_rock | 0 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 1 | 594 |

## KNN

```
                          precision    recall  f1-score   support

       apontando_para_si      0.896     0.600     0.719       600
          dedo_levantado      0.660     0.885     0.756       600
            dedo_na_boca      0.573     0.815     0.673       600
      garras_para_frente      0.895     0.722     0.799       600
      mao_aberta_na_boca      0.804     0.757     0.779       600
     mao_fechada_na_boca      0.824     0.585     0.684       600
          mao_na_gravata      0.709     0.725     0.717       400
            mao_no_ombro      0.932     0.940     0.936       600
maos_abertas_para_frente      0.811     0.970     0.883       600
          maos_na_cabeca      0.940     0.948     0.944       400
             maos_unidas      0.963     1.000     0.981       600
                  neutro      0.881     0.738     0.803       600
           sinal_de_rock      0.894     0.900     0.897       600

                accuracy                          0.813      7400
               macro avg      0.829     0.814     0.813      7400
            weighted avg      0.830     0.813     0.812      7400
```

### Matriz de confusão

| real \ previsto | apontando_para_si | dedo_levantado | dedo_na_boca | garras_para_frente | mao_aberta_na_boca | mao_fechada_na_boca | mao_na_gravata | mao_no_ombro | maos_abertas_para_frente | maos_na_cabeca | maos_unidas | neutro | sinal_de_rock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apontando_para_si | 360 | 207 | 2 | 0 | 2 | 22 | 0 | 0 | 0 | 0 | 0 | 5 | 2 |
| dedo_levantado | 34 | 531 | 2 | 0 | 1 | 0 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| dedo_na_boca | 0 | 0 | 489 | 0 | 78 | 33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| garras_para_frente | 3 | 6 | 0 | 433 | 0 | 0 | 0 | 0 | 126 | 0 | 0 | 32 | 0 |
| mao_aberta_na_boca | 0 | 0 | 125 | 0 | 454 | 17 | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| mao_fechada_na_boca | 0 | 0 | 229 | 0 | 20 | 351 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_na_gravata | 0 | 35 | 0 | 0 | 0 | 0 | 290 | 41 | 0 | 0 | 0 | 1 | 33 |
| mao_no_ombro | 0 | 0 | 0 | 0 | 0 | 0 | 33 | 564 | 0 | 0 | 1 | 0 | 2 |
| maos_abertas_para_frente | 0 | 0 | 0 | 18 | 0 | 0 | 0 | 0 | 582 | 0 | 0 | 0 | 0 |
| maos_na_cabeca | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 379 | 0 | 21 | 0 |
| maos_unidas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 |
| neutro | 5 | 25 | 2 | 33 | 10 | 3 | 4 | 0 | 10 | 24 | 14 | 443 | 27 |
| sinal_de_rock | 0 | 0 | 5 | 0 | 0 | 0 | 48 | 0 | 0 | 0 | 6 | 1 | 540 |

## Regras

```
                          precision    recall  f1-score   support

       apontando_para_si      0.843     0.645     0.731       600
          dedo_levantado      0.960     0.967     0.963       600
            dedo_na_boca      0.855     1.000     0.922       600
      garras_para_frente      0.015     0.015     0.015       600
      mao_aberta_na_boca      0.828     0.923     0.873       600
     mao_fechada_na_boca      0.686     0.728     0.707       600
          mao_na_gravata      0.652     0.403     0.498       400
            mao_no_ombro      0.676     0.980     0.800       600
maos_abertas_para_frente      0.000     0.000     0.000       600
          maos_na_cabeca      0.944     0.975     0.959       400
             maos_unidas      0.903     0.963     0.932       600
                  neutro      0.475     0.745     0.580       600
           sinal_de_rock      0.934     0.990     0.961       600

                accuracy                          0.720      7400
               macro avg      0.675     0.718     0.688      7400
            weighted avg      0.668     0.720     0.686      7400
```

### Matriz de confusão

| real \ previsto | apontando_para_si | dedo_levantado | dedo_na_boca | garras_para_frente | mao_aberta_na_boca | mao_fechada_na_boca | mao_na_gravata | mao_no_ombro | maos_abertas_para_frente | maos_na_cabeca | maos_unidas | neutro | sinal_de_rock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apontando_para_si | 387 | 0 | 0 | 2 | 0 | 26 | 1 | 28 | 0 | 1 | 1 | 154 | 0 |
| dedo_levantado | 0 | 580 | 7 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| dedo_na_boca | 0 | 0 | 600 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| garras_para_frente | 3 | 21 | 1 | 9 | 82 | 94 | 17 | 0 | 0 | 0 | 32 | 303 | 38 |
| mao_aberta_na_boca | 0 | 0 | 0 | 0 | 554 | 46 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_fechada_na_boca | 69 | 0 | 94 | 0 | 0 | 437 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_na_gravata | 0 | 0 | 0 | 0 | 0 | 0 | 161 | 239 | 0 | 0 | 0 | 0 | 0 |
| mao_no_ombro | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 588 | 0 | 0 | 0 | 4 | 4 |
| maos_abertas_para_frente | 0 | 0 | 0 | 562 | 13 | 0 | 0 | 0 | 0 | 1 | 0 | 24 | 0 |
| maos_na_cabeca | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 390 | 10 | 0 | 0 |
| maos_unidas | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 2 | 0 | 0 | 578 | 0 | 0 |
| neutro | 0 | 3 | 0 | 8 | 0 | 31 | 64 | 7 | 0 | 21 | 19 | 447 | 0 |
| sinal_de_rock | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 594 |
