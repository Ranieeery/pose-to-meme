# Relatório de treino

Sessões (3): sessao_02, sessao_03, sessao_04. Amostras: 7400.

Todas as métricas vêm de sessões que o modelo avaliado não viu (leave-one-session-out). As regras não são treinadas, então todas as sessões são inéditas para elas.

## Aviso: sessões incompletas

Estas sessões distorcem as métricas. Classes ausentes:

- sessao_02: mao_na_gravata, maos_na_cabeca

## Comparação

| Classificador | Acurácia | F1 macro | Pior sessão | Acurácia após threshold | Meme indevido em `neutro` | Latência (ms) |
|---|---|---|---|---|---|---|
| MLP | 0.874 | 0.872 | 0.791 | 0.838 | 0.157 | 0.298 |
| KNN | 0.764 | 0.767 | 0.707 | 0.744 | 0.263 | 1.096 |
| Regras | 0.720 | 0.688 | 0.673 | 0.720 | 0.255 | 0.032 |

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

       apontando_para_si      0.859     0.738     0.794       600
          dedo_levantado      0.814     0.948     0.876       600
            dedo_na_boca      0.815     0.838     0.827       600
      garras_para_frente      0.901     0.715     0.797       600
      mao_aberta_na_boca      0.871     0.932     0.900       600
     mao_fechada_na_boca      0.940     0.920     0.930       600
          mao_na_gravata      0.715     0.782     0.747       400
            mao_no_ombro      0.903     0.950     0.926       600
maos_abertas_para_frente      0.798     1.000     0.888       600
          maos_na_cabeca      0.935     0.978     0.956       400
             maos_unidas      0.997     1.000     0.998       600
                  neutro      0.907     0.733     0.811       600
           sinal_de_rock      0.931     0.833     0.880       600

                accuracy                          0.874      7400
               macro avg      0.876     0.874     0.872      7400
            weighted avg      0.879     0.874     0.873      7400
```

### Matriz de confusão

| real \ previsto | apontando_para_si | dedo_levantado | dedo_na_boca | garras_para_frente | mao_aberta_na_boca | mao_fechada_na_boca | mao_na_gravata | mao_no_ombro | maos_abertas_para_frente | maos_na_cabeca | maos_unidas | neutro | sinal_de_rock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apontando_para_si | 443 | 114 | 7 | 0 | 1 | 14 | 0 | 0 | 0 | 0 | 0 | 21 | 0 |
| dedo_levantado | 9 | 569 | 20 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| dedo_na_boca | 0 | 2 | 503 | 0 | 77 | 15 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| garras_para_frente | 37 | 0 | 0 | 429 | 0 | 0 | 0 | 0 | 120 | 0 | 0 | 14 | 0 |
| mao_aberta_na_boca | 0 | 0 | 33 | 0 | 559 | 6 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_fechada_na_boca | 0 | 0 | 46 | 0 | 2 | 552 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_na_gravata | 0 | 0 | 0 | 0 | 0 | 0 | 313 | 50 | 0 | 0 | 0 | 0 | 37 |
| mao_no_ombro | 0 | 0 | 0 | 0 | 0 | 0 | 28 | 570 | 0 | 0 | 2 | 0 | 0 |
| maos_abertas_para_frente | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 | 0 | 0 |
| maos_na_cabeca | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 391 | 0 | 9 | 0 |
| maos_unidas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 |
| neutro | 27 | 14 | 8 | 47 | 0 | 0 | 5 | 0 | 32 | 27 | 0 | 440 | 0 |
| sinal_de_rock | 0 | 0 | 0 | 0 | 1 | 0 | 87 | 11 | 0 | 0 | 0 | 1 | 500 |

## KNN

```
                          precision    recall  f1-score   support

       apontando_para_si      0.895     0.597     0.716       600
          dedo_levantado      0.658     0.890     0.756       600
            dedo_na_boca      0.443     0.648     0.526       600
      garras_para_frente      0.855     0.648     0.737       600
      mao_aberta_na_boca      0.721     0.675     0.697       600
     mao_fechada_na_boca      0.753     0.523     0.618       600
          mao_na_gravata      0.591     0.700     0.641       400
            mao_no_ombro      0.960     0.910     0.934       600
maos_abertas_para_frente      0.761     0.957     0.848       600
          maos_na_cabeca      0.938     0.953     0.945       400
             maos_unidas      0.951     1.000     0.975       600
                  neutro      0.897     0.697     0.784       600
           sinal_de_rock      0.816     0.782     0.798       600

                accuracy                          0.764      7400
               macro avg      0.787     0.768     0.767      7400
            weighted avg      0.789     0.764     0.766      7400
```

### Matriz de confusão

| real \ previsto | apontando_para_si | dedo_levantado | dedo_na_boca | garras_para_frente | mao_aberta_na_boca | mao_fechada_na_boca | mao_na_gravata | mao_no_ombro | maos_abertas_para_frente | maos_na_cabeca | maos_unidas | neutro | sinal_de_rock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apontando_para_si | 358 | 214 | 9 | 0 | 1 | 14 | 1 | 0 | 0 | 0 | 0 | 1 | 2 |
| dedo_levantado | 30 | 534 | 0 | 0 | 4 | 0 | 32 | 0 | 0 | 0 | 0 | 0 | 0 |
| dedo_na_boca | 0 | 0 | 389 | 0 | 120 | 72 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| garras_para_frente | 0 | 9 | 0 | 389 | 0 | 0 | 0 | 0 | 168 | 0 | 0 | 34 | 0 |
| mao_aberta_na_boca | 0 | 0 | 176 | 0 | 405 | 17 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_fechada_na_boca | 0 | 0 | 257 | 0 | 29 | 314 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mao_na_gravata | 0 | 35 | 0 | 0 | 0 | 0 | 280 | 23 | 0 | 0 | 0 | 1 | 61 |
| mao_no_ombro | 0 | 0 | 1 | 0 | 0 | 0 | 42 | 546 | 0 | 0 | 0 | 0 | 11 |
| maos_abertas_para_frente | 0 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | 574 | 0 | 0 | 0 | 0 |
| maos_na_cabeca | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 381 | 0 | 11 | 0 |
| maos_unidas | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 600 | 0 | 0 |
| neutro | 4 | 20 | 35 | 40 | 2 | 0 | 6 | 0 | 12 | 25 | 25 | 418 | 13 |
| sinal_de_rock | 0 | 0 | 12 | 0 | 1 | 0 | 111 | 0 | 0 | 0 | 6 | 1 | 469 |

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
