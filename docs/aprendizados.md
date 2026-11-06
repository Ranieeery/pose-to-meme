# Aprendizados

Anotações sobre os tópicos que o projeto cobriu e o que cada um ensinou. Os números vêm das medições feitas durante o desenvolvimento. As decisões finais estão no [README](../README.md).

## Resumo: o que mais importou

1. **Dados consistentes valem mais que modelo melhor.**
   - **O erro:** mudar a instrução de como gravar o `neutro` entre uma sessão e outra.
   - **O efeito:** o MLP caiu para 0,533 de acurácia, abaixo das regras (0,709), e o `neutro` ficou com recall 0.
   - **A correção:** nenhum ajuste de modelo resolveria. Foi preciso gravar uma terceira sessão no padrão certo, e a acurácia foi para 0,874.
2. **O jeito de avaliar decide o que você enxerga.** Num split aleatório, frames quase idênticos caem no treino e no teste, e o problema acima teria passado despercebido. O leave-one-session-out é que o revelou.
3. **Meça antes de otimizar e compare na mesma condição.** Um benchmark isolado chegou a dizer que a otimização do render tinha piorado as coisas. Comparando as duas versões alternadas no mesmo processo, ela era de 4 a 13 vezes mais rápida.
4. **Margens pequenas são ruído.** No CLIP, os erros perdiam por menos de 0,01 de similaridade, e só mudar como os prompts eram processados já trocava o resultado.
5. **Resultado negativo também é resultado.** Classificar a pose pelo meme mais próximo foi testado, deu 0,42 contra 0,92, e ficou registrado como descartado.

## 1. Visão computacional e tracking

### MediaPipe
- **API:** usar a API de Tasks (`PoseLandmarker`, `HandLandmarker`, arquivos `.task`), e não a `mp.solutions`, que é legada. A maioria dos tutoriais ainda usa a antiga.
- **Modos de execução:**
  - **VIDEO**, no app: exige timestamps estritamente crescentes e aproveita o tracking entre frames.
  - **IMAGE**, para fotos soltas, como os memes.
- **Dedos:** o `PoseLandmarker` tem só 3 pontos por mão, que não dizem se um dedo está esticado ou dobrado. Para isso é preciso o `HandLandmarker`, com 21 pontos.
- **Em paralelo:** em sequência, pose e mãos passavam do orçamento de um frame a 30 FPS. Em paralelo, o tracking dura o que demora o mais lento dos dois.

### Armadilhas encontradas
- **Câmera virtual ignora a resolução pedida:** o app pede 640x480 e recebe 1280x720. Nada no pipeline pode assumir tamanho ou proporção fixos.
- **Frame espelhado confunde a lateralidade:** o rótulo esquerda/direita do modelo de mãos falha. A solução foi ligar cada mão ao punho da pose mais próximo.
- **Mão fantasma:** em 41 frames de uma sessão, o modelo detectou uma mão sobre o ombro que não existia.
  - **Como identificar:** ela ficava a cerca de 0,6 unidade de corpo do punho e tinha metade do tamanho de uma mão real.
  - **Por que não filtrar com um corte simples por distância:** ele descartaria mãos reais em poses em que o punho da pose é que está errado.
- **Ruído de dedo:** o polegar aparecia esticado em 22% a 28% dos frames de `dedo_na_boca`, embora estivesse dobrado. Como o ruído era igual em todas as sessões, as regras simplesmente ignoram o polegar.
- **Fotos são mais difíceis que vídeo:** nos memes, o MediaPipe achou pessoa em 10 de 12. Falhou num macaco e numa foto recortada. Em várias, mãos não foram detectadas, ou foram detectadas a mais.

## 2. Engenharia de features

- **Normalização em "unidades de corpo":**
  1. corrigir x pela proporção do frame, porque o MediaPipe normaliza cada eixo separadamente;
  2. levar a origem para o centro dos ombros;
  3. dividir pela distância entre os ombros.

  O resultado não depende da posição no quadro, da distância da câmera nem da proporção do frame. Os testes verificam cada uma dessas invariâncias.
- **Vetor de tamanho fixo com indicador de presença:** são 125 valores. Uma mão não detectada vira zeros com a flag de presença em 0, e a amostra continua valendo.
- **Uma única fonte de verdade contra o training-serving skew:** coleta, treino e app chamam a mesma função do `features.py`. Se cada etapa calculasse as features do seu jeito, o modelo seria avaliado num formato e usado em outro, sem erro aparente.
- **O custo de guardar features em vez de landmarks:** os CSVs ficam compactos e prontos para treinar. Por outro lado, mudar a definição das features exige reaplicar a mudança aos vetores guardados, quando isso é possível, ou gravar de novo.
- **Acesso por nome:** o `FeatureView` dá acesso por nome (`point("left_wrist")`, `hand("right")`). Assim, nenhum outro módulo depende das posições dentro do vetor.

## 3. Coleta de dados

- **Sessão é a unidade:** cada execução da coleta é uma sessão completa, com todas as classes. Sessões em dias e condições diferentes (luz, roupa, distância) são o que mede generalização.
- **Protocolo escrito:** o [guia de gravação](gravacao.md) fixa como fazer cada pose. Isso evita que o mesmo nome acabe ensinando dois gestos diferentes.
- **Consistência entre sessões:** a definição de cada classe não pode mudar no meio da coleta. O `neutro` gravado sem mãos numa sessão e com mãos em outra quebrou a avaliação (veja o Resumo).
- **Nome é semântica:** duas classes tinham nomes que não correspondiam ao gesto ("mãos na boca" era, na verdade, um "amém"). Renomear exigiu migrar os rótulos já gravados, com backup antes.
- **Avisar sobre sessões incompletas:** uma sessão sem algumas classes distorce as métricas sem nenhum sinal. O `train` passou a avisar quais sessões estão incompletas e o que falta em cada uma.
- **Usabilidade da coleta:** contagem regressiva, amostrar um frame a cada dois, parar sozinho na meta e mostrar a tecla de cada classe na tela. Tudo isso reduz frames com rótulo errado.

## 4. Avaliação de modelos

- **Split por sessão, nunca aleatório:** frames consecutivos são quase idênticos, então o split aleatório mede memorização. A avaliação usa leave-one-session-out: cada sessão é testada por um modelo treinado nas outras.
- **Holdout ou leave-one-session-out?**
  - **Leave-one-session-out:** usa todas as sessões como teste e mostra a variação entre elas.
  - **Holdout:** uma sessão reservada só para teste. Só faz sentido quando há ajuste de hiperparâmetros, para não escolher o ajuste olhando para o teste.
- **Métricas além da acurácia:**
  - F1 macro: classes com o mesmo peso;
  - pior sessão: generalização;
  - acurácia depois do threshold: o que o usuário vê;
  - **meme indevido em `neutro`:** o erro que o usuário mais percebe;
  - latência por predição.
- **Baselines dão contexto:** KNN e regras geométricas mostram se o MLP realmente agrega. Na primeira avaliação, ele não agregava.
- **Avaliar não é escolher:** o modelo salvo é sempre o MLP. Como o resultado não escolhe o modelo, a métrica mede o MLP em vez de servir para escolhê-lo, o que inflaria o número.
- **Comparar modelos classe a classe:** o `compare_metrics` rejeita um modelo novo se alguma classe existente perder mais de 0,05 de F1. Assim, uma pose nova não piora as antigas sem ninguém ver.
- **Cuidado com dados de treino na demonstração:** o modelo salvo viu todas as sessões, e uma demonstração com ele pareceria melhor do que é. O GIF usa um MLP treinado sem a sessão exibida.

## 5. Aumento de dados

- **Espelhamento:** o x muda de sinal e os pontos e mãos de um lado trocam com os do outro, junto com as flags de presença. Uma pose gravada mais com uma mão passa a ensinar também a outra.
- **Só no treino:** só os dados de treino de cada fold são espelhados. A sessão de teste é avaliada como foi gravada, e um teste com um classificador espião garante que nenhuma cópia dela entre no treino.
- **Teste de correção mais forte:** espelhar o vetor tem que dar exatamente as features de uma pessoa espelhada.
- **Resultado:** acurácia de 0,874 para 0,922 e pior sessão de 0,791 para 0,872.
  - Os maiores ganhos foram nas poses de uma mão: `mao_na_gravata` (+0,16 de F1) e `dedo_na_boca` (+0,14).
  - O custo foi pequeno: o meme indevido em `neutro` foi de 0,157 para 0,165.

## 6. Regras geométricas

- **Papel:** são o baseline e o fallback quando não há modelo treinado. Explicáveis e quase sem custo (0,03 ms), mas limitadas: 0,72 de acurácia, contra 0,92 do MLP.
- **Limites medidos nos dados, não chutados:**
  - **Mãos unidas:** os punhos ficam a 0,30–0,42 um do outro, contra 1,0 ou mais nas outras poses com mão na boca. Limite: 0,6.
  - **Mão no peito:** o quanto a mão atravessa o corpo separa "mão na gravata" (−0,18) de "mão no ombro" (5º percentil −0,04). Limite: −0,1.
- **Onde falham:** gestos parecidos que dependem de detalhes finos, como garras e mãos abertas para frente. É o tipo de fronteira que o modelo aprende melhor que uma regra.

## 7. Zero-shot com CLIP

- **Como funciona:** imagem e texto viram vetores no mesmo espaço. Cada meme vai para a classe de prompt com maior similaridade cosseno, sem nenhum treino.
- **O modelo maior fez diferença:** o `clip-ViT-B-32` acertou 7 de 12 memes; o `clip-ViT-L-14`, de 9 a 11. Como o CLIP só roda na indexação, o modelo maior não custa FPS.
- **O que não ajudou no B-32:** média de prompts e correção do viés por classe. O problema era a capacidade do modelo de distinguir gestos finos de mão.
- **Média de templates:** o embedding de cada classe é a média do prompt aplicado a 4 templates ("a photo of a person {}", e assim por diante). É uma prática comum com CLIP e deixa o resultado mais estável.
- **Margem pequena é ruído:** os erros perdiam por menos de 0,01. Processar os prompts todos juntos ou um a um já mudava o placar entre 9 e 10 acertos.
- **O cosseno absoluto não é calibrado:** imagens sem pessoa pontuaram de 0,145 a 0,197, e memes reais de 0,178 a 0,240. Nenhum limite separa as duas faixas; uma "classe nula" que concorra com as demais funcionaria melhor.
- **Escape manual:** `memes.overrides` corrige à mão os erros, e o `index_memes` mostra a margem de cada meme, para indicar onde revisar.
- **Avaliação otimista:** os mesmos 12 memes serviram para escolher o modelo e para medir. Isso precisa ser dito junto com o número.
- **Cache por conteúdo:** os embeddings são invalidados pelo hash dos arquivos mais o nome do modelo. Basta colocar um meme na pasta, e um app que abre sem mudanças não carrega o CLIP (41 s com o modelo, 0,5 s com o cache).

## 8. Experimentos descartados e custo-benefício

- **Classificar pelo meme mais próximo, sem treino:** acurácia de 0,42, contra 0,92. O `neutro` não se separa: a distância mediana foi 8,2 nos frames de pose e 8,5 nos de `neutro`. Um único exemplo por classe, vindo de uma foto, não cobre a variação de várias sessões gravadas.
- **Pose do meme como segunda opinião:** o MLP aplicado à pose de cada meme acertou 6 de 10, e errou em memes diferentes dos que o CLIP errava. Funcionaria como complemento, mas não compensou:
  - aumentaria o acoplamento da indexação;
  - o sinal é fraco;
  - os memes atuais já estavam corrigidos.

  Lição: viável não é o mesmo que necessário.
- **Modelo `lite` de pose:** ganhou só 1 ms, porque o tracking é limitado pelo modelo de mãos, que roda em paralelo e é o mesmo nas duas variantes.

## 9. Desempenho

- **Medir por etapa:** média e p95, descartando os primeiros frames de aquecimento. O p95 mostra os picos que a média esconde.
- **Gargalo:** o tracking ocupa três quartos do frame (~26 ms). Features, classificador e estabilizador somam menos de 1 ms.
- **Teto da câmera:** o app gasta 34,9 ms por frame, para uma câmera de 30 FPS (33 ms). O tempo de captura inclui a espera pelo próximo frame.
- **Otimização do render**, de 7,1 para 1,4 ms em média e de 19,9 para 1,8 ms no p95:
  - **mistura do alpha só quando o meme muda:** era a parte cara, porque o meme com transparência era misturado ao painel a cada frame. Só isso levou o meme transparente de 21,7 para 1,6 ms;
  - **buffers reaproveitados:** a imagem da janela e o frame reduzido são criados uma vez só;
  - **interpolação mais barata:** `INTER_LINEAR` gravando direto no buffer (0,9 ms), no lugar de `INTER_AREA` (5,8 ms).
- **Benchmark justo:** a máquina tem ruído. Rodar as duas versões alternadas no mesmo processo evitou uma conclusão errada.
- **Variação entre execuções:** o tracking varia com a cena, principalmente com quantas mãos estão visíveis. Execuções curtas enganam, e a mais longa é a mais confiável.

## 10. Interface e experiência de uso

- **Layout em duas colunas:** a câmera fica limpa, só com o esqueleto, e um painel ao lado mostra as informações em cima e o meme embaixo.
- **Nunca esticar a imagem:** no Windows, uma janela redimensionável do OpenCV estica a imagem para preencher a janela. A solução foi uma janela de tamanho fixo e a câmera reduzida por igual nos dois eixos para caber na tela.
- **Som de confirmação com boa UX:**
  - um único plim curto, gerado por código: duas notas subindo, com começo suave e fim em silêncio, para não estalar;
  - toca uma vez por aparição, com intervalo mínimo entre toques, para trocas rápidas de pose não virarem uma sequência de plins;
  - toca em segundo plano, sem travar o loop;
  - liga e desliga por tecla, com o estado visível no painel.
- **Demonstração honesta:** o painel do GIF mostra o palpite do modelo, a pose aceita pelo app e o rótulo verdadeiro. Quando a pose some por confiança abaixo do threshold, quem assiste entende o motivo.

## 11. Engenharia de software

- **TDD com dependências reais isoladas:** os testes unitários não usam webcam, `.task` nem o download do CLIP.
  - **Substitutos:** landmarks sintéticos e um encoder falso, injetado no índice de memes.
  - **Testes de integração:** os que usam os modelos reais ficam marcados e pulados por padrão.
- **Config estrito:** o `config.yaml` é a única fonte de valores.
  - As dataclasses não têm valores padrão, e uma chave faltando é erro.
  - Cada valor é validado com uma mensagem que diz o caminho da chave.
- **Sem números mágicos no código:** thresholds e caminhos ficam no config. Valores puramente cosméticos, como cores e o timbre do som, ficam em constantes nomeadas.
- **CI no Windows:** no Linux, o `torch` do PyPI vem com CUDA e ocupa vários GB; no Windows, vem a versão para CPU. Antes de criar o CI, os testes rodaram num clone limpo, para provar que não dependiam de arquivos fora do git.
- **Pegadinhas do Windows:**
  - o `cv2.imread` não abre caminhos com acento (solução: `np.fromfile` mais `cv2.imdecode`);
  - o `winsound` não toca da memória em modo assíncrono (solução: gravar o WAV em arquivo);
  - finais de linha CRLF/LF exigem atenção ao gerar arquivos;
  - no Windows, o OpenCV ignora o "manter proporção" de janelas redimensionáveis.

## 12. Privacidade, direitos e git

- **Metadados escondidos:** figurinhas de WhatsApp carregavam nomes de pacote e de quem criou, e duas fotos tinham blocos do Photoshop. A limpeza removeu só os blocos de metadados, com pixels verificados como idênticos.
- **O histórico guarda tudo:** remover do arquivo atual não basta, porque os commits antigos continuam com o conteúdo.
  - **O que foi feito:** reescrita do histórico trocando cada versão das imagens pela versão limpa, com backup em bundle e verificação de mensagens, autores e datas.
  - **O envio:** `--force-with-lease`, que recusa se o remoto tiver mudanças desconhecidas.
  - **O limite:** o GitHub ainda pode servir commits antigos pelo hash, e forks guardam o histórico antigo.
- **Dados pessoais na documentação:** modelo do processador, caminhos locais e nome de usuário identificam a pessoa. Foram tirados da documentação.
- **Demonstração sem rosto:** o boneco de palito reconstruído das features mostra o projeto sem expor ninguém.
- **Direitos de imagem:** os memes são imagens de terceiros, de figuras públicas e virais (jogadores, cantores, celebridades, um macaco famoso). O README deixa claro que os direitos são dos autores, que o uso é só para demonstração e que as imagens podem ser removidas a pedido.

## 13. Processo e uso de IA

- **Marcos pequenos com critério de pronto:** cada etapa (tracking, features, coleta, treino, CLIP, desempenho) terminava com testes passando, verificação manual e commits separados por natureza.
- **Decisões registradas:** o que não pode mudar sem discussão fica no CLAUDE.md, e o motivo de cada escolha fica no README.
- **Checklist reproduzível para mudanças recorrentes:** a skill `/nova-pose` transforma "adicionar uma pose" num roteiro com comparação automática contra o modelo anterior.
- **Revisar o que a IA afirma:** os números foram conferidos medindo. Uma instrução do assistente de IA causou o problema do `neutro`, ao mudar como gravar a classe no meio da coleta. O erro foi identificado e corrigido com uma sessão nova; nada no código teria resolvido.
