# Sarsa semi-gradiente com Tile-coding

Autoria: João Arthur Gaia da Rocha Almeida · 2026.2 · PPGI071 Aprendizado por Reforço (UFAL)
Referência: Sutton & Barto (2020), § 10.1 / Lapan (2024), cap. Z

## Motivação
Imagine o cenário em que se tem um espaço de estados contínuo, como por exemplo a velocidade e posição de um carro.
ou ainda que o espaço de estados seja discreto, mas com uma alta dimensionalidade, como por exemplo o gerenciamento
de um computador, onde voga o estado de cada núcleo do processador, da RAM, do cache, consumo energético, entradas e saídas,
gerenciamento de arquivos... e por aí vai.
Representar esses cenários é computacionalmente custoso, principalmente para métodos tabulares tradicionais, como ação valor,
bandit e implementações incrementais. Uma alternativa viável é SARSA semi-gradient com tile-coding.
Para fins didáticos, utilizarei o acrônimo SARSA-SG-TC para me referir ao algoritmo, ressalto que nenhuma das referências utilizadas
usa esse termo.

## Onde isto se encaixa
### Classificação
Justamente por não ser um algoritmo tabular, o SARSA-SG-TC se sai bem nesses ambientes contínuos ou complexos. Ele faz isso pois
aproxima o cenário, não o representando por uma tabela, e sim com uma função, o que faz com que seja um algoritmo de aproximação.
Também trata-se de um método on-policy, visto que trabalha com uma única política que vai sendo melhorada durante a execução, diferente de estratégias
off-policy, que utilizam duas políticas, uma fortemente melhorada e outra exploratória.
Por fim, por não se basear em nenhum modelo que calcule probabilidades de transição e recompensa, e sim agir diretamente no ambiente ele é um algoritmo model-free.
<classificação: tabular/aproximação, model-free/based, on/off-policy,
 predição/controle, episódico/contínuo — cada uma com justificativa>
<genealogia: de qual método da disciplina descende e o que muda na atualização>
<tabela de símbolos: notação da fonte → notação da disciplina>
<hipóteses: o que o método assume e o que quebra quando a hipótese cai>

## A ideia
<em palavras, antes de qualquer símbolo>

## Formalização
<derivação passo a passo, na notação da disciplina>

## Pseudocódigo
<caixa no estilo do livro>

## Implementação
<as decisões que o pseudocódigo esconde>

## Experimento: <a pergunta>
<figura, o que ela mostra, por que era esse o resultado esperado>

## Limites
<onde falha; qual método resolve isso>

## Referências
<bibliografia com capítulo e seção; artigos; implementações consultadas>
<divergências entre fontes e qual versão foi seguida>
