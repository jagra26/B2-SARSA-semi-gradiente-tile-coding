# Sarsa semi-gradiente com Tile-coding

Autoria: João Arthur Gaia da Rocha Almeida · 2026.2 · PPGI071 Aprendizado por Reforço (UFAL)

Referência: Sutton & Barto (2020), § 10.1 / Lapan (2024), cap. Z

## Motivação
Imagine o cenário em que se tem um espaço de estados contínuo, como por exemplo a velocidade e posição de um carro.
Ou ainda que o espaço de estados seja discreto, mas com uma alta dimensionalidade, como por exemplo o gerenciamento
de um computador, onde voga o estado de cada núcleo do processador, da RAM, do cache, consumo energético, entradas e saídas,
gerenciamento de arquivos...

Representar esses cenários é computacionalmente custoso, principalmente para métodos tabulares tradicionais, como ação valor,
bandit e implementações incrementais. Uma alternativa viável é SARSA semi-gradient com tile-coding.

Para fins didáticos, utilizarei o acrônimo SARSA-SG-TC para me referir ao algoritmo, ressalto que nenhuma das referências utilizadas
usa esse termo.

## Onde isto se encaixa
### Classificação
Justamente por não ser um algoritmo tabular, o SARSA-SG-TC se sai bem nesses ambientes contínuos ou complexos. Ele faz isso pois
mapeia sua política com uma função e não com uma tabela, o que o classifica como uma estratégia de aproximação.

E, por não agir diretamente no ambiente, e não em um modelo, o SARSA-SG-TC é model-free.

Também trata-se de um método on-policy, visto que o aprendizado dele se dá direto com o resultado das ações no ambiente. Sem nenhum tipo de política prévia.



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
