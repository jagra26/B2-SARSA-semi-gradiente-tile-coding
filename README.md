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

Por fim, como a política só é avaliada ao final de cada episódio, e não ao final de cada passo no espaço de estados, trata-se de um algoritmo episódico.

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

## Experimento: <a pergunta></a>

<figura, o que ela mostra, por que era esse o resultado esperado>

## Limites

<onde falha; qual método resolve isso>

## Referências

* Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). Bradford Books. Cap. 9, 10. Sec. 9.5.4, 10.1.
* Jabrah Tutorials (2023). Reinforcement Learning. mountain_car. [github.com/JabrahTutorials/ReinforcementLearning.git](https://github.com/JabrahTutorials/ReinforcementLearning.git) , [www.youtube.com/watch?v=Lphlyuvocz0](https://www.youtube.com/watch?v=Lphlyuvocz0)
* O'neill, M. (2018). Reinforcement Learning Tutorial: Semi-gradient n-step Sarsa and Sarsa(λ) Theory and Implementation. [michaeloneill.github.io/RL-tutorial.html](https://michaeloneill.github.io/RL-tutorial.html)

A principal referência foi Sutton & Barto, que também é referência para as demais, a base teórica e figuras a serem reproduzidas foram tiradas dessa fonte. A biblioteca de tile coding é a mesma desenvolvida por Sutton.

Jabrah Tutorials tem uma aula muito boa sobre o assunto e foi a principal referência em código. Contudo, algumas mudanças significativas foram feitas, principalmente para o ajuste do tile coding e geração de figuras.

O'Neill tem uma forma muito didática de explicar, além de fazer muitas comparações com métodos mais tradicionais. Também possui uma implementação do problema, que considerei mais complexa que a de Jabrah. Porém, sua visualização da figura cost to go, foi útil para a criação das figuras desse trabalho.

<bibliografia com capítulo e seção; artigos; implementações consultadas>
<divergências entre fontes e qual versão foi seguida>
