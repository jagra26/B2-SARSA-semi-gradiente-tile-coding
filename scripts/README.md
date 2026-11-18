# Uso do programa pela linha de comando

Execute os comandos a partir da pasta [scripts](scripts)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py --help
```

## Treinamento direto

Sem informar um subcomando, o programa inicia a execução com os valores padrão:

```bash
python3 main.py
```

Opções disponíveis no modo direto:

| Argumento | Função | Padrão |
|---|---|---:|
| `--taxa-aprendizado VALOR` | Define a taxa de aprendizado. | `0.0125` |
| `--episodios QUANTIDADE` | Define o número de episódios. | `500` |
| `--gerar-figuras` | Solicita a geração das figuras do cost-to-go. | ativado |
| `--sem-gerar-figuras` | Desativa a geração das figuras. | - |
| `--salvar-pesos` | Solicita o salvamento dos pesos e do tile-coding. | ativado |
| `--sem-salvar-pesos` | Desativa o salvamento dos pesos. | - |
| `--terminar-ao-limitar-passos` | Termina um episódio ao atingir o limite de passos. | ativado |
| `--sem-terminar-ao-limitar-passos` | Permite continuar além do limite configurado. | - |
| `--limite-passos QUANTIDADE` | Define o limite de passos por episódio. | `1000` |
| `--semente NUMERO` | Define a semente do modo direto. | `10` |

Exemplo:

```bash
python3 main.py \
  --taxa-aprendizado 0.0625 \
  --episodios 9000 \
  --sem-gerar-figuras \
  --limite-passos 1000
```

## Executar o agente treinado

Utilize esse comando para visualizar o modelo treinado resolvendo o problema proposto.

```bash
python3 main.py executar --semente 42
```

Argumentos:

| Argumento | Função | Padrão |
|---|---|---:|
| `--semente NUMERO` | Semente usada ao reiniciar o ambiente. | `10` |

## Treinar explicitamente

Utilize esse comando para um novo treinamento, com a flag "--gerar-figuras" reproduz-se a figura 10.1 de Sutton e Barto. Use "--salvar-pesos" para guardar os resultados do treinamento, sobrescrevendo os anteriores. 

```bash
python3 main.py treinar \
  --taxa-aprendizado 0.0625 \
  --episodios 9000 \
  --gerar-figuras \
  --salvar-pesos \
  --sem-terminar-ao-limitar-passos \
  --limite-passos 1000
```

| Argumento | Função | Padrão |
|---|---|---:|
| `--taxa-aprendizado VALOR` | Taxa de aprendizado. | `0.0125` |
| `--episodios QUANTIDADE` | Quantidade de episódios de treinamento. | `500` |
| `--gerar-figuras` / `--sem-gerar-figuras` | Ativa ou desativa figuras do cost-to-go. | ativado |
| `--salvar-pesos` / `--sem-salvar-pesos` | Ativa ou desativa os arquivos `weights.pkl` e `tilings.pkl`. | ativado |
| `--terminar-ao-limitar-passos` / `--sem-terminar-ao-limitar-passos` | Controla a interrupção pelo limite de passos. | ativado |
| `--limite-passos QUANTIDADE` | Máximo de passos por episódio. | `1000` |

## Gerar curvas de aprendizado

Utilize esse comando para gerar a figura 10.2 de Sutton e Barto, 

```bash
python3 main.py curvas-aprendizado \
  --execucoes 100 \
  --episodios 500 \
  --taxas-aprendizado 0.0125 0.025 0.0625 \
  --gerar-grafico
```

| Argumento | Função | Padrão |
|---|---|---:|
| `--execucoes QUANTIDADE` | Repetições de treinamento para cada taxa. | `3` |
| `--episodios QUANTIDADE` | Episódios em cada execução. | `10` |
| `--taxas-aprendizado VALORES` | Lista de taxas separadas por espaços. | `0.0125 0.025 0.0625` |
| `--gerar-grafico` / `--sem-gerar-grafico` | Gera ou não o gráfico final. | ativado |

Para consultar a ajuda completa ou a ajuda de um subcomando:

```bash
python3 main.py --help
python3 main.py treinar --help
python3 main.py curvas-aprendizado --help
```
