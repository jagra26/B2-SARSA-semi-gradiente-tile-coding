import time
import random
import argparse
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

random.seed(10)

from semi_gradient_sarsa import SemiGradientSarsa


def execute(seed=10):

    env_ = gym.make("MountainCar-v0", render_mode=None)

    sarsa = SemiGradientSarsa(env_)
    sarsa.load_params()

    env = gym.make("MountainCar-v0", render_mode='human')
    state, info = env.reset(seed=seed)

    for i in range(300):
        action = sarsa.select_action(state, eps_greedy=False)
        print(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        # Render the env
        env.render()

        # Wait a bit before the next frame unless you want to see a crazy fast video
        time.sleep(0.001)

        if terminated:
            time.sleep(1)
            break

        state = next_state

    env.close()



def train(alpha, eps, generate_figs=True, save_weights=True, end_steps=True, step_limit=1000):
    # add `.env` at the end to ignore internal truncation
    env = gym.make("MountainCar-v0", render_mode=None)
    sarsa = SemiGradientSarsa(env, num_eps=eps, alpha=alpha)
    run_result = sarsa.train(end_steps=end_steps, step_limit=step_limit,
                                generate_figs=generate_figs, save_weights=save_weights)
    return run_result

def learning_curves_data(runs = 3, eps = 10, alphas = [0.1/8, 0.2/8, 0.5/8]):
    result = []
    alpha_results = []
    for a in alphas:
        for run in range(runs):
            print("RUN: " + str(run))
            run_result = train(a, eps, generate_figs=False, save_weights=False)
            result.append([a, run, run_result])
    for alpha in alphas:
        alpha_results.append(np.zeros(eps))
    for line in result:
        print(line)
        for i in range(len(alphas)):
            if line[0] == alphas[i]:
                alpha_results[i] = np.add(alpha_results[i], np.divide(line[-1], runs))
    return alpha_results

def learning_curves_plot(alpha_results, runs, eps, alpha_labels=["α = 0.1/8", "α = 0.2/8", "α = 0.5/8"], 
                         colors=['#1f4fa3', '#2e8b2e', '#cc2222'], 
                         yticks=[100, 200, 400, 1000]):
    fig, ax = plt.subplots()

    for i in range(len(alpha_results)):
        ax.plot(range(1,eps+1), alpha_results[i], label=alpha_labels[i], color=colors[i])

    ax.set_yscale('log')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_title("Mountain Car")
    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Steps per episode - log scale - averaged over '+ str(runs) + ' runs', fontsize=8)

    ax.legend()

    print(alpha_results)
    plt.tight_layout()
    plt.savefig('../media/Learning_curves.png', dpi=300, bbox_inches='tight')


def create_parser():
    parser = argparse.ArgumentParser(
        description="Executa, treina ou gera os resultados do SARSA semi-gradiente."
    )
    parser.add_argument(
        "--taxa-aprendizado", "--alpha", dest="alpha", type=float, default=0.1 / 8,
        help="Taxa de aprendizado usada pelo modo direto de treinamento.",
    )
    parser.add_argument(
        "--episodios", "--eps", dest="eps", type=int, default=500,
        help="Número de episódios usado pelo modo direto de treinamento.",
    )
    parser.add_argument(
        "--gerar-figuras", dest="generate_figs", action="store_true",
        help="Gera figuras do cost-to-go.",
    )
    parser.add_argument(
        "--sem-gerar-figuras", dest="generate_figs", action="store_false",
        help="Não gera figuras do cost-to-go.",
    )
    parser.add_argument(
        "--salvar-pesos", dest="save_weights", action="store_true",
        help="Salva pesos e tile-coding.",
    )
    parser.add_argument(
        "--sem-salvar-pesos", dest="save_weights", action="store_false",
        help="Não salva pesos nem tile-coding.",
    )
    parser.add_argument(
        "--terminar-ao-limitar-passos", dest="end_steps", action="store_true",
        help="Interrompe episodios ao atingir o limite de passos.",
    )
    parser.add_argument(
        "--sem-terminar-ao-limitar-passos", dest="end_steps", action="store_false",
        help="Não interrompe episódios ao atingir o limite de passos.",
    )
    parser.add_argument(
        "--limite-passos", dest="step_limit", type=int, default=1000,
    )
    parser.add_argument(
        "--semente", "--seed", dest="seed", type=int, default=10,
    )
    parser.set_defaults(
        generate_figs=True,
        save_weights=True,
        end_steps=True,
    )

    subparsers = parser.add_subparsers(dest="command")

    execute_parser = subparsers.add_parser(
        "executar", aliases=["execute"], help="Executa o agente treinado em uma janela renderizada."
    )
    execute_parser.add_argument(
        "--semente", "--seed", dest="seed", type=int, default=10,
        help="Semente usada ao reiniciar o ambiente.",
    )

    train_parser = subparsers.add_parser(
        "treinar", aliases=["train"], help="Treina o agente e opcionalmente salva pesos e figuras."
    )
    train_parser.add_argument(
        "--taxa-aprendizado", "--alpha", dest="alpha", type=float, default=0.1 / 8,
        help="Taxa de aprendizado (padrão: 0.1/8).",
    )
    train_parser.add_argument(
        "--episodios", "--eps", dest="eps", type=int, default=500,
        help="Número de episódios de treinamento.",
    )
    train_parser.add_argument(
        "--gerar-figuras", dest="generate_figs", action="store_true",
        help="Gera figuras do cost-to-go.",
    )
    train_parser.add_argument(
        "--sem-gerar-figuras", dest="generate_figs", action="store_false",
        help="Não gera figuras do cost-to-go.",
    )
    train_parser.add_argument(
        "--salvar-pesos", dest="save_weights", action="store_true",
        help="Salva pesos e tile-coding em arquivos .pkl.",
    )
    train_parser.add_argument(
        "--sem-salvar-pesos", dest="save_weights", action="store_false",
        help="Não salva pesos nem tile-coding.",
    )
    train_parser.add_argument(
        "--terminar-ao-limitar-passos", dest="end_steps", action="store_true",
        help="Interrompe episódios ao atingir o limite de passos.",
    )
    train_parser.add_argument(
        "--sem-terminar-ao-limitar-passos", dest="end_steps", action="store_false",
        help="Não interrompe episódios ao atingir o limite de passos.",
    )
    train_parser.add_argument(
        "--limite-passos", "--step-limit", dest="step_limit", type=int, default=1000,
        help="Limite de passos por episódio.",
    )
    train_parser.set_defaults(
        generate_figs=True,
        save_weights=True,
        end_steps=True,
    )

    curves_parser = subparsers.add_parser(
        "curvas-aprendizado", aliases=["learning-curves"],
        help="Coleta dados e gera o gráfico das curvas de aprendizado.",
    )
    curves_parser.add_argument(
        "--execucoes", "--runs", dest="runs", type=int, default=3,
        help="Número de execuções por valor de taxa de aprendizado.",
    )
    curves_parser.add_argument(
        "--episodios", "--eps", dest="eps", type=int, default=10,
        help="Número de episódios considerado em cada execução.",
    )
    curves_parser.add_argument(
        "--taxas-aprendizado", "--alphas", dest="alphas", nargs="+", type=float,
        default=[0.1 / 8, 0.2 / 8, 0.5 / 8],
        help="Valores de taxa de aprendizado separados por espaço.",
    )
    curves_parser.add_argument(
        "--gerar-grafico", dest="plot", action="store_true",
        help="Gera o gráfico após coletar os dados.",
    )
    curves_parser.add_argument(
        "--sem-gerar-grafico", dest="plot", action="store_false",
        help="Não gera o gráfico após coletar os dados.",
    )
    curves_parser.set_defaults(plot=True)

    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command in ("executar", "execute") or args.command is None:
        execute(seed=args.seed)
    elif args.command in ("treinar", "train"):
        return train(
            alpha=args.alpha,
            eps=args.eps,
            generate_figs=args.generate_figs,
            save_weights=args.save_weights,
            end_steps=args.end_steps,
            step_limit=args.step_limit,
        )
    elif args.command in ("curvas-aprendizado", "learning-curves"):
        alpha_results = learning_curves_data(
            runs=args.runs,
            eps=args.eps,
            alphas=args.alphas,
        )
        if args.plot:
            learning_curves_plot(alpha_results, args.runs, args.eps)
        return alpha_results


if __name__ == "__main__":
    main()
