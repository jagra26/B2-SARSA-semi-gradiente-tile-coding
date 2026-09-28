import gymnasium as gym
from semi_gradient_sarsa import SemiGradientSarsa
import numpy as np
import matplotlib.pyplot as plt


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
    plt.savefig('../media/Learning curves.png', dpi=300, bbox_inches='tight')
eps = 500
runs = 100
learning_curves_plot(learning_curves_data(runs = runs, eps=eps), runs, eps)

train(0.5/8, 9000, end_steps=False)