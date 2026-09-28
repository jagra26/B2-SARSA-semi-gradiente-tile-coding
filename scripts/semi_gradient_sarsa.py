import pickle
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

random.seed(18)

from tile_coding import *


class SemiGradientSarsa(object):
    def __init__(
        self,
        env,
        alpha=0.01/8, 
        eps=0.1,
        gamma=1,
        n_tilings = 8,
        num_eps=100) -> None:
        self.alpha = alpha
        self.eps = eps
        self.gamma = gamma
        self.num_eps = num_eps
        self.n_tilings = n_tilings
        self.env = env
        self.plot_episodes = {1, 12, 104, 500, 1000, 9000}
        # escala para o tile coding
        obs_low = env.observation_space.low
        obs_high = env.observation_space.high
        # queremos ~8-16 tiles por dimensão
        self.scale = 8.0 / (obs_high - obs_low)   # array de shape (2,)

        # IHT generoso
        self.tile_coding = IHT(4096 * 8)
        self.w = np.zeros(self.tile_coding.size)

    def q_func(self, feature_vector):
        return np.dot(self.w, feature_vector)

    def update_weight(self, reward, current_q, future_q, feature_vector, terminal):
        if terminal:
            w_update = self.alpha * (reward - current_q)
        else:
            w_update = self.alpha * (reward + self.gamma *future_q - current_q)
        self.w += np.multiply(w_update, feature_vector)


    def train(self, generate_figs=True, save_weights=True, end_steps = True, step_limit = 1000):
        state, info = self.env.reset()
        action, q = self.select_action(state)
        episodes = 0
        steps = 0
        results = []

        total_reward = 0

        while episodes < self.num_eps:
            steps += 1

            feature_vec = self.hash_feature_vector(state, action)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or (end_steps and steps == step_limit):
                results.append(steps)
                if episodes % 50 == 0:
                    print("episode:", episodes, 'completed', 'reward:', total_reward)
                
                self.update_weight(reward, q, None, feature_vec, True)
                state, info = self.env.reset()
                action, q = self.select_action(state)
                total_reward = 0
                steps = 0
                episodes += 1
                if episodes in self.plot_episodes and generate_figs:
                    self.plot_cost_to_go(episode=episodes)

                continue

            next_action, next_q = self.select_action(next_state)
            self.update_weight(reward, q, next_q, feature_vec, False)
            state = next_state
            action = next_action
            q = next_q

        if save_weights:    
            self.save_params()
        return results
        
    
    def save_params(self):
        print(self.w)
        pickle.dump(self.w, open('weights.pkl', 'wb'))
        pickle.dump(self.tile_coding, open('tilings.pkl', 'wb'))
    
    def load_params(self):
        self.w = pickle.load(open('weights.pkl', 'rb'))
        self.tile_coding = pickle.load(open('tilings.pkl', 'rb'))

    def one_hot_encode(self, indices):
        size = len(self.w)
        one_hot_vec = np.zeros(size)
        for i in indices:
            one_hot_vec[i] = 1
        return one_hot_vec

    def hash_feature_vector(self, state, action):
        # speed you up
        scaled = np.asarray(state) * self.scale
        feature_ind = np.array(tiles(self.tile_coding, self.n_tilings, scaled.tolist(), [action]))
        feature_vec = self.one_hot_encode(feature_ind)
        return feature_vec

    def select_action(self, state, eps_greedy = True):
        num_actions = self.env.action_space.n
        actions = range(num_actions)
        action_val_dict = {}
        for action in actions:
            feature_vector = self.hash_feature_vector(state, action)
            q_val = self.q_func(np.array(feature_vector))

            action_val_dict[action] = q_val
        
        greedy_action = max(action_val_dict, key=action_val_dict.get)
        
        if not eps_greedy:
            return greedy_action

        non_greedy_actions = list(set(range(num_actions)) - {greedy_action})
        
        prob_explorative_action = self.eps / num_actions
        prob_greedy_action = 1 - self.eps + prob_explorative_action

        action = np.random.choice([greedy_action] + non_greedy_actions,
                    p=[prob_greedy_action]+[prob_explorative_action for _ in range(len(non_greedy_actions))])
        return action, action_val_dict[action]

    def predict(self, state):
        """Retorna q̂(s, a, w) para todas as ações. state é um array-like de shape (2,)."""
        num_actions = self.env.action_space.n
        q_values = np.zeros(num_actions)
        for action in range(num_actions):
            feature_vector = self.hash_feature_vector(np.asarray(state), action)
            q_values[action] = self.q_func(np.array(feature_vector))
        return q_values

    def plot_cost_to_go(self, episode=None, n=60, save_dir='../media'):
        low = self.env.observation_space.low
        high = self.env.observation_space.high

        pos = np.linspace(low[0], high[0], n)
        vel = np.linspace(low[1], high[1], n)
        X, Y = np.meshgrid(pos, vel)

        states = np.stack([X, Y], axis=2).reshape(-1, 2)
        q_all = np.array([self.predict(s) for s in states])
        Z = -np.max(q_all, axis=1).reshape(X.shape)

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')

        # --- estilo Sutton: face branca + grade preta ---
        ax.plot_surface(
            X, Y, Z,
            rstride=2, cstride=2,        # controla densidade das linhas
            color='white',               # face branca
            edgecolor='black',           # arestas pretas
            linewidth=0.3,
            antialiased=True,
            shade=False,                 # sem sombreamento -> visual "chapado"
        )

        # rótulos e limites como no livro
        ax.set_xlabel('Position', labelpad=8)
        ax.set_ylabel('Velocity', labelpad=8)
        ax.set_zlabel('cost-to-go', labelpad=8)

        # remove o fundo cinza padrão
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')

        ax.grid(True)
        ax.set_title(f'Episode {episode}' if episode is not None else 'Cost-to-go')

        fname = f'{save_dir}/cost_to_go_ep{episode}.png' if episode is not None \
                else f'{save_dir}/cost_to_go.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[plot] salvo em {fname}')