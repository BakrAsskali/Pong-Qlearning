import game as g
import matplotlib.pylab as plt
import numpy as np
import tkinter as tk


class UserInterface:
    def __init__(self):
        self.root = tk.Tk()
        self.var = tk.IntVar()
        self.selected_game_type = None

    def select_mode(self):
        self.selected_game_type = self.var.get()
        self.root.destroy()

    def run(self):
        tk.Label(self.root, text="Choose game mode:").pack()
        tk.Radiobutton(self.root, text="Agent RL vs Human", variable=self.var, value=1).pack(anchor='w')
        tk.Radiobutton(self.root, text="Agent RL vs Agent AI", variable=self.var, value=2).pack(anchor='w')
        tk.Radiobutton(self.root, text="Agent RL vs Agent RL", variable=self.var, value=3).pack(anchor='w')
        tk.Button(self.root, text="Submit", command=self.select_mode).pack()

        self.root.mainloop()


def plot_agent_reward(rewards):
    """ Function to plot agent's accumulated reward vs. iteration """
    plt.plot(np.cumsum(rewards))
    plt.title('Agent Cumulative Reward vs. Iteration')
    plt.ylabel('Reward')
    plt.xlabel('Episode')
    plt.show()


class GameLearning:
    def __init__(self, game_type, alpha=0.5, gamma=0.9, epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.game = g.Game(game_type)
        self.games_played = 0

    def begin_playing(self, episodes):
        self.game.play(episodes)


if __name__ == '__main__':
    ui = UserInterface()
    ui.run()

    if ui.selected_game_type is not None:
        gl = GameLearning(ui.selected_game_type)
        gl.begin_playing(1000)
    else:
        print("No game type selected.")
