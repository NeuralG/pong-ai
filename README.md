# pong-ai

A classic Pong game implemented with Pygame and an agent trained using Deep Q-Learning (DQL) with PyTorch.

This project provides a complete environment and agent for learning the optimal gameplay to play the game.

## Installation

1.  Clone the repo:

    ```bash
    git clone https://github.com/NeuralG/pong-ai.git
    cd pong-ai
    ```

2.  Install dependencies:

    ```bash
    pip install pygame torch numpy matplotlib
    ```

## Usage

You can run the project in human mode or agent mode.

### Agent Mode (AI)

To start with agent mode run the following command or run `main.py`. AI will learn from its gameplay and will save the state/weights once per 50 games. It will save it on `/model` on working directory by default.

```bash
python main.py
```

### Human Mode (Debugging)

To start with human mode where you play the game instead, run the following command or run `game.py`. It is useful for debugging the environment. Use left/right arrow to play.

```bash
python game.py
```

## The tech stacks I used

-   Environment: **Pygame**
-   Agent Logic: **Deep Q-Learning (DQL)**
-   Model: **PyTorch** with **optim.Adam** and **nn.MSELoss**
-   Persistence: **Pickle** and **torch.save** for saving model weights and agent state.
-   Plotting: **Matplotlib**

## Parameters

-   gamma = 0.99
-   max_memory = 250_000
-   batch_size = 1500
-   learning_rate = 1e-4
-   decay_rate = 0.005
-   $\epsilon_{min} = 0.01$
-   $\epsilon_{start} = 1.00$

Epsilon decays exponentially with the following formula:

$$\epsilon_t = \epsilon_{\min} + (\epsilon_{\text{start}} - \epsilon_{\min}) \cdot e^{-D \cdot N}$$
