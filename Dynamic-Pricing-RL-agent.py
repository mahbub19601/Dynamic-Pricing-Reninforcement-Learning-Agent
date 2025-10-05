import numpy as np
import matplotlib.pyplot as plt

class MarketEnvironment:
    """
    A simulated market environment for dynamic pricing.
    The environment simulates customer demand based on the price set by the agent.
    Higher prices lead to lower demand.
    """
    def __init__(self):
        # The agent can choose from one of these five price points.
        # These are the "actions" the agent can take.
        self.price_options = [10, 15, 20, 25, 30]
        self.num_actions = len(self.price_options)
        self._base_demand = 100  # Base number of customers interested at a low price.
        self._price_sensitivity = 2.5 # How much demand drops as price increases.

    def _calculate_demand(self, price):
        """
        Calculates the number of sales based on a given price.
        Introduces some randomness to simulate real-world market fluctuations.
        """
        if price <= 0:
            return 0
        
        # A simple demand function: demand decreases as price increases.
        mean_demand = self._base_demand - self._price_sensitivity * price
        
        # Add some random noise to make the simulation more realistic.
        noise = np.random.normal(0, 5) 
        
        # Demand cannot be negative.
        demand = max(0, mean_demand + noise)
        return int(demand)

    def step(self, action_index):
        """
        Executes one time step in the environment.
        1. The agent chooses an action (a price index).
        2. The environment calculates the demand for that price.
        3. The environment returns the reward (revenue).
        """
        if not (0 <= action_index < self.num_actions):
            raise ValueError(f"Invalid action index: {action_index}")

        # Get the actual price from the index.
        price = self.price_options[action_index]
        
        # Calculate how many units were sold at this price.
        demand = self._calculate_demand(price)
        
        # The reward is the total revenue from sales in this step.
        reward = price * demand
        
        # In this simple model, the state doesn't change, so we return a dummy value.
        # This setup is known as a "multi-armed bandit" problem.
        next_state = 0
        
        return next_state, reward

class QLearningAgent:
    """
    A reinforcement learning agent that uses Q-learning to find the optimal price.
    """
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995

        # The Q-table stores the agent's learned values for each action.
        # It's initialized to zeros. The agent's goal is to update this table
        # so it reflects the true long-term reward of each action.
        self.q_table = np.zeros(num_actions)

    def choose_action(self):
        """
        Decides on an action using the epsilon-greedy strategy.
        - With probability epsilon, it explores by choosing a random action.
        - Otherwise, it exploits its current knowledge by choosing the best-known action.
        """
        if np.random.random() < self.epsilon:
            # Explore: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_table)

    def update_q_table(self, action, reward):
        """
        Updates the Q-value for a given action using the Bellman equation.
        This is the core of the Q-learning algorithm.
        """
        # For a stateless (bandit) problem, the Q-learning update rule simplifies.
        # We move the current Q-value towards the newly observed reward.
        old_value = self.q_table[action]
        new_value = reward 
        
        self.q_table[action] = old_value + self.lr * (new_value - old_value)

    def decay_epsilon(self):
        """
        Decreases the exploration rate over time.
        As the agent learns more, it should explore less and exploit more.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def train_agent():
    """
    Main training loop for the reinforcement learning agent.
    """
    num_episodes = 1000
    
    env = MarketEnvironment()
    agent = QLearningAgent(num_actions=env.num_actions)
    
    rewards_history = []
    
    print("--- Starting Training ---")
    for episode in range(num_episodes):
        # Agent chooses a price (action).
        action = agent.choose_action()
        
        # Environment responds with the result of that action.
        _, reward = env.step(action)
        
        # Agent learns from the experience.
        agent.update_q_table(action, reward)
        
        # Reduce exploration rate.
        agent.decay_epsilon()
        
        # Log results for plotting.
        rewards_history.append(reward)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Epsilon: {agent.epsilon:.3f}")

    print("\n--- Training Finished ---")
    
    # Final Results
    print("\nLearned Q-values (expected revenue for each price):")
    for i, price in enumerate(env.price_options):
        print(f"  Price: ${price} -> Expected Revenue: ${agent.q_table[i]:.2f}")
        
    best_action_index = np.argmax(agent.q_table)
    best_price = env.price_options[best_action_index]
    print(f"\nOptimal price found by the agent: ${best_price}")

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Revenue ($)')
    plt.grid(True)
    
    # Calculate a moving average to show the trend
    moving_avg = np.convolve(rewards_history, np.ones(50)/50, mode='valid')
    plt.plot(moving_avg, linewidth=3, color='red', label='50-episode Moving Average')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    train_agent()
