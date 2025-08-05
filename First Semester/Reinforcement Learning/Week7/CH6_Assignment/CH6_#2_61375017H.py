import numpy as np
import matplotlib.pyplot as plt

reward_matrix = np.array([
    # A   B   C   D   E   F
    [-1, -5, -5, -5, 10, -5], # A (state 0)
    [-5, -1, -1, -3, -5, 10], # B (state 1)
    [-5, 10, -1, -3, -5, -5], # C (state 2)
    [-5, -10, 10, -1, -3, -5], # D (state 3)
    [-3, -5, -5, 10, -1, -3], # E (state 4)
    [-5, -3, -5, -5, -3, -1], # F (state 5)
])

# reward_matrix = np.array([
#     # A   B   C   D   E   F
#     [-1, -5, -5, -5, 50, -5], # A (state 0)
#     [-5, -1, -1, -3, -5, 50], # B (state 1)
#     [-5, 50, -1, -3, -5, -5], # C (state 2)
#     [-5, -3, 50, -1, -3, -5], # D (state 3)
#     [-3, -5, -5, 50, -1, -3], # E (state 4)
#     [-5, -3, -5, -5, -3, 50], # F (state 5)
# ])

valid_actions = {
    0: [4],         # A
    1: [2, 3, 5],   # B
    2: [1, 3],      # C
    3: [1, 4],      # D
    4: [0, 3, 5],   # E
    5: [1, 4],      # F
}

# Global variables for current state (robot starts in Room C, state 2)
current_state = 0
visited_B = False
visited_C = False
visited_D = False

# Reset the environment (set the robot back to Room C)
def reset_environment():
    global current_state, visited_C, visited_B
    current_state = 0  # Reset to Room A
    visited_C = False  # Reset the flag
    visited_B = False
    visited_D = False
    return current_state

# Perform a step in the environment
def step(action):
    global current_state, visited_C, visited_B, visited_D
    reward = reward_matrix[current_state, action]  # Get reward for the action

    if current_state == 2:
        visited_C = True
    if current_state == 1:
        visited_B = True
    if current_state == 3:
        visited_D = True
    
    current_state = action  # Move to the next state (room)
    if current_state == 1 and visited_D and not visited_C:
        current_state = 2
        reward = -1
    elif current_state == 1 and visited_C:
        current_state = 5
        reward = 50
    
    if current_state == 5 and visited_D and visited_C and visited_B:
        reward = 50
    elif current_state == 5 and visited_D and visited_C and not visited_B:
        current_state = 1
        reward = -1

    done = (current_state == 5 and visited_C)  # If reached F with reward 50, done
    return current_state, reward, done

# Q-Learning algorithm without using class
def q_learning(alpha=0.1, gamma=0.9, epsilon=0.01, episodes=5000, max_steps=100):
    q_table = np.zeros((6, 6))  # Initialize Q-table for 6 states (rooms) and 6 possible actions
    rewards_per_episode = []
    max_q_per_episode = []

    for episode in range(episodes):
        state = reset_environment()  # Start each episode from Room C
        total_reward = 0

        for step_count in range(max_steps):
            # Epsilon-greedy action selection
            if np.random.randn() > epsilon:
                # Exploit: choose the action with the highest Q-value for the current state, but it must be valid
                valid_q_values = [q_table[state, a] for a in valid_actions[state]]
                action = valid_actions[state][np.argmax(valid_q_values)]
            else:
                # Explore: choose a random valid action
                action = np.random.choice(valid_actions[state])

            # Take the action and observe the reward and the next state
            next_state, reward, done = step(action)
            total_reward += reward

            # Q-Learning update rule
            best_next_action = np.argmax([q_table[next_state, a] for a in valid_actions[next_state]])  # Best action for the next state
            q_table[state, action] += alpha * (
                reward + gamma * q_table[next_state, valid_actions[next_state][best_next_action]] - q_table[state, action]
            )

            state = next_state  # Move to the next state

            if done:
                break
        
        rewards_per_episode.append(total_reward)
        max_q_per_episode.append(np.max(q_table))  # Track the max Q-value to observe convergence

    return q_table, rewards_per_episode, max_q_per_episode

# Plotting Q-value convergence
def plot_convergence(max_q_values):
    plt.plot(max_q_values)
    plt.xlabel("Episodes")
    plt.ylabel("Max Q-value")
    plt.title("Q-value Convergence Over Episodes")
    plt.grid(True)
    plt.show()

# Testing the learned policy (test the agent)
def test_agent(q_table, max_steps=100):
    state = reset_environment()  # Start from Room C (state 2)
    print(f"Starting at Room A (State {state})")

    steps = 0
    while True:
        # Choose the best valid action based on the learned Q-table
        valid_q_values = [q_table[state, a] for a in valid_actions[state]]
        action = valid_actions[state][np.argmax(valid_q_values)]

        next_state, reward, done = step(action)
        print(f"Step {steps + 1}: Moved to Room {chr(65 + next_state)} (State {next_state}), Reward: {reward}")
        state = next_state
        steps += 1
        if done or steps >= max_steps:
            break
    print(f"Finished after {steps} steps.")

if __name__ == "__main__":
    # Train the Q-learning agent
    q_table, rewards, max_q_values = q_learning()

    print(q_table)
    # Plot Q-value convergence
    plot_convergence(max_q_values)

    # Test the trained agent
    test_agent(q_table)