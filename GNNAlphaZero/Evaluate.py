from CircuitEnv import CircuitEnv
from SelfPlay import self_play_episode

def evaluate_agent(agent, all_monomials, target_vector, init_vectors, num_episodes=100, max_steps=10):
    successes = 0
    total_steps = 0

    for episode in range(num_episodes):
        env = CircuitEnv(all_monomials, target_vector, max_steps=max_steps)
        graph = env.reset(init_vectors)

        done = False
        steps = 0

        while not done:
            action = agent.select_action(graph)
            graph, reward, done = env.step(action)
            steps += 1

            # Optional: early stopping if already matches
            if reward >= 0.99:
                done = True

            if steps >= max_steps:
                done = True

        if reward >= 0.99:
            successes += 1
            total_steps += steps  # only count steps if success

    success_rate = successes / num_episodes * 100
    avg_steps = total_steps / successes if successes > 0 else float('inf')

    print(f"Evaluation Results:")
    print(f"Success Rate: {success_rate:.2f}% ({successes}/{num_episodes})")
    print(f"Average Steps (only successes): {avg_steps:.2f}")
