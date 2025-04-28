from CircuitEnv import CircuitEnv
from GNNPolicyValueNet import GNNPolicyValueNet
from MCTSAgent import MCTSAgent
from SelfPlay import self_play_episode
from generator import generate_random_polynomials
from Trainer import Trainer
from Evaluate import evaluate_agent


def main():
    # 1. Create monomial basis
    _, all_monomials = generate_random_polynomials(n=5, m=6, C=4, num_polynomials=1)

    # 2. Initialize random starting vectors
    init_vectors, _ = generate_random_polynomials(n=5, m=6, C=4, num_polynomials=2)

    # 3. Target polynomial
    target_vector, _ = generate_random_polynomials(n=5, m=6, C=6, num_polynomials=1)
    target_vector = target_vector[0]

    # 4. Environment and Agent
    model = GNNPolicyValueNet(in_dim=len(all_monomials), hidden_dim=128)
    agent = MCTSAgent(model, simulations_per_move=30)
    env = CircuitEnv(all_monomials, target_vector)

    # 5. Self-play: collect many experiences
    all_experiences = []
    num_episodes = 100  # can adjust this

    for episode in range(num_episodes):
        experiences = self_play_episode(env, agent, init_vectors)
        all_experiences.extend(experiences)

        if (episode + 1) % 10 == 0:
            print(f"Completed {episode+1}/{num_episodes} episodes.")

    print(f"Collected {len(all_experiences)} experiences.")

    # 6. Train the model
    trainer = Trainer(model)
    trainer.train(all_experiences, batch_size=8, epochs=10)

    print("Training complete.")


    # 7. Evaluation
    print("Starting evaluation...")
    evaluate_agent(agent, all_monomials, target_vector, init_vectors, num_episodes=100, max_steps=10)


if __name__ == "__main__":
    main()
