def self_play_episode(env, agent, init_vectors):
    """
    Play one episode and collect experiences.
    """
    graph = env.reset(init_vectors)
    experiences = []

    done = False
    while not done:
        action = agent.select_action(graph)
        graph, reward, done = env.step(action)
        experiences.append((graph, action, reward))

    return experiences
