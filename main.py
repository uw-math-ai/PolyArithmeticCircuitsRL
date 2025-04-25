from net import *
from gen import *
from data_loader import *

# index_to_monomial, polynomials = generate_random_polynomials(n=10, m=5, C=5, i=3)
# save_polynomial_data(index_to_monomial, polynomials, "poly_data_10_5_5", format="pickle")

# Load saved data
with open("poly_data_10_5_5.pkl", "rb") as f:
    data = pickle.load(f)

# Train the model
model, trainer = train_model(data["index_to_monomial"], data["polynomials"], n=10, m=5, C=5)


# state = create_random_problem(data["index_to_monomial"], data["polynomials"])

# solution_state = trainer.solve_problem(state)

# print(solution_state.operation_history)

# term_size = len(data["polynomials"][0])
# max_terms = 10

# # Max number of terms we can have after C operations
# max_possible_terms = 10  # Initial variables + constant + C new terms
# action_size = get_action_space_size(max_possible_terms)
    

# model = PolynomialCircuitNet(term_size, max_terms, action_size)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Create trainer
# trainer = AlphaZeroTrainer(model, optimizer, mcts_simulations=50)

# trainer.load_model("model_episode_100.pt")

# def create_fn():
#     return create_random_problem(data["index_to_monomial"], data["polynomials"], 5,5,5)

# avg_value, solution_rate = trainer.evaluate(create_fn, 30)

# print(avg_value)
# print(solution_rate)


