import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import random
from collections import deque
import pickle
import time

# State Representation for Polynomial Arithmetic Circuits
class CircuitState:
    def __init__(self, index_to_monomial, target_polynomial, variables, max_complexity, m):
        """
            variables: List of initial variables (as polynomial vectors)
        """
        self.index_to_monomial = index_to_monomial
        self.target_polynomial = target_polynomial
        self.max_complexity = max_complexity
        self.m = m  
        self.monomial_to_index = {monomial: idx for idx, monomial in index_to_monomial.items()}


        self.available_terms = variables.copy()
        self.operations_used = 0
        self.operation_history = []
        
        self.num_terms = len(variables)  # Track how many terms we have
    
    def apply_operation(self, op_type, term1_idx, term2_idx):
        """
            op_type: 0 for add, 1 for multiply 2 for subtract
            term1_idx, term2_idx: Indices of terms to operate on
            
        Returns:
            New state after applying the operation
        """
        
        new_state = CircuitState(
            self.index_to_monomial,
            self.target_polynomial,
            self.available_terms[:self.num_terms],  # Only pass original variables
            self.max_complexity,
            self.m
        )
        
        # Copy the current state
        new_state.available_terms = self.available_terms.copy()
        new_state.operations_used = self.operations_used + 1
        new_state.operation_history = self.operation_history.copy()
        new_state.num_terms = self.num_terms
        
        # Get the terms to operate on
        term1 = self.available_terms[term1_idx]
        term2 = self.available_terms[term2_idx]
        
        # Apply the operation
        if op_type == 0:  # add
            result = add_polynomials(term1, term2, 2)
        elif op_type == 1:  # multiply
            result = multiply_polynomials(term1, term2, self.index_to_monomial, self.monomial_to_index, 2)
        elif op_type == 2:  # subtract
            result = subtract_polynomials(term1, term2, 2)
        
        if result is None:
            # This can happen if the degree exceeds n
            new_state.available_terms.append([0]*len(self.target_polynomial)) # I add f(x)=0
            new_state.num_terms += 1
            new_state.operation_history.append((op_type, term1_idx, term2_idx))
            return new_state
        else:
            # Add the result to available terms
            new_state.available_terms.append(result)
            new_state.num_terms += 1
            new_state.operation_history.append((op_type, term1_idx, term2_idx))
            return new_state
    
    def encode_for_nn(self):
        """
        Encode the state for input to the neural network.
        
        Returns:
            Tensor representation of the state
        """
        # Convert polynomial vectors to fixed-size representation
        max_poly_size = len(self.target_polynomial)
        
        # Fixed size (m + C)
        max_terms = self.m + self.max_complexity  
        
        term_vectors = []
        # Add all available terms (m initial and i-1 intermediary)
        for term in self.available_terms:
            if len(term) < max_poly_size:
                padded = term + [0] * (max_poly_size - len(term))
            else:
                padded = term[:max_poly_size]
            term_vectors.append(padded)
        

        # Pad to exactly m + C terms
        while len(term_vectors) < max_terms:
            term_vectors.append([0] * max_poly_size)
        
        if term_vectors > max_terms:
            print("error: too many terms")

        # 2. Encode target polynomial
        target_vector = self.target_polynomial
        if len(target_vector) < max_poly_size:
            target_vector = target_vector + [0] * (max_poly_size - len(target_vector))
        else:
            target_vector = target_vector[:max_poly_size]
        
        # 3. Encode complexity information
        complexity_vector = [
            self.operations_used / self.max_complexity,  # Used complexity (normalized)
            (self.max_complexity - self.operations_used) / self.max_complexity  # Remaining complexity
        ]
        
        # Combine into a single tensor
        state_tensor = {
            'terms': np.array(term_vectors, dtype=np.float32),
            'target': np.array(target_vector, dtype=np.float32),
            'complexity': np.array(complexity_vector, dtype=np.float32)
        }
        
        return state_tensor
    
    def get_valid_actions(self):
        """
        Get all valid actions from the current state.
        
        Returns:
            List of valid (op_type, term1_idx, term2_idx) actions
        """
        if self.operations_used >= self.max_complexity:
            return []
            
        valid_actions = []
        
        # Consider all pairs of available terms
        for i in range(len(self.available_terms)):
            for j in range(i, len(self.available_terms)):  # j >= i to avoid duplicates
                # Add operation
                valid_actions.append((0, i, j))
                
                # Multiply operation
                valid_actions.append((1, i, j))
            
            # subtraction (i-j and j-i are different)
            valid_actions.append((2, i, j))
        
        return valid_actions
    
    def is_solution(self):
        """Check if current state contains the target polynomial"""
        for term in self.available_terms:
            if polynomials_equal(term, self.target_polynomial):
                return True
        return False
    
    def evaluate(self):
        """
        Evaluate the current state.
        
        Returns:
            Value between -1 and 1. Higher is better.
        """
        if self.is_solution():
            return 1.0 - 0.5 * (self.operations_used / self.max_complexity) # Reward for using fewer operations
        
        # haven't found the solution but used all operations
        if self.operations_used >= self.max_complexity:
            return -1.0
        
        # Otherwise, evaluate based on how close we are to the target
        closest_distance = polynomial_distance(self.available_terms[-1], self.target_polynomial)
        
        # Normalize distance between -1 and 0
        max_possible_distance = max(sum(abs(x) for x in self.target_polynomial), 1)
        normalized_distance = -closest_distance / max_possible_distance
        
        return normalized_distance * 0.9  # so that it is always less than finding a solution

class PolynomialCircuitNet(nn.Module):
    def __init__(self, term_size, max_terms, action_size):
        super(PolynomialCircuitNet, self).__init__()
        
        self.term_size = term_size
        self.max_terms = max_terms
        
        # Term encoder (processes each polynomial separately)
        self.term_encoder = nn.Sequential(
            nn.Linear(term_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(term_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Combine all term encodings
        self.terms_aggregator = nn.Sequential(
            nn.Linear(64 * max_terms, 256),
            nn.ReLU()
        )
        
        # Common trunk
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64 + 2, 256),  # terms + target + complexity
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 and 1
        )
    
    def forward(self, state_dict):
        # Process each term with the term encoder
        terms = state_dict['terms']
        batch_size = terms.shape[0]
        
        # Reshape for batch processing
        terms_reshaped = terms.reshape(-1, self.term_size)
        encoded_terms = self.term_encoder(terms_reshaped)
        
        # Reshape to batch_size x max_terms x feature_dim
        feature_dim = 64
        encoded_terms = encoded_terms.reshape(batch_size, self.max_terms, feature_dim)

        # Flatten and aggregate term encodings
        terms_flat = encoded_terms.reshape(batch_size, -1)
        terms_features = self.terms_aggregator(terms_flat)
        
        # Process target polynomial
        target = state_dict['target']
        target_features = self.target_encoder(target)
        
        # Get complexity features
        complexity_features = state_dict['complexity']
        
        # Combine all features
        combined_features = torch.cat([terms_features, target_features, complexity_features], dim=1)
        trunk_output = self.trunk(combined_features)
        
        # Policy and value outputs
        policy_logits = self.policy_head(trunk_output)
        value = self.value_head(trunk_output)
        
        return F.softmax(policy_logits, dim=1), value


class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.expanded = False
    
    def value(self):
        """Get average value from visits"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """Select child according to PUCT formula"""
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            # UCB score calculation
            exploit = child.value()
            explore = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            ucb_score = exploit + explore
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self, policy):
        """Expand using policy network predictions"""
        valid_actions = self.state.get_valid_actions()
        
        # Create children for each valid action
        for action in valid_actions:
            # Convert action to index for policy lookup
            action_idx = action_to_index(action, self.state.num_terms)
            
            # Get prior probability from policy
            prior = policy[action_idx]
            
            # Apply action to create new state
            new_state = self.state.apply_operation(*action)
            
            # Create child node
            child = MCTSNode(new_state, parent=self, action=action, prior=prior)
            self.children.append(child)
        
        self.expanded = True
    
    def backup(self, value):
        """Update node statistics"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

class MCTS:
    def __init__(self, model, num_simulations=100, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, state):
        """Run MCTS search from the given state"""
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            # Selection phase
            node = root
            search_path = [node]
            
            while node.expanded and node.children:
                node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # Leaf node
            leaf = search_path[-1]

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            
            # Expansion and evaluation phase
            # Get value from neural network
            if not leaf.state.is_solution() and leaf.state.operations_used < leaf.state.max_complexity:
                # Convert state to tensor
                state_tensors = leaf.state.encode_for_nn()
                state_tensors = {
                    k: torch.FloatTensor(np.array([v])).to(device) for k, v in state_tensors.items()
                }
                
                # Get policy and value from neural network
                with torch.no_grad():
                    policy, value = self.model(state_tensors)
                
                # Expand node
                leaf.expand(policy[0].cpu().numpy())
                value = value.cpu().item()
            else:
                # Terminal state, evaluate directly
                value = leaf.state.evaluate()
            
            # Backup phase
            for node in reversed(search_path):
                node.backup(value)
        
        max_possible_terms = state.m + state.max_complexity
        max_action_space = get_action_space_size(max_possible_terms)

        
        # Initialize action_probs with the maximum size
        action_probs = np.zeros(max_action_space)
        
        # Fill in probabilities for valid actions
        for child in root.children:
            action_idx = action_to_index(child.action, state.num_terms)
            if action_idx < max_action_space:  # Safety check
                action_probs[action_idx] = child.visit_count
        
        # Normalize only the part that corresponds to current state
        current_action_space = get_action_space_size(state.num_terms)
        if np.sum(action_probs[:current_action_space]) > 0:
            action_probs[:current_action_space] /= np.sum(action_probs[:current_action_space])
        
        return action_probs

class AlphaZeroTrainer:
    def __init__(self, model, optimizer, replay_buffer_size=10000, batch_size=32, mcts_simulations=100):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.mcts = MCTS(model, num_simulations=mcts_simulations)
    
    def self_play_episode(self, initial_state):
        """
        Play one episode and collect training data
        
        Returns:
            List of (state, policy, value) tuples
        """
        experience = []
        state = initial_state

        max_possible_terms = state.m + state.max_complexity
        max_action_space = get_action_space_size(max_possible_terms)
        
        while not state.is_solution() and state.operations_used < state.max_complexity:
            # Run MCTS to get action probabilities
            action_probs = self.mcts.search(state)
            if len(action_probs) < max_action_space:
                # Create a padded version with zeros
                padded_probs = np.zeros(max_action_space)
                padded_probs[:len(action_probs)] = action_probs
                action_probs = padded_probs

            
            # Save state and policy
            experience.append((state.encode_for_nn(), action_probs))
            
            # Choose action 
            # During training, add exploration noise
            valid_actions = state.get_valid_actions()
            valid_indices = [action_to_index(a, state.num_terms) for a in valid_actions]
            
            # Mask invalid actions
            masked_probs = np.zeros_like(action_probs)
            for idx in valid_indices:
                masked_probs[idx] = action_probs[idx]
            
            # Normalize with extra safeguards
            sum_probs = masked_probs.sum()
            if sum_probs > 0:
                # First normalization
                masked_probs = masked_probs / sum_probs
                
                # Force exact sum of 1.0 to avoid floating point errors
                masked_probs = masked_probs / np.sum(masked_probs)
            else:
                # If all probabilities are zero 
                for idx in valid_indices:
                    masked_probs[idx] = 1.0 / len(valid_indices)
            
            # Double-check the sum is very close to 1 (not always necessary)
            # if not 0.9999 < np.sum(masked_probs) < 1.0001:
            #     print(f"Warning: Probabilities sum to {np.sum(masked_probs)}")
            #     # Force normalization again
            #     masked_probs = masked_probs / np.sum(masked_probs)
            
            # Choose based on probabilities
            action_idx = np.random.choice(len(action_probs), p=masked_probs)
                
            action = index_to_action(action_idx, state.num_terms)
            
            state = state.apply_operation(*action)
        
        if state is not None:
            final_value = state.evaluate()
        else:
            final_value = -1.0  # Penalty for invalid state
        
        # Update experience with values
        for i in range(len(experience)):
            # All states get the same value (outcome of the episode)
            experience[i] = (experience[i][0], experience[i][1], final_value)
        
        return experience, final_value
    
    def train(self, num_episodes, create_problem_fn):
        """
        Train the model through self-play
        
        Args:
            num_episodes: Number of episodes to train
            create_problem_fn: Function that creates a new problem
        """
        for episode in range(num_episodes):
            initial_state = create_problem_fn()
            
            experiences, final_value = self.self_play_episode(initial_state)
            
            self.replay_buffer.extend(experiences)
            
            if len(self.replay_buffer) >= self.batch_size:
                loss = self.train_batch()
                print(f"Episode {episode+1}, Value: {final_value:.3f}, Loss: {loss:.3f}")
            else:
                print(f"Episode {episode+1}, Value: {final_value:.3f}, Buffer size: {len(self.replay_buffer)}")
            
            if (episode + 1) % 50 == 0:
                self.save_model(f"model_episode_{episode+1}.pt")
                self.evaluate(create_problem_fn, 10)
    
    def train_batch(self):
        """
        Train on a batch from the replay buffer
        
        Returns:
            Loss value
        """
        # Sample batch
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, policy_batch, value_batch = zip(*minibatch)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare inputs
        state_tensors = {
            'terms': torch.FloatTensor(np.stack([s['terms'] for s in state_batch])).to(device),
            'target': torch.FloatTensor(np.stack([s['target'] for s in state_batch])).to(device),
            'complexity': torch.FloatTensor(np.stack([s['complexity'] for s in state_batch])).to(device)
        }
        
        policy_batch = torch.FloatTensor(np.stack(policy_batch)).to(device)
        value_batch = torch.FloatTensor(np.stack(value_batch)).unsqueeze(1).to(device)
        
        # Forward pass
        self.optimizer.zero_grad()
        policy_pred, value_pred = self.model(state_tensors)
        
        # Loss calculation
        policy_loss = -torch.mean(torch.sum(policy_batch * torch.log(policy_pred + 1e-8), dim=1))
        value_loss = F.mse_loss(value_pred, value_batch)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def evaluate(self, create_problem_fn, num_problems=10):
        """
        Evaluate model performance on test problems
        """
        total_value = 0
        found_solutions = 0
        
        for _ in range(num_problems):
            state = create_problem_fn()
            solution_state = self.solve_problem(state)
            
            if solution_state is not None:
                value = solution_state.evaluate()
                total_value += value
                
                if solution_state.is_solution():
                    found_solutions += 1
        
        avg_value = total_value / num_problems
        solution_rate = found_solutions / num_problems
        
        print(f"Evaluation: Avg value: {avg_value:.3f}, Solution rate: {solution_rate:.2f}")
        return avg_value, solution_rate
    
    def solve_problem(self, initial_state):
        """
        Solve a problem using the trained model and MCTS
        
        Returns:
            Final state
        """
        state = initial_state
        
        while not state.is_solution() and state.operations_used < state.max_complexity:
            # Run MCTS with temperature=0 
            action_probs = self.mcts.search(state)
            
            # Choose the best action
            valid_actions = state.get_valid_actions()
            valid_indices = [action_to_index(a, state.num_terms) for a in valid_actions]
            
            # Get the best valid action
            best_idx = -1
            best_prob = -1
            
            for idx in valid_indices:
                if action_probs[idx] > best_prob:
                    best_prob = action_probs[idx]
                    best_idx = idx
            
            if best_idx == -1:
                # No valid actions
                break
            
            action = index_to_action(best_idx, state.num_terms)
            
            state = state.apply_operation(*action)



        return state
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        
    def load_model(self, filepath):
        """Load model from file"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(filepath, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def subtract_polynomials(poly1, poly2, mod):
    """Subtract poly2 from poly1"""
    max_len = max(len(poly1), len(poly2))
    result = [0] * max_len
    
    # Copy poly1
    for i in range(len(poly1)):
        result[i] = poly1[i]
    
    for i in range(len(poly2)):
        result[i] = (result[i] + poly2[i]) % mod  
    
    return result

# Helper functions for working with polynomials and actions
def add_polynomials(poly1, poly2, mod):
    """Add two polynomial vectors (binary coefficients)"""
    max_len = max(len(poly1), len(poly2))
    result = [0] * max_len
    
    # Copy poly1
    for i in range(len(poly1)):
        result[i] = poly1[i]
    
    # Add poly2
    for i in range(len(poly2)):
        result[i] = (result[i] + poly2[i]) % mod
    
    return result

def multiply_polynomials(poly1, poly2, index_to_monomial, monomial_to_index, mod):
    result = [0] * len(poly1)
    
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            if poly1[i] == 0 or poly2[j] == 0:
                continue
                
            # Get the monomials
            monomial1 = index_to_monomial[i]
            monomial2 = index_to_monomial[j]
            
            # Add exponents component-wise
            new_monomial = tuple(m1 + m2 for m1, m2 in zip(monomial1, monomial2))
            
            # Use dictionary lookup 
            if monomial_to_index and new_monomial in monomial_to_index:
                new_idx = monomial_to_index[new_monomial]
                result[new_idx] = (result[new_idx] + (poly1[i] * poly2[j])) % mod
            else:
                return None
    
    return result

def polynomials_equal(poly1, poly2):
    """Check if two polynomials are equal"""
    max_len = max(len(poly1), len(poly2))
    
    for i in range(max_len):
        val1 = poly1[i] if i < len(poly1) else 0
        val2 = poly2[i] if i < len(poly2) else 0
        
        if val1 != val2:
            return False
    
    return True

def polynomial_distance(poly1, poly2):
    """Compute distance between polynomials vectors. We should change this"""
    max_len = max(len(poly1), len(poly2))
    
    distance = 0
    for i in range(max_len):
        val1 = poly1[i] if i < len(poly1) else 0
        val2 = poly2[i] if i < len(poly2) else 0
        
        if val1 != val2:
            distance += 1
    
    return distance

def action_to_index(action, num_terms):
    """Convert action tuple (op_type, term1_idx, term2_idx) to index"""
    op_type, term1_idx, term2_idx = action
    
    symmetric_pair_count = num_terms * (num_terms + 1) // 2 # this is num_terms Choose 2
    
    if op_type < 2:  # Add or multiply (symmetric)
        # Ensure term1_idx <= term2_idx for consistent indexing of symmetric operations
        if term1_idx > term2_idx:
            term1_idx, term2_idx = term2_idx, term1_idx
        


        # this function is a bijection between ordered pairs (i,j) where i≤j
        # and integers starting from 0 to number of ordered pairs (i,j) where i≤j
        
        # The formula can be derived from:
        # term1_idx * num_terms: Starting point for each "row" of the upper triangular matrix
        # -term1_idx * (term1_idx - 1) // 2: Adjustment for the triangular structure
        # +term2_idx - term1_idx: Position within the current row
        pair_idx = term1_idx * num_terms - term1_idx * (term1_idx - 1) // 2 + term2_idx - term1_idx
        
        # Index accounting for addition or multiplication 
        return op_type * symmetric_pair_count + pair_idx
    else:  # Subtract (asymmetric)
        # Calculate base index after all symmetric operations
        base_idx = 2 * symmetric_pair_count
        
        # term1_idx * num_terms + term2_idx is again a bijection from ordered pair to integer
        # from 0 to number of ordered pair 
        return base_idx + term1_idx * num_terms + term2_idx

def index_to_action(idx, num_terms):
    """Convert index to action tuple (op_type, term1_idx, term2_idx)"""
    symmetric_pair_count = num_terms * (num_terms + 1) // 2
    total_symmetric_ops = 2 * symmetric_pair_count
    
    if idx < total_symmetric_ops:
        # symmetric operation add or multiply
        op_type = idx // symmetric_pair_count
        pair_idx = idx % symmetric_pair_count
        
        # Reverse the pair indexing using quadratic formula
        term1_idx = int((1 + math.sqrt(1 + 8 * pair_idx)) / 2)
        term2_idx = pair_idx - (term1_idx * (term1_idx - 1)) // 2 + term1_idx

        if term1_idx >= num_terms:
            term1_idx = num_terms - 1
        
        if term2_idx >= num_terms:
            term2_idx = num_terms - 1
        
    else:
        # asymmetric operation (subtract)
        op_type = 2  # Subtract
        
        # Get the index within the subtraction operations
        asymmetric_idx = idx - total_symmetric_ops
        
        # Convert to term indices
        term1_idx = asymmetric_idx // num_terms
        term2_idx = asymmetric_idx % num_terms
    
    return (op_type, term1_idx, term2_idx)


def get_action_space_size(num_terms):
    """Calculate total number of possible actions"""
    # Calculate symmetric operations (add, multiply)
    symmetric_ops = 2 * (num_terms * (num_terms + 1) // 2)
    
    # Calculate asymmetric operations (subtract)
    asymmetric_ops = num_terms * num_terms
    
    return symmetric_ops + asymmetric_ops

def create_random_problem(index_to_monomial, polynomials, n=10, m=5, C=15):
    """
    Create a random problem using the saved polynomial data
        
    Returns:
        A CircuitState object representing the problem
    """
    # Randomly select a target polynomial
    target_idx = random.randint(0, len(polynomials) - 1)
    target_polynomial = polynomials[target_idx]
    
    # Create initial variables (one-hot vectors)
    variables = []
    for i in range(m):
        var = [0] * len(index_to_monomial)
        # Find the monomial that represents this variable
        for idx, monomial in index_to_monomial.items():
            if sum(monomial) <= 1 and i < len(monomial) and monomial[i] == 1: 
                var[idx] = 1
                break
        variables.append(var)
    
    
    # Create initial state
    return CircuitState(index_to_monomial, target_polynomial, variables, C,m)

# Main training function
def train_model(index_to_monomial, polynomials, n=5, m=5, C=5, filepath=None, num_episodes=1000):
    """Train a model to find efficient polynomial circuits"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    term_size = len(polynomials[0])
    max_terms = m +C 
    
    max_possible_terms = m + C  
    action_size = get_action_space_size(max_possible_terms)
    
    model = PolynomialCircuitNet(term_size, max_terms, action_size)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    trainer = AlphaZeroTrainer(model, optimizer, mcts_simulations=50)
    
    if filepath:
        trainer.load_model(filepath) 

    def create_problem():
        return create_random_problem(index_to_monomial, polynomials, n, m, C)
    
    # Train
    trainer.train(num_episodes, create_problem)
    
    # Save final model
    trainer.save_model("final_model.pt")
    
    return model, trainer
