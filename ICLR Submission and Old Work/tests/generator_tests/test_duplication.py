import sys
import os
import time

# Add project root to path to allow importing 'transformer'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformer.generator import generate_random_polynomials

def test_duplication_rate(n=3, C=5, num_to_generate=100, mod=5):
    """
    Analyzes the duplication rate of the polynomial generator.

    This test generates a specified number of unique polynomials and reports
    how many attempts were needed. A high number of attempts compared to the
    number of generated polynomials indicates a high duplication rate.
    """
    print("\n--- Starting Duplication Rate Analysis ---")
    print(f"Parameters: n={n}, C={C}, mod={mod}")
    print(f"Aiming to generate {num_to_generate} unique polynomials...")

    start_time = time.time()

    # The generator function internally handles deduplication.
    # We can assess the rate by checking how many attempts it takes.
    # The function returns the actual polynomials and the attempts count.
    polynomials, _, attempts = generate_random_polynomials(
        n=n,
        C=C,
        num_polynomials=num_to_generate,
        mod=mod,
        return_attempts=True,
    )

    end_time = time.time()
    duration = end_time - start_time

    num_generated = len(polynomials)
    num_duplicates_found = attempts - num_generated

    print("\n--- Sample of Generated Unique Polynomials ---")
    for i, p in enumerate(polynomials[:5]):
        print(f"  {i+1}: {p}")
    if num_generated > 5:
        print(f"  ... and {num_generated - 5} more.")

    print(f"\n--- Analysis Results ---")
    print(f"Time taken: {duration:.2f} seconds")
    print(f"Total generation attempts: {attempts}")
    print(f"Successfully generated unique polynomials: {num_generated}")
    print(f"Discarded duplicate polynomials: {num_duplicates_found}")

    if attempts > 0:
        duplication_percentage = (num_duplicates_found / attempts) * 100
        print(f"Duplication rate: {duplication_percentage:.2f}%")
    else:
        print("No polynomials were generated.")

    # This is a test assertion to ensure the generator is behaving as expected.
    # It should either generate the requested number of polynomials or stop early
    # if it hits its internal attempt limit, but it should never return more.
    assert num_generated <= num_to_generate
    print("\nAssertion passed: The number of generated polynomials is correct.")
    print("--- End of Analysis ---")

if __name__ == "__main__":
    # You can run this file directly to see the analysis.
    # Example with small parameters:
    test_duplication_rate(n=3, C=10, num_to_generate=1000, mod=2)
    
    # Example with larger parameters that might have a higher duplication rate:
    # test_duplication_rate(n=4, C=8, num_to_generate=1000, mod=5)
