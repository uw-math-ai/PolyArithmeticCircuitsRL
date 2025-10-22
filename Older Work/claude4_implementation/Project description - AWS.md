Our current project description (as appears on the math AI page)

**Reinforcement Learning for Polynomials.**

* **Lead mentor: Jarod Alper**  
* **Description: We will use reinforcement learning to train computers to search for efficient arithmetic circuits computing specific polynomials. This is in some sense a simplified version of proof generation, where computers are trained to search for proofs of theorems, except that in this case the search space is considerably smaller. At the same time, this project may shed some light on some explicit questions in algebraic complexity theory, and specifically the algebraic analogue of P vs NP. Our inspiration is the AlphaZero algorithm for two-player games, which we will adapt to a single-player game. The problem we will attack is to find efficient ways to compute a polynomial f(x\_1, ..., x\_n) using an arithmetic circuit consisting of \+ and x gates together with scalar multiplication. Our strategy is to use Monte Carlo Tree Search to train a deep neural network to learn an effective evaluation function (giving a score of how close the current state is from computing the polynomial) and a policy (a probability distribution over the next moves, with higher probability given to moves that simplify the expression). The computer is rewarded when it finds efficient arithmetic circuits. We aim to submit to one of the major ML conferences such as ICML or ICLR.**

Here’s what GPT wrote for us:

# **Reinforcement Learning for Polynomials**

**Principal Investigator and Lead Mentor:** Jarod Alper (University of Washington)

An *arithmetic circuit* is a directed acyclic graph that computes a polynomial from inputs x\_1, …, x\_n​ and constants using addition, multiplication, and scalar multiplication. Its *size* is the number of gates and its *depth* is the number of computational layers. Designing smaller or shallower circuits for explicit families of polynomials is a central task in algebraic complexity, with connections to major open questions such as **P vs NP** and its algebraic analogue **VP vs VNP**. For many concrete polynomials, however, concise and verifiably correct circuits that improve on standard constructions are still not known.

As a part of the Math AI Lab, this project asks whether *reinforcement learning* (RL) can automate part of that design process. In RL, an agent learns by trial and error to choose actions that lead to higher rewards. The AlphaZero line of work by Google Deepmind showed that combining a neural network with *Monte Carlo Tree Search* (MCTS) can guide exploration effectively in large decision spaces for two-player games such as Go and chess. Our setting is not of competitive nature, but the core idea—using learned guidance to search a vast space of possibilities—is applicable. We adapt it to a *single-player* setting in which each move applies a sound algebraic transformation that preserves equality with a target polynomial, and a “win” is a correct circuit with low size or depth.

We formalize circuit construction as a sequential decision process. A *state* records a partially constructed circuit or an algebraically equivalent expression for the target polynomial. An *action* applies a verified identity (e.g., factorization, the distributive law, or reuse of a common subexpression). A neural network produces two outputs: a *policy* suggesting which transformation to try next and a *value* estimating how promising the current state is with respect to reaching a correct, low-cost circuit. MCTS uses these predictions to focus exploration while still testing alternatives. Training alternates between search—producing trajectories of transformations—and supervised updates to the policy and value models based on the outcomes of those trajectories. The *reward* favors exact circuits with fewer gates or smaller depth, with the option to weight multiplications more heavily than additions. (Michael: are these statements factual?)

Correctness is essential. Every proposed terminal circuit is checked for exact equality with the target polynomial using symbolic methods and randomized modular evaluation; only circuits that pass these tests are accepted. We begin with small, well-understood benchmarks—such as elementary symmetric polynomials and low-degree determinants—so that results can be compared against known constructions. Michael: did we do these? Seems like good advice. As the models improve, we scale the instance sizes and introduce additional structured families. Experiments report success rate, circuit size and depth relative to classical baselines, and transfer across related polynomials. These outcomes speak directly to empirical aspects of algebraic complexity and may provide data relevant to questions such as **VP vs VNP**.

**Necessity of AWS Credits and Expected Speedups.**  
 The training and search workload consists of thousands of short, parallel jobs: batched neural-network inference during MCTS rollouts, periodic network training steps, and CPU-bound symbolic verification. This pattern benefits from elastic, heterogeneous clusters with fast checkpointing and shared storage—precisely what AWS provides. Without on-demand GPUs, the wall-clock time to run even a modest experimental suite becomes prohibitive.

We anticipate the following, based on standard RL/MCTS pipelines and pilots on local hardware:

* **Single-GPU acceleration (per job).** Moving policy/value inference and training from a 16-core CPU node to a single modern cloud GPU (e.g., A10 or equivalent) typically yields **10–20×** higher training step throughput and **5–10×** higher batched inference throughput within MCTS. This shift makes the search **model-bound** rather than **CPU-bound**, increasing useful rollouts per hour.

* **Small GPU pool vs. one GPU.** Distributing self-play/inference across **8 GPUs** yields an additional **6–7.5×** throughput in rollouts (allowing for modest scaling loss from synchronization), turning week-long sweeps into runs that complete in **1–2 days**.

* **CPU scaling for verification.** Symbolic equality checks and modular evaluations scale nearly linearly with CPU cores. A pool of \~128 vCPUs (e.g., a few c7i instances) gives about **8–10×** the verification throughput of a single 16-core server, keeping the CPU queue from bottlenecking the faster GPU-driven search.

Michael: discuss. We need to make sure that every claim we make in this proposal is factual.

Putting these together, an experimental set that would require roughly **2–3 weeks** on a single local GPU and one CPU server can be executed in **2–4 days** with a modest AWS cluster (≈8 GPUs for search/inference, ≈100–150 vCPUs for verification), while remaining checkpoint-friendly and robust to Spot interruptions. This compression of wall-clock time is essential for running multiple ablations (action sets, cost models, curriculum schedules) and for producing verified circuits across several polynomial families within the credit period.

We will containerize the stack and orchestrate runs with AWS Batch or SageMaker to submit many short jobs that exploit Spot capacity. S3 will hold datasets, checkpoints, and circuit artifacts; EFS or FSx will provide shared scratch for active jobs; and standard monitoring will enforce budget limits. Because each job is independent and writes incremental results, interruptions do not compromise correctness or progress.

**Outputs and Evaluation.**  
 We will produce (i) verified circuits that reduce gate counts or depth for selected cases; (ii) a small, public benchmark suite for arithmetic-circuit synthesis; and (iii) code to reproduce the experiments. We will prepare a paper for a major machine-learning venue (ICML or ICLR) describing the approach, empirical results, and released artifacts. The team, led by Professor Alper, has the required expertise in reinforcement learning, algebra, and software engineering to execute the plan and to maintain a clear pipeline from training to verification to analysis.

