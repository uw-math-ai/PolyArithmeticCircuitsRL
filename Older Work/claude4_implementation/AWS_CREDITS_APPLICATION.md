# AWS Credits Application: PolyArithmeticCircuitsRL

## Executive Summary

We request AWS credits to support a 6-month research project on reinforcement learning for polynomial arithmetic circuits. The project implements AlphaZero-style Monte Carlo Tree Search to discover efficient circuit constructions, addressing fundamental questions in algebraic complexity theory.

**Key Metrics:**
- **Total Credits Requested:** $652 (with experimentation buffer)
- **Training Time:** 8.2 days on recommended A10G instances
- **Model Size:** 4.2M parameters
- **CPU Baseline:** 148 days (impractical without GPU acceleration)

## Project Description

### Research Objectives
1. **Primary Goal:** Implement reinforcement learning (RL) to discover efficient arithmetic circuits for polynomial evaluation
2. **Technical Approach:** AlphaZero-style MCTS combined with transformer neural networks
3. **Evaluation:** Comprehensive benchmarking on standard polynomials (elementary symmetric, determinants, Chebyshev)
4. **Output:** Publication-ready results for top-tier ML conferences (ICML/ICLR)

### Scientific Impact
- Addresses open problems in algebraic complexity theory
- Novel application of RL to mathematical optimization
- Potential to discover new computational structures
- Educational value for mathematical AI community

## Technical Architecture

### Model Specifications
```
Architecture: GNN + Transformer hybrid
Parameters: 4.2M total
- Transformer encoder: 3.8M parameters
- Graph neural network: 0.4M parameters
Memory requirement: 470MB (model + gradients)
Training data: Self-generated through MCTS self-play
```

### Implementation Status
✅ **Complete:** Core transformer model, MCTS algorithm, benchmark suite, verification pipeline
✅ **Tested:** All components verified through comprehensive smoke tests (12/12 passing)
✅ **Ready:** Full integration pipeline ready for cloud deployment

## Computational Requirements

### Why GPU Acceleration is Essential

| Training Phase | CPU Time | GPU Time (A10G) | Speedup |
|----------------|----------|-----------------|---------|
| Supervised Learning | 21.1 hours | 1.2 hours | 18x |
| PPO Reinforcement Learning | 3,527 hours | 196 hours | 18x |
| **Total** | **148 days** | **8.2 days** | **18x** |

**Critical Point:** CPU training would take 148 days, making iterative research impossible. GPU acceleration reduces this to practical 8.2 days.

### Memory Requirements
- Model: 470MB
- Training data: Variable (generated online)
- Checkpoints: ~2GB per experiment
- Total storage: ~50GB for full project

## AWS Cost Analysis

### Recommended Configuration: A10G (g5.xlarge)
- **Why A10G:** Optimal balance of performance and cost for this workload
- **Instance Cost:** $1.006/hour
- **Training Hours:** 197.1 hours
- **Compute Cost:** $198
- **Additional Costs:** $60 (storage, transfer, monitoring)
- **Training Total:** $258

### Complete Project Budget

| Component | Duration | Cost | Description |
|-----------|----------|------|-------------|
| **Development Phase** | 3 months | $176 | Model development and testing (T4 instances) |
| **Main Training** | 8.2 days | $258 | Primary model training (A10G instances) |
| **Experimentation** | Various | $129 | Ablation studies and model variants |
| **Infrastructure** | 6 months | $89 | Storage, monitoring, data transfer |
| **TOTAL** | | **$652** | Full project with experimentation buffer |

### Cost Optimization Strategies
- **Spot Instances:** 50-70% cost reduction when available
- **Early Stopping:** Training may converge before maximum epochs
- **Staged Training:** Start with supervised learning (1.2 hours) before full RL
- **Development Efficiency:** Use T4 instances for debugging and validation

## Alternative Solutions Considered

| Option | Cost | Time | Viability |
|--------|------|------|-----------|
| **Local CPU** | $0 | 148 days | ❌ Impractical for research timeline |
| **Local GPU** | $0 | N/A | ❌ No access to sufficient hardware |
| **Google Colab** | $50/month | N/A | ❌ 12-hour session limits insufficient |
| **Other Clouds** | Similar | Similar | ⚠️ AWS preferred for academic ecosystem |
| **AWS GPUs** | $652 | 8.2 days | ✅ Only viable option for this scale |

## Educational and Research Value

### Academic Benefits
- **Publication Output:** Conference paper with novel RL application
- **Open Source:** All code will be publicly available
- **Educational:** Demonstrates RL in mathematical domains
- **Reproducible:** Comprehensive verification ensures reliability

### Broader Impact
- **Algorithmic Complexity:** Potential insights into computational lower bounds
- **AI for Mathematics:** Advances AI applications in pure mathematics
- **Methodology:** Reusable framework for similar optimization problems

## Project Timeline

### Phase 1: Development (Months 1-2)
- Model refinement and validation
- Comprehensive testing on T4 instances
- Cost: $176

### Phase 2: Main Training (Month 3)
- Large-scale supervised training (1.2 hours)
- Full PPO reinforcement learning (196 hours)
- Cost: $258

### Phase 3: Experimentation (Months 4-5)
- Ablation studies and model variants
- Hyperparameter optimization
- Cost: $129

### Phase 4: Analysis (Month 6)
- Results analysis and paper writing
- Minimal additional compute costs
- Cost: $89 (infrastructure only)

## Risk Mitigation

### Technical Risks
- **Convergence Issues:** Staged training approach reduces risk
- **Memory Limitations:** Model sized to fit within GPU memory constraints
- **Implementation Bugs:** Comprehensive testing already completed

### Cost Risks
- **Overruns:** 20% buffer included in estimates
- **Early Completion:** Likely to finish under budget due to potential early convergence
- **Spot Interruptions:** Checkpoint system enables recovery from interruptions

## Deliverables

### Technical Outputs
1. **Complete RL Framework:** Open-source implementation
2. **Benchmark Results:** Performance on standard polynomial evaluation tasks
3. **Trained Models:** Pre-trained checkpoints for community use
4. **Documentation:** Comprehensive usage and reproduction guides

### Academic Outputs
1. **Conference Publication:** Submission to ICML/ICLR 2025
2. **Technical Report:** Detailed methodology and results
3. **Code Repository:** Public GitHub repository with reproducible experiments

## Justification Summary

This project represents a novel application of reinforcement learning to fundamental problems in mathematical computation. The **$652 credit request** enables:

1. **Practical Timeline:** Reduces 148-day CPU training to 8.2 days
2. **Iterative Research:** Enables experimentation and refinement
3. **Publication Quality:** Sufficient compute for comprehensive evaluation
4. **Community Value:** Open-source framework for future research

The project addresses genuine computational constraints while providing significant educational and research value to the broader AI and mathematics communities.

---

**Contact Information:**
- Project Repository: [PolyArithmeticCircuitsRL](https://github.com/uw-math-ai/PolyArithmeticCircuitsRL)
- Implementation Status: Complete and tested
- Timeline: Ready to deploy immediately upon credit approval