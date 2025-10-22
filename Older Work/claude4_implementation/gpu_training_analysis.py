"""
GPU Training Estimates and AWS Cloud Cost Analysis for PolyArithmeticCircuitsRL

This script provides detailed estimates for GPU training times and AWS cloud costs
to support cloud credits applications and deployment planning.
"""

import numpy as np
from typing import Dict, Tuple
import json

# Import existing estimator
try:
    from training_estimate import TrainingEstimator

    IMPORTS_OK = True
except:
    IMPORTS_OK = False


class GPUTrainingEstimator:
    """Estimate GPU training times and cloud costs."""

    def __init__(self):
        if IMPORTS_OK:
            self.base_estimator = TrainingEstimator()
            self.config = self.base_estimator.config
        else:
            self.config = self._mock_config()

        # GPU performance data (relative to CPU baseline)
        self.gpu_specs = {
            "T4": {
                "name": "NVIDIA Tesla T4",
                "memory_gb": 16,
                "speedup_vs_cpu": 12,
                "aws_instance": "g4dn.xlarge",
                "hourly_cost": 0.526,
                "description": "Entry-level GPU, good for development",
            },
            "V100": {
                "name": "NVIDIA Tesla V100",
                "memory_gb": 32,
                "speedup_vs_cpu": 20,
                "aws_instance": "p3.2xlarge",
                "hourly_cost": 3.06,
                "description": "High-performance GPU, excellent for training",
            },
            "A10G": {
                "name": "NVIDIA A10G",
                "memory_gb": 24,
                "speedup_vs_cpu": 18,
                "aws_instance": "g5.xlarge",
                "hourly_cost": 1.006,
                "description": "Modern GPU, balanced performance/cost",
            },
            "A100": {
                "name": "NVIDIA A100",
                "memory_gb": 40,
                "speedup_vs_cpu": 25,
                "aws_instance": "p4d.xlarge",
                "hourly_cost": 4.077,
                "description": "Top-tier GPU, fastest training",
            },
        }

        # CPU baseline estimates (from CPU analysis)
        self.cpu_baseline = {
            "supervised_hours": 21.1,
            "ppo_hours": 3527.1,
            "total_hours": 3548.2,
        }

    def _mock_config(self):
        """Mock config if imports fail."""

        class MockConfig:
            def __init__(self):
                self.epochs = 50
                self.ppo_iterations = 2000
                self.steps_per_batch = 4096
                self.batch_size = 128

        return MockConfig()

    def estimate_gpu_training_times(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate training times for different GPU types.

        Returns:
            Dictionary with timing estimates for each GPU
        """
        results = {}

        for gpu_id, gpu_info in self.gpu_specs.items():
            speedup = gpu_info["speedup_vs_cpu"]

            supervised_hours = self.cpu_baseline["supervised_hours"] / speedup
            ppo_hours = self.cpu_baseline["ppo_hours"] / speedup
            total_hours = supervised_hours + ppo_hours

            results[gpu_id] = {
                "name": gpu_info["name"],
                "aws_instance": gpu_info["aws_instance"],
                "speedup": speedup,
                "supervised_hours": supervised_hours,
                "ppo_hours": ppo_hours,
                "total_hours": total_hours,
                "total_days": total_hours / 24,
                "memory_gb": gpu_info["memory_gb"],
                "description": gpu_info["description"],
            }

        return results

    def estimate_aws_costs(
        self, training_time_hours: float = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate AWS costs for different instance types.

        Args:
            training_time_hours: Override default training time

        Returns:
            Dictionary with cost estimates for each instance type
        """
        results = {}

        for gpu_id, gpu_info in self.gpu_specs.items():
            if training_time_hours is None:
                # Use GPU-specific training time
                gpu_times = self.estimate_gpu_training_times()
                hours = gpu_times[gpu_id]["total_hours"]
            else:
                hours = training_time_hours

            hourly_cost = gpu_info["hourly_cost"]

            # Calculate costs
            compute_cost = hours * hourly_cost

            # Storage costs (EBS, S3)
            storage_cost = 30  # ~$30 for datasets, checkpoints, logs

            # Data transfer (minimal for training)
            transfer_cost = 10  # ~$10 for data in/out

            # Buffer for other services (CloudWatch, etc.)
            other_cost = 20  # ~$20 buffer

            total_cost = compute_cost + storage_cost + transfer_cost + other_cost

            results[gpu_id] = {
                "instance_type": gpu_info["aws_instance"],
                "hourly_cost": hourly_cost,
                "training_hours": hours,
                "compute_cost": compute_cost,
                "storage_cost": storage_cost,
                "transfer_cost": transfer_cost,
                "other_cost": other_cost,
                "total_cost": total_cost,
                "cost_per_day": total_cost / (hours / 24) if hours > 0 else 0,
            }

        return results

    def estimate_parallel_training_costs(self, num_gpus: int = 4) -> Dict[str, float]:
        """
        Estimate costs for distributed training across multiple GPUs.

        Args:
            num_gpus: Number of GPUs for distributed training

        Returns:
            Cost estimates for distributed setup
        """
        # Use A10G as baseline for multi-GPU (good performance/cost ratio)
        base_gpu = "A10G"
        gpu_info = self.gpu_specs[base_gpu]

        # Distributed training speedup (not linear due to communication overhead)
        if num_gpus == 2:
            parallel_speedup = 1.7
        elif num_gpus == 4:
            parallel_speedup = 3.0
        elif num_gpus == 8:
            parallel_speedup = 5.5
        else:
            parallel_speedup = num_gpus * 0.75  # Rough estimate

        # Calculate training time with parallelization
        single_gpu_hours = self.cpu_baseline["total_hours"] / gpu_info["speedup_vs_cpu"]
        parallel_hours = single_gpu_hours / parallel_speedup

        # Multi-GPU instance costs
        if num_gpus <= 1:
            instance_cost = gpu_info["hourly_cost"]
        elif num_gpus <= 4:
            instance_cost = gpu_info["hourly_cost"] * num_gpus * 1.1  # Slight premium
        else:
            # Larger instances (p4d.24xlarge for 8x A100)
            instance_cost = 32.77  # p4d.24xlarge hourly rate

        compute_cost = parallel_hours * instance_cost
        infrastructure_cost = 100  # Additional costs for distributed setup

        return {
            "num_gpus": num_gpus,
            "parallel_speedup": parallel_speedup,
            "training_hours": parallel_hours,
            "training_days": parallel_hours / 24,
            "hourly_instance_cost": instance_cost,
            "compute_cost": compute_cost,
            "infrastructure_cost": infrastructure_cost,
            "total_cost": compute_cost + infrastructure_cost,
        }

    def estimate_development_costs(self) -> Dict[str, float]:
        """
        Estimate costs for development and experimentation phase.

        Returns:
            Cost estimates for development work
        """
        # Assume 3 months of development with intermittent training
        # Using T4 instances for cost-effective development

        dev_hours_per_week = 20  # 20 hours of GPU time per week
        dev_weeks = 12  # 3 months
        total_dev_hours = dev_hours_per_week * dev_weeks

        t4_cost = self.gpu_specs["T4"]["hourly_cost"]

        return {
            "development_weeks": dev_weeks,
            "gpu_hours_per_week": dev_hours_per_week,
            "total_gpu_hours": total_dev_hours,
            "compute_cost": total_dev_hours * t4_cost,
            "storage_cost": 50,  # Extended storage for experiments
            "total_cost": total_dev_hours * t4_cost + 50,
        }

    def generate_aws_credit_justification(self) -> Dict:
        """
        Generate comprehensive justification for AWS credits application.

        Returns:
            Structured data for credits application
        """
        gpu_times = self.estimate_gpu_training_times()
        aws_costs = self.estimate_aws_costs()
        parallel_costs = self.estimate_parallel_training_costs(4)
        dev_costs = self.estimate_development_costs()

        # Calculate total project costs
        training_cost = aws_costs["A10G"]["total_cost"]  # Balanced choice
        development_cost = dev_costs["total_cost"]
        experimentation_cost = training_cost * 0.5  # Multiple experiments

        total_project_cost = training_cost + development_cost + experimentation_cost

        return {
            "project_overview": {
                "title": "Reinforcement Learning for Polynomial Arithmetic Circuits",
                "description": "Research project using RL to discover efficient arithmetic circuits for polynomials",
                "duration_months": 6,
                "research_goals": [
                    "Implement AlphaZero-style MCTS for circuit construction",
                    "Evaluate on standard benchmarks (elementary symmetric, determinants)",
                    "Compare against classical algebraic complexity constructions",
                    "Generate training data through self-play",
                    "Publish results at ML conference (ICML/ICLR)",
                ],
            },
            "computational_requirements": {
                "model_size": "4.2M parameters",
                "training_data": "Self-generated through MCTS self-play",
                "cpu_baseline": f"{self.cpu_baseline['total_hours']:.0f} hours ({self.cpu_baseline['total_hours'] / 24:.0f} days)",
                "gpu_speedup_needed": "Essential for practical training times",
                "preferred_gpu": "A10G (balanced performance/cost) or V100 (high performance)",
            },
            "cost_breakdown": {
                "main_training_run": {
                    "instance": aws_costs["A10G"]["instance_type"],
                    "hours": f"{gpu_times['A10G']['total_hours']:.1f}",
                    "cost": f"${aws_costs['A10G']['total_cost']:.0f}",
                },
                "development_phase": {
                    "duration": "3 months",
                    "cost": f"${dev_costs['total_cost']:.0f}",
                },
                "experimentation": {
                    "description": "Multiple model configurations and ablations",
                    "cost": f"${experimentation_cost:.0f}",
                },
                "total_project_cost": f"${total_project_cost:.0f}",
            },
            "justification_points": [
                f"CPU training would take {self.cpu_baseline['total_hours'] / 24:.0f} days - impractical for research",
                f"GPU reduces training to {gpu_times['A10G']['total_days']:.1f} days - enables iterative research",
                "Project addresses fundamental questions in algebraic complexity theory",
                "Implementation includes comprehensive verification and benchmarking",
                "Results will be published and code open-sourced",
                "Educational value for understanding RL in mathematical domains",
            ],
            "alternatives_considered": {
                "local_cpu": f"{self.cpu_baseline['total_hours'] / 24:.0f} days - too slow",
                "local_gpu": "Not available/insufficient for this scale",
                "google_colab": "Limited to 12-hour sessions, insufficient for full training",
                "other_clouds": "AWS preferred for academic discounts and ecosystem",
            },
        }

    def print_comprehensive_analysis(self):
        """Print complete GPU training and cost analysis."""
        print("=" * 80)
        print("GPU TRAINING ESTIMATES AND AWS COST ANALYSIS")
        print("=" * 80)

        # GPU training times
        gpu_times = self.estimate_gpu_training_times()
        print(f"\n🚀 GPU TRAINING TIME ESTIMATES:")
        print(
            f"{'GPU':<12} {'Instance':<15} {'Speedup':<8} {'Total Time':<12} {'Days':<8} {'Memory':<8}"
        )
        print("-" * 70)
        for gpu_id, data in gpu_times.items():
            print(
                f"{gpu_id:<12} {data['aws_instance']:<15} {data['speedup']:<8}x "
                f"{data['total_hours']:<11.1f}h {data['total_days']:<7.1f} {data['memory_gb']:<7}GB"
            )

        # AWS costs
        aws_costs = self.estimate_aws_costs()
        print(f"\n💰 AWS TRAINING COSTS:")
        print(
            f"{'GPU':<12} {'Instance':<15} {'Hours':<8} {'$/hour':<8} {'Total Cost':<12}"
        )
        print("-" * 60)
        for gpu_id, data in aws_costs.items():
            print(
                f"{gpu_id:<12} {data['instance_type']:<15} {data['training_hours']:<7.1f}h "
                f"${data['hourly_cost']:<7.2f} ${data['total_cost']:<11.0f}"
            )

        # Parallel training costs
        parallel_costs = self.estimate_parallel_training_costs(4)
        print(f"\n⚡ DISTRIBUTED TRAINING (4 GPUs):")
        print(
            f"  Training time: {parallel_costs['training_hours']:.1f} hours ({parallel_costs['training_days']:.1f} days)"
        )
        print(f"  Speedup: {parallel_costs['parallel_speedup']:.1f}x")
        print(f"  Total cost: ${parallel_costs['total_cost']:.0f}")

        # Development costs
        dev_costs = self.estimate_development_costs()
        print(f"\n🛠️ DEVELOPMENT PHASE COSTS:")
        print(f"  Duration: {dev_costs['development_weeks']} weeks")
        print(f"  GPU hours: {dev_costs['total_gpu_hours']} hours")
        print(f"  Total cost: ${dev_costs['total_cost']:.0f}")

        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")

        # Best value option
        best_value = min(
            aws_costs.items(),
            key=lambda x: x[1]["total_cost"] / gpu_times[x[0]]["speedup"],
        )
        print(
            f"  💡 Best Value: {best_value[0]} ({self.gpu_specs[best_value[0]]['name']})"
        )
        print(
            f"     - Training time: {gpu_times[best_value[0]]['total_days']:.1f} days"
        )
        print(f"     - Total cost: ${aws_costs[best_value[0]]['total_cost']:.0f}")

        # Development option
        print(f"  🚀 Development: T4 (cost-effective for testing)")
        print(f"     - Hourly cost: ${self.gpu_specs['T4']['hourly_cost']}")
        print(f"     - Good for model validation and debugging")

        # Production option
        print(f"  🏆 Production: A100 (fastest training)")
        print(f"     - Training time: {gpu_times['A100']['total_days']:.1f} days")
        print(f"     - Total cost: ${aws_costs['A100']['total_cost']:.0f}")

        # Total project estimate
        total_cost = aws_costs["A10G"]["total_cost"] + dev_costs["total_cost"]
        print(f"\n📊 TOTAL PROJECT COST ESTIMATE:")
        print(f"  Development + Main Training: ${total_cost:.0f}")
        print(f"  With experimentation (+50%): ${total_cost * 1.5:.0f}")

        print(f"\n⚠️ IMPORTANT NOTES:")
        print(f"  - Costs include compute, storage, and data transfer")
        print(f"  - Spot instances can reduce costs by 50-70%")
        print(f"  - Training may converge earlier than estimated")
        print(f"  - Multiple experiments needed for robust results")


def generate_aws_credits_proposal():
    """Generate detailed AWS credits application proposal."""
    estimator = GPUTrainingEstimator()
    justification = estimator.generate_aws_credit_justification()

    print("\n" + "=" * 80)
    print("AWS CREDITS APPLICATION PROPOSAL")
    print("=" * 80)

    print(f"\n📋 PROJECT OVERVIEW:")
    overview = justification["project_overview"]
    print(f"  Title: {overview['title']}")
    print(f"  Duration: {overview['duration_months']} months")
    print(f"  Description: {overview['description']}")

    print(f"\n🎯 RESEARCH GOALS:")
    for goal in overview["research_goals"]:
        print(f"  • {goal}")

    print(f"\n⚙️ COMPUTATIONAL REQUIREMENTS:")
    reqs = justification["computational_requirements"]
    print(f"  Model size: {reqs['model_size']}")
    print(f"  CPU baseline: {reqs['cpu_baseline']}")
    print(f"  Preferred GPU: {reqs['preferred_gpu']}")

    print(f"\n💰 COST BREAKDOWN:")
    costs = justification["cost_breakdown"]
    print(
        f"  Main training: {costs['main_training_run']['cost']} ({costs['main_training_run']['hours']} on {costs['main_training_run']['instance']})"
    )
    print(
        f"  Development: {costs['development_phase']['cost']} ({costs['development_phase']['duration']})"
    )
    print(f"  Experimentation: {costs['experimentation']['cost']}")
    print(f"  TOTAL: {costs['total_project_cost']}")

    print(f"\n✅ JUSTIFICATION:")
    for point in justification["justification_points"]:
        print(f"  • {point}")

    print(f"\n🔍 ALTERNATIVES CONSIDERED:")
    alts = justification["alternatives_considered"]
    for alt, reason in alts.items():
        print(f"  • {alt.replace('_', ' ').title()}: {reason}")

    return justification


def main():
    """Run comprehensive GPU analysis."""
    estimator = GPUTrainingEstimator()
    estimator.print_comprehensive_analysis()

    # Generate credits proposal
    proposal = generate_aws_credits_proposal()

    # Save proposal to file
    with open("aws_credits_proposal.json", "w") as f:
        json.dump(proposal, f, indent=2)

    print(f"\n💾 Detailed proposal saved to 'aws_credits_proposal.json'")
    print(f"📋 Use this data for your AWS credits application!")


if __name__ == "__main__":
    main()
