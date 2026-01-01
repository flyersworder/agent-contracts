"""Good Enough Experiment: Demonstrating Q_min-based Early Stopping.

This experiment shows how Agent Contracts enable agents to recognize
"good enough" and stop voluntarily, optimizing for human benefit
rather than engagement.

Key concept: Define quality threshold (Q_min) in the contract, agent
self-evaluates after each iteration, stops when Q >= Q_min.

Reference: Ethan Flory's observation about AI optimization for engagement
vs human benefit - "AI might help you iterate 30 times to draft the perfect
email over half an hour... But would you rather it tell you to stop after
5 minutes and send the email, that it's good enough?"
"""
