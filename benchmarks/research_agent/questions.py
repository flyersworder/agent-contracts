"""Research question dataset for benchmark evaluation."""

from dataclasses import dataclass


@dataclass
class ResearchQuestion:
    """A research question with expected decomposition.

    Attributes:
        id: Unique identifier
        question: Main research question
        domain: Subject domain (AI/ML, Systems, Economics, etc.)
        expected_subquestions: Expected sub-questions from decomposition
        difficulty: Difficulty level (1-5)
    """

    id: str
    question: str
    domain: str
    expected_subquestions: list[str]
    difficulty: int


RESEARCH_QUESTIONS = [
    ResearchQuestion(
        id="q1_transformers_vs_ssm",
        question=(
            "What are the key differences between transformer and state-space models "
            "for long-context language modeling, and what are the tradeoffs?"
        ),
        domain="AI/ML",
        expected_subquestions=[
            "How do transformers handle long contexts?",
            "How do state-space models (like Mamba) handle long contexts?",
            "What are the computational complexity differences?",
            "What are the practical performance tradeoffs?",
        ],
        difficulty=4,
    ),
    ResearchQuestion(
        id="q2_consensus_algorithms",
        question="How do consensus algorithms like Raft and Paxos differ, and when should you use each?",
        domain="Distributed Systems",
        expected_subquestions=[
            "What is the core mechanism of Raft?",
            "What is the core mechanism of Paxos?",
            "What are the failure modes of each?",
            "What are typical use cases for each?",
        ],
        difficulty=4,
    ),
    ResearchQuestion(
        id="q3_financial_crisis",
        question="What were the causes of the 2008 financial crisis, and what lessons were learned?",
        domain="Economics",
        expected_subquestions=[
            "What was the role of subprime mortgages?",
            "How did derivatives and CDOs contribute?",
            "What regulatory failures occurred?",
            "What reforms were implemented afterward?",
        ],
        difficulty=3,
    ),
    ResearchQuestion(
        id="q4_crispr",
        question="How does CRISPR-Cas9 gene editing work, and what are its limitations?",
        domain="Biology",
        expected_subquestions=[
            "What is the mechanism of CRISPR-Cas9?",
            "How is it used for gene editing?",
            "What are the off-target effects?",
            "What are current therapeutic applications?",
        ],
        difficulty=3,
    ),
    ResearchQuestion(
        id="q5_microservices_vs_monolith",
        question="What are the tradeoffs between microservices and monolithic architectures?",
        domain="Software Engineering",
        expected_subquestions=[
            "What are the benefits of microservices?",
            "What are the benefits of monolithic architectures?",
            "What are the operational costs of each?",
            "When should you choose each approach?",
        ],
        difficulty=3,
    ),
]
