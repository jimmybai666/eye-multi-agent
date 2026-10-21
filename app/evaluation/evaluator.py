"""Evaluator for running agent evaluation pipelines.

评测统计逻辑（指标计算等）"""

from app.evaluation.metrics import EvaluationMetrics, calculate_diagnosis_accuracy, calculate_completeness
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """Evaluates agent performance on test cases."""

    def __init__(self):
        self.results: list[EvaluationMetrics] = []

    async def evaluate_single(self, graph, test_case: dict) -> EvaluationMetrics:
        """Evaluate the graph on a single test case.

        Args:
            graph: The compiled LangGraph.
            test_case: Dict with 'input', 'image_path', and 'expected' keys.

        Returns:
            Evaluation metrics for this test case.
        """
        import time

        start = time.time()
        result = await graph.ainvoke({
            "input": test_case["input"],
            "image_path": test_case.get("image_path"),
            "messages": [],
            "next_agent": "supervisor",
            "is_complete": False,
            "iteration_count": 0,
            "max_iterations": 5,
        })
        elapsed = time.time() - start

        metrics = EvaluationMetrics(
            accuracy=calculate_diagnosis_accuracy(
                result.get("final_diagnosis", ""),
                test_case.get("expected", ""),
            ),
            completeness=calculate_completeness(result.get("clinical_result", {})),
            response_time=elapsed,
            iteration_count=result.get("iteration_count", 0),
        )

        self.results.append(metrics)
        logger.info(f"Evaluation complete: accuracy={metrics.accuracy}, time={metrics.response_time:.2f}s")
        return metrics

    def summary(self) -> dict:
        """Get summary statistics of all evaluations."""
        if not self.results:
            return {"count": 0}

        return {
            "count": len(self.results),
            "avg_accuracy": sum(r.accuracy for r in self.results) / len(self.results),
            "avg_completeness": sum(r.completeness for r in self.results) / len(self.results),
            "avg_response_time": sum(r.response_time for r in self.results) / len(self.results),
            "avg_iterations": sum(r.iteration_count for r in self.results) / len(self.results),
        }
