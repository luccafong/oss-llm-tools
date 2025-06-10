import os
import subprocess
import sys
from typing import Optional, Tuple
import argparse
import logging
import json

logger = logging.getLogger(__name__)


def evaluate_model(model_name: str, task_name: str,  metric_name: str, lm_eval_engine: str = "vllm", stop_with_exception: bool=True, limit: int = -1, eval_args: dict = {}, model_args ={},) -> float:
    """Evaluate the model using lm_eval API with VLLM backend."""
    import lm_eval
    logger.info(f"Evaluating {model_name} on {task_name} using VLLM...")

    try:
        model_args["pretrained"] = model_name
        default_args = {
            "tensor_parallel_size": 1,
            "dtype": "auto",
            "seed": 0,
        }

        model_args = {**default_args, **model_args}
        

        results = lm_eval.simple_evaluate(
            model=lm_eval_engine,
            model_args=model_args,
            batch_size="auto",
            tasks=[task_name],
            limit=limit,
            **eval_args,
        )

        # Extract the main metric (adjust based on your task)
        print(f'Results: {results["results"]}')
        if metric_name:
            score = results["results"][task_name][metric_name]
        else:
            score = [v.item() for k, v in results["results"][task_name].items() if "stderr" not in k and "alia" not in k ][0]
        logger.info(f"Evaluation score: {score}")
        return score
    except Exception as e:
        if stop_with_exception:
            raise e
        logger.error(f"Evaluation failed: {e}")
        return -1.0

def run_command(cmd: str, cwd: Optional[str] = None) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout

def install_environment() -> bool:
    """Install the environment with VLLM_USE_PRECOMPILED."""
    logger.info("Installing environment...")
    success, output = run_command("uv pip install lm_eval")
    if not success:
        logger.error(f"Installation failed: {output}")
    return success

def git_bisect(good: str, bad: str) -> bool:
    """Initialize git bisect with given good and bad commits."""
    logger.info(f"Starting bisect between good={good} and bad={bad}")
    success, _ = run_command(f"git bisect start {bad} {good}")
    return success

def git_checkout(commit: str) -> bool:
    """Checkout a specific commit."""
    logger.info(f"Checking out commit: {commit}")
    success, output = run_command(f"git checkout {commit}")
    if not success:
        logger.error(f"Checkout failed: {output}")
    return success

def bisect_commit(
    good_commit: str,
    bad_commit: str,
    model_name: str,
    task_name: str,
    target_score: float,
    metric_name: str,
    model_args: dict = {},
    eval_args: dict = {},
    lm_eval_engine: str = "vllm",
    stop_with_exception: bool=True,
    threshold: float = 0.02,
    limit: int = -1,
) -> Optional[str]:
    """Perform binary search to find where the score crosses the threshold."""

    if not install_environment():
        return None

    # Initialize bisect
    if not git_bisect(good_commit, bad_commit):
        return None

    while True:
        # Get current commit hash
        success, output = run_command("git rev-parse HEAD")
        if not success:
            return None
        current_commit = output.strip()
        logger.info(f"Testing commit: {current_commit}")

        # Evaluate model
        score = evaluate_model(
            model_name,
            task_name,
            metric_name,
            lm_eval_engine=lm_eval_engine,
            stop_with_exception=stop_with_exception,
            limit=limit,
            model_args=model_args,
            eval_args=eval_args,
        )
        if score < 0:
            return None

        # Determine if this commit is good or bad
        if score - target_score < -threshold:
            is_good = False
            logger.info("Marking as bad (score < target)")
        else:
            is_good = True
            logger.info("Marking as good (score >= target)")

        # Continue bisect
        cmd = "git bisect good" if is_good else "git bisect bad"
        success, output = run_command(cmd)
        if not success:
            return None

        # Check if bisect is complete
        if "first bad commit" in output.lower():
            run_command("git bisect reset")
            return output.split("\n")[0].split()[-1]

def main():
    parser = argparse.ArgumentParser(description="Bisect commits based on model performance.")
    parser.add_argument("--good", required=True, help="Known good commit hash")
    parser.add_argument("--bad", required=True, help="Known bad commit hash")
    parser.add_argument("--model", required=True, help="Model name (e.g. google/gemma-3-1b-it)")
    parser.add_argument("--task", default="gsm8k", help="Task name (default: gsm8k)")
    parser.add_argument("--target", type=float, required=True, help="Target score threshold")
    parser.add_argument("--threshold", type=float, default=0.02, help="Acceptable margin around target")
    parser.add_argument("--metric", default="", help="Metric name (default: em)")
    parser.add_argument("--stop_with_exception", action="store_true", help="Stop bisect if evaluation fails")
    parser.add_argument("--limit", type=int, default=-1, help="Limit the number of samples to evaluate")
    parser.add_argument("--engine", default="vllm", help="LM eval engine (default: vllm)")
    parser.add_argument("--model_args", type=str, default="{}", help="Model arguments as a JSON string")
    parser.add_argument("--eval_args", type=str, default="{}", help="Evaluation arguments as a JSON string")
    parser.add_argument("--bisect_log", type=str, nargs='?', help="Log file for bisect results")

    args = parser.parse_args()


    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.bisect_log:
        file_handler = logging.FileHandler(args.bisect_log)
        logger.addHandler(file_handler)
    try:
        result = bisect_commit(
            good_commit=args.good,
            bad_commit=args.bad,
            model_name=args.model,
            task_name=args.task,
            target_score=args.target,
            threshold=args.threshold,
            metric_name=args.metric,
            model_args=json.loads(args.model_args),
            eval_args=json.loads(args.eval_args),
            lm_eval_engine=args.engine,
            stop_with_exception=args.stop_with_exception,
            limit=args.limit,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        result = None
        run_command("git bisect reset")
        raise e

    if result:
        print(f"\nFound target commit: {result}")
    else:
        print("\nBisect failed.")

if __name__ == "__main__":
    main()
