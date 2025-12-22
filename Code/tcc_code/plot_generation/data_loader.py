import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .config import LOGS_DIR, METRICS_TO_PLOT


def find_most_recent_experiment(algorithm: str, strategy: str) -> Optional[Path]:
    algo_dir = LOGS_DIR / algorithm
    if not algo_dir.exists():
        return None

    matching_dirs = [d for d in algo_dir.iterdir()
                     if d.is_dir() and strategy in d.name]

    if not matching_dirs:
        return None

    most_recent = max(matching_dirs, key=lambda d: d.stat().st_mtime)
    return most_recent


def load_metrics_json(exp_dir: Path) -> Optional[Dict]:
    metrics_file = exp_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics_by_round(data: Dict) -> Tuple[List[int], Dict[str, List[float]]]:
    metrics_by_round = data.get("metrics_by_round", {})

    rounds = sorted([int(r) for r in metrics_by_round.keys()])

    metrics = {metric: [] for metric in METRICS_TO_PLOT}

    for round_num in rounds:
        round_data = metrics_by_round[str(round_num)]
        for metric in METRICS_TO_PLOT:
            metrics[metric].append(round_data.get(metric, 0.0))

    return rounds, metrics


def load_all_experiments() -> Dict[str, Dict]:
    from .config import ALGORITHMS, STRATEGIES

    all_data = {}

    for algorithm in ALGORITHMS:
        for strategy in STRATEGIES:
            exp_dir = find_most_recent_experiment(algorithm, strategy)
            if not exp_dir:
                continue

            data = load_metrics_json(exp_dir)
            if not data:
                continue

            rounds, metrics = extract_metrics_by_round(data)
            exp_info = data['experiment_info']

            key = f"{algorithm}_{strategy}"
            all_data[key] = {
                'rounds': rounds,
                'metrics': metrics,
                'exp_info': exp_info,
                'data': data,
                'exp_dir': exp_dir
            }

    return all_data
