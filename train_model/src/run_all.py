from __future__ import annotations

import argparse
from types import SimpleNamespace

from src.report import generate_report
from src.train_deep import MODEL_NAMES, train_model
from src.train_random_forest import train_random_forest
from src.utils import config_path_default, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(config_path_default()))
    parser.add_argument("--models", default="random_forest,cnn,lstm,transformer")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show tqdm batch progress inside each epoch.")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    requested = [item.strip().lower() for item in args.models.split(",") if item.strip()]

    if "randomforest" in requested:
        requested[requested.index("randomforest")] = "random_forest"
    if "rf" in requested:
        requested[requested.index("rf")] = "random_forest"
    if "1d-cnn" in requested:
        requested[requested.index("1d-cnn")] = "cnn"
    if "transformer" not in requested and "tranformer" in requested:
        requested[requested.index("tranformer")] = "transformer"

    if "random_forest" in requested:
        rf_args = SimpleNamespace(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            seed=args.seed,
            skip_tsne=args.skip_tsne,
        )
        train_random_forest(cfg, rf_args)

    deep_args = SimpleNamespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.num_workers,
        skip_tsne=args.skip_tsne,
        progress=args.progress,
        no_progress=args.no_progress,
    )
    for model_name in MODEL_NAMES:
        if model_name in requested:
            train_model(model_name, cfg, deep_args)

    if not args.skip_report:
        generate_report(cfg)


if __name__ == "__main__":
    main()
