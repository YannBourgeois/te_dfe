#!/usr/bin/env python3
"""
model_select.py — shared --model handling for every pipeline script.

Model selection must happen BEFORE any `import te_dfe.*`, because te_dfe.config
dispatches to the simple/context overlay at import time based on the
TE_DFE_MODEL environment variable. So each script does, at the very top of
main() (before importing te_dfe):

    from model_select import select_model
    args, _ = make_parser().parse_known_args()
    select_model(args.model)        # sets TE_DFE_MODEL; CLI overrides config
    import te_dfe...                # now safe — config is locked to the choice

add_model_arg(parser) adds a uniform `--model {simple,context}` flag. If --model
is omitted, the existing TE_DFE_MODEL env var (or config default) is used, so
config can still set the default and the CLI overrides it when given.
"""
import argparse
import os
import sys


def prescan_model(argv=None):
    """
    Scan sys.argv for `--model X` (or `--model=X`) and set TE_DFE_MODEL before
    te_dfe is imported. Call this at the top of a script, BEFORE importing
    te_dfe. CLI value overrides any pre-existing env var / config default.
    Returns the resolved model (or None if not given and no env var).
    """
    argv = argv if argv is not None else sys.argv[1:]
    model = None
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
        elif a.startswith("--model="):
            model = a.split("=", 1)[1]
    if model:
        if model not in ("simple", "context"):
            raise SystemExit(f"--model must be simple|context, got {model!r}")
        os.environ["TE_DFE_MODEL"] = model
    return os.environ.get("TE_DFE_MODEL")


def add_model_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model", choices=["simple", "context"], default=None,
        help="Which model to run. Overrides the TE_DFE_MODEL env var / config "
             "default. If omitted, uses TE_DFE_MODEL or the config default "
             "(simple).")
    return parser


def select_model(model):
    """
    Lock the model for this process by setting TE_DFE_MODEL, BEFORE te_dfe is
    imported. CLI value (if given) overrides any pre-existing env var. Returns
    the resolved model name.
    """
    if model:                                   # explicit --model wins
        os.environ["TE_DFE_MODEL"] = model
    return os.environ.get("TE_DFE_MODEL", "simple")
