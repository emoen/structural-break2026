"""Guarded launcher for the Bayesian white-GLOC posterior screen.

Running without a mode prints the prepared experiment and performs no computation.
Use --smoke only for plumbing validation. Use --full only after reviewing the smoke
output, checking disk space, and confirming that no other 15-job harness is active.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true", help="run the correctness-only smoke check")
    modes.add_argument("--full", action="store_true", help="run the declared three-seed screen")
    return parser.parse_args()


def print_plan():
    print("Bayesian white-GLOC posterior screen is prepared but not launched.")
    print("Baseline: base79 + robust calibration79 + WBLOCK8 + white-GLOC7")
    print("Candidate: baseline + Bayesian posterior8")
    print("Seeds: 42, 7, 123; promotion: mean_d >= 0.0015 and positive 3/3")
    print("Smoke: env N_JOBS=15 .venv/bin/python _bayes_white_gloc.py --smoke")
    print("Full:  env N_JOBS=15 .venv/bin/python _bayes_white_gloc.py --full")


def main():
    args = parse_args()
    if not args.smoke and not args.full:
        print_plan()
        return

    os.environ["GLOC6"] = "1"
    os.environ["WBLOCK"] = "1"
    os.environ["BAYES_GLOC"] = "1"
    os.environ["BASELINE"] = "czrb_all79,wblock,white_gloc"
    os.environ["CANDIDATES"] = "white_bayes"
    os.environ["CALIB_OUTPUT_STEM"] = "_bayes_white_gloc"
    os.environ["SMOKE"] = "1" if args.smoke else "0"

    import _calib_rankerrf as experiment

    experiment.main()


if __name__ == "__main__":
    main()
