"""Guarded screen for calibrated Gaussian and integrated-Student GLOC posteriors.

With no mode, print the plan and exit. --smoke runs correctness-scale fitting;
--full runs the declared three-seed ranker/random-forest screen.
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--full", action="store_true")
    return parser.parse_args()


def print_plan():
    print("Bayesian refinement screen is prepared but not launched.")
    print("Baseline: base79 + robust calibration79 + WBLOCK8 + white-GLOC7")
    print("Candidates: calibrated Gaussian uncertainty5, integrated-Student uncertainty5, union10")
    print("Seeds: 42, 7, 123; promotion: mean_d >= 0.0015 and positive 3/3")
    print("Smoke: env N_JOBS=15 .venv/bin/python _bayes_refine.py --smoke")
    print("Full:  env N_JOBS=15 .venv/bin/python _bayes_refine.py --full")


def main():
    args = parse_args()
    if not args.smoke and not args.full:
        print_plan()
        return

    os.environ["GLOC6"] = "1"
    os.environ["WBLOCK"] = "1"
    os.environ["BAYES_REFINE"] = "1"
    os.environ["BASELINE"] = "czrb_all79,wblock,white_gloc"
    os.environ["CANDIDATES"] = "white_gauss_cal,white_student_cal,white_both_cal"
    os.environ["CALIB_OUTPUT_STEM"] = "_bayes_refine"
    os.environ["SMOKE"] = "1" if args.smoke else "0"

    import _calib_rankerrf as experiment

    experiment.main()


if __name__ == "__main__":
    main()
