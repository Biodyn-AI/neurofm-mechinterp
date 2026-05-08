#!/usr/bin/env python
"""Download one published BrainOmni checkpoint bundle."""

from __future__ import annotations

import argparse

from neurofm_mechinterp.checkpoints import download_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["tiny", "base", "braintokenizer"],
        required=True,
        help="Checkpoint bundle to download.",
    )
    args = parser.parse_args()
    path = download_variant(args.variant)
    print(path)


if __name__ == "__main__":
    main()

