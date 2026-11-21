#!/usr/bin/env python3
"""Parallel SARU Picker crYOLO launcher based on the 2.9.0 interface."""
from app import main as application_main


def main(argv=None):
    return application_main(argv, default_backend="cryolo")


if __name__ == "__main__":
    raise SystemExit(main())
