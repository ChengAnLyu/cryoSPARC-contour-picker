#!/usr/bin/env python3
"""Compatibility launcher for SARU Picker 3.0.0, whose default is now crYOLO."""
from app import main as application_main


def main(argv=None):
    return application_main(argv, default_backend="cryolo")


if __name__ == "__main__":
    raise SystemExit(main())
