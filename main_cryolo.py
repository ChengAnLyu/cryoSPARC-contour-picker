#!/usr/bin/env python3
"""Compatibility entry point for the unified four-profile SARU Picker 3.0.0."""
from app import main as application_main


def main(argv=None):
    return application_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
