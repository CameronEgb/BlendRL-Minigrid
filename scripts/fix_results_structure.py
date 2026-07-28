#!/usr/bin/env python3
"""
Wrapper script to reorganize results directory.
Delegates to scripts/reorganize_results.py.
"""
from scripts.reorganize_results import main as reorganize_main

if __name__ == "__main__":
    reorganize_main()
