#!/usr/bin/env python3
"""
pytest script for ArXplorer test suite.
Usage:
    python scripts/run_tests.py               # Run all tests
    python scripts/run_tests.py --fast        # Run only fast unit tests
    python scripts/run_tests.py --coverage    # Generate coverage report
    python scripts/run_tests.py --verbose     # Verbose output
"""

import sys
import subprocess
from pathlib import Path


def run_tests(
    fast_only: bool = False,
    coverage: bool = True,
    verbose: bool = False,
    html_report: bool = True,
    fail_under: int = 70
):
    """
    Run pytest with specified options.
    
    Args:
        fast_only: Skip slow integration tests
        coverage: Generate coverage report
        verbose: Verbose test output
        html_report: Generate HTML coverage report
        fail_under: Minimum coverage percentage required
    """
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    
    if not tests_dir.exists():
        print(f"Tests directory not found: {tests_dir}")
        return 1
    
    # Base pytest command
    cmd = ["pytest", str(tests_dir)]
    
    # Add markers for fast-only mode
    if fast_only:
        cmd.extend(["-m", "not slow"])
    
    # Coverage options
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=xml",
        ])
        if html_report:
            cmd.append("--cov-report=html")
    
    # Verbosity
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Show short traceback on failures
    cmd.append("--tb=short")
    
    # Color output
    cmd.append("--color=yes")
    
    print("=" * 60)
    print("Running ArXplorer Test Suite")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run pytest from project root
    result = subprocess.run(cmd, cwd=project_root)
    
    # Check coverage threshold if enabled
    if coverage and result.returncode == 0:
        print()
        print("=" * 60)
        print(f"Checking coverage threshold (>= {fail_under}%)")
        print("=" * 60)
        threshold_cmd = ["coverage", "report", f"--fail-under={fail_under}"]
        threshold_result = subprocess.run(threshold_cmd, cwd=project_root)
        
        if threshold_result.returncode != 0:
            print()
            print(f"overage below {fail_under}% threshold")
            return threshold_result.returncode
        else:
            print()
            print(f"Coverage meets {fail_under}% threshold")
    
    return result.returncode


def main():
    """Parse arguments and run tests."""
    args = sys.argv[1:]
    
    fast_only = "--fast" in args
    no_coverage = "--no-coverage" in args
    verbose = "--verbose" in args or "-v" in args
    no_html = "--no-html" in args
    
    # Check for help
    if "--help" in args or "-h" in args:
        print(__doc__)
        return 0
    
    exit_code = run_tests(
        fast_only=fast_only,
        coverage=not no_coverage,
        verbose=verbose,
        html_report=not no_html,
        fail_under=70
    )
    
    if exit_code == 0:
        print()
        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        if not no_coverage:
            print()
            print("Coverage reports generated:")
            print("  - Terminal: (displayed above)")
            print("  - XML: coverage.xml")
            if not no_html:
                print("  - HTML: htmlcov/index.html")
    else:
        print()
        print("=" * 60)
        print("Tests failed")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
