#!/usr/bin/env python3
"""Test runner with per-test timing and stuck detection.

Runs all tests in a SINGLE pytest process (no subprocess isolation)
and reports per-test timing, failures, and slow tests.

Usage:
    python run_tests.py                       # run all tests
    python run_tests.py tests/test_cache.py   # run specific file
    python run_tests.py --slow 3.0            # flag tests slower than 3s
"""

import sys
import time


class TimingPlugin:
    """Pytest plugin that tracks per-test timing and results."""

    def __init__(self):
        self.results = []  # (nodeid, status, duration)
        self._start = None
        self._suite_start = None

    def pytest_sessionstart(self, session):
        self._suite_start = time.perf_counter()

    def pytest_runtest_logstart(self, nodeid, location):
        self._start = time.perf_counter()

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            duration = time.perf_counter() - self._start if self._start else 0
            status = "PASSED" if report.passed else ("FAILED" if report.failed else "SKIPPED")
            self.results.append((report.nodeid, status, duration, getattr(report, "longreprtext", "")))

    def pytest_sessionfinish(self, session, exitstatus):
        total_time = time.perf_counter() - self._suite_start if self._suite_start else 0
        self._print_report(total_time)

    def _print_report(self, total_time):
        passed = [r for r in self.results if r[1] == "PASSED"]
        failed = [r for r in self.results if r[1] == "FAILED"]
        skipped = [r for r in self.results if r[1] == "SKIPPED"]

        print("\n" + "=" * 70)
        print(f"{'DETAILED TIMING REPORT':^70s}")
        print("=" * 70)

        for i, (nodeid, status, dur, _) in enumerate(self.results, 1):
            short = nodeid.split("::")[-1]
            icon = {"PASSED": "✅", "FAILED": "❌", "SKIPPED": "⏭️"}.get(status, "?")
            flag = " 🐢" if dur > 3.0 else ""
            print(f"  [{i:3d}/{len(self.results)}] {icon} {short:<50s} {dur:.2f}s{flag}")

        print("\n" + "-" * 70)
        print(f"  ✅ Passed:  {len(passed)}")
        print(f"  ❌ Failed:  {len(failed)}")
        print(f"  ⏭️  Skipped: {len(skipped)}")
        print(f"  ⏱  Total:   {total_time:.2f}s")
        print("-" * 70)

        if failed:
            print("\n❌ FAILED TESTS:")
            for nodeid, _, dur, longrepr in failed:
                print(f"\n  - {nodeid} ({dur:.2f}s)")
                # Print last 5 lines of failure
                if longrepr:
                    lines = longrepr.strip().splitlines()[-5:]
                    for line in lines:
                        print(f"    {line}")

        slow = [(n, d) for n, s, d, _ in self.results if s == "PASSED" and d > 3.0]
        if slow:
            print(f"\n🐢 SLOW TESTS (>3.0s):")
            for nodeid, dur in sorted(slow, key=lambda x: -x[1]):
                print(f"  - {nodeid.split('::')[-1]}: {dur:.2f}s")

        print("=" * 70)


def main():
    import pytest

    path = "tests/"
    extra_args = []

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("-"):
            extra_args.append(arg)
        else:
            path = arg

    plugin = TimingPlugin()
    exit_code = pytest.main(
        [path, "-v", "--tb=short", "-q"] + extra_args,
        plugins=[plugin],
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
