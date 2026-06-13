from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any


def _worker(code: str, entry_point: str, tests: list[dict[str, Any]], queue: mp.Queue) -> None:
    try:
        namespace: dict[str, Any] = {}
        exec(code, namespace, namespace)
        fn = namespace.get(entry_point)
        if not callable(fn):
            queue.put({"error_type": "missing_entry_point", "error_message": entry_point})
            return
        results = []
        for test in tests:
            args = test.get("args", [])
            expected = test.get("expected")
            try:
                actual = fn(*args)
                passed = actual == expected
                results.append({"args": args, "expected": expected, "actual": actual, "passed": passed})
            except Exception as exc:  # noqa: BLE001 - executor should report candidate errors.
                results.append({"args": args, "expected": expected, "actual": None, "passed": False, "error": repr(exc)})
        queue.put({"assert_results": results})
    except Exception as exc:  # noqa: BLE001
        queue.put({"error_type": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})


def run_function_tests(code: str, entry_point: str, tests: list[dict[str, Any]], timeout_seconds: float = 2.0) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker, args=(code, entry_point, tests, queue))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "passed": False,
            "pass_rate": 0.0,
            "passed_tests": 0,
            "total_tests": len(tests),
            "timeout": True,
            "assert_results": [],
        }
    if queue.empty():
        return {
            "passed": False,
            "pass_rate": 0.0,
            "passed_tests": 0,
            "total_tests": len(tests),
            "error_type": "empty_executor_result",
            "assert_results": [],
        }
    payload = queue.get()
    results = payload.get("assert_results", [])
    passed_tests = sum(1 for item in results if item.get("passed"))
    total_tests = len(tests)
    return {
        **payload,
        "passed": passed_tests == total_tests and total_tests > 0,
        "pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "timeout": False,
    }

