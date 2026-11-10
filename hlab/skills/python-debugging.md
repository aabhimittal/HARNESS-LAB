---
name: python-debugging
description: Find and fix bugs in Python code, using failing tests and tracebacks.
---
1. Reproduce first: run the failing test or script and read the full traceback, bottom frame first.
2. Locate: the bug is usually where expected and actual values first diverge. Add a temporary `print` or run
   `python -c "from mod import f; print(f(...))"` on the smallest input that fails.
3. Fix the cause, not the symptom. Do not edit tests to make them pass unless the task says so.
4. Re-run every test after the fix; one fix can break another case.
5. Common culprits: off-by-one in ranges/slices, wrong operator, mutable default arguments, integer vs float
   division, string vs int comparisons from parsed input, forgetting to return a value.
