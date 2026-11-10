---
name: verification
description: Check that work meets every requirement before declaring a task done.
---
- Re-read the task and list each requirement (file names, paths, exact formats, sorting, edge cases).
- Check outputs by running them, not by reading your own code: run the tests, `cat` the output file,
  execute the script with the documented arguments.
- Test an edge case the task mentions (empty input, missing file, duplicates).
- Exact formats matter: no extra whitespace or headers unless asked; check with `cat -A file`.
