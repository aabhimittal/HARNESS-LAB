---
name: searching-files
description: Locate files and content in large directory trees with find, grep and sort.
---
- Content: `grep -rn PATTERN DIR` (add `-I` to skip binaries, `-l` for file names only, `-o` for just the match,
  `-E` for extended regex). Hidden files are searched by grep -r; `ls` hides them.
- Names: `find DIR -type f -name '*.log'`; sizes: `find DIR -type f -printf '%s %p\n' | sort -n | tail`.
- Huge output: pipe through `head`, `wc -l` or redirect to a file and grep that, instead of printing it all.
- Deduplicate with `sort -u`; count with `sort | uniq -c | sort -rn`.
