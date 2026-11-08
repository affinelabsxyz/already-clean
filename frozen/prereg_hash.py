"""sha256 of one prereg section: from the line starting with HEADING to the line before the next '## ' heading (or EOF).
Usage: python prereg_hash.py notes/FORM_PREREG.md "## Amendment 4 (2026-10-17"   -> prints <sha256>  <n lines>"""
import hashlib, sys
path, head = sys.argv[1], sys.argv[2]
lines = open(path, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(lines) if l.startswith(head))
j = next((k for k in range(i + 1, len(lines)) if lines[k].startswith("## ")), len(lines))
sec = "\n".join(lines[i:j]).rstrip("\n") + "\n"
print(hashlib.sha256(sec.encode("utf-8")).hexdigest(), j - i)
