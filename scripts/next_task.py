#!/usr/bin/env python3
"""Find, show, or tick tasks in todo.md (stdlib only).

  python scripts/next_task.py Yashwant          next task I drive whose dependencies are done
  python scripts/next_task.py Sushil
  python scripts/next_task.py --show T2-005     print one task block
  python scripts/next_task.py --tick T2-005     tick every acceptance-criteria box of a task
  python scripts/next_task.py --status          counts per person
A task is DONE when all its acceptance-criteria boxes are ticked.
"""
import re
import sys
from pathlib import Path

FILE = Path("todo.md")
HEAD = re.compile(r"^#### (T\d+-\d+): (.*)$", re.MULTILINE)


def load():
    text = FILE.read_text(encoding="utf-8")
    ms = list(HEAD.finditer(text))
    tasks = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        block = text[m.start():end]
        block = re.split(r"\n---\n", block)[0].rstrip()
        deps = re.search(r"\*\*Dependencies:\*\*\s*(.*?)\s*$", block, re.MULTILINE)
        dep_ids = re.findall(r"T\d+-\d+", deps.group(1)) if deps else []
        drv = re.search(r"\*\*Driver / Navigator:\*\*\s*(\w+)", block)
        boxes = re.findall(r"^- \[( |x)\]", block, re.MULTILINE)
        tasks.append({
            "id": m.group(1),
            "title": m.group(2),
            "block": block,
            "deps": dep_ids,
            "driver": drv.group(1) if drv else "",
            "boxes": boxes,
            "done": bool(boxes) and all(b == "x" for b in boxes),
        })
    return text, tasks


def main(argv):
    global FILE
    if "--file" in argv:
        i = argv.index("--file"); FILE = Path(argv[i + 1]); del argv[i:i + 2]
    text, tasks = load()
    by_id = {t["id"]: t for t in tasks}
    if not argv:
        print(__doc__); return 1
    if argv[0] == "--show":
        print(by_id[argv[1]]["block"]); return 0
    if argv[0] == "--status":
        for who in ("Yashwant", "Sushil", "Both"):
            mine = [t for t in tasks if t["driver"] == who]
            print(f"{who}: {sum(t['done'] for t in mine)}/{len(mine)} done")
        return 0
    if argv[0] == "--tick":
        t = by_id[argv[1]]
        new_block = t["block"].replace("- [ ]", "- [x]")
        with open(FILE, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.replace(t["block"], new_block, 1))
        print(f"ticked {t['id']}: {t['title']}"); return 0
    who = argv[0]
    waiting = []
    for t in tasks:
        if t["done"] or t["driver"] != who:
            continue
        missing = [d for d in t["deps"] if not by_id[d]["done"]]
        if not missing:
            print(t["block"]); return 0
        waiting.append((t["id"], missing))
    joint = [t["id"] for t in tasks if t["driver"] == "Both" and not t["done"]
             and all(by_id[d]["done"] for d in t["deps"])]
    if joint:
        print("No solo task ready. Joint task(s) ready (do these together):", ", ".join(joint))
    elif waiting:
        tid, missing = waiting[0]
        owners = {m: by_id[m]["driver"] for m in missing}
        print(f"Blocked. {tid} waits for {owners} (run `git pull --rebase`, then re-run).")
    else:
        print(f"All tasks for {who} are done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
