#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HCS world v3 — organisms live ON the Mandelbrot set.

The habitat is the complex plane. Each organism carries a coordinate (cr, ci)
in its genome. Fertility at a point is the escape time of the Mandelbrot
iteration: empty space is poor, the frozen interior is sterile, and the
boundary — an infinitely long line of measure zero — is the only rich ground.

Resolution deepens every generation (maxiter grows). Points that were frozen
interior become boundary as you look closer, so new territory opens up
mid-run and the population can colonise it. That is the fractal lesson: every
zoom reveals the same shape, and a new place to live.

Population is held at an optimum K by logistic selection, not a hard cap.

Genome: strategy (code), cr/ci (habitat), mu (copy fidelity), eff (metabolism),
brood (children), diet (hunger). Predators eat neighbours' energy and code.

Mutation never escapes the strategy expression: AST is whitelisted first.

Usage:  python3 world.py --gens 80 --K 30
"""
import base64
import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POP = ROOT / "pop"
DEAD = ROOT / "dead"
LOG = ROOT / "world.jsonl"
MAP = ROOT / "positions.jsonl"
REPORT = ROOT / "report.md"

COST = 5
CASE_PTS = 2
ALL_PTS = 4
MAX_GENE = 240
TIME_LIMIT = 6
UPKEEP = 6.0       # upkeep is quadratic in efficiency: real metabolisms have diminishing returns
EFF_EXP = 0.5      # gain scales as sqrt(efficiency): hard diminishing returns
HUNT_COST = 4
HUNT_SHARE = 0.45
MAXITER_0 = 60
MAXITER_STEP = 3

CASES = [
    [2, 4, 6],
    [1, 3, 5, 7],
    [0, 8, 9, 10],
    [-2, -4, 3],
    [-1, 5, 12],
    [7, 7, 7, 8, 8],
]
EXPECTED = [sum(x * x for x in c if x % 2 == 0) for c in CASES]

SEED = {
    "id": "g000-root",
    "gen": 0,
    "parent": None,
    "energy": 18,
    "mu": 0.16,
    "eff": 1.0,
    "brood": 1,
    "diet": 0.05,
    "cr": -0.75,
    "ci": 0.11,
    "strategy": "sum(x*x for x in c if x>0 and x%2==0)",
    "lineage": ["g000-root"],
}


def escape(cr, ci, maxiter):
    zr = zi = 0.0
    for n in range(maxiter):
        zr, zi = zr * zr - zi * zi + cr, 2.0 * zr * zi + ci
        if zr * zr + zi * zi > 4.0:
            return n
    return maxiter


def fertility(cr, ci, maxiter):
    n = escape(cr, ci, maxiter)
    if n >= maxiter:          # frozen interior: stable, but nothing happens
        return 0.30
    return round(0.35 + 2.40 * (n / maxiter), 4)


TEMPLATE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HCS organism. Self-reading, self-rewriting. Do not edit by hand."""
import ast
import base64
import builtins
import copy
import math
import json
import os
import random
import re
import sys

_G = "__GENOME__"

COST = 5
MAX_GENE = 240
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789_+-*/%<>()[],. =!"
FUNCS = {"sum", "len", "min", "max", "abs", "sorted", "range", "all", "any"}
VARS = {"c", "x", "y"}
NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.GeneratorExp, ast.comprehension, ast.Call, ast.Name,
    ast.Load, ast.Store, ast.Del, ast.Constant, ast.ListComp, ast.List,
    ast.Tuple, ast.Set, ast.Subscript, ast.Slice, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.In, ast.NotIn, ast.keyword,
)


def genome():
    return json.loads(base64.b64decode(_G))


def verify(tree):
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            return "banned node %s" % type(node).__name__
        if isinstance(node, ast.Name) and node.id not in (FUNCS | VARS):
            return "banned name %s" % node.id
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS:
                return "banned call"
    return None


def compile_strategy(src):
    tree = ast.parse(src, mode="eval")
    bad = verify(tree)
    if bad:
        raise SyntaxError(bad)
    scope = {f: builtins.__dict__[f] for f in FUNCS}
    code = compile(tree, "<strategy>", "eval")

    def run(c):
        return eval(code, {"__builtins__": {}, "c": c}, dict(scope))
    return run


def mutate(g):
    s = g["strategy"]
    if random.random() < g["mu"]:
        for _ in range(random.randint(1, 3)):
            op = random.choice(["cut", "cut", "swap", "ins", "del"])
            if op == "cut" and len(s) > 8:
                i = random.randrange(len(s) - 4)
                s = s[:i] + s[min(len(s), i + random.randint(1, 9)):]
            elif op == "swap" and len(s) > 2:
                i = random.randrange(len(s) - 1)
                s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            elif op == "ins":
                i = random.randrange(len(s) + 1)
                s = s[:i] + random.choice(ALPHABET) + s[i:]
            elif op == "del" and len(s) > 6:
                i = random.randrange(len(s))
                s = s[:i] + s[i + 1:]
        g["strategy"] = s[:MAX_GENE]

    g["mu"] = round(min(0.60, max(0.02, g["mu"] + random.gauss(0, 0.02))), 4)
    g["eff"] = round(max(0.02, g["eff"] * math.exp(random.gauss(0, 0.07))), 4)
    g["brood"] = max(1, min(3, g["brood"] + random.choice([0, 0, 0, 1, -1])))
    g["diet"] = round(min(1.0, max(0.0, g["diet"] + random.gauss(0, 0.07))), 3)

    # habitat drift: mostly a crawl along the boundary, rarely a leap
    step = 0.004 if random.random() > 0.08 else 0.05
    g["cr"] = round(g["cr"] + random.gauss(0, step), 7)
    g["ci"] = round(g["ci"] + random.gauss(0, step), 7)
    return g


def write_child(parent, energy, outdir):
    src = open(__file__, encoding="utf-8").read()
    child = mutate(copy.deepcopy(parent))
    child["gen"] = parent["gen"] + 1
    child["parent"] = parent["id"]
    child["energy"] = max(2, energy)
    child["id"] = "%03d-%04x" % (child["gen"], random.getrandbits(16))
    child["lineage"] = (parent["lineage"] + [child["id"]])[-24:]
    blob = base64.b64encode(json.dumps(child, sort_keys=True).encode()).decode()
    text = re.sub(r'^_G = ".*"$', '_G = "%s"' % blob, src, count=1, flags=re.M)
    with open(os.path.join(outdir, child["id"] + ".py"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return child["id"]


def main():
    g = genome()
    if "--reproduce" in sys.argv:
        i = sys.argv.index("--reproduce")
        energy, outdir = int(sys.argv[i + 1]), sys.argv[i + 2]
        n = max(1, min(3, int(g.get("brood", 1))))
        share = max(2, energy // (1 + n))
        kids = [write_child(g, share, outdir) for _ in range(n)]
        print(json.dumps({"children": kids}))
        return 0

    if "--spawn" in sys.argv:
        i = sys.argv.index("--spawn")
        n, outdir = int(sys.argv[i + 1]), sys.argv[i + 2]
        kids = [write_child(g, 12, outdir) for _ in range(n)]
        print(json.dumps({"children": kids}))
        return 0

    payload = json.load(sys.stdin)
    answers, reason = [], None
    try:
        fn = compile_strategy(g["strategy"])
    except Exception as exc:
        fn, reason = None, "%s: %s" % (type(exc).__name__, exc)
    for case in payload["cases"]:
        if fn is None:
            answers.append(None)
            continue
        try:
            answers.append(fn(case))
        except Exception:
            answers.append(None)
    print(json.dumps({"id": g["id"], "gen": g["gen"], "answers": answers,
                      "strategy": g["strategy"], "mu": g["mu"],
                      "eff": g.get("eff", 1.0), "brood": g.get("brood", 1),
                      "diet": g.get("diet", 0.0),
                      "cr": g.get("cr", 0.0), "ci": g.get("ci", 0.0),
                      "lineage": g["lineage"], "dead_reason": reason}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def seed_file(path, genome=None):
    g = genome or SEED
    blob = base64.b64encode(json.dumps(g, sort_keys=True).encode()).decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.replace("__GENOME__", blob), encoding="utf-8")
    return path


def read_genome(path):
    blob = re.search(r'^_G = "(.*)"$', path.read_text(encoding="utf-8"), re.M).group(1)
    return json.loads(base64.b64decode(blob))


def save_genome(path, g):
    blob = base64.b64encode(json.dumps(g, sort_keys=True).encode()).decode()
    src = path.read_text(encoding="utf-8")
    path.write_text(re.sub(r'^_G = ".*"$', '_G = "%s"' % blob, src, count=1, flags=re.M),
                    encoding="utf-8")


def run_organism(path, args=None, stdin=None):
    try:
        p = subprocess.run([sys.executable, "-I", str(path)] + (args or []),
                           input=stdin, capture_output=True, text=True,
                           timeout=TIME_LIMIT, cwd=str(path.parent),
                           env={"PATH": "/usr/bin:/bin"})
        if p.returncode != 0:
            return {"answers": [], "error": (p.stderr or "").strip()[-160:]}
        return json.loads(p.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"answers": [], "error": "timeout"}
    except Exception as exc:
        return {"answers": [], "error": str(exc)[:160]}


def score(answers):
    right = sum(1 for got, want in zip(answers, EXPECTED) if got == want)
    return right * CASE_PTS + (ALL_PTS if right == len(EXPECTED) else 0), right


def main():
    gens, K, soup = 80, 60, 20
    if "--gens" in sys.argv:
        gens = int(sys.argv[sys.argv.index("--gens") + 1])
    if "--K" in sys.argv:
        K = int(sys.argv[sys.argv.index("--K") + 1])
    if "--soup" in sys.argv:
        soup = int(sys.argv[sys.argv.index("--soup") + 1])

    for d in (POP, DEAD):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    LOG.unlink(missing_ok=True)
    MAP.unlink(missing_ok=True)

    cur = POP / "gen000"
    cur.mkdir(parents=True)
    seed_file(cur / "g000-root.py")
    run_organism(cur / "g000-root.py", ["--spawn", str(max(1, soup - 1)), str(cur)])
    print("primordial soup: %d organisms on the boundary" % soup)

    history, log, mlog = [], open(LOG, "w", encoding="utf-8"), open(MAP, "w", encoding="utf-8")
    t0, best_ever = time.time(), None

    for gen in range(gens):
        maxiter = MAXITER_0 + gen * MAXITER_STEP
        files = sorted(cur.glob("*.py"))
        if not files:
            print("EXTINCTION at generation %d" % gen)
            history.append({"gen": gen, "extinct": True})
            break

        # --- 1. predation: only foreign code is worth eating ---------------
        gmap = {f: read_genome(f) for f in files}
        energy = {f: gmap[f].get("energy", 0) for f in files}
        hunts, hunted, refused = 0, set(), 0
        for f in files:
            g = gmap[f]
            if len(files) < 2 or random.random() >= g.get("diet", 0.0):
                continue
            mine = set(g.get("lineage", []))
            prey = [x for x in files if x != f
                    and gmap[x]["strategy"] != g["strategy"]
                    and not (mine & set(gmap[x].get("lineage", [])))]
            if not prey:
                refused += 1          # nothing but kin around: hunger unspent
                continue
            victim = random.choice(prey)
            share = int(energy[victim] * HUNT_SHARE)
            if share < 2:
                continue
            energy[f] += share - HUNT_COST
            energy[victim] -= share
            code = gmap[victim]["strategy"]
            if random.random() < 0.5:                     # digestion error
                code = code[:max(1, len(code) - 2)]
            g["strategy"] = code[:MAX_GENE]               # horizontal transfer
            save_genome(f, g)
            gmap[f] = g
            hunted.add(f)
            hunts += 1

        # --- 2. the environment bites ------------------------------------
        rows = []
        for f in files:
            out = run_organism(f, stdin=json.dumps({"cases": CASES}))
            pts, right = score(out.get("answers") or [])
            g = gmap[f]
            eff = g.get("eff", 1.0)
            fert = fertility(g["cr"], g["ci"], maxiter)
            # a hunter spends the generation hunting: foraging suffers
            pts_eff = int(round(pts * 0.6)) if f in hunted else pts
            gain = pts_eff * fert * (eff ** EFF_EXP)      # sqrt: hard diminishing returns
            cost = UPKEEP * (eff ** 2)                    # quadratic upkeep
            net = energy[f] + round(gain) - round(cost)
            rows.append({"file": f, "g": g, "pts": pts, "right": right,
                         "fert": fert, "energy": net,
                         "error": out.get("error") or out.get("dead_reason"),
                         "strategy": out.get("strategy", g["strategy"])})

        # --- 3. optimum population, not a cap ----------------------------
        rows.sort(key=lambda r: (-r["energy"], -r["right"]))
        mean_brood = sum(r["g"].get("brood", 1) for r in rows) / len(rows) or 1
        n_surv = max(3, min(len(rows), int(round(K / max(1.0, mean_brood)))))
        survivors = [r for r in rows if r["energy"] >= COST][:n_surv]
        for r in [x for x in rows if x not in survivors]:
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        nxt = POP / ("gen%03d" % (gen + 1))
        nxt.mkdir(exist_ok=True)
        born = 0
        for r in survivors:
            out = run_organism(r["file"], ["--reproduce", str(r["energy"]), str(nxt)])
            born += len(out.get("children") or [])
            shutil.move(str(r["file"]), str(DEAD / f"{gen:03d}-{r['file'].name}"))

        for r in rows:
            mlog.write(json.dumps({"gen": gen, "cr": r["g"]["cr"], "ci": r["g"]["ci"],
                                   "fert": r["fert"], "energy": r["energy"],
                                   "alive": r in survivors}) + "\n")
        mlog.flush()

        best = rows[0]
        carriers = sum(1 for r in rows if r["strategy"] == best["strategy"])
        rec = {"gen": gen, "maxiter": maxiter, "pop": len(rows),
               "survivors": len(survivors), "dead": len(rows) - len(survivors),
               "born": born, "hunts": hunts, "refused": refused,
               "best_energy": best["energy"], "best_right": best["right"],
               "best_fert": best["fert"], "best_cr": best["g"]["cr"],
               "best_ci": best["g"]["ci"],
               "perfect": sum(1 for r in rows if r["right"] == len(CASES)),
               "carriers": carriers,
               "mean_fert": round(sum(r["fert"] for r in rows) / len(rows), 3),
               "mean_energy": round(sum(r["energy"] for r in rows) / len(rows), 1),
               "mean_mu": round(sum(r["g"].get("mu", 0) for r in rows) / len(rows), 4),
               "mean_eff": round(sum(r["g"].get("eff", 1) for r in rows) / len(rows), 3),
               "mean_brood": round(sum(r["g"].get("brood", 1) for r in rows) / len(rows), 2),
               "mean_diet": round(sum(r["g"].get("diet", 0) for r in rows) / len(rows), 3),
               "errors": sum(1 for r in rows if r["error"]),
               "refused": refused,
               "best_strategy": best["strategy"],
               "best_lineage": best["g"].get("lineage", [])[-6:]}
        history.append(rec)
        if best_ever is None or (rec["best_right"], rec["best_energy"]) > \
                (best_ever["best_right"], best_ever["best_energy"]):
            best_ever = rec
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print("gen %02d it %3d pop %2d surv %2d born %2d hunt %2d kin-refused %2d | best %d/%d E%3d fert %.2f | "
              "mu %.3f eff %.2f brood %.1f diet %.2f | err %d"
              % (gen, maxiter, rec["pop"], rec["survivors"], born, hunts, refused, best["right"],
                 len(CASES), best["energy"], rec["mean_fert"], rec["mean_mu"],
                 rec["mean_eff"], rec["mean_brood"], rec["mean_diet"], rec["errors"]))
        cur = nxt

    log.close()
    mlog.close()
    print("\n--- %d generations, %.1fs ---" % (len(history), time.time() - t0))
    if best_ever:
        print("fittest: %s/%d energy %d at (%.6f, %.6f) fertility %.2f"
              % (best_ever["best_right"], len(CASES), best_ever["best_energy"],
                 best_ever["best_cr"], best_ever["best_ci"], best_ever["best_fert"]))
        print("strategy: %s" % best_ever["best_strategy"])

    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("# HCS world v3 — life on the Mandelbrot boundary\n\n")
        fh.write("| gen | maxiter | pop | surv | born | best | perfect | carrier |"
                 " fert | energy | eff | brood | diet | mu | err |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for h in history:
            if h.get("extinct"):
                fh.write("| %s | — | 0 | — | — | — | — | — | — | — | — | — | — | — | **extinction** |\n" % h["gen"])
                continue
            fh.write("| %s | %s | %s | %s | %s | %s/%d | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                h["gen"], h["maxiter"], h["pop"], h["survivors"], h["born"],
                h["best_right"], len(CASES), h["perfect"], h["carriers"],
                h["mean_fert"], h["mean_energy"], h["mean_eff"], h["mean_brood"],
                h["mean_diet"], h["mean_mu"], h["errors"]))
        if best_ever:
            fh.write("\n**fittest** `%s` at (%.6f, %.6f), fertility %.2f\n\n"
                     % (best_ever["best_strategy"], best_ever["best_cr"],
                        best_ever["best_ci"], best_ever["best_fert"]))
            fh.write("**lineage** `%s`\n" % " → ".join(best_ever["best_lineage"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
