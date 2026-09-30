#!/usr/bin/env bash
# Verify the renamed package builds, installs and imports as `domherre`.
# Tests the BUILT WHEEL in a throwaway venv, not the src/ tree.
# Read-only w.r.t. your repo, except for dist/ which it rebuilds.
set -uo pipefail

PKG=domherre
OLD=koltrast
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

pass=0; fail=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
no()   { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; }
section() { printf '\n\033[1m%s\033[0m\n' "$1"; }

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" || exit 1
echo "repo: $PWD"

command -v uv >/dev/null 2>&1 || { echo "error: uv not on PATH"; exit 1; }

# ---------------------------------------------------------------- build
section "1. build"
rm -rf dist/
if uv build >"$WORK/build.log" 2>&1; then ok "uv build"; else
  no "uv build"; sed 's/^/        /' "$WORK/build.log" | tail -20; exit 1
fi

WHL=$(ls dist/*.whl 2>/dev/null | head -1)
SDIST=$(ls dist/*.tar.gz 2>/dev/null | head -1)
[[ -n "$WHL"   ]] && ok "wheel produced: $(basename "$WHL")"   || { no "no wheel"; exit 1; }
[[ -n "$SDIST" ]] && ok "sdist produced: $(basename "$SDIST")" || no "no sdist"

case "$(basename "$WHL")" in
  ${PKG}-*) ok "wheel filename carries '$PKG'" ;;
  *)        no "wheel filename is $(basename "$WHL"), expected ${PKG}-*" ;;
esac

# ------------------------------------------------------ wheel contents
section "2. wheel contents"
python3 - "$WHL" "$PKG" "$OLD" << 'PY' && ok "wheel layout" || no "wheel layout"
import sys, zipfile
whl, pkg, old = sys.argv[1], sys.argv[2], sys.argv[3]
z = zipfile.ZipFile(whl)
names = z.namelist()
bad = []

tops = {n.split("/")[0] for n in names if "/" in n}
mods = {t for t in tops if not t.endswith((".dist-info", ".data"))}
if mods != {pkg}:
    bad.append(f"top-level packages in wheel: {sorted(mods) or 'NONE'} (expected ['{pkg}'])")

if f"{pkg}/py.typed" not in names:
    bad.append(f"{pkg}/py.typed missing -- the Typing :: Typed classifier is a lie")

pyfiles = [n for n in names if n.startswith(f"{pkg}/") and n.endswith(".py")]
if not pyfiles:
    bad.append("wheel contains no .py files -- empty package")

meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
if meta:
    txt = z.read(meta).decode("utf-8", "replace")
    nm = next((l.split(":",1)[1].strip() for l in txt.splitlines()
               if l.lower().startswith("name:")), None)
    if nm != pkg:
        bad.append(f"METADATA Name: {nm!r} (expected {pkg!r})")
else:
    bad.append("no METADATA in wheel")

for n in names:
    if old in n.lower():
        bad.append(f"path contains '{old}': {n}")
    if n.endswith((".py", ".txt", ".md", "METADATA", "RECORD", "py.typed")):
        try:
            if old.encode() in z.read(n).lower():
                bad.append(f"'{old}' found inside {n}")
        except Exception:
            pass

print(f"        modules={sorted(mods)}  files={len(pyfiles)}")
for b in bad:
    print("        " + b)
sys.exit(1 if bad else 0)
PY

# ------------------------------------------------------------- metadata
section "3. metadata"
if uv run --with twine --no-project twine check dist/* >"$WORK/tw.log" 2>&1; then
  ok "twine check"
else
  no "twine check"; sed 's/^/        /' "$WORK/tw.log" | tail -10
fi

# ------------------------------------------- install into a clean venv
section "4. clean-venv install"
uv venv "$WORK/venv" -q 2>/dev/null
PY_BIN="$WORK/venv/bin/python"
if uv pip install -q --python "$PY_BIN" "$WHL" >"$WORK/inst.log" 2>&1; then
  ok "wheel installs"
else
  no "wheel installs"; sed 's/^/        /' "$WORK/inst.log" | tail -20; exit 1
fi

# ------------------------------------------------------- runtime smoke
section "5. runtime (inside the clean venv)"
"$PY_BIN" - "$PKG" "$OLD" "$PWD" << 'PY'
import importlib, importlib.util, os, sys, traceback

pkg, old, repo = sys.argv[1], sys.argv[2], sys.argv[3]
P, F = [], []
def ok(m): P.append(m); print(f"  \033[32mPASS\033[0m  {m}")
def no(m): F.append(m); print(f"  \033[31mFAIL\033[0m  {m}")

try:
    m = importlib.import_module(pkg)
    ok(f"import {pkg}")
except Exception as e:
    no(f"import {pkg}: {e}"); traceback.print_exc(); sys.exit(1)

loc = os.path.realpath(m.__file__ or "")
if os.path.realpath(repo) in loc:
    no(f"resolved to the repo tree, not the installed wheel: {loc}")
else:
    ok(f"resolved to installed wheel: {loc}")

try:
    importlib.import_module(old)
    no(f"'{old}' is STILL importable -- stale install polluting the env")
except ModuleNotFoundError:
    ok(f"'{old}' is gone")

names = list(getattr(m, "__all__", []))
if not names:
    no("__all__ is empty or missing")
else:
    missing = [n for n in names if not hasattr(m, n)]
    if missing:
        no(f"__all__ names not resolvable: {missing}")
    else:
        ok(f"all {len(names)} __all__ names resolve")

for sub in ("backends", "core", "_frames", "_personnummer"):
    try:
        importlib.import_module(f"{pkg}.{sub}"); ok(f"import {pkg}.{sub}")
    except ModuleNotFoundError:
        print(f"  \033[33mSKIP\033[0m  {pkg}.{sub} (not present)")
    except Exception as e:
        no(f"import {pkg}.{sub}: {e}")

# pure-python paths: no model download needed
if hasattr(m, "is_valid_personnummer"):
    cases = [("850101-0006", True), ("198501010006", True), ("850101-0007", False)]
    bad = [(s, e) for s, e in cases if bool(m.is_valid_personnummer(s)) is not e]
    ok("is_valid_personnummer checksum") if not bad else no(f"checksum wrong for {bad}")

if hasattr(m, "find_personnummer"):
    got = m.find_personnummer("ring 850101-0006 eller order 0701234567")
    n = len(list(got))
    ok("find_personnummer finds 1, ignores the order number") if n == 1 \
        else no(f"find_personnummer returned {n} spans, expected 1")

# the renamed env var must be the one the code reads
if importlib.util.find_spec(f"{pkg}._frames"):
    src = open(importlib.import_module(f"{pkg}._frames").__file__, encoding="utf-8").read()
    if f"{old.upper()}_QUIET" in src:
        no(f"_frames.py still reads {old.upper()}_QUIET")
    elif f"{pkg.upper()}_QUIET" in src:
        ok(f"_frames.py reads {pkg.upper()}_QUIET")

print(f"\n  runtime: {len(P)} pass, {len(F)} fail")
sys.exit(1 if F else 0)
PY
[[ $? -eq 0 ]] && ok "runtime smoke" || no "runtime smoke"

# ----------------------------------------------------------- sdist
section "6. sdist"
tar xzf "$SDIST" -C "$WORK" 2>/dev/null
D=$(find "$WORK" -maxdepth 1 -type d -name "${PKG}-*" | head -1)
if [[ -n "$D" ]]; then
  [[ -d "$D/src/$PKG" ]] && ok "sdist contains src/$PKG" || no "sdist missing src/$PKG"
  if grep -rqil "$OLD" "$D" 2>/dev/null; then
    no "'$OLD' appears in the sdist:"; grep -ril "$OLD" "$D" | sed "s|$D|        .|"
  else
    ok "no '$OLD' anywhere in the sdist"
  fi
else
  no "sdist did not unpack"
fi

# ----------------------------------------------------------- summary
section "summary"
printf '  %d passed, %d failed\n' "$pass" "$fail"
if (( fail )); then
  echo "  NOT ready to publish"; exit 1
else
  echo "  wheel is sound -- ready for twine upload / a GitHub Release"
fi
