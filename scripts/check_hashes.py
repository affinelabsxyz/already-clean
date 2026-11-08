"""Check that every frozen file matches the hash recorded in its pre-registration.

Scripts are hashed byte for byte. Pre-registration sections are hashed with frozen/prereg_hash.py, which normalises
line endings. Run from the repository root: python scripts/check_hashes.py
"""
import hashlib
import subprocess
import sys

FILES = {
    "frozen/cleanspill_probe.py": "ed8a7d6700934882a3317bcc9bbc644c97511be9597990440830c9267acb3ebd",
    "frozen/sum_cleanspill.py": "4a76138f80e15bd5dc5f587357878fde91497af1f77478d56bc896dc0929c7a8",
    "frozen/quietleak_probe.py": "bc0e8251ffbb2ef1e16b68ec8e8e108bdd0cc7038283d0dee61132bd42b02ce4",
    "frozen/sum_quietleak.py": "e9ba0e197d13cf9751153115a8c8f7968063079461cfcbc049f7d4d50553df1e",
    "frozen/rlleak_probe.py": "0fbfc4d7b4d7258fa218704866b524fe3784a3acfeb34e4335ed1c73bb01c330",
    "frozen/sum_rlleak.py": "8ab186a4b57f689683425b35f1e318e1ff5404111d05b0a402652d992b145e87",
}
SECTIONS = {
    ("frozen/CLEANSPILL_PREREG.md", "## Frozen design (small tier)"): "0fbf0dca421b55add2e866a4a53667501a91a3e4ad20a4fbbb3f5d9b2d317412",
    ("frozen/QUIETLEAK_PREREG.md", "## Frozen design"): "30075e63e3e64aede466194e1dbce1298b8a8b89843a4a563bc67977cbcd0a4a",
    ("frozen/RLLEAK_PREREG.md", "## Frozen design"): "b96f8551e58efbc1d21e5adbaefa6d9772e148cc183e4ee142cea8c47e35f236",
}
ok = True
for path, want in FILES.items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(("ok  " if got == want else "BAD ") + path, got[:8])
    ok &= got == want
for (path, head), want in SECTIONS.items():
    got = subprocess.run([sys.executable, "frozen/prereg_hash.py", path, head], capture_output=True, text=True).stdout.split()[0]
    print(("ok  " if got == want else "BAD ") + f"{path} [{head}]", got[:8])
    ok &= got == want
sys.exit(0 if ok else 1)
