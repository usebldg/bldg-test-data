"""Scrub personal / infrastructure data from an IFC (STEP text) without touching geometry or semantics.

    python -I tools/ifc_scrub.py in.ifc out.ifc short_name      -> writes out.ifc, prints a JSON record of the changes

Changes (record lists kinds, never the removed values):
* the header's comment block (/* ... */: producer, host name, database path with the user's profile) is removed
* FILE_NAME: name -> '<short_name>.ifc', author -> (''), organization -> (''), authorization -> ''
* IfcPerson: Identification / FamilyName / GivenName -> 'anonymised' where set; MiddleNames / PrefixTitles /
  SuffixTitles -> $; Addresses -> $ (an IfcTelecomAddress / IfcPostalAddress only a person used is deleted)
* IfcTelecomAddress holding e-mail addresses / phone numbers that persons or organizations use: deleted, references $
Fails if an e-mail address or a user-profile / server path is left anywhere.
"""
import json
import re
import sys

src, dst, short = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(src, "rb").read()
txt = raw.decode("latin-1")                     # STEP is 7-bit; latin-1 round-trips any byte
rec = []

head_end = txt.index("DATA;")
head, data = txt[:head_end], txt[head_end:]

# 1. header comment block(s)
comments = re.findall(r"/\*.*?\*/", head, re.S)
if comments:
    head = re.sub(r"\s*/\*.*?\*/", "", head, flags=re.S)
    rec.append("header comment removed (the exporting computer's host name, a database path in the user's profile, "
               "the toolkit's licence id)")


def split_args(s: str) -> list[str]:
    """Top-level comma-separated arguments of a STEP parameter list (without the outer parentheses)."""
    out, depth, cur, i, n = [], 0, [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "'":                            # a string: '' is an escaped quote
            j = i + 1
            while True:
                j = s.index("'", j)
                if j + 1 < n and s[j + 1] == "'":
                    j += 2
                    continue
                break
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


# 2. FILE_NAME
m = re.search(r"FILE_NAME\s*\((.*?)\);\s*\n", head, re.S)
args = split_args(m.group(1))
assert len(args) == 7, args
what = []
if args[0].strip() not in ("''", f"'{short}.ifc'"):
    what.append("name")
for k, label in ((2, "author"), (3, "organization")):
    if args[k].strip() not in ("('')", "()", "$"):
        what.append(label)
if args[6].strip() not in ("''", "$"):
    what.append("authorization")
args[0], args[2], args[3], args[6] = f"'{short}.ifc'", "('')", "('')", "''"
head = head[:m.start(1)] + ",".join(args) + head[m.end(1):]
rec.append("FILE_NAME: name set to the file's name; author, organization and authorization emptied"
           + (f" (were set: {', '.join(what)})" if what else ""))

# 3. persons and their addresses
LINE = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\((.*)\);\s*$", re.M)
ents = {int(mm.group(1)): (mm.group(2), mm.group(3), mm.start(), mm.end()) for mm in LINE.finditer(data)}
refcount: dict[int, int] = {}
for eid, (cls, body, *_rest) in ents.items():
    for r in re.findall(r"#(\d+)", re.sub(r"'(?:[^']|'')*'", "", body)):
        refcount[int(r)] = refcount.get(int(r), 0) + 1

edits: dict[int, str | None] = {}               # entity id -> new body, None = delete the line
n_person = 0
drop_addr: set[int] = set()
for eid, (cls, body, *_r) in ents.items():
    if cls != "IFCPERSON":
        continue
    a = split_args(body)
    changed = False
    for k in (0, 1, 2):
        if a[k].strip() not in ("$", "''"):
            if a[k] != "'anonymised'":
                a[k] = "'anonymised'"
                changed = True
    for k in (3, 4, 5):
        if a[k].strip() != "$":
            a[k] = "$"
            changed = True
    if a[7].strip() != "$":
        drop_addr |= {int(x) for x in re.findall(r"#(\d+)", a[7])}
        a[7] = "$"
        changed = True
    if changed:
        edits[eid] = ",".join(a)
        n_person += 1
if n_person:
    rec.append(f"IfcPerson: {n_person} anonymised (identification / names set to 'anonymised')")

# telecom addresses with e-mail / phone numbers (used by persons or organizations)
n_tel = 0
for eid, (cls, body, *_r) in ents.items():
    if cls == "IFCTELECOMADDRESS" and re.search(r"'[^']+'", body):
        drop_addr.add(eid)
for eid in sorted(drop_addr):
    cls = ents[eid][0]
    users = [u for u, (c2, b2, *_r) in ents.items() if re.search(rf"#{eid}(?!\d)", re.sub(r"'(?:[^']|'')*'", "", b2))]
    for u in users:                              # take the address out of every Addresses list that holds it
        if u in edits and edits[u] is None:
            continue
        c2, b2 = ents[u][0], edits.get(u, ents[u][1])
        assert c2 in ("IFCPERSON", "IFCORGANIZATION"), (eid, c2)
        a = split_args(b2)
        k = 7 if c2 == "IFCPERSON" else 3
        left = [x for x in re.findall(r"#\d+", a[k]) if x != f"#{eid}"]
        a[k] = f"({','.join(left)})" if left else "$"
        edits[u] = ",".join(a)
    edits[eid] = None
    n_tel += cls == "IFCTELECOMADDRESS"
if n_tel:
    rec.append(f"IfcTelecomAddress: {n_tel} with e-mail address / phone number deleted")
n_post = len([e for e in drop_addr if ents[e][0] == "IFCPOSTALADDRESS"])
if n_post:
    rec.append(f"IfcPostalAddress: {n_post} of a person deleted")

out, pos = [], 0
for eid in sorted(edits, key=lambda e: ents[e][2]):
    cls, body, s0, e0 = ents[eid]
    out.append(data[pos:s0])
    if edits[eid] is not None:
        out.append(f"#{eid}= {cls}({edits[eid]});")
        pos = e0
    else:                                        # drop the line and its line break
        pos = e0 + (1 if data[e0:e0 + 1] == "\n" else 2 if data[e0:e0 + 2] == "\r\n" else 0)
out.append(data[pos:])
new = head + "".join(out)

# 4. nothing personal or infrastructural may be left
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
PATHS = re.compile(r"[A-Za-z]:\\\\?Users|\\\\\\\\[A-Za-z0-9]|/Users/|/home/", re.I)
left = EMAIL.findall(new) + PATHS.findall(new)
assert not left, left
gone = {e for e, b in edits.items() if b is None}
refs = {int(r) for r in re.findall(r"#(\d+)", re.sub(r"'(?:[^']|'')*'", "", new[new.index("DATA;"):]))}
defined = {int(d) for d in re.findall(r"^#(\d+)\s*=", new, re.M)}
assert not (gone & refs), gone & refs           # nothing still points at a deleted entity
assert refs <= defined, sorted(refs - defined)[:10]
open(dst, "wb").write(new.encode("latin-1"))
print(json.dumps({"file": dst, "changes": rec}))
