"""Scan an IFC (text) for strings that look personal or infrastructural: emails, paths, hosts, user ids, and the
values of person-ish properties. Prints unique hits with the entity type that holds them."""
import re, sys
from collections import defaultdict
path = sys.argv[1]
txt = open(path, encoding="latin-1").read()
hits = defaultdict(set)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WIN = re.compile(r"[A-Za-z]:\\\\?[^',)]{3,}|\\\\\\\\[^',)]{3,}")
UNIX = re.compile(r"(?<![A-Za-z])/(?:Users|home|mnt|Volumes|srv)/[^',)]+")
for m in EMAIL.finditer(txt): hits["email"].add(m.group(0))
for m in WIN.finditer(txt): hits["winpath"].add(m.group(0)[:160])
for m in UNIX.finditer(txt): hits["unixpath"].add(m.group(0)[:160])
for m in re.finditer(r"^\* (Host|Database)[^\n]*", txt, re.M): hits["comment"].add(m.group(0)[:160])
PROP = re.compile(r"IFCPROPERTYSINGLEVALUE\('([^']*)',[^,]*,(IFC[A-Z]+\('((?:[^']|'')*)'\))", re.I)
KEYS = re.compile(r"author|designer|drawn|checked|approved|client|owner|issued|organi[sz]ation|creat|user|name of|person|engineer|architect|contact|phone|e-?mail|bearbeiter|gepr|gezeichnet|verfasser|planer|bauherr", re.I)
for m in PROP.finditer(txt):
    k, v = m.group(1), m.group(3)
    if KEYS.search(k) and v.strip() and v.strip() not in ("-", "?"):
        hits["prop"].add(f"{k} = {v[:100]}")
for ent in ("IFCPERSON", "IFCORGANIZATION", "IFCTELECOMADDRESS", "IFCPOSTALADDRESS", "IFCACTOR", "IFCPERSONANDORGANIZATION", "IFCPROJECT", "IFCAPPLICATION"):
    for m in re.finditer(rf"^#\d+=\s*{ent}\((.*)\);\s*$", txt, re.M):
        hits[ent.lower()].add(m.group(1)[:200])
for k in sorted(hits):
    vals = sorted(hits[k])
    print(f"  {k} ({len(vals)}):")
    for v in vals[:25]: print("     ", v)
