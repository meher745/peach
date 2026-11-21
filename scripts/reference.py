#!/usr/bin/env python3
"""Slow, literal reference implementation of rules.md, for cross-checking. Writes CSVs.
Usage: reference.py DATA_DIR OUT_DIR"""
import bisect, calendar, glob, json, os, re, sys
from collections import defaultdict

DAY = 86_400_000
data, out = sys.argv[1], sys.argv[2]

def lines(path):
    with open(path, 'rb') as f:
        raw = f.read()
    parts = raw.split(b'\n')
    if parts and parts[-1] == b'':
        parts.pop()
    return parts[1:]  # header

def parse(line, n):
    fields = line.split(b',')
    if len(fields) != n:
        return None
    res = []
    for f in fields:
        if b'"' in f:
            if len(f) >= 2 and f[:1] == b'"' and f[-1:] == b'"' and b'"' not in f[1:-1]:
                f = f[1:-1]
            else:
                return None
        res.append(f if f != b'' else None)
    return res

F1 = re.compile(rb'^([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(Z|([+-])([0-9]{2}):([0-9]{2}))$')
def ts(b):
    if b is None:
        return None
    if re.fullmatch(rb'[0-9]+', b):
        v = int(b)
        return v if v < 2**63 else None
    m = F1.match(b)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    if not (1 <= mo <= 12) or h > 23 or mi > 59 or s > 59 or y < 1:
        return None
    if d < 1 or d > calendar.monthrange(y, mo)[1]:
        return None
    days = calendar.timegm((y, mo, 1, 0, 0, 0)) // 86400
    t = ((days + d - 1) * 86400 + h * 3600 + mi * 60 + s) * 1000
    if m.group(7) != b'Z':
        oh, om = int(m.group(9)), int(m.group(10))
        if oh > 23 or om > 59:
            return None
        sign = 1 if m.group(8) == b'+' else -1
        t -= sign * (oh * 60 + om) * 60000
    return t

man = json.load(open(os.path.join(data, 'manifest.json')))
ws, we = ts(man['window_start'].encode()), ts(man['window_end'].encode())

products = {}
for l in lines(os.path.join(data, 'products.csv')):
    r = parse(l, 2)
    if r: products[r[0]] = r[1]
campaigns = {}
for l in lines(os.path.join(data, 'campaigns.csv')):
    r = parse(l, 2)
    if r: campaigns[r[0]] = r[1]

parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
for p in sorted(glob.glob(os.path.join(data, 'links', '*.csv'))):
    for l in lines(p):
        r = parse(l, 2)
        if not r or r[0] is None or r[1] is None:
            continue
        a, b = find(r[0]), find(r[1])
        if a != b:
            parent[a] = b
name = {}
for x in list(parent):
    rt = find(x)
    if rt not in name or x < name[rt]:
        name[rt] = x
def person(u):
    return name[find(u)] if u in parent else u

best = {}
for p in sorted(glob.glob(os.path.join(data, 'events', '*.csv'))):
    for l in lines(p):
        r = parse(l, 7)
        if not r or r[0] is None:
            continue
        ing = ts(r[3])
        if r[0] not in best or ing < best[r[0]][0]:
            best[r[0]] = (ing, r)
evs = defaultdict(list)
for ing, r in best.values():
    eid, u, t, _, typ, prod, price = r
    t = ts(t)
    if u is None or t is None or not (ws <= t < we) or typ not in (b'view', b'cart', b'purchase') or prod is None:
        continue
    pr = 0
    if typ == b'purchase':
        if price is None or not re.fullmatch(rb'[0-9]+', price) or not (1 <= int(price) <= 10_000_000):
            continue
        pr = int(price)
    if ing - t > DAY:
        continue
    evs[person(u)].append((t, typ == b'purchase', pr, products.get(prod, b'unknown')))

clicks = {}
for p in sorted(glob.glob(os.path.join(data, 'clicks', '*.csv'))):
    for l in lines(p):
        r = parse(l, 4)
        if not r or r[0] is None:
            continue
        clicks.setdefault(r[0], r)
pcl = defaultdict(list)
for cid, u, c, t in clicks.values():
    t = ts(t)
    if u is None or t is None or not (ws - 7 * DAY <= t < we) or c not in campaigns:
        continue
    pcl[person(u)].append((t, cid, c))

ud = {}
cd = defaultdict(lambda: [0, 0, 0, 0])
for u, es in evs.items():
    es.sort(key=lambda e: e[0])
    cl = sorted(pcl.get(u, []))
    cts = [c[0] for c in cl]
    sessions = []
    for e in es:
        if sessions and e[0] - sessions[-1][-1][0] <= 1_800_000:
            sessions[-1].append(e)
        else:
            sessions.append([e])
    for s in sessions:
        st = s[0][0]
        day = st // DAY
        rev = sum(e[2] for e in s)
        purch = any(e[1] for e in s)
        lo = bisect.bisect_left(cts, st - 7 * DAY)
        hi = bisect.bisect_right(cts, st)
        win = cl[lo:hi]
        if win:
            mt = max(c[0] for c in win)
            camp = min((c for c in win if c[0] == mt), key=lambda c: c[1])[2]
        else:
            camp = b'organic'
        a = cd[(camp, day)]
        a[0] += 1; a[1] += purch; a[2] += rev
        for c in set(c[2] for c in win):
            cd[(c, day)][3] += 1
        row = ud.setdefault((u, day), [0, 0, 0, defaultdict(int)])
        row[0] += 1; row[1] += len(s); row[2] += rev
        for e in s:
            if e[1]:
                row[3][e[3]] += e[2]

import datetime
def ds(d):
    return (datetime.date(1970, 1, 1) + datetime.timedelta(days=d)).isoformat()
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, 'user_daily.csv'), 'w') as f:
    f.write('user_id,day,sessions,events,revenue_cents,top_category\n')
    for (u, d), (s, n, rv, cats) in ud.items():
        top = ''
        if cats:
            top = min(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0].decode()
        f.write(f'{u.decode()},{ds(d)},{s},{n},{rv},{top}\n')
with open(os.path.join(out, 'campaign_daily.csv'), 'w') as f:
    f.write('campaign_id,channel,day,sessions,purchasing_sessions,revenue_cents,assisted_sessions\n')
    for (c, d), a in cd.items():
        ch = b'organic' if c == b'organic' else campaigns[c]
        f.write(f'{c.decode()},{ch.decode()},{ds(d)},{a[0]},{a[1]},{a[2]},{a[3]}\n')
