//! Self-contained synthetic dataset generator for build-time PGO training only.
//! Compiled directly with `rustc -O src/pgo_gen.rs` inside the Docker builder
//! stage and never linked into the runtime binary.
//!
//! Matches the statistical profile of `rules.md`: 50K products / 20 categories,
//! 2K campaigns / 5 channels, 45% unlinked + 55% 2..8-alias clusters + whales &
//! shared-device clusters, 58% F1 / 32% F3 / 10% F2 timestamps, 87% view / 10%
//! cart / 3% purchase, 20% quoted fields, 3% event dups, 1% late events.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

struct Rng(u64);
impl Rng {
    #[inline(always)]
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    #[inline(always)]
    fn u(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// 93% 9-byte and 7% 8-byte base36 IDs, matching the real generator.
fn b36(prefix: u8, v: u64, out: &mut String) {
    const DIG: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    out.clear();
    out.push(prefix as char);
    let mut x = v.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (v >> 17);
    let len = if (x & 15) == 0 { 7 } else { 8 };
    let mut buf = [0u8; 8];
    for i in (0..len).rev() {
        buf[i] = DIG[(x % 36) as usize];
        x /= 36;
    }
    for &c in &buf[..len] {
        out.push(c as char);
    }
}

fn fmt_utc(sec_from_oct1: i64, out: &mut String) {
    let day = 1 + (sec_from_oct1 / 86_400);
    let rem = sec_from_oct1 % 86_400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let s = rem % 60;
    use std::fmt::Write as _;
    out.clear();
    let _ = write!(out, "2026-10-{day:02}T{h:02}:{m:02}:{s:02}Z");
}

#[inline(always)]
fn q(rng: &mut Rng, s: &str, w: &mut impl Write) {
    if rng.u(5) == 0 {
        let _ = w.write_all(b"\"");
        let _ = w.write_all(s.as_bytes());
        let _ = w.write_all(b"\"");
    } else {
        let _ = w.write_all(s.as_bytes());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0x5eed_cafe_1234_5678);
    let mut rng = Rng(seed);

    for sub in ["events", "clicks", "links"] {
        fs::create_dir_all(dir.join(sub)).unwrap();
    }
    fs::write(
        dir.join("manifest.json"),
        "{\"window_start\":\"2026-10-08T00:00:00Z\",\"window_end\":\"2026-10-22T00:00:00Z\"}\n",
    )
    .unwrap();

    // products.csv (50,000 products, 20 categories)
    {
        let mut w = BufWriter::with_capacity(1 << 16, File::create(dir.join("products.csv")).unwrap());
        writeln!(w, "product_id,category").unwrap();
        for i in 0..50_000u32 {
            writeln!(w, "p{i},cat_{:02}", i % 20).unwrap();
        }
    }
    // campaigns.csv (2,000 campaigns, 5 channels)
    let chans = ["search", "social", "email", "display", "video"];
    {
        let mut w = BufWriter::with_capacity(1 << 16, File::create(dir.join("campaigns.csv")).unwrap());
        writeln!(w, "campaign_id,channel").unwrap();
        for i in 0..2_000usize {
            writeln!(w, "c_{i},{}", chans[i % chans.len()]).unwrap();
        }
    }

    // Realistic person & alias structure:
    // ~30,000 people -> ~60,000 user IDs:
    //   45% have 1 ID (unlinked -> Tier 1 in-place solve!)
    //   30% have 2 IDs, 13% have 3 IDs, 12% have 4..8 IDs (linked -> Tier 2!)
    //   3 whales (indices 0..3) with 4 IDs each taking ~4.5% of events
    //   2 shared-device clusters (100 people linked together each)
    const N_PEOPLE: usize = 30_000;
    const N_EVENTS: usize = 900_000;
    const N_CLICKS: usize = 225_000;
    const OCT1_MS: i64 = 1_790_812_800_000; // 2026-10-01T00:00:00Z

    let mut person_start: Vec<u32> = Vec::with_capacity(N_PEOPLE + 1);
    let mut next_uid = 1u32;
    let mut link_edges: Vec<(u32, u32)> = Vec::with_capacity(45_000);
    for p in 0..N_PEOPLE {
        person_start.push(next_uid);
        let n_ids = if p < 3 {
            4
        } else {
            let r = rng.u(100);
            if r < 45 {
                1
            } else if r < 75 {
                2
            } else if r < 88 {
                3
            } else {
                4 + (rng.u(5) as u32)
            }
        };
        for k in 1..n_ids {
            let parent = next_uid + (rng.u(k as u64) as u32);
            link_edges.push((next_uid + k, parent));
        }
        next_uid += n_ids;
    }
    person_start.push(next_uid);
    // 2 shared-device clusters of 100 people each
    for cl in 0..2usize {
        let base_p = 100 + cl * 200;
        let hub = person_start[base_p];
        for off in 1..100usize {
            link_edges.push((person_start[base_p + off], hub));
        }
    }
    let total_uids = (next_uid - 1) as u64;

    let mut s1 = String::with_capacity(32);
    let mut s2 = String::with_capacity(32);
    let mut ts_s = String::with_capacity(32);
    let mut ing_s = String::with_capacity(32);
    let mut camp_s = String::with_capacity(16);
    let mut prod_s = String::with_capacity(16);

    // links (shuffle slightly + 5% repeats)
    {
        let mut w = BufWriter::with_capacity(1 << 16, File::create(dir.join("links/part-00000.csv")).unwrap());
        writeln!(w, "user_id_a,user_id_b").unwrap();
        for &(a, b) in &link_edges {
            b36(b'u', a as u64, &mut s1);
            b36(b'u', b as u64, &mut s2);
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b"\n");
            if rng.u(20) == 0 {
                let _ = w.write_all(s2.as_bytes());
                let _ = w.write_all(b",");
                let _ = w.write_all(s1.as_bytes());
                let _ = w.write_all(b"\n");
            }
        }
    }

    // clicks
    {
        let mut w = BufWriter::with_capacity(1 << 16, File::create(dir.join("clicks/part-00000.csv")).unwrap());
        writeln!(w, "click_id,user_id,campaign_id,ts").unwrap();
        for i in 0..N_CLICKS {
            let uid = if rng.u(20) == 0 {
                // 5% clicks on whale users
                let p = rng.u(3) as usize;
                let lo = person_start[p];
                let hi = person_start[p + 1];
                lo + (rng.u((hi - lo) as u64) as u32)
            } else if rng.u(7) == 0 {
                // ~14% clicks on random uids (many 0-event-in-shard -> trains flush_miss_clicks)
                (rng.u(total_uids) + 1) as u32
            } else {
                let p = rng.u(N_PEOPLE as u64) as usize;
                let lo = person_start[p];
                let hi = person_start[p + 1];
                if hi - lo == 1 || rng.u(2) == 0 { lo } else { lo + (rng.u((hi - lo) as u64) as u32) }
            };
            b36(b'k', i as u64 + 1, &mut s1);
            b36(b'u', uid as u64, &mut s2);
            let camp_id = if rng.u(10) < 3 { rng.u(4) } else { 4 + rng.u(1996) };
            use std::fmt::Write as _;
            camp_s.clear();
            let _ = write!(camp_s, "c_{camp_id}");
            let sec = rng.u(21 * 86_400) as i64; // Oct 1..22
            let tf = rng.u(100);
            if tf < 58 {
                fmt_utc(sec, &mut ts_s);
            } else if tf < 90 {
                ts_s.clear();
                let _ = write!(ts_s, "{}", OCT1_MS + sec * 1000 + (rng.u(1000) as i64));
            } else {
                fmt_utc(sec, &mut ts_s);
                ts_s.pop();
                ts_s.push_str("-04:00");
            }
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &camp_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ts_s, &mut w);
            let _ = w.write_all(b"\n");
        }
    }

    // events (split across 4 parts so parallel chunking is exercised)
    let per_part = N_EVENTS / 4;
    for part in 0..4usize {
        let mut w = BufWriter::with_capacity(
            1 << 16,
            File::create(dir.join(format!("events/part-{part:05}.csv"))).unwrap(),
        );
        writeln!(w, "event_id,user_id,ts,ingest_ts,event_type,product_id,price_cents").unwrap();
        for j in 0..per_part {
            let gid = (part * per_part + j) as u64;
            let eid_num = if rng.u(33) == 0 && gid > 200 { gid - rng.u(200) - 1 } else { gid + 1 };
            // 5% on 3 whales (>15k events each -> timebins), 25% on active 10% of people, 70% across all people
            let p = {
                let r = rng.u(100);
                if r < 5 {
                    rng.u(3) as usize
                } else if r < 30 {
                    3 + (rng.u(3000) as usize)
                } else {
                    rng.u(N_PEOPLE as u64) as usize
                }
            };
            let lo = person_start[p];
            let hi = person_start[p + 1];
            let uid = if hi - lo == 1 || rng.u(2) == 0 { lo } else { lo + (rng.u((hi - lo) as u64) as u32) };
            b36(b'e', eid_num, &mut s1);
            b36(b'u', uid as u64, &mut s2);
            // Cluster timestamps slightly per person so sessions have multiple events
            let base_sec = ((p as u64 * 7919) % (13 * 86_400)) as i64;
            let off_sec = if rng.u(3) == 0 {
                rng.u(14 * 86_400) as i64
            } else {
                (base_sec + rng.u(2 * 86_400) as i64) % (14 * 86_400)
            };
            let sec = 7 * 86_400 + off_sec; // Oct 8..22
            let late = if rng.u(100) == 0 { 95_000 } else { rng.u(600) as i64 };
            let ing_sec = (sec + late).min(29 * 86_400);
            use std::fmt::Write as _;
            let tf = rng.u(100);
            if tf < 58 {
                fmt_utc(sec, &mut ts_s);
            } else if tf < 90 {
                ts_s.clear();
                let _ = write!(ts_s, "{}", OCT1_MS + sec * 1000 + (rng.u(1000) as i64));
            } else {
                fmt_utc(sec, &mut ts_s);
                ts_s.pop();
                ts_s.push_str("-04:00");
            }
            fmt_utc(ing_sec, &mut ing_s);
            let kind = rng.u(100);
            let (et, price) = if kind < 87 {
                ("view", "")
            } else if kind < 97 {
                ("cart", "")
            } else {
                ("purchase", "2499")
            };
            prod_s.clear();
            let _ = write!(prod_s, "p{}", rng.u(50_000));
            q(&mut rng, &s1, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &s2, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ts_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &ing_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, et, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, &prod_s, &mut w);
            let _ = w.write_all(b",");
            q(&mut rng, price, &mut w);
            let _ = w.write_all(b"\n");
        }
    }
}
