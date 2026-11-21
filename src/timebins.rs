//! Exact session-preserving event ordering without full timestamp sorting.
//!
//! Within any half-open `GAP_MS`-wide interval `[b*W, (b+1)*W)`, every positive
//! pairwise difference is strictly `< GAP_MS`. Placing the bin's minimum
//! timestamp first and its maximum timestamp last preserves every session start
//! and every cross-bin gap `> GAP_MS`; interior order cannot create or merge a
//! session and does not affect count, revenue, or category sums.

use crate::{Ev8, GAP_MS};

const WIDTH: u32 = GAP_MS as u32;
const BINS: usize = (u32::MAX / WIDTH) as usize + 1;
const MIN_EVENTS: usize = 257;

pub(crate) struct Scratch {
    cursors: Vec<u32>,
}

impl Scratch {
    pub fn new() -> Self {
        Self { cursors: vec![0; BINS] }
    }

    #[inline]
    pub fn order<'a>(&mut self, rows: &'a mut [Ev8], tmp: &'a mut Vec<Ev8>) -> &'a [Ev8] {
        let n = rows.len();
        if n < MIN_EVENTS {
            rows.sort_unstable();
            return rows;
        }
        let cptr = self.cursors.as_mut_ptr();
        let (mut first, mut last) = (BINS, 0usize);
        for e in rows.iter() {
            let b = (e.ts() / WIDTH) as usize;
            unsafe { *cptr.add(b) += 1 };
            first = first.min(b);
            last = last.max(b);
        }
        if first == last {
            unsafe { *cptr.add(first) = 0 };
            endpoints(rows);
            return rows;
        }
        let mut total = 0u32;
        for b in first..=last {
            unsafe {
                let p = cptr.add(b);
                let cnt = *p;
                *p = total;
                total += cnt;
            }
        }
        tmp.clear();
        tmp.reserve(n);
        let out = tmp.as_mut_ptr();
        for &e in rows.iter() {
            let b = (e.ts() / WIDTH) as usize;
            unsafe {
                let p = cptr.add(b);
                let idx = *p as usize;
                *p = (idx + 1) as u32;
                out.add(idx).write(e);
            }
        }
        unsafe { tmp.set_len(n) };
        let mut start = 0usize;
        for b in first..=last {
            let end = unsafe {
                let p = cptr.add(b);
                let v = *p as usize;
                *p = 0;
                v
            };
            if end - start >= 2 {
                endpoints(unsafe { tmp.get_unchecked_mut(start..end) });
            }
            start = end;
        }
        tmp
    }
}

#[inline(always)]
fn endpoints(rows: &mut [Ev8]) {
    let n = rows.len();
    if n < 2 {
        return;
    }
    let (mut min_i, mut max_i) = (0usize, 0usize);
    let mut min_t = unsafe { rows.get_unchecked(0).ts() };
    let mut max_t = min_t;
    for i in 1..n {
        let t = unsafe { rows.get_unchecked(i).ts() };
        if t < min_t {
            min_t = t;
            min_i = i;
        }
        if t > max_t {
            max_t = t;
            max_i = i;
        }
    }
    rows.swap(0, min_i);
    if max_i == 0 {
        max_i = min_i;
    }
    rows.swap(n - 1, max_i);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Debug, PartialEq, Eq)]
    struct Agg {
        start: u32,
        end: u32,
        events: usize,
        revenue: u64,
        cats: BTreeMap<u8, u64>,
    }

    impl Agg {
        fn new() -> Self {
            Self { start: u32::MAX, end: 0, events: 0, revenue: 0, cats: BTreeMap::new() }
        }
        fn add(&mut self, e: Ev8) {
            self.start = self.start.min(e.ts());
            self.end = self.end.max(e.ts());
            self.events += 1;
            if e.pc() != 0 {
                let p = (e.pc() >> 8) as u64;
                self.revenue += p;
                *self.cats.entry(e.pc() as u8).or_default() += p;
            }
        }
    }

    fn walk(rows: &[Ev8]) -> Vec<Agg> {
        let mut out: Vec<Agg> = Vec::new();
        let mut prev = i64::MIN;
        for &e in rows {
            if out.is_empty() || e.ts() as i64 - prev > GAP_MS {
                out.push(Agg::new());
            }
            out.last_mut().unwrap().add(e);
            prev = e.ts() as i64;
        }
        out
    }

    #[test]
    fn timebins_matches_full_sort_sessions() {
        let mut sc = Scratch::new();
        let mut tmp = Vec::new();
        let times = [
            0, WIDTH - 1, WIDTH, WIDTH + 1, 2 * WIDTH + 1, 3 * WIDTH + 2,
            86_399_999, 86_400_000, 86_400_001, 86_400_000 + WIDTH + 1,
            1_209_599_999, u32::MAX - 1, u32::MAX,
        ];
        for n in [0, 1, 2, 24, 256, 257, 513, 1024] {
            let mut rows: Vec<_> = (0..n)
                .map(|i| {
                    Ev8::new(
                        times[i % times.len()],
                        if i % 3 == 0 { 0 } else { ((i as u32 % 99 + 1) << 8) | (i as u32 % 3) },
                    )
                })
                .collect();
            for mode in 0..3 {
                if mode == 1 {
                    rows.reverse();
                } else if mode == 2 && !rows.is_empty() {
                    let l = rows.len();
                    rows.rotate_left(l / 3);
                }
                let mut sorted = rows.clone();
                sorted.sort_unstable();
                let want = walk(&sorted);
                let mut inp = rows.clone();
                let got_rows = sc.order(&mut inp, &mut tmp);
                assert_eq!(walk(got_rows), want);
                assert!(sc.cursors.iter().all(|&c| c == 0));
            }
        }
    }
}
