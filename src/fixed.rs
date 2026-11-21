//! Two-pass, schema-fixed CSV tokenizer. Same contract as `csv::scan`.
//!
//! Pass 1 runs over one batch (up to 64 KB) and writes the offset of every comma
//! and newline into flat `u32` arrays. It also stores a quote bitmap per 64-byte
//! block with running quote counts. Pass 2 walks the newlines. Under this dialect
//! every comma separates fields, so a line has exactly N fields iff its
//! (N-1)th comma comes before the newline and its Nth comes after it: one compare
//! pair against the comma array. Its quoting is legal iff every quoted field
//! opens and closes on its own boundaries and the line holds no other quote,
//! i.e. the line's quote count is twice the number of quoted fields.
//!
//! A line that fails either test is malformed by the rules and is dropped,
//! exactly as `csv::scan` drops it. The only lines handed to the generic scanner
//! are those longer than a batch.
//!
//! Field-cutting idea after Dremix10/shopping-sessions-fast (stage1/stage2).
//! This is an independent implementation.

const BATCH: usize = 16 << 10;
const BLOCKS: usize = BATCH / 64 + 3;

pub struct Tok {
    cm: Vec<u32>,
    nl: Vec<u32>,
    qb: Vec<u64>,
    qp: Vec<u32>,
    ncm: usize,
    nnl: usize,
}

impl Default for Tok {
    fn default() -> Self {
        Tok {
            // At most one separator per byte, plus unconditional-write slack and sentinels.
            cm: vec![0; BATCH + 128],
            nl: vec![0; BATCH + 128],
            qb: vec![0; BLOCKS],
            qp: vec![0; BLOCKS],
            ncm: 0,
            nnl: 0,
        }
    }
}

#[inline(always)]
fn masks_scalar(b: &[u8; 64]) -> (u64, u64, u64) {
    let (mut c, mut n, mut q) = (0u64, 0u64, 0u64);
    for (i, &x) in b.iter().enumerate() {
        c |= ((x == b',') as u64) << i;
        n |= ((x == b'\n') as u64) << i;
        q |= ((x == b'"') as u64) << i;
    }
    (c, n, q)
}

/// Appends the bit offsets of `m` (plus `base`) at `out[*n..]`. The first `K` slots
/// are written unconditionally, so `out` needs `K` slack entries past the end.
#[inline(always)]
unsafe fn spill<const K: usize>(out: *mut u32, n: &mut usize, base: u32, mut m: u64) {
    let c = m.count_ones();
    let o = out.add(*n);
    for i in 0..K {
        *o.add(i) = base + m.trailing_zeros();
        m &= m.wrapping_sub(1);
    }
    let mut i = K;
    while m != 0 {
        *o.add(i) = base + m.trailing_zeros();
        m &= m - 1;
        i += 1;
    }
    *n += c as usize;
}

impl Tok {
    #[inline(always)]
    fn index_scalar<const K_CM: usize>(&mut self, d: &[u8]) {
        debug_assert!(d.len() <= BATCH);
        let nb = d.len().div_ceil(64);
        let (cm, nl) = (self.cm.as_mut_ptr(), self.nl.as_mut_ptr());
        let (mut ncm, mut nnl, mut qc) = (0usize, 0usize, 0u32);
        for b in 0..nb {
            let off = b * 64;
            let (c, n, q) = if off + 64 <= d.len() {
                masks_scalar(d[off..off + 64].try_into().unwrap())
            } else {
                let mut t = [0u8; 64];
                t[..d.len() - off].copy_from_slice(&d[off..]);
                masks_scalar(&t)
            };
            unsafe {
                *self.qb.get_unchecked_mut(b) = q;
                *self.qp.get_unchecked_mut(b) = qc;
                qc += q.count_ones();
                spill::<K_CM>(cm, &mut ncm, off as u32, c);
                spill::<2>(nl, &mut nnl, off as u32, n);
            }
        }
        unsafe {
            *self.qb.get_unchecked_mut(nb) = 0;
            *self.qp.get_unchecked_mut(nb) = qc;
            *self.qb.get_unchecked_mut(nb + 1) = 0;
            *self.qp.get_unchecked_mut(nb + 1) = qc;
        }
        for i in 0..16 {
            unsafe { *self.cm.get_unchecked_mut(ncm + i) = u32::MAX };
        }
        self.ncm = ncm;
        self.nnl = nnl;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,popcnt")]
    unsafe fn index_avx2<const K_CM: usize>(&mut self, d: &[u8]) {
        use core::arch::x86_64::*;
        debug_assert!(d.len() <= BATCH);
        let len = d.len();
        let full = len / 64;
        let nb = len.div_ceil(64);
        let p = d.as_ptr();
        let (cm, nl) = (self.cm.as_mut_ptr(), self.nl.as_mut_ptr());
        let (mut ncm, mut nnl, mut qc) = (0usize, 0usize, 0u32);
        let v_cm = _mm256_set1_epi8(b',' as i8);
        let v_nl = _mm256_set1_epi8(b'\n' as i8);
        let v_qt = _mm256_set1_epi8(b'"' as i8);
        for b in 0..full {
            let off = b * 64;
            let qptr = p.add(off);
            let v0 = _mm256_loadu_si256(qptr as *const __m256i);
            let v1 = _mm256_loadu_si256(qptr.add(32) as *const __m256i);
            let c = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_cm)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_cm)) as u32 as u64) << 32);
            let n = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_nl)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_nl)) as u32 as u64) << 32);
            let q = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt)) as u32 as u64)
                | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt)) as u32 as u64) << 32);
            *self.qb.get_unchecked_mut(b) = q;
            *self.qp.get_unchecked_mut(b) = qc;
            qc += q.count_ones();
            spill::<K_CM>(cm, &mut ncm, off as u32, c);
            spill::<2>(nl, &mut nnl, off as u32, n);
        }
        if full < nb {
            let off = full * 64;
            let mut t = [0u8; 64];
            t[..len - off].copy_from_slice(&d[off..]);
            let (c, n, q) = masks_scalar(&t);
            *self.qb.get_unchecked_mut(full) = q;
            *self.qp.get_unchecked_mut(full) = qc;
            qc += q.count_ones();
            spill::<K_CM>(cm, &mut ncm, off as u32, c);
            spill::<2>(nl, &mut nnl, off as u32, n);
        }
        *self.qb.get_unchecked_mut(nb) = 0;
        *self.qp.get_unchecked_mut(nb) = qc;
        *self.qb.get_unchecked_mut(nb + 1) = 0;
        *self.qp.get_unchecked_mut(nb + 1) = qc;
        for i in 0..16 {
            *self.cm.get_unchecked_mut(ncm + i) = u32::MAX;
        }
        self.ncm = ncm;
        self.nnl = nnl;
    }

    #[inline(always)]
    fn index<const N: usize>(&mut self, d: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::csv::simd_backend() == "avx2" {
                unsafe {
                    if N >= 6 {
                        self.index_avx2::<6>(d);
                    } else if N >= 4 {
                        self.index_avx2::<4>(d);
                    } else {
                        self.index_avx2::<2>(d);
                    }
                }
                return;
            }
        }
        if N >= 6 {
            self.index_scalar::<6>(d);
        } else if N >= 4 {
            self.index_scalar::<4>(d);
        } else {
            self.index_scalar::<2>(d);
        }
    }

    /// Running prefix count of quotes in `[0, pos)` of the indexed batch.
    #[inline(always)]
    fn quotes_before(&self, pos: u32) -> u32 {
        let w = (pos >> 6) as usize;
        unsafe { *self.qp.get_unchecked(w) + (*self.qb.get_unchecked(w) & ((1u64 << (pos & 63)) - 1)).count_ones() }
    }

    /// Pass 2 over one indexed batch `d`; returns the bytes consumed (through the
    /// last newline, or all of `d` when `last` and it has no final newline).
    #[inline(always)]
    fn rows<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(&mut self, d: &'a [u8], last: bool, f: &mut F) -> usize {
        let mut nnl = self.nnl;
        if last && (nnl == 0 || self.nl[nnl - 1] as usize + 1 < d.len()) {
            self.nl[nnl] = d.len() as u32;
            nnl += 1;
        }
        let p = d.as_ptr();
        let (cm, nlv) = (self.cm.as_ptr(), self.nl.as_ptr());
        let (mut ls, mut ci, mut prev_q) = (0u32, 0usize, 0u32);
        for li in 0..nnl {
            let nl = unsafe { *nlv.add(li) };
            let cur_q = self.quotes_before(nl);
            let q = cur_q - prev_q;
            prev_q = cur_q;
            let (cl, cn) = unsafe { (*cm.add(ci + N - 2), *cm.add(ci + N - 1)) };
            if cl < nl && cn > nl {
                let mut b = [(0u32, 0u32); N];
                b[0] = (ls, unsafe { *cm.add(ci) });
                for k in 1..N - 1 {
                    b[k] = unsafe { (*cm.add(ci + k - 1) + 1, *cm.add(ci + k)) };
                }
                b[N - 1] = (cl + 1, nl);
                ci += N - 1;
                let ok = if q == 0 {
                    true
                } else if (q & 1) != 0 {
                    false
                } else {
                    // Unchecked, branchless outer-quote strip & validation:
                    // For interior fields 1..N-1, s >= 1 and e < d.len() always; when s == e,
                    // *p.add(s) and *p.add(e-1) are both ',', so open=0, close=0 without branching!
                    let (mut bad, mut nq) = (0u32, 0u32);
                    {
                        let (s, e) = b[0];
                        if s < e {
                            let open = (unsafe { *p.add(s as usize) } == b'"') as u32;
                            let close = (unsafe { *p.add(e as usize - 1) } == b'"') as u32;
                            bad |= (open ^ close) | (open & ((e - s < 2) as u32));
                            b[0] = (s + open, e - open);
                            nq += open;
                        }
                    }
                    for k in 1..N - 1 {
                        let (s, e) = unsafe { *b.get_unchecked(k) };
                        let open = (unsafe { *p.add(s as usize) } == b'"') as u32;
                        let close = (unsafe { *p.add(e as usize - 1) } == b'"') as u32;
                        bad |= (open ^ close) | (open & ((e - s < 2) as u32));
                        unsafe { *b.get_unchecked_mut(k) = (s + open, e - open) };
                        nq += open;
                    }
                    {
                        let (s, e) = b[N - 1];
                        if s < e {
                            let open = (unsafe { *p.add(s as usize) } == b'"') as u32;
                            let close = (unsafe { *p.add(e as usize - 1) } == b'"') as u32;
                            bad |= (open ^ close) | (open & ((e - s < 2) as u32));
                            b[N - 1] = (s + open, e - open);
                            nq += open;
                        }
                    }
                    bad == 0 && q == (nq << 1)
                };
                if ok {
                    let arr: [&'a [u8]; N] =
                        std::array::from_fn(|k| unsafe { d.get_unchecked(b[k].0 as usize..b[k].1 as usize) });
                    f(&arr);
                }
            } else {
                // Wrong field count: malformed. Skip this line's commas.
                while unsafe { *cm.add(ci) } < nl {
                    ci += 1;
                }
            }
            ls = nl + 1;
        }
        (ls as usize).min(d.len())
    }
}

/// Scans `data` (which must start at a line boundary) and calls `f` with the
/// unquoted fields of every well-formed line with exactly N fields.
#[inline(always)]
pub fn scan<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(tok: &mut Tok, data: &'a [u8], mut f: F) {
    assert!((2..=15).contains(&N));
    let mut pos = 0usize;
    while pos < data.len() {
        let end = (pos + BATCH).min(data.len());
        let last = end == data.len();
        let d = &data[pos..end];
        tok.index::<N>(d);
        if tok.nnl == 0 && !last {
            // One line longer than a batch: hand it to the generic scanner.
            let e = match data[end..].iter().position(|&c| c == b'\n') {
                Some(i) => end + i + 1,
                None => data.len(),
            };
            crate::csv::scan::<N, _>(&data[pos..e], |x| f(x));
            pos = e;
            continue;
        }
        pos += tok.rows::<N, F>(d, last, &mut f);
        if last {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect<const N: usize>(d: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut t = Tok::default();
        scan::<N, _>(&mut t, d, |f| out.push(f.iter().map(|x| x.to_vec()).collect()));
        out
    }
    fn oracle<const N: usize>(d: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        crate::csv::scan_byte_oracle::<N, _>(d, |f| out.push(f.iter().map(|x| x.to_vec()).collect()));
        out
    }

    #[test]
    fn matches_oracle_on_edge_lines() {
        let lines: &[&[u8]] = &[
            b"a,b,c,d", b"\"a\",b,\"\",d", b"a,b,c", b"a,b,c,d,e", b"", b"\"", b"a\"b,c,d,e", b"\"a,b,c,d",
            b"\"a\"x,b,c,d", b"a,b,c,d\r", b"\"\",\"\",\"\",\"\"", b",,,", b"\"a\"\"b\",c,d,e", b" a , b ,c,d",
            b"a,\"b\",c,\"d", b"x\"\",b,c,d", b"\"\"\",b,c,d",
        ];
        let mut rng = 0x1234_5678u64;
        for round in 0..400 {
            let mut d = Vec::new();
            for _ in 0..(round % 37 + 1) {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                d.extend_from_slice(lines[(rng >> 33) as usize % lines.len()]);
                d.push(b'\n');
            }
            if round % 3 == 0 {
                d.pop(); // no final newline
            }
            assert_eq!(collect::<4>(&d), oracle::<4>(&d), "round {round}");
            assert_eq!(collect::<2>(&d), oracle::<2>(&d), "round {round}");
        }
    }

    #[test]
    fn batch_boundaries_and_long_lines() {
        let mut d = Vec::new();
        let mut i = 0u64;
        while d.len() < 3 * BATCH + 777 {
            let pad = "x".repeat((i % 97) as usize);
            d.extend_from_slice(format!("e{i},\"u{pad}\",{},z\n", i * 7).as_bytes());
            if i % 501 == 0 {
                d.extend_from_slice(format!("long,{},q,w\n", "y".repeat(BATCH + 3)).as_bytes());
            }
            i += 1;
        }
        d.extend_from_slice(b"tail,a,b,c");
        assert_eq!(collect::<4>(&d), oracle::<4>(&d));
    }
}
