//! Strict CSV line scanner for the competition dialect.
//!
//! Only `\n` ends a line, commas always separate fields, and a quote is legal only
//! as the first and last byte of a field. Well-formed lines with exactly N fields
//! are passed to the callback with their (unquoted) field values.
//!
//! Runtime-dispatched 64-byte AVX2 scanner on x86_64 (`x86-64-v3`) with an exact
//! 8-byte SWAR fallback for non-AVX2 CPUs and forced-backend testing (`FORCE_SIMD=scalar`).

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;

#[inline(always)]
fn eqz(y: u64) -> u64 {
    !(((y & LO7).wrapping_add(LO7)) | y | LO7)
}

#[inline(always)]
fn special_mask(w: u64) -> u64 {
    eqz(w ^ 0x2c2c_2c2c_2c2c_2c2c) | eqz(w ^ 0x2222_2222_2222_2222) | eqz(w ^ 0x0a0a_0a0a_0a0a_0a0a)
}

/// Independent byte-by-byte reference scanner (no shared SWAR/SIMD primitives); used by tests.
#[allow(dead_code)]
pub fn scan_byte_oracle<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], mut f: F) {
    let mut start = 0usize;
    let len = data.len();
    while start < len {
        let end = match data[start..].iter().position(|&b| b == b'\n') {
            Some(off) => start + off,
            None => len,
        };
        let line = &data[start..end];
        if !line.is_empty() {
            let mut fields = [&[][..]; N];
            let mut nf = 0usize;
            let mut bad = false;
            for part in line.split(|&b| b == b',') {
                if nf >= N {
                    bad = true;
                    break;
                }
                if part.len() >= 2 && part[0] == b'"' && part[part.len() - 1] == b'"' {
                    let inner = &part[1..part.len() - 1];
                    if inner.contains(&b'"') {
                        bad = true;
                        break;
                    }
                    fields[nf] = inner;
                } else {
                    if part.contains(&b'"') {
                        bad = true;
                        break;
                    }
                    fields[nf] = part;
                }
                nf += 1;
            }
            if !bad && nf == N {
                f(&fields);
            }
        }
        start = if end < len { end + 1 } else { len };
    }
}

struct St<const N: usize> {
    fs: usize,
    nf: usize,
    q: u32,
    qf: usize,
    ql: usize,
    bad: bool,
    fields: [(usize, usize); N],
}

impl<const N: usize> St<N> {
    #[inline(always)]
    fn end_field(&mut self, pos: usize) {
        if self.nf < N {
            let r = if self.q == 0 {
                (self.fs, pos)
            } else if self.q == 2 && self.qf == self.fs && self.ql + 1 == pos {
                (self.fs + 1, pos - 1)
            } else {
                self.bad = true;
                (0, 0)
            };
            self.fields[self.nf] = r;
        }
        self.nf += 1;
        self.fs = pos + 1;
        self.q = 0;
    }

    #[inline(always)]
    fn end_line<'a, F: FnMut(&[&'a [u8]; N])>(&mut self, data: &'a [u8], f: &mut F) {
        if !self.bad && self.nf == N {
            let arr: [&'a [u8]; N] =
                std::array::from_fn(|k| unsafe { data.get_unchecked(self.fields[k].0..self.fields[k].1) });
            f(&arr);
        }
        self.nf = 0;
        self.bad = false;
    }

    #[inline(always)]
    fn on<'a, F: FnMut(&[&'a [u8]; N])>(&mut self, data: &'a [u8], pos: usize, f: &mut F) {
        match unsafe { *data.get_unchecked(pos) } {
            b',' => self.end_field(pos),
            b'\n' => {
                self.end_field(pos);
                self.end_line(data, f);
            }
            _ => {
                if self.q == 0 {
                    self.qf = pos;
                }
                self.ql = pos;
                self.q += 1;
            }
        }
    }
}

/// Reference SWAR scanner (per-special-byte state machine); used by tests.
#[allow(dead_code)]
#[inline(always)]
pub fn scan_ref<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], mut f: F) {
    let len = data.len();
    let p = data.as_ptr();
    let mut st = St::<N> { fs: 0, nf: 0, q: 0, qf: 0, ql: 0, bad: false, fields: [(0, 0); N] };
    let mut i = 0usize;
    while i + 8 <= len {
        let w = u64::from_le(unsafe { (p.add(i) as *const u64).read_unaligned() });
        let mut m = special_mask(w);
        while m != 0 {
            let pos = i + (m.trailing_zeros() as usize >> 3);
            st.on(data, pos, &mut f);
            m &= m - 1;
        }
        i += 8;
    }
    if i < len {
        let mut buf = [0u8; 8];
        buf[..len - i].copy_from_slice(&data[i..]);
        let mut m = special_mask(u64::from_le_bytes(buf));
        while m != 0 {
            let pos = i + (m.trailing_zeros() as usize >> 3);
            st.on(data, pos, &mut f);
            m &= m - 1;
        }
    }
    if st.fs < len || st.nf > 0 {
        st.end_field(len);
        st.end_line(data, &mut f);
    }
}

#[cfg(target_arch = "x86_64")]
static USE_AVX2: OnceLock<bool> = OnceLock::new();

#[inline(always)]
pub fn simd_backend() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if *USE_AVX2.get_or_init(|| {
            if std::env::var("FORCE_SIMD").ok().as_deref() == Some("scalar") {
                false
            } else {
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("bmi1")
                    && is_x86_feature_detected!("bmi2")
                    && is_x86_feature_detected!("popcnt")
            }
        }) {
            return "avx2";
        }
    }
    "scalar"
}

/// Scans `data` (which must start at a line boundary) line by line, calling `f`
/// with the unquoted fields of every well-formed line with exactly N fields.
#[inline(always)]
pub fn scan<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], f: F) {
    #[cfg(target_arch = "x86_64")]
    {
        if *USE_AVX2.get_or_init(|| {
            if std::env::var("FORCE_SIMD").ok().as_deref() == Some("scalar") {
                false
            } else {
                is_x86_feature_detected!("avx2")
                    && is_x86_feature_detected!("bmi1")
                    && is_x86_feature_detected!("bmi2")
                    && is_x86_feature_detected!("popcnt")
            }
        }) {
            unsafe { scan_avx2::<N, F>(data, f) };
            return;
        }
    }
    scan_scalar::<N, F>(data, f);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2")]
unsafe fn scan_avx2<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], mut f: F) {
    use core::arch::x86_64::*;

    debug_assert!(N < 16);
    let len = data.len();
    let p = data.as_ptr();
    let mut fields = [(0usize, 0usize); 16];
    let mut fs = 0usize;
    let mut nf = 0usize;
    let mut qcnt = 0u32;
    let mut qlast = usize::MAX;
    let mut bad = false;
    let mut i = 0usize;

    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);

    macro_rules! process_64 {
        ($cm:expr, $nlm:expr, $qm:expr) => {{
            let nlm: u64 = $nlm;
            let mut sm: u64 = $cm | nlm;
            let mut qm: u64 = $qm;
            if qm == 0 && qcnt == 0 {
                // Fast unquoted path: zero quote arithmetic or extra loads.
                while sm != 0 {
                    let b = sm.trailing_zeros();
                    let pos = i + b as usize;
                    *fields.get_unchecked_mut(nf.min(N)) = (fs, pos);
                    nf += 1;
                    fs = pos + 1;
                    if (nlm >> b) & 1 != 0 {
                        if !bad && nf == N {
                            let arr: [&'a [u8]; N] = std::array::from_fn(|k| {
                                data.get_unchecked(fields[k].0..fields[k].1)
                            });
                            f(&arr);
                        }
                        nf = 0;
                        bad = false;
                    }
                    sm &= sm - 1;
                }
            } else if qcnt == 0 && sm != 0 && {
                let lz = sm.leading_zeros();
                let hi_mask = if lz == 0 { !0u64 } else { (1u64 << (64 - lz)) - 1 };
                let valid_edge = (sm << 1) | ((fs == i) as u64) | (sm >> 1);
                (qm & (!hi_mask | !valid_edge)) == 0
            } {
                // Fast edge-quoted path (>95% of quoted blocks): zero interior quotes,
                // zero quotes carried in or out. Validate each field with 2 edge checks.
                while sm != 0 {
                    let b = sm.trailing_zeros();
                    let pos = i + b as usize;
                    let (f0, f1) = if pos == fs {
                        (fs, pos)
                    } else {
                        let q0 = *p.add(fs) == b'"';
                        let q1 = *p.add(pos - 1) == b'"';
                        let quoted = q0 && q1 && pos > fs + 1;
                        bad |= !((!q0 && !q1) || quoted);
                        let qu = quoted as usize;
                        (fs + qu, pos - qu)
                    };
                    *fields.get_unchecked_mut(nf.min(N)) = (f0, f1);
                    nf += 1;
                    fs = pos + 1;
                    if (nlm >> b) & 1 != 0 {
                        if !bad && nf == N {
                            let arr: [&'a [u8]; N] = std::array::from_fn(|k| {
                                data.get_unchecked(fields[k].0..fields[k].1)
                            });
                            f(&arr);
                        }
                        nf = 0;
                        bad = false;
                    }
                    sm &= sm - 1;
                }
            } else {
                while sm != 0 {
                    let b = sm.trailing_zeros();
                    let below = (1u64 << b) - 1;
                    let qf = qm & below;
                    qm &= !below;
                    qcnt += qf.count_ones();
                    if qf != 0 {
                        qlast = i + (qf.leading_zeros() ^ 63) as usize;
                    }
                    let pos = i + b as usize;
                    let q0 = *p.add(fs) == b'"';
                    let quoted = qcnt == 2 && q0 && qlast == pos.wrapping_sub(1) && pos > fs + 1;
                    bad |= !(qcnt == 0 || quoted);
                    let qu = quoted as usize;
                    *fields.get_unchecked_mut(nf.min(N)) = (fs + qu, pos - qu);
                    nf += 1;
                    fs = pos + 1;
                    qcnt = 0;
                    if (nlm >> b) & 1 != 0 {
                        if !bad && nf == N {
                            let arr: [&'a [u8]; N] = std::array::from_fn(|k| {
                                data.get_unchecked(fields[k].0..fields[k].1)
                            });
                            f(&arr);
                        }
                        nf = 0;
                        bad = false;
                    }
                    sm &= sm - 1;
                }
                if qm != 0 {
                    qcnt += qm.count_ones();
                    qlast = i + (qm.leading_zeros() ^ 63) as usize;
                }
            }
        }};
    }

    while i + 64 <= len {
        let v0 = _mm256_loadu_si256(p.add(i) as *const __m256i);
        let v1 = _mm256_loadu_si256(p.add(i + 32) as *const __m256i);

        let nl0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_nl)) as u32 as u64;
        let nl1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_nl)) as u32 as u64;
        let cm0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_cm)) as u32 as u64;
        let cm1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_cm)) as u32 as u64;
        let qt0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt)) as u32 as u64;
        let qt1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt)) as u32 as u64;

        process_64!(cm0 | (cm1 << 32), nl0 | (nl1 << 32), qt0 | (qt1 << 32));
        i += 64;
    }

    if i < len {
        let mut buf = [0u8; 64];
        let rem = len - i;
        std::ptr::copy_nonoverlapping(p.add(i), buf.as_mut_ptr(), rem);
        let bp = buf.as_ptr();
        let v0 = _mm256_loadu_si256(bp as *const __m256i);
        let v1 = _mm256_loadu_si256(bp.add(32) as *const __m256i);

        let nl0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_nl)) as u32 as u64;
        let nl1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_nl)) as u32 as u64;
        let cm0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_cm)) as u32 as u64;
        let cm1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_cm)) as u32 as u64;
        let qt0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v0, v_qt)) as u32 as u64;
        let qt1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v1, v_qt)) as u32 as u64;

        // Zero-padding never matches ',', '\n', or '"', so all set bits are strictly < rem.
        process_64!(cm0 | (cm1 << 32), nl0 | (nl1 << 32), qt0 | (qt1 << 32));
    }

    if fs < len || nf > 0 {
        let pos = len;
        let q0 = fs < len && *p.add(fs) == b'"';
        let quoted = qcnt == 2 && q0 && qlast == pos.wrapping_sub(1) && pos > fs + 1;
        bad |= !(qcnt == 0 || quoted);
        let qu = quoted as usize;
        fields[nf.min(N)] = (fs + qu, pos - qu);
        nf += 1;
        if !bad && nf == N {
            let arr: [&'a [u8]; N] = std::array::from_fn(|k| &data[fields[k].0..fields[k].1]);
            f(&arr);
        }
    }
}

/// Scalar 8-byte SWAR scanner (arm64 & non-AVX2 fallback).
#[inline(always)]
pub fn scan_scalar<'a, const N: usize, F: FnMut(&[&'a [u8]; N])>(data: &'a [u8], mut f: F) {
    assert!(N < 16);
    let len = data.len();
    let p = data.as_ptr();
    let mut fields = [(0usize, 0usize); 16];
    let mut fs = 0usize;
    let mut nf = 0usize;
    let mut qcnt = 0u32;
    let mut qlast = usize::MAX;
    let mut bad = false;
    let mut i = 0usize;

    macro_rules! word {
        ($w:expr) => {{
            let w: u64 = $w;
            let nlm = eqz(w ^ 0x0a0a_0a0a_0a0a_0a0a);
            let mut sm = eqz(w ^ 0x2c2c_2c2c_2c2c_2c2c) | nlm;
            let mut qm = eqz(w ^ 0x2222_2222_2222_2222);
            while sm != 0 {
                let b = sm.trailing_zeros();
                let below = (1u64 << b) - 1;
                let qf = qm & below;
                qm &= !below;
                qcnt += qf.count_ones();
                let cand = i + ((qf.leading_zeros() ^ 63) >> 3) as usize;
                qlast = if qf != 0 { cand } else { qlast };
                let pos = i + (b >> 3) as usize;
                let q0 = unsafe { *p.add(fs) } == b'"';
                let quoted = qcnt == 2 && q0 && qlast == pos.wrapping_sub(1) && pos > fs + 1;
                bad |= !(qcnt == 0 || quoted);
                let qu = quoted as usize;
                unsafe { *fields.get_unchecked_mut(nf.min(N)) = (fs + qu, pos - qu) };
                nf += 1;
                fs = pos + 1;
                qcnt = 0;
                if (nlm >> b) & 1 != 0 {
                    if !bad && nf == N {
                        let arr: [&'a [u8]; N] =
                            std::array::from_fn(|k| unsafe { data.get_unchecked(fields[k].0..fields[k].1) });
                        f(&arr);
                    }
                    nf = 0;
                    bad = false;
                }
                sm &= sm - 1;
            }
            qcnt += qm.count_ones();
            let cand = i + ((qm.leading_zeros() ^ 63) >> 3) as usize;
            qlast = if qm != 0 { cand } else { qlast };
        }};
    }

    while i + 8 <= len {
        word!(u64::from_le(unsafe { (p.add(i) as *const u64).read_unaligned() }));
        i += 8;
    }
    if i < len {
        let mut buf = [0u8; 8];
        buf[..len - i].copy_from_slice(&data[i..]);
        word!(u64::from_le_bytes(buf));
    }
    if fs < len || nf > 0 {
        let pos = len;
        let q0 = fs < len && data[fs] == b'"';
        let quoted = qcnt == 2 && q0 && qlast == pos.wrapping_sub(1) && pos > fs + 1;
        bad |= !(qcnt == 0 || quoted);
        let qu = quoted as usize;
        fields[nf.min(N)] = (fs + qu, pos - qu);
        nf += 1;
        if !bad && nf == N {
            let arr: [&'a [u8]; N] = std::array::from_fn(|k| &data[fields[k].0..fields[k].1]);
            f(&arr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn lines<const N: usize>(s: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = vec![];
        scan::<N, _>(s, |f| out.push(f.iter().map(|x| x.to_vec()).collect()));
        out
    }
    #[test]
    fn dialect() {
        let r = lines::<2>(b"a,b\n\"c\",\"\"\n\n\"x,y\nq\"r,s\n\" t\",u \r\na,b,c\na\nlast,line");
        let exp: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"c".to_vec(), b"".to_vec()],
            vec![b" t".to_vec(), b"u \r".to_vec()],
            vec![b"last".to_vec(), b"line".to_vec()],
        ];
        assert_eq!(r, exp);
        assert_eq!(lines::<2>(b"\"\",\"\"\"\n").len(), 0);
        assert_eq!(lines::<2>(b"\",a\n").len(), 0);
        assert_eq!(lines::<2>(b"a,\n").len(), 1);
        assert_eq!(lines::<2>(b"a,").len(), 1);
    }

    #[test]
    fn fuzz_equivalence_all_arities() {
        let alphabet = b"ab\",\n\r \"\",xyz0123456789";
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..80_000 {
            let n = (rnd() % 180) as usize;
            let s: Vec<u8> = (0..n).map(|_| alphabet[(rnd() % alphabet.len() as u64) as usize]).collect();
            macro_rules! check_arity {
                ($N:literal) => {{
                    let mut a = vec![];
                    scan::<$N, _>(&s, |f| a.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>()));
                    let mut b = vec![];
                    scan_byte_oracle::<$N, _>(&s, |f| b.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>()));
                    assert_eq!(a, b, "arity={} input={:?}", $N, String::from_utf8_lossy(&s));
                    let mut c = vec![];
                    scan_scalar::<$N, _>(&s, |f| c.push(f.iter().map(|x| x.to_vec()).collect::<Vec<_>>()));
                    assert_eq!(c, b, "scalar arity={} input={:?}", $N, String::from_utf8_lossy(&s));
                }};
            }
            check_arity!(2);
            check_arity!(4);
            check_arity!(7);
        }
    }
}
