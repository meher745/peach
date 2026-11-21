//! Shopping Sessions pipeline. See rules.md.
//!
//! Strings (IDs) are turned into u64 keys: a string of at most 15 bytes is packed
//! exactly (bytes big-endian, length in the low byte), so key order equals byte-wise
//! string order. Longer strings get a 128-bit hash with low byte 0xFF and their bytes
//! are kept in a side table for the rare places that need them.

use hashbrown::HashMap;
use std::cmp::Ordering;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering as AO};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

const DAY: i64 = 86_400_000;
const GAP: i64 = 1_800_000;
const WEEK: i64 = 7 * DAY;
const NB: usize = 1024; // partitions
const CHUNK: usize = 8 << 20;

// ---------------------------------------------------------------- keys
//
// IDs become u64 keys. A string of at most 10 bytes, all from the 64-symbol alphabet
// `-0-9A-Z_a-z`, is encoded exactly: 6 bits per character from the top (codes in byte
// order, so key order equals byte-wise string order) and its length in the low 4 bits.
// Any other string gets a 60-bit hash with 15 in the low 4 bits, and its bytes are kept in
// a side table for the places that need them (names, click_id ties). 0 is the empty string.

#[inline(always)]
fn mix(k: u64) -> u64 {
    let mut x = k ^ 0x9E37_79B9_7F4A_7C15;
    x ^= x >> 32;
    x = x.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    x ^= x >> 32;
    x = x.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    x ^= x >> 32;
    x
}

#[inline(always)]
fn part(k: u64) -> usize {
    let mut x = k ^ 0x2545_F491_4F6C_DD1D;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^= x >> 33;
    (x as usize) & (NB - 1)
}

#[derive(Default)]
struct KH(u64);
impl Hasher for KH {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0.rotate_left(5) ^ x as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
    #[inline(always)]
    fn write_u64(&mut self, k: u64) {
        self.0 = mix(k);
    }
}
type KMap<V> = HashMap<u64, V, BuildHasherDefault<KH>>;

const ALPHA: &[u8; 64] = b"-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz";
static ENC: [u8; 256] = {
    let mut t = [0xFFu8; 256];
    let mut i = 0;
    while i < 64 {
        t[ALPHA[i] as usize] = i as u8;
        i += 1;
    }
    t
};

fn hash60(b: &[u8]) -> u64 {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325 ^ b.len() as u64;
    let mut h2: u64 = 0x8422_2325_cbf2_9ce4;
    for &x in b {
        h1 = (h1 ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        h2 = (h2.rotate_left(7) ^ x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    (mix(h1 ^ mix(h2)) & !0xF) | 0xF
}

#[inline(always)]
fn pack(b: &[u8]) -> u64 {
    let n = b.len();
    if n <= 10 {
        let mut k = 0u64;
        let mut bad = 0u8;
        for (i, &c) in b.iter().enumerate() {
            let e = ENC[c as usize];
            bad |= e;
            k |= ((e & 63) as u64) << (58 - 6 * i);
        }
        if bad & 0xC0 == 0 {
            return k | n as u64;
        }
    }
    hash60(b)
}

#[inline(always)]
fn is_long(k: u64) -> bool {
    k & 0xF == 0xF
}

static LONG: RwLock<Option<HashMap<u64, Vec<u8>>>> = RwLock::new(None);

fn register_long(k: u64, b: &[u8]) {
    let mut g = LONG.write().unwrap();
    g.get_or_insert_with(HashMap::new).entry(k).or_insert_with(|| b.to_vec());
}

#[inline(always)]
fn pack_reg(b: &[u8]) -> u64 {
    let k = pack(b);
    if is_long(k) {
        register_long(k, b);
    }
    k
}

fn unpack(k: u64) -> Vec<u8> {
    if !is_long(k) {
        let n = (k & 0xF) as usize;
        (0..n).map(|i| ALPHA[((k >> (58 - 6 * i)) & 63) as usize]).collect()
    } else {
        LONG.read().unwrap().as_ref().unwrap().get(&k).unwrap().clone()
    }
}

#[inline]
fn cmp_str(a: u64, b: u64) -> Ordering {
    if !is_long(a) && !is_long(b) {
        a.cmp(&b)
    } else {
        unpack(a).cmp(&unpack(b))
    }
}

// ---------------------------------------------------------------- link table

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
}

/// Open-addressing map ID key -> person key; key 0 (empty string) marks a free slot.
struct LinkTable {
    slots: Vec<[u64; 2]>,
    mask: usize,
    filter: Vec<u64>, // one bit per hash bucket: set when some key maps there
    fshift: u32,
}

impl LinkTable {
    fn new(n: usize) -> Self {
        let cap = (n * 2).next_power_of_two().max(2);
        // ~20 filter bits per key (~5% false positives), at least 64
        let fbits = (n * 20).next_power_of_two().max(64);
        LinkTable {
            slots: vec![[0, 0]; cap],
            mask: cap - 1,
            filter: vec![0; fbits / 64],
            fshift: 64 - fbits.trailing_zeros(),
        }
    }
    #[inline(always)]
    fn fbit(&self, k: u64) -> usize {
        // top bits of a second mix, independent of the slot index
        (mix(k ^ 0x5851_F42D_4C95_7F2D) >> self.fshift) as usize
    }
    /// False means the key is certainly not in the table.
    #[inline(always)]
    fn maybe(&self, k: u64) -> bool {
        let b = self.fbit(k);
        unsafe { (*self.filter.get_unchecked(b >> 6) >> (b & 63)) & 1 != 0 }
    }
    #[inline(always)]
    fn slot(&self, k: u64) -> usize {
        mix(k) as usize & self.mask
    }
    fn insert(&mut self, k: u64, v: u64) {
        let b = self.fbit(k);
        self.filter[b >> 6] |= 1 << (b & 63);
        let mut i = self.slot(k);
        loop {
            if self.slots[i][0] == 0 || self.slots[i][0] == k {
                self.slots[i] = [k, v];
                return;
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn prefetch(&self, k: u64) {
        prefetch(self.slots.as_ptr().wrapping_add(self.slot(k)));
    }
    #[inline(always)]
    fn get(&self, k: u64) -> u64 {
        let mut i = self.slot(k);
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            if s[0] == k {
                return s[1];
            }
            if s[0] == 0 {
                return k;
            }
            i = (i + 1) & self.mask;
        }
    }
}

// ---------------------------------------------------------------- CSV

#[inline(always)]
fn fix(f: &[u8], quotes: u32) -> Option<&[u8]> {
    if quotes == 0 {
        Some(f)
    } else if quotes == 2 && f.len() >= 2 && f[0] == b'"' && f[f.len() - 1] == b'"' {
        Some(&f[1..f.len() - 1])
    } else {
        None
    }
}

/// Splits one line into exactly N fields, removing whole-field quotes. False if malformed.
#[inline(always)]
fn split<'a, const N: usize>(line: &'a [u8], out: &mut [&'a [u8]; N]) -> bool {
    let mut n = 0;
    let mut start = 0;
    let mut quotes = 0u32;
    for (i, &c) in line.iter().enumerate() {
        if c == b',' {
            if n == N - 1 {
                return false;
            }
            match fix(&line[start..i], quotes) {
                Some(f) => out[n] = f,
                None => return false,
            }
            n += 1;
            start = i + 1;
            quotes = 0;
        } else if c == b'"' {
            quotes += 1;
        }
    }
    if n != N - 1 {
        return false;
    }
    match fix(&line[start..], quotes) {
        Some(f) => out[N - 1] = f,
        None => return false,
    }
    true
}

/// Calls `f` for every line in `data[start..end]`.
#[inline]
fn for_lines(data: &[u8], mut f: impl FnMut(&[u8])) {
    let mut pos = 0;
    let n = data.len();
    while pos < n {
        match memchr::memchr(b'\n', &data[pos..]) {
            Some(o) => {
                f(&data[pos..pos + o]);
                pos += o + 1;
            }
            None => {
                f(&data[pos..]);
                pos = n;
            }
        }
    }
}


// ---------------------------------------------------------------- SIMD line scanner

/// Bitmasks of (comma | newline, newline, quote) for 64 bytes at `p`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let c = _mm_set1_epi8(b',' as i8);
        let n = _mm_set1_epi8(b'\n' as i8);
        let q = _mm_set1_epi8(b'"' as i8);
        let (mut mc, mut mn, mut mq) = (0u64, 0u64, 0u64);
        for k in 0..4 {
            let v = _mm_loadu_si128(p.add(16 * k) as *const __m128i);
            let ec = _mm_cmpeq_epi8(v, c);
            let en = _mm_cmpeq_epi8(v, n);
            mc |= (_mm_movemask_epi8(_mm_or_si128(ec, en)) as u32 as u64) << (16 * k);
            mn |= (_mm_movemask_epi8(en) as u32 as u64) << (16 * k);
            mq |= (_mm_movemask_epi8(_mm_cmpeq_epi8(v, q)) as u32 as u64) << (16 * k);
        }
        (mc, mn, mq)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    use std::arch::aarch64::*;
    unsafe {
        const BM: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let bm = vld1q_u8(BM.as_ptr());
        let v = [vld1q_u8(p), vld1q_u8(p.add(16)), vld1q_u8(p.add(32)), vld1q_u8(p.add(48))];
        #[inline(always)]
        unsafe fn mm(bm: uint8x16_t, c: [uint8x16_t; 4]) -> u64 {
            unsafe {
                let t0 = vandq_u8(c[0], bm);
                let t1 = vandq_u8(c[1], bm);
                let t2 = vandq_u8(c[2], bm);
                let t3 = vandq_u8(c[3], bm);
                let s0 = vpaddq_u8(t0, t1);
                let s1 = vpaddq_u8(t2, t3);
                let s = vpaddq_u8(s0, s1);
                let s = vpaddq_u8(s, s);
                vgetq_lane_u64(vreinterpretq_u64_u8(s), 0)
            }
        }
        let c = vdupq_n_u8(b',');
        let n = vdupq_n_u8(b'\n');
        let q = vdupq_n_u8(b'"');
        let en = [vceqq_u8(v[0], n), vceqq_u8(v[1], n), vceqq_u8(v[2], n), vceqq_u8(v[3], n)];
        let es = [
            vorrq_u8(vceqq_u8(v[0], c), en[0]),
            vorrq_u8(vceqq_u8(v[1], c), en[1]),
            vorrq_u8(vceqq_u8(v[2], c), en[2]),
            vorrq_u8(vceqq_u8(v[3], c), en[3]),
        ];
        let eq = [vceqq_u8(v[0], q), vceqq_u8(v[1], q), vceqq_u8(v[2], q), vceqq_u8(v[3], q)];
        (mm(bm, es), mm(bm, en), mm(bm, eq))
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn masks64(p: *const u8) -> (u64, u64, u64) {
    let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
    for i in 0..64 {
        let x = unsafe { *p.add(i) };
        if x == b',' || x == b'\n' {
            a |= 1 << i;
        }
        if x == b'\n' {
            b |= 1 << i;
        }
        if x == b'"' {
            c |= 1 << i;
        }
    }
    (a, b, c)
}

#[allow(dead_code)]
#[inline(always)]
fn count_bits(m: &[[u64; 3]], a: usize, b: usize) -> u32 {
    // quote bits in [a, b)
    if a >= b {
        return 0;
    }
    let (wa, wb) = (a >> 6, (b - 1) >> 6);
    let lo = !0u64 << (a & 63);
    let hi = !0u64 >> (63 - ((b - 1) & 63));
    if wa == wb {
        return (m[wa][2] & lo & hi).count_ones();
    }
    let mut c = (m[wa][2] & lo).count_ones() + (m[wb][2] & hi).count_ones();
    for w in &m[wa + 1..wb] {
        c += w[2].count_ones();
    }
    c
}

/// Quote bits in [a, b).
#[inline(always)]
fn quote_count(m: &[[u64; 3]], a: usize, b: usize) -> u32 {
    if a >= b {
        return 0;
    }
    let (wa, wb) = (a >> 6, (b - 1) >> 6);
    let lo = !0u64 << (a & 63);
    let hi = !0u64 >> (63 - ((b - 1) & 63));
    unsafe {
        if wa == wb {
            return (m.get_unchecked(wa)[2] & lo & hi).count_ones();
        }
        let mut c = (m.get_unchecked(wa)[2] & lo).count_ones() + (m.get_unchecked(wb)[2] & hi).count_ones();
        for w in wa + 1..wb {
            c += m.get_unchecked(w)[2].count_ones();
        }
        c
    }
}

/// Calls `f` with the N unquoted fields of every well-formed line in `buf`.
/// A line is well formed when it has exactly N fields and its quote count equals
/// twice the number of fields that start and end with a quote (length >= 2): then
/// those fields hold exactly two quotes each and every other field none.
#[inline(always)]
fn scan_lines<'a, const N: usize>(buf: &'a [u8], bits: &mut Vec<[u64; 3]>, mut f: impl FnMut(&[&'a [u8]; N])) {
    // Work in cache-sized windows that end right after a newline, so the bytes the
    // masks were built from are still in L1/L2 when the lines are cut.
    const WIN: usize = 32 << 10;
    let mut s = 0;
    while s < buf.len() {
        let mut e = (s + WIN).min(buf.len());
        if e < buf.len() {
            e = match memchr::memchr(b'\n', &buf[e..]) {
                Some(o) => e + o + 1,
                None => buf.len(),
            };
        }
        scan_window(&buf[s..e], bits, &mut f);
        s = e;
    }
}

#[inline(always)]
fn scan_window<'a, const N: usize>(buf: &'a [u8], bits: &mut Vec<[u64; 3]>, f: &mut impl FnMut(&[&'a [u8]; N])) {
    let len = buf.len();
    let words = len.div_ceil(64);
    bits.clear();
    bits.reserve(words);
    let full = len / 64;
    for w in 0..full {
        let (a, b, c) = unsafe { masks64(buf.as_ptr().add(w * 64)) };
        bits.push([a, b, c]);
    }
    if full < words {
        let mut tmp = [0u8; 64];
        tmp[..len - full * 64].copy_from_slice(&buf[full * 64..]);
        let (a, b, c) = unsafe { masks64(tmp.as_ptr()) };
        bits.push([a, b, c]);
    }
    let bits: &[[u64; 3]] = bits;
    let mut pos = [0usize; 16];
    let mut k = 0usize;
    let mut ls = 0usize;
    let mut line = |ls: usize, le: usize, pos: &[usize; 16], k: usize| {
        if k != N - 1 {
            return;
        }
        let q = quote_count(bits, ls, le);
        let mut out: [&'a [u8]; N] = [&[]; N];
        let mut nq = 0u32;
        let mut a = ls;
        for i in 0..N {
            let b = if i + 1 < N { pos[i] } else { le };
            // branchless: sq = 1 when the field is wrapped in quotes (length >= 2)
            unsafe {
                let first = *buf.get_unchecked(a.min(len - 1)) == b'"';
                let last = *buf.get_unchecked(b.max(1) - 1) == b'"';
                let sq = (first & last & (b >= a + 2)) as usize;
                out[i] = buf.get_unchecked(a + sq..b - sq);
                nq += sq as u32;
            }
            a = b + 1;
        }
        if q == 2 * nq {
            f(&out);
        }
    };
    for w in 0..words {
        let [mut m, nl, _] = bits[w];
        while m != 0 {
            let tz = m.trailing_zeros() as usize;
            m &= m - 1;
            let p = w * 64 + tz;
            if (nl >> tz) & 1 == 0 {
                pos[k & 15] = p;
                k += 1;
            } else {
                line(ls, p, &pos, k);
                ls = p + 1;
                k = 0;
            }
        }
    }
    if ls < len {
        line(ls, len, &pos, k);
    }
}

// ---------------------------------------------------------------- time

const fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    era * 146_097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719_468
}

/// Days from the epoch to Jan 1 of 1900 + i.
static YEAR_DAYS: [i32; 400] = {
    let mut t = [0i32; 400];
    let mut i = 0;
    while i < 400 {
        t[i] = days_from_civil(1900 + i as i64, 1, 1) as i32;
        i += 1;
    }
    t
};
const MONTH_CUM: [i32; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

#[inline(always)]
fn fast_days(y: i64, mo: i64, d: i64, leap: bool) -> i64 {
    if (1900..2300).contains(&y) {
        (YEAR_DAYS[(y - 1900) as usize] + MONTH_CUM[mo as usize] + (leap && mo > 2) as i32) as i64 + d - 1
    } else {
        days_from_civil(y, mo, d)
    }
}

/// Value of 8 ASCII digits (first digit in the lowest byte), already validated.
#[inline(always)]
fn p8(w: u64) -> u64 {
    let x = w & 0x0F0F_0F0F_0F0F_0F0F;
    let x = (x.wrapping_mul(10) + (x >> 8)) & 0x00FF_00FF_00FF_00FF;
    let x = (x.wrapping_mul(100) + (x >> 16)) & 0x0000_FFFF_0000_FFFF;
    (x.wrapping_mul(10000) + (x >> 32)) & 0xFFFF_FFFF
}

/// Digits-only value of `b` (1..=16 bytes), None if any byte is not an ASCII digit.
#[inline(always)]
fn digits16(b: &[u8]) -> Option<u64> {
    let n = b.len();
    if n > 8 {
        // two direct loads (overlapping when n < 16); no stack copy, so no
        // store-to-load forwarding stall
        let w1 = u64::from_le_bytes(b[n - 8..].try_into().unwrap());
        let sh = 8 * (16 - n) as u32;
        let head = u64::from_le_bytes(b[..8].try_into().unwrap());
        let pad = if sh == 0 { 0 } else { 0x3030_3030_3030_3030u64 >> (64 - sh) };
        let w0 = if sh == 0 { head } else { (head << sh) | pad };
        if !(digits_ok(w0, !0) & digits_ok(w1, !0)) {
            return None;
        }
        return Some(p8(w0) * 100_000_000 + p8(w1));
    }
    let mut buf = [b'0'; 16];
    buf[16 - b.len()..].copy_from_slice(b);
    let w0 = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let w1 = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if !(digits_ok(w0, !0) & digits_ok(w1, !0)) {
        return None;
    }
    Some(p8(w0) * 100_000_000 + p8(w1))
}

#[inline]
fn dg(b: u8) -> Option<i64> {
    let v = b.wrapping_sub(b'0');
    if v < 10 { Some(v as i64) } else { None }
}
#[inline]
fn d2(b: &[u8], i: usize) -> Option<i64> {
    Some(dg(b[i])? * 10 + dg(b[i + 1])?)
}

#[inline(always)]
fn digits_ok(w: u64, d: u64) -> bool {
    const HI: u64 = 0xF0F0_F0F0_F0F0_F0F0;
    const T: u64 = 0x3030_3030_3030_3030;
    const SIX: u64 = 0x0606_0606_0606_0606;
    (w & d & HI) == (d & T) && ((w.wrapping_add(d & SIX)) & d & HI) == (d & T)
}

/// `YYYY-MM-DDTHH:MM:SS` at the start of `b` (b.len() >= 20) as epoch ms, checked.
#[inline(always)]
fn civil_ms(b: &[u8]) -> Option<i64> {
    let w0 = u64::from_le_bytes(b[0..8].try_into().unwrap()); // YYYY-MM-
    let w1 = u64::from_le_bytes(b[8..16].try_into().unwrap()); // DDTHH:MM
    let w2 = u32::from_le_bytes(b[16..20].try_into().unwrap()) as u64; // :SS?
    const D0: u64 = 0x00FF_FF00_FFFF_FFFF;
    const S0: u64 = 0xFF00_00FF_0000_0000;
    const V0: u64 = ((b'-' as u64) << 56) | ((b'-' as u64) << 32);
    const D1: u64 = 0xFFFF_00FF_FF00_FFFF;
    const S1: u64 = 0x0000_FF00_00FF_0000;
    const V1: u64 = ((b':' as u64) << 40) | ((b'T' as u64) << 16);
    const D2: u64 = 0x0000_0000_00FF_FF00;
    let ok = digits_ok(w0, D0)
        & digits_ok(w1, D1)
        & digits_ok(w2, D2)
        & ((w0 & S0) == V0)
        & ((w1 & S1) == V1)
        & ((w2 & 0xFF) == b':' as u64);
    if !ok {
        return None;
    }
    let x0 = w0 & 0x0F0F_0F0F_0F0F_0F0F;
    let x1 = w1 & 0x0F0F_0F0F_0F0F_0F0F;
    let x2 = w2 & 0x0F0F_0F0F;
    let g = |x: u64, i: u32| ((x >> (8 * i)) & 0xFF) as i64;
    let y = g(x0, 0) * 1000 + g(x0, 1) * 100 + g(x0, 2) * 10 + g(x0, 3);
    let mo = g(x0, 5) * 10 + g(x0, 6);
    let d = g(x1, 0) * 10 + g(x1, 1);
    let h = g(x1, 3) * 10 + g(x1, 4);
    let mi = g(x1, 6) * 10 + g(x1, 7);
    let sec = g(x2, 1) * 10 + g(x2, 2);
    if !(1..=12).contains(&mo) || h > 23 || mi > 59 || sec > 59 || d < 1 {
        return None;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let dim = match mo {
        2 => 28 + leap as i64,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if d > dim {
        return None;
    }
    Some((fast_days(y, mo, d, leap) * 86_400 + h * 3600 + mi * 60 + sec) * 1000)
}

/// `ingest_ts`, which rules.md guarantees is valid F1: shape-checked, digits not
/// re-validated. Anything not shaped like F1 goes through the full parser.
#[inline(always)]
fn parse_ingest(b: &[u8]) -> i64 {
    if b.len() == 20 && b[19] == b'Z' && b[4] == b'-' && b[10] == b'T' {
        let w0 = u64::from_le_bytes(b[0..8].try_into().unwrap()) & 0x0F0F_0F0F_0F0F_0F0F;
        let w1 = u64::from_le_bytes(b[8..16].try_into().unwrap()) & 0x0F0F_0F0F_0F0F_0F0F;
        let w2 = u32::from_le_bytes(b[16..20].try_into().unwrap()) as u64 & 0x0F0F_0F0F;
        let g = |x: u64, i: u32| ((x >> (8 * i)) & 0xFF) as i64;
        let y = g(w0, 0) * 1000 + g(w0, 1) * 100 + g(w0, 2) * 10 + g(w0, 3);
        let mo = g(w0, 5) * 10 + g(w0, 6);
        if (1900..2300).contains(&y) && (1..=12).contains(&mo) {
            let d = g(w1, 0) * 10 + g(w1, 1);
            let h = g(w1, 3) * 10 + g(w1, 4);
            let mi = g(w1, 6) * 10 + g(w1, 7);
            let sec = g(w2, 1) * 10 + g(w2, 2);
            let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
            return (fast_days(y, mo, d, leap) * 86_400 + h * 3600 + mi * 60 + sec) * 1000;
        }
    }
    parse_ts(b).unwrap_or(i64::MAX)
}

/// The local dates a timestamp in the click window can carry (window ± 1 day for
/// F2 offsets), grouped by their `YYYY-MM-` prefix. A date outside the table is either
/// impossible or outside the window, so it is bad either way.
struct DateTab {
    pre: [u64; 4],
    base: [i64; 4], // day number = base + day of month
    lo: [i64; 4],
    hi: [i64; 4],
    n: usize,
}

static DATES: std::sync::OnceLock<DateTab> = std::sync::OnceLock::new();

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + (m <= 2) as i64, m, d)
}

fn build_dates(from_ms: i64, to_ms: i64) -> DateTab {
    let mut t = DateTab { pre: [0; 4], base: [0; 4], lo: [0; 4], hi: [0; 4], n: 0 };
    for day in from_ms.div_euclid(DAY) - 1..=(to_ms - 1).div_euclid(DAY) + 1 {
        let (y, m, d) = civil_from_days(day);
        let txt = format!("{y:04}-{m:02}-");
        let pre = u64::from_le_bytes(txt.as_bytes()[..8].try_into().unwrap());
        if t.n > 0 && t.pre[t.n - 1] == pre {
            t.hi[t.n - 1] = d;
        } else {
            assert!(t.n < 4, "window spans too many months");
            t.pre[t.n] = pre;
            t.base[t.n] = day - d;
            t.lo[t.n] = d;
            t.hi[t.n] = d;
            t.n += 1;
        }
    }
    t
}

/// `YYYY-MM-DDTHH:MM:SS` (b.len() >= 20) via the date table; None if bad or far outside the window.
#[inline(always)]
fn table_ms(b: &[u8], t: &DateTab) -> Option<i64> {
    let w0 = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let k = (0..t.n).find(|&k| t.pre[k] == w0)?;
    let d = d2(b, 8)?;
    if d < t.lo[k] || d > t.hi[k] || b[10] != b'T' {
        return None;
    }
    let w = u64::from_le_bytes(b[11..19].try_into().unwrap()); // HH:MM:SS
    const D: u64 = 0xFFFF_00FF_FF00_FFFF;
    const S: u64 = 0x0000_FF00_00FF_0000;
    const V: u64 = ((b':' as u64) << 40) | ((b':' as u64) << 16);
    if !(digits_ok(w, D) & ((w & S) == V)) {
        return None;
    }
    let x = w & 0x0F0F_0F0F_0F0F_0F0F;
    let g = |i: u32| ((x >> (8 * i)) & 0xFF) as i64;
    let (h, mi, sec) = (g(0) * 10 + g(1), g(3) * 10 + g(4), g(6) * 10 + g(7));
    if h > 23 || mi > 59 || sec > 59 {
        return None;
    }
    Some(((t.base[k] + d) * 86_400 + h * 3600 + mi * 60 + sec) * 1000)
}

/// parse_ts for event and click `ts`: same results for every timestamp inside the
/// click window; may say None for valid ones far outside it (rejected anyway).
#[inline(always)]
fn window_ts(b: &[u8], t: &DateTab) -> Option<i64> {
    let n = b.len();
    if n == 20 && b[4] == b'-' {
        if b[19] != b'Z' {
            return None;
        }
        return table_ms(b, t);
    }
    if n == 25 && b[4] == b'-' {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        return Some(table_ms(b, t)? - sign * (oh * 60 + om) * 60_000);
    }
    parse_ts(b)
}

#[inline]
fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 0 {
        return None;
    }
    // F1/F2 have '-' at index 4; anything else can only be F3 (digits only).
    if !((n == 20 || n == 25) && b[4] == b'-') {
        if n <= 16 {
            return digits16(b).map(|v| v as i64);
        }
        if !b.iter().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let mut i = 0;
        while i < n && b[i] == b'0' {
            i += 1;
        }
        if n - i > 19 {
            return None;
        }
        let mut v: u64 = 0;
        for &c in &b[i..] {
            v = v * 10 + (c - b'0') as u64;
        }
        if v > i64::MAX as u64 {
            return None;
        }
        return Some(v as i64);
    }
    if n == 20 {
        if b[19] != b'Z' {
            return None;
        }
        return civil_ms(b);
    }
    if n == 25 {
        let sign = match b[19] {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if b[22] != b':' {
            return None;
        }
        let oh = d2(b, 20)?;
        let om = d2(b, 23)?;
        if oh > 23 || om > 59 {
            return None;
        }
        let t = civil_ms(b)?;
        return Some(t - sign * (oh * 60 + om) * 60_000);
    }
    None
}

#[inline]
fn parse_price(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let mut i = 0;
    while i < b.len() && b[i] == b'0' {
        i += 1;
    }
    if b.len() - i > 8 {
        return None;
    }
    let mut v: u32 = 0;
    for &c in &b[i..] {
        v = v * 10 + (c - b'0') as u32;
    }
    if (1..=10_000_000).contains(&v) { Some(v) } else { None }
}

// ---------------------------------------------------------------- records


#[derive(Clone, Copy)]
struct Ev {
    person: u64, // user ID, then person; dense person id inside the sessions phase
    ts: u32,
    pk: u32, // price (24 bits) | category (7 bits) << 24 | purchase << 31
}

impl Ev {
    #[inline(always)]
    fn price(&self) -> u32 {
        self.pk & 0xFF_FFFF
    }
    #[inline(always)]
    fn cat(&self) -> usize {
        ((self.pk >> 24) & 0x7F) as usize
    }
    #[inline(always)]
    fn purchase(&self) -> bool {
        self.pk >> 31 != 0
    }
}

#[derive(Clone, Copy)]
struct Click {
    person: u64,
    click: u64,
    ts: u32, // ms after window_start - 7 days
    camp: u32,
}

struct UserRow {
    person: u64,
    day: i32,
    sessions: i64,
    events: i64,
    revenue: i64,
    top: i32,
}

// ---------------------------------------------------------------- helpers

fn workers<S: Send, I: Fn() -> S + Sync, F: Fn(&mut S, usize) + Sync>(
    nthreads: usize,
    ntasks: usize,
    init: I,
    f: F,
) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nthreads)
            .map(|_| {
                sc.spawn(|| {
                    let mut s = init();
                    loop {
                        let t = next.fetch_add(1, AO::Relaxed);
                        if t >= ntasks {
                            break;
                        }
                        f(&mut s, t);
                    }
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Like `workers`, but each thread calls `fin(state, thread index)` once it runs out of tasks.
fn workers_fin<S: Send, I: Fn() -> S + Sync, F: Fn(&mut S, usize) + Sync, G: Fn(&mut S, usize) + Sync>(
    nthreads: usize,
    ntasks: usize,
    init: I,
    f: F,
    fin: G,
) -> Vec<S> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nthreads)
            .map(|ti| {
                let (next, init, f, fin) = (&next, &init, &f, &fin);
                sc.spawn(move || {
                    let mut s = init();
                    loop {
                        let t = next.fetch_add(1, AO::Relaxed);
                        if t >= ntasks {
                            break;
                        }
                        f(&mut s, t);
                    }
                    fin(&mut s, ti);
                    s
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Writes one user_daily Parquet file.
fn write_user_file(path: &Path, rows: &[UserRow], cats_s: &[String]) {
    use arrow_array::builder::{Date32Builder, Int64Builder, StringBuilder};
    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;
    let mut uid = StringBuilder::with_capacity(rows.len(), rows.len() * 10);
    let mut day = Date32Builder::with_capacity(rows.len());
    let mut se = Int64Builder::with_capacity(rows.len());
    let mut evs = Int64Builder::with_capacity(rows.len());
    let mut rv = Int64Builder::with_capacity(rows.len());
    let mut top = StringBuilder::with_capacity(rows.len(), rows.len() * 8);
    let mut nb = [0u8; 10];
    for r in rows {
        let k = r.person;
        if !is_long(k) {
            // decode into a stack buffer: no allocation per row
            let n = (k & 0xF) as usize;
            for (i, c) in nb[..n].iter_mut().enumerate() {
                *c = ALPHA[((k >> (58 - 6 * i)) & 63) as usize];
            }
            uid.append_value(unsafe { std::str::from_utf8_unchecked(&nb[..n]) }); // alphabet is ASCII
        } else {
            let name = unpack(k);
            match std::str::from_utf8(&name) {
                Ok(x) => uid.append_value(x),
                Err(_) => uid.append_value(String::from_utf8_lossy(&name)),
            }
        }
        day.append_value(r.day);
        se.append_value(r.sessions);
        evs.append_value(r.events);
        rv.append_value(r.revenue);
        if r.top >= 0 {
            top.append_value(&cats_s[r.top as usize]);
        } else {
            top.append_null();
        }
    }
    let batch = RecordBatch::try_from_iter(vec![
        ("user_id", Arc::new(uid.finish()) as ArrayRef),
        ("day", Arc::new(day.finish()) as ArrayRef),
        ("sessions", Arc::new(se.finish()) as ArrayRef),
        ("events", Arc::new(evs.finish()) as ArrayRef),
        ("revenue_cents", Arc::new(rv.finish()) as ArrayRef),
        ("top_category", Arc::new(top.finish()) as ArrayRef),
    ])
    .unwrap();
    let f = File::create(path).unwrap();
    let mut wr = ArrowWriter::try_new(f, batch.schema(), Some(parquet_props())).unwrap();
    wr.write(&batch).unwrap();
    wr.close().unwrap();
}

fn parquet_props() -> parquet::file::properties::WriterProperties {
    parquet::file::properties::WriterProperties::builder()
        .set_compression(parquet::basic::Compression::UNCOMPRESSED)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
        .build()
}

/// [worker][bucket] -> [bucket] (worker lists), each bucket taken once by its task.
fn transpose<T>(v: Vec<Vec<Vec<T>>>) -> Vec<Mutex<Vec<Vec<T>>>> {
    let mut out: Vec<Mutex<Vec<Vec<T>>>> = (0..NB).map(|_| Mutex::new(Vec::new())).collect();
    for w in v {
        for (b, x) in w.into_iter().enumerate() {
            out[b].get_mut().unwrap().push(x);
        }
    }
    out
}

fn csv_files(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".csv") && !name.starts_with('.') {
                let p = e.path();
                if p.is_file() {
                    v.push(p);
                }
            }
        }
    }
    v.sort();
    v
}

struct Src {
    files: Vec<File>,
    lens: Vec<usize>,
    maps: Vec<Option<memmap2::Mmap>>,
}

fn use_mmap() -> bool {
    std::env::var("IO_MODE").map_or(true, |v| v != "pread")
}

fn open_all(paths: &[PathBuf]) -> Src {
    let files: Vec<File> = paths.iter().map(|p| File::open(p).unwrap()).collect();
    let lens: Vec<usize> = files.iter().map(|f| f.metadata().unwrap().len() as usize).collect();
    let maps = files
        .iter()
        .zip(&lens)
        .map(|(f, &n)| {
            if n == 0 || !use_mmap() {
                return None;
            }
            let m = unsafe { memmap2::MmapOptions::new().map(f).unwrap() };
            let _ = m.advise(memmap2::Advice::WillNeed);
            Some(m)
        })
        .collect();
    Src { files, lens, maps }
}

/// Nominal (file, start, end) byte ranges. A range owns the lines that start inside it.
fn chunks(src: &Src) -> Vec<(usize, usize, usize)> {
    chunks_sized(src, CHUNK)
}

fn chunks_sized(src: &Src, size: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for (fi, &n) in src.lens.iter().enumerate() {
        let mut s = 0;
        while s < n {
            let e = (s + size).min(n);
            out.push((fi, s, e));
            s = e;
        }
    }
    // biggest first for better balance
    out.sort_by_key(|c| std::cmp::Reverse(c.2 - c.1));
    out
}

fn read_at(f: &File, buf: &mut Vec<u8>, off: usize, len: usize) {
    use std::os::unix::fs::FileExt;
    let old = buf.len();
    buf.resize(old + len, 0);
    let mut done = 0;
    while done < len {
        let k = f.read_at(&mut buf[old + done..], (off + done) as u64).unwrap();
        if k == 0 {
            buf.truncate(old + done);
            return;
        }
        done += k;
    }
}

/// Reads the lines owned by chunk `c` (those starting inside [s, e), never the
/// file's header line) into `buf` and returns them.
fn load_chunk<'a>(src: &Src, c: (usize, usize, usize), buf: &'a mut Vec<u8>) -> &'a [u8] {
    let (fi, s, e) = c;
    let f = &src.files[fi];
    let n = src.lens[fi];
    if let Some(m) = &src.maps[fi] {
        let d: &'a [u8] = unsafe { std::slice::from_raw_parts(m.as_ptr(), n) };
        let r0 = s.saturating_sub(1);
        let Some(i) = memchr::memchr(b'\n', &d[r0..]) else { return &d[..0] };
        let begin = r0 + i + 1;
        if begin >= e {
            return &d[..0];
        }
        let end = match memchr::memchr(b'\n', &d[e - 1..]) {
            Some(o) => e + o,
            None => n,
        };
        return &d[begin..end];
    }
    buf.clear();
    let r0 = s.saturating_sub(1);
    read_at(f, buf, r0, (e + 4096).min(n) - r0);
    let Some(i) = memchr::memchr(b'\n', buf) else { return &buf[..0] };
    let begin = i + 1;
    if r0 + begin >= e {
        return &buf[..0];
    }
    let from = e - 1 - r0;
    let mut scanned = from;
    loop {
        if let Some(o) = memchr::memchr(b'\n', &buf[scanned..]) {
            let end = scanned + o + 1;
            return &buf[begin..end];
        }
        scanned = buf.len();
        if r0 + buf.len() >= n {
            return &buf[begin..];
        }
        let at = r0 + buf.len();
        read_at(f, buf, at, 65536.min(n - at));
    }
}

fn read_small(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_default()
}

fn small_csv<const N: usize>(path: &Path, mut f: impl FnMut(&[&[u8]; N])) {
    let d = read_small(path);
    let Some(h) = memchr::memchr(b'\n', &d) else { return };
    for_lines(&d[h + 1..], |line| {
        let mut fs: [&[u8]; N] = [&[]; N];
        if split(line, &mut fs) {
            f(&fs);
        }
    });
}

fn manifest_ts(text: &str, key: &str) -> i64 {
    let at = text.find(&format!("\"{key}\"")).expect("manifest key");
    let rest = &text[at + key.len() + 2..];
    let q1 = rest.find('"').unwrap();
    let rest = &rest[q1 + 1..];
    let q2 = rest.find('"').unwrap();
    parse_ts(rest[..q2].as_bytes()).expect("manifest ts")
}

/// Joins linked IDs; maps every linked ID whose person name differs to that name.
fn build_link_table(pairs: Vec<Vec<(u64, u64)>>) -> LinkTable {
    let link_map: LinkTable = {
        let total: usize = pairs.iter().map(|v| v.len()).sum();
        let mut idx: KMap<u32> = KMap::with_capacity_and_hasher(total, Default::default());
        let mut keys: Vec<u64> = Vec::with_capacity(total);
        let mut parent: Vec<u32> = Vec::with_capacity(total);
        fn find(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                let g = p[p[x as usize] as usize];
                p[x as usize] = g;
                x = g;
            }
            x
        }
        let mut get = |k: u64, keys: &mut Vec<u64>, parent: &mut Vec<u32>| -> u32 {
            *idx.entry(k).or_insert_with(|| {
                keys.push(k);
                parent.push(parent.len() as u32);
                (keys.len() - 1) as u32
            })
        };
        for v in &pairs {
            for &(a, b) in v {
                let ia = get(a, &mut keys, &mut parent);
                let ib = get(b, &mut keys, &mut parent);
                let ra = find(&mut parent, ia);
                let rb = find(&mut parent, ib);
                if ra != rb {
                    // keep smaller index as root (arbitrary but deterministic)
                    if ra < rb {
                        parent[rb as usize] = ra;
                    } else {
                        parent[ra as usize] = rb;
                    }
                }
            }
        }
        drop(pairs);
        let n = keys.len();
        let mut best: Vec<u64> = keys.clone();
        for i in 0..n {
            let r = find(&mut parent, i as u32) as usize;
            if cmp_str(keys[i], best[r]) == Ordering::Less {
                best[r] = keys[i];
            }
        }
        let moved = (0..n).filter(|&i| best[find(&mut parent, i as u32) as usize] != keys[i]).count();
        let mut m = LinkTable::new(moved);
        for i in 0..n {
            let r = find(&mut parent, i as u32) as usize;
            if best[r] != keys[i] {
                m.insert(keys[i], best[r]);
            }
        }
        m
    };
    link_map
}

// ---------------------------------------------------------------- main

fn main() {
    let t0 = Instant::now();
    let data = PathBuf::from(std::env::var("DATA_DIR").unwrap_or("/data".into()));
    let out = PathBuf::from(std::env::var("OUTPUT_DIR").unwrap_or("/output".into()));
    let nthreads: usize = std::env::var("NCPU")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |n| n.get()));
    let lap = |what: &str| eprintln!("{:8.3}s {what}", t0.elapsed().as_secs_f64());

    let manifest = std::fs::read_to_string(data.join("manifest.json")).unwrap();
    let ws = manifest_ts(&manifest, "window_start");
    let we = manifest_ts(&manifest, "window_end");
    let cws = ws - WEEK;
    let ws_s = ws.div_euclid(1000);
    let dates: &DateTab = DATES.get_or_init(|| build_dates(cws, we));
    assert!(we - cws < (1i64 << 32), "window too long for u32 offsets");
    let ws_day = ws.div_euclid(DAY);
    let ndays = ((we - 1).div_euclid(DAY) - ws_day + 1).max(1) as usize;

    // products, categories
    let mut prod_raw: Vec<(u64, Vec<u8>)> = Vec::new();
    small_csv::<2>(&data.join("products.csv"), |f| {
        prod_raw.push((pack(f[0]), f[1].to_vec()));
    });
    let mut cats: Vec<Vec<u8>> = prod_raw.iter().map(|p| p.1.clone()).collect();
    cats.push(b"unknown".to_vec());
    cats.sort();
    cats.dedup();
    let ncat = cats.len();
    assert!(ncat <= 127, "category index must fit 7 bits");
    let unknown = cats.binary_search(&b"unknown".to_vec()).unwrap() as u16;
    let mut products: KMap<u16> = KMap::default();
    for (k, c) in &prod_raw {
        products.insert(*k, cats.binary_search(c).unwrap() as u16);
    }

    // campaigns
    let mut camp_ids: Vec<Vec<u8>> = Vec::new();
    let mut camp_ch: Vec<Vec<u8>> = Vec::new();
    let mut campaigns: KMap<u32> = KMap::default();
    small_csv::<2>(&data.join("campaigns.csv"), |f| {
        let k = pack(f[0]);
        if !campaigns.contains_key(&k) {
            campaigns.insert(k, camp_ids.len() as u32);
            camp_ids.push(f[0].to_vec());
            camp_ch.push(f[1].to_vec());
        }
    });
    let ncamp = camp_ids.len();
    let organic = ncamp as u32;
    camp_ids.push(b"organic".to_vec());
    camp_ch.push(b"organic".to_vec());

    const BATCH: usize = 64;
    let link_cell: std::sync::OnceLock<LinkTable> = std::sync::OnceLock::new();

    // events + clicks parse (links are joined meanwhile on their own thread)
    let esrc = open_all(&csv_files(&data.join("events")));
    let csrc = open_all(&csv_files(&data.join("clicks")));
    let ech = chunks(&esrc);
    let cch = chunks(&csrc);
    let ne = ech.len();
    let lsrc = open_all(&csv_files(&data.join("links")));
    let lch = chunks_sized(&lsrc, 1 << 20);
    let nl = lch.len();
    let link_parts: Mutex<Vec<Vec<(u64, u64)>>> = Mutex::new(Vec::new());
    let links_left = AtomicUsize::new(nl);
    // Dedup tuples: everything dedup needs, and where the full record went.
    #[derive(Clone, Copy)]
    struct Tup {
        eid: u64,
        ingest: u32,
        loc: u32, // index into the worker's record array; u32::MAX for an invalid copy
    }
    const NE: usize = 256; // event_id partitions
    struct P1 {
        wid: usize,
        fill: Vec<usize>,      // tuples written into this worker's arena region per partition
        spill: Vec<Vec<Tup>>,  // overflow when a region is full
        rec: Vec<Ev>,   // valid events in parse order; person holds the user ID until resolved
        alive: Vec<u8>, // 1 per record; dedup clears losers
        cl: Vec<Vec<Click>>,
        buf: Vec<u8>,
        bits: Vec<[u64; 3]>,
        pendc: Vec<Click>,
    }
    fn flush_cl(pend: &mut Vec<Click>, cl: &mut [Vec<Click>], lt: &LinkTable) {
        for c in pend.iter() {
            lt.prefetch(c.person);
        }
        for mut c in pend.drain(..) {
            if lt.maybe(c.person) {
                c.person = lt.get(c.person);
            }
            cl[part(c.person)].push(c);
        }
    }
    let ebytes: usize = esrc.lens.iter().sum();
    let cbytes: usize = csrc.lens.iter().sum();
    // One arena for all dedup tuples: a fixed region per (worker, partition), so the
    // memory is touched once here and reused afterwards for the person shuffle.
    let capb = std::env::var("CAPB") // test knob: tiny regions exercise the spill path
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(ebytes / 70 / nthreads / NE * 13 / 10 + 64);
    let arena_cap = nthreads * NE * capb;
    let mut arena: Vec<Tup> = Vec::with_capacity(arena_cap);
    struct TupPtr(*mut Tup);
    unsafe impl Sync for TupPtr {}
    impl TupPtr {
        fn get(&self) -> *mut Tup {
            self.0
        }
    }
    let arena_ptr = TupPtr(arena.as_mut_ptr());
    let next_wid = AtomicUsize::new(0);
    let rcap = ebytes / 70 / nthreads + 1024;
    let ccap = cbytes / 40 / nthreads / NB + 64;
    let p1: Vec<P1> = std::thread::scope(|sc| {
    sc.spawn(|| {
        // union-find once every link chunk is parsed (they are the first tasks)
        while links_left.load(AO::Acquire) > 0 {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        let parts = std::mem::take(&mut *link_parts.lock().unwrap());
        let _ = link_cell.set(build_link_table(parts));
        lap("links joined");
    });
    workers(
        nthreads,
        nl + ne + cch.len(),
        || P1 {
            wid: next_wid.fetch_add(1, AO::Relaxed),
            fill: vec![0; NE],
            spill: (0..NE).map(|_| Vec::new()).collect(),
            rec: Vec::with_capacity(rcap),
            alive: Vec::with_capacity(rcap),
            cl: (0..NB).map(|_| Vec::with_capacity(ccap)).collect(),
            buf: Vec::new(),
            bits: Vec::new(),
            pendc: Vec::with_capacity(BATCH),
        },
        |st, t| {
            if t < nl {
                let mut v: Vec<(u64, u64)> = Vec::new();
                scan_lines::<2>(load_chunk(&lsrc, lch[t], &mut st.buf), &mut st.bits, |f| {
                    if f[0].is_empty() || f[1].is_empty() {
                        return;
                    }
                    v.push((pack_reg(f[0]), pack_reg(f[1])));
                });
                link_parts.lock().unwrap().push(v);
                links_left.fetch_sub(1, AO::Release);
                return;
            }
            let t = t - nl;
            if t < ne {
                scan_lines::<7>(load_chunk(&esrc, ech[t], &mut st.buf), &mut st.bits, |f| {
                    if f[0].is_empty() {
                        return;
                    }
                    let eid = pack(f[0]);
                    let ingest = parse_ingest(f[3]);
                    let ing32 = (ingest.div_euclid(1000) - ws_s + (1i64 << 31)).clamp(0, u32::MAX as i64) as u32;
                    let mut loc = u32::MAX;
                    'v: {
                        if f[1].is_empty() || f[5].is_empty() {
                            break 'v;
                        }
                        let Some(ts) = window_ts(f[2], dates) else { break 'v };
                        if ts < ws || ts >= we || ingest.saturating_sub(ts) > DAY {
                            break 'v;
                        }
                        let mut e = Ev { person: 0, ts: (ts - ws) as u32, pk: 0 };
                        match f[4] {
                            b"view" | b"cart" => {}
                            b"purchase" => {
                                let Some(p) = parse_price(f[6]) else { break 'v };
                                let cat = *products.get(&pack(f[5])).unwrap_or(&unknown) as u32;
                                e.pk = p | cat << 24 | 1 << 31;
                            }
                            _ => break 'v,
                        }
                        e.person = pack_reg(f[1]); // the ID; the person is looked up after dedup
                        loc = st.rec.len() as u32;
                        st.rec.push(e);
                        st.alive.push(1);
                    }
                    let pb = part(eid) & (NE - 1);
                    let tup = Tup { eid, ingest: ing32, loc };
                    if st.fill[pb] < capb {
                        // region (wid, pb) belongs to this worker alone
                        unsafe { arena_ptr.get().add((st.wid * NE + pb) * capb + st.fill[pb]).write(tup) };
                        st.fill[pb] += 1;
                    } else {
                        st.spill[pb].push(tup);
                    }
                });
            } else {
                scan_lines::<4>(load_chunk(&csrc, cch[t - ne], &mut st.buf), &mut st.bits, |f| {
                    if f[0].is_empty() || f[1].is_empty() {
                        return;
                    }
                    let Some(ts) = window_ts(f[3], dates) else { return };
                    if ts < cws || ts >= we {
                        return;
                    }
                    let Some(&camp) = campaigns.get(&pack(f[2])) else { return };
                    let person = pack_reg(f[1]);
                    let click = pack_reg(f[0]);
                    st.pendc.push(Click { person, click, ts: (ts - cws) as u32, camp });
                    if st.pendc.len() == BATCH {
                        flush_cl(&mut st.pendc, &mut st.cl, link_cell.wait());
                    }
                });
                flush_cl(&mut st.pendc, &mut st.cl, link_cell.wait());
            }
        },
    )
    });
    lap("events+clicks parsed");
    let link_map = link_cell.get().unwrap();
    let nw = p1.len();

    // Dedup per event_id partition: the first copy seen is held in the table; each
    // later copy loses to it or replaces it (smaller ingest), and the loser's alive
    // byte is cleared on the spot. Equal ingest: the held copy stays.
    struct Shared(Vec<*mut u8>);
    unsafe impl Sync for Shared {}
    impl Shared {
        fn at(&self, w: usize) -> *mut u8 {
            self.0[w]
        }
    }
    // per worker id: region fill counts and spills
    let mut fills: Vec<Vec<usize>> = vec![Vec::new(); nw];
    let mut spills: Vec<Vec<Vec<Tup>>> = (0..nw).map(|_| Vec::new()).collect();
    let mut recs: Vec<Mutex<Vec<Ev>>> = Vec::with_capacity(nw);
    let mut alives: Vec<Vec<u8>> = Vec::with_capacity(nw);
    let mut clv: Vec<Vec<Vec<Click>>> = Vec::with_capacity(nw);
    let mut p1 = p1;
    p1.sort_by_key(|p| p.wid); // worker index == wid everywhere below
    for p in p1 {
        assert!(p.rec.len() < (1 << 27) - 1, "too many records for one worker");
        fills[p.wid] = p.fill;
        spills[p.wid] = p.spill;
        recs.push(Mutex::new(p.rec));
        alives.push(p.alive);
        clv.push(p.cl);
    }
    let clicks_by = transpose(clv);
    let alive_ptr = Shared(alives.iter_mut().map(|a| a.as_mut_ptr()).collect());
    // loc packs the worker (5 bits) and record index (27 bits); DEAD marks an invalid copy
    const DEAD: u32 = (1 << 27) - 1;
    assert!(nw <= 32);
    workers(
        nthreads,
        NE,
        || (Vec::<Tup>::new(), Vec::<Tup>::new()),
        |(tab, sub), b| {
            // this partition's tuples: per worker, its arena region then its spill
            let mine: Vec<[&[Tup]; 2]> = (0..nw)
                .map(|w| {
                    let r = unsafe { std::slice::from_raw_parts(arena_ptr.get().add((w * NE + b) * capb), fills[w][b]) };
                    [r, &spills[w][b][..]]
                })
                .collect();
            let total: usize = mine.iter().map(|v| v[0].len() + v[1].len()).sum();
            let kill = |wl: u32| {
                let loc = wl & DEAD;
                if loc != DEAD {
                    // distinct bytes per record; only this task decides about this event
                    unsafe { alive_ptr.at((wl >> 27) as usize).add(loc as usize).write(0) };
                }
            };
            let pk = |w: usize, t: &Tup| Tup {
                eid: t.eid,
                ingest: t.ingest,
                loc: ((w as u32) << 27) | if t.loc == u32::MAX { DEAD } else { t.loc },
            };
            // split into cache-sized pieces by high hash bits (table slots use low bits)
            let sbits = (total / 16384).next_power_of_two().trailing_zeros().min(8);
            let ns = 1usize << sbits;
            let mut soff = vec![0usize; ns + 1];
            sub.clear();
            sub.reserve(total);
            if ns == 1 {
                for (w, v) in mine.iter().enumerate() {
                    for part in v {
                        sub.extend(part.iter().map(|t| pk(w, t)));
                    }
                }
                soff[1] = total;
            } else {
                let sh = 64 - sbits;
                for v in &mine {
                    for t in v.iter().flat_map(|x| x.iter()) {
                        soff[(mix(t.eid) >> sh) as usize + 1] += 1;
                    }
                }
                for i in 0..ns {
                    soff[i + 1] += soff[i];
                }
                let mut fill = soff.clone();
                let dst = sub.as_mut_ptr();
                for (w, v) in mine.iter().enumerate() {
                    for t in v.iter().flat_map(|x| x.iter()) {
                        let k = (mix(t.eid) >> sh) as usize;
                        unsafe { dst.add(fill[k]).write(pk(w, t)) };
                        fill[k] += 1;
                    }
                }
                unsafe { sub.set_len(total) };
            }
            drop(mine);
            for si in 0..ns {
                let piece = &sub[soff[si]..soff[si + 1]];
                let cap = (piece.len() * 2).next_power_of_two().max(16);
                let mask = cap - 1;
                tab.clear();
                tab.resize(cap, Tup { eid: 0, ingest: 0, loc: 0 });
                for t in piece {
                    let mut i = mix(t.eid) as usize & mask;
                    loop {
                        let sl = unsafe { tab.get_unchecked_mut(i) };
                        if sl.eid == 0 {
                            *sl = *t;
                            break;
                        }
                        if sl.eid == t.eid {
                            if t.ingest < sl.ingest {
                                kill(sl.loc);
                                *sl = *t;
                            } else {
                                kill(t.loc);
                            }
                            break;
                        }
                        i = (i + 1) & mask;
                    }
                }
            }
        },
    );
    drop(spills);
    lap("events deduped");

    // Person shuffle, count then scatter: each worker resolves the person of its alive
    // records (prefetch ring over the link table), counts them per person partition,
    // then writes them straight into one shared array at its own offsets.
    let counts: Vec<Mutex<Vec<usize>>> = (0..nw).map(|_| Mutex::new(Vec::new())).collect();
    workers(nthreads, nw, || (), |_, w| {
        let mut rec = recs[w].lock().unwrap();
        let alive = &alives[w];
        let mut c = vec![0usize; NB];
        const RING: usize = 64;
        let mut ring = [0u32; RING];
        let mut nr = 0usize;
        for i in 0..rec.len() {
            if alive[i] == 0 {
                continue;
            }
            if !link_map.maybe(rec[i].person) {
                c[part(rec[i].person)] += 1; // not linked: the ID is the person
                continue;
            }
            link_map.prefetch(rec[i].person);
            let slot = &mut ring[nr & (RING - 1)];
            if nr >= RING {
                let j = *slot as usize;
                let p = link_map.get(rec[j].person);
                rec[j].person = p;
                c[part(p)] += 1;
            }
            *slot = i as u32;
            nr += 1;
        }
        for k in nr.saturating_sub(RING)..nr {
            let j = ring[k & (RING - 1)] as usize;
            let p = link_map.get(rec[j].person);
            rec[j].person = p;
            c[part(p)] += 1;
        }
        *counts[w].lock().unwrap() = c;
    });
    let counts: Vec<Vec<usize>> = counts.into_iter().map(|m| m.into_inner().unwrap()).collect();
    let mut off: Vec<Vec<usize>> = vec![vec![0; NB]; nw];
    let mut bstart = vec![0usize; NB + 1];
    let mut acc = 0;
    for b in 0..NB {
        bstart[b] = acc;
        for w in 0..nw {
            off[w][b] = acc;
            acc += counts[w][b];
        }
    }
    bstart[NB] = acc;
    // reuse the arena's pages (tuples are dead now; same size and alignment) when big enough
    const _: () = assert!(std::mem::size_of::<Tup>() == std::mem::size_of::<Ev>() && std::mem::align_of::<Tup>() == std::mem::align_of::<Ev>());
    let mut all: Vec<Ev> = if acc <= arena_cap {
        let mut a = std::mem::ManuallyDrop::new(std::mem::take(&mut arena));
        unsafe { Vec::from_raw_parts(a.as_mut_ptr() as *mut Ev, 0, a.capacity()) }
    } else {
        Vec::with_capacity(acc)
    };
    struct EvPtr(*mut Ev);
    unsafe impl Sync for EvPtr {}
    impl EvPtr {
        fn get(&self) -> *mut Ev {
            self.0
        }
    }
    let dst = EvPtr(all.as_mut_ptr());
    workers(nthreads, nw, || (), |_, w| {
        let rec = recs[w].lock().unwrap();
        let alive = &alives[w];
        let mut pos = off[w].clone();
        for i in 0..rec.len() {
            if alive[i] != 0 {
                let e = rec[i];
                let b = part(e.person);
                // disjoint ranges per (bucket, worker)
                unsafe { dst.get().add(pos[b]).write(e) };
                pos[b] += 1;
            }
        }
    });
    unsafe { all.set_len(acc) };
    drop(recs);
    drop(alives);
    lap("persons shuffled");


    // sessionize + attribute per person partition
    struct W {
        camp: Vec<(u32, u32, u64)>, // [day][campaign]: credited sessions, purchasing, revenue
        asst: Vec<u32>,      // [day][campaign]: assisted sessions
        cnt: Vec<u32>,
        pos: Vec<u32>,
        active: Vec<u32>,
        catsum: Vec<i64>,
        touched: Vec<u16>,
        rows: Vec<UserRow>,
        ra: Vec<Ev>,
        rb: Vec<Ev>,
    }
    let ncampo = ncamp + 1;
    // biggest partitions first (whale people make some ~20x the mean), so no thread
    // is left alone with a big one at the end
    let mut order: Vec<usize> = (0..NB).collect();
    order.sort_by_key(|&b| std::cmp::Reverse(bstart[b + 1] - bstart[b]));
    let ud = out.join("user_daily");
    std::fs::create_dir_all(&ud).unwrap();
    let cats_s: Vec<String> = cats.iter().map(|c| String::from_utf8_lossy(c).into_owned()).collect();
    let p3: Vec<W> = workers_fin(
        nthreads,
        NB,
        || W {
            camp: vec![(0, 0, 0); ncampo * ndays],
            asst: vec![0; ncampo * ndays],
            cnt: vec![0; ncampo],
            pos: vec![0; ncampo],
            active: Vec::new(),
            catsum: vec![0; ncat],
            touched: Vec::new(),
            rows: Vec::new(),
            ra: Vec::new(),
            rb: Vec::new(),
        },
        |w, t| {
            let b = order[t];
            // Group by person with time order inside each person: dense person ids in
            // first-seen order, then a stable LSD radix sort on ts (3 x 11 bits) and a
            // final stable pass by person id. All histograms come from one pre-pass.
            let mine: &[Ev] = &all[bstart[b]..bstart[b + 1]];
            let n: usize = mine.len();
            let mut pm: KMap<u32> = KMap::with_capacity_and_hasher(n / 8 + 16, Default::default());
            let mut persons: Vec<u64> = Vec::new();
            let mut eoff: Vec<usize> = Vec::new();
            let mut pidv: Vec<u32> = Vec::with_capacity(n);
            let mut h0 = vec![0usize; 2048];
            let mut h1 = vec![0usize; 2048];
            let mut h2 = vec![0usize; 1024];
            for e in mine {
                let id = *pm.entry(e.person).or_insert_with(|| {
                    persons.push(e.person);
                    eoff.push(0);
                    (persons.len() - 1) as u32
                });
                eoff[id as usize] += 1;
                pidv.push(id);
                h0[(e.ts & 2047) as usize] += 1;
                h1[((e.ts >> 11) & 2047) as usize] += 1;
                h2[(e.ts >> 22) as usize] += 1;
            }
            let np = persons.len();
            fn excl(h: &mut [usize]) -> usize {
                let mut acc = 0;
                for x in h.iter_mut() {
                    let c = *x;
                    *x = acc;
                    acc += c;
                }
                acc
            }
            excl(&mut h0);
            excl(&mut h1);
            excl(&mut h2);
            let total = excl(&mut eoff);
            eoff.push(total);
            let mut ra = std::mem::take(&mut w.ra);
            let mut rb = std::mem::take(&mut w.rb);
            ra.clear();
            rb.clear();
            ra.reserve(n);
            rb.reserve(n);
            {
                let pa = ra.as_mut_ptr();
                let pb = rb.as_mut_ptr();
                for (k, e) in mine.iter().enumerate() {
                    let mut e = *e;
                    e.person = pidv[k] as u64;
                    let d = (e.ts & 2047) as usize;
                    unsafe { pa.add(h0[d]).write(e) };
                    h0[d] += 1;
                }
                for k in 0..n {
                    let e = unsafe { *pa.add(k) };
                    let d = ((e.ts >> 11) & 2047) as usize;
                    unsafe { pb.add(h1[d]).write(e) };
                    h1[d] += 1;
                }
                for k in 0..n {
                    let e = unsafe { *pb.add(k) };
                    let d = (e.ts >> 22) as usize;
                    unsafe { pa.add(h2[d]).write(e) };
                    h2[d] += 1;
                }
                let mut fill = eoff.clone();
                for k in 0..n {
                    let e = unsafe { *pa.add(k) };
                    let d = e.person as usize;
                    unsafe { pb.add(fill[d]).write(e) };
                    fill[d] += 1;
                }
                unsafe { rb.set_len(n) };
            }
            drop(pidv);
            let ev: &[Ev] = &rb;

            let mine = std::mem::take(&mut *clicks_by[b].lock().unwrap());
            let mut coff = vec![0usize; np + 1];
            let mut cpid: Vec<u32> = Vec::with_capacity(mine.iter().map(|o| o.len()).sum());
            for o in &mine {
                for c in o {
                    let id = pm.get(&c.person).map_or(u32::MAX, |&x| x);
                    if id != u32::MAX {
                        coff[id as usize] += 1;
                    }
                    cpid.push(id);
                }
            }
            let mut acc = 0;
            for x in coff.iter_mut() {
                let c = *x;
                *x = acc;
                acc += c;
            }
            let mut cl: Vec<Click> = Vec::with_capacity(acc);
            {
                let mut fill = coff.clone();
                let dst = cl.as_mut_ptr();
                let mut k = 0;
                for o in &mine {
                    for c in o {
                        let id = cpid[k];
                        if id != u32::MAX {
                            unsafe { dst.add(fill[id as usize]).write(*c) };
                            fill[id as usize] += 1;
                        }
                        k += 1;
                    }
                }
                unsafe { cl.set_len(acc) };
            }
            drop(mine);
            // sort each person's clicks by (ts, click_id desc) and drop duplicate click_ids
            let mut wpos = 0;
            for pi in 0..np {
                let (s0, s1) = (coff[pi], coff[pi + 1]);
                coff[pi] = wpos;
                let r = &mut cl[s0..s1];
                r.sort_unstable_by(|x, y| x.ts.cmp(&y.ts).then_with(|| cmp_str(y.click, x.click)));
                let mut last: Option<u64> = None;
                for k in s0..s1 {
                    let c = cl[k];
                    if last == Some(c.click) {
                        continue;
                    }
                    last = Some(c.click);
                    cl[wpos] = c;
                    wpos += 1;
                }
            }
            coff[np] = wpos;

            for pi in 0..np {
                let person = persons[pi];
                let (i, ee) = (eoff[pi], eoff[pi + 1]);
                let (cs, ce) = (coff[pi], coff[pi + 1]);
                let (mut lo, mut hi) = (cs, cs);

                let mut cur_day = i64::MIN;
                let (mut ds, mut dev, mut drev) = (0i64, 0i64, 0i64);
                macro_rules! flush {
                    () => {
                        if ds > 0 {
                            let mut top = -1i32;
                            let mut topv = 0i64;
                            for &c in &w.touched {
                                let v = w.catsum[c as usize];
                                if v > topv || (v == topv && (c as i32) < top) {
                                    topv = v;
                                    top = c as i32;
                                }
                                w.catsum[c as usize] = 0;
                            }
                            w.touched.clear();
                            w.rows.push(UserRow { person, day: cur_day as i32, sessions: ds, events: dev, revenue: drev, top });
                            ds = 0;
                            dev = 0;
                            drev = 0;
                        }
                    };
                }
                let mut j = i;
                while j < ee {
                    let start = ws + ev[j].ts as i64;
                    let d = start.div_euclid(DAY);
                    if d != cur_day {
                        flush!();
                        cur_day = d;
                    }
                    let mut last = start;
                    let mut cnt = 0i64;
                    let mut rev = 0i64;
                    let mut purch = false;
                    while j < ee && ws + ev[j].ts as i64 - last <= GAP {
                        let e = &ev[j];
                        last = ws + e.ts as i64;
                        cnt += 1;
                        if e.purchase() {
                            purch = true;
                            rev += e.price() as i64;
                            let c = e.cat();
                            if w.catsum[c] == 0 {
                                w.touched.push(c as u16);
                            }
                            w.catsum[c] += e.price() as i64;
                        }
                        j += 1;
                    }
                    ds += 1;
                    dev += cnt;
                    drev += rev;

                    while hi < ce && cl[hi].ts as i64 <= start - cws {
                        let c = cl[hi].camp as usize;
                        if w.cnt[c] == 0 {
                            w.pos[c] = w.active.len() as u32;
                            w.active.push(c as u32);
                        }
                        w.cnt[c] += 1;
                        hi += 1;
                    }
                    while lo < hi && (cl[lo].ts as i64) < start - cws - WEEK {
                        let c = cl[lo].camp as usize;
                        w.cnt[c] -= 1;
                        if w.cnt[c] == 0 {
                            let p = w.pos[c] as usize;
                            let last_c = *w.active.last().unwrap();
                            w.active[p] = last_c;
                            w.pos[last_c as usize] = p as u32;
                            w.active.pop();
                        }
                        lo += 1;
                    }
                    let credited = if hi > lo { cl[hi - 1].camp } else { organic } as usize;
                    let di = (d - ws_day) as usize;
                    let a = &mut w.camp[di * ncampo + credited];
                    // per worker: at most one count per session, far below u32::MAX
                    a.0 += 1;
                    a.1 += purch as u32;
                    a.2 += rev as u64;
                    let row = &mut w.asst[di * ncampo..(di + 1) * ncampo];
                    for &c in &w.active {
                        row[c as usize] += 1;
                    }
                }
                flush!();
                for k in lo..hi {
                    w.cnt[cl[k].camp as usize] = 0;
                }
                w.active.clear();
            }
            w.ra = ra;
            w.rb = rb;
        },
        |w, ti| {
            // write this worker's user_daily rows while others finish their partitions
            if !w.rows.is_empty() || ti == 0 {
                write_user_file(&ud.join(format!("part-{ti}.parquet")), &w.rows, &cats_s);
            }
            w.rows = Vec::new();
        },
    );
    lap("sessions done");

    // outputs
    let parts: Vec<(Vec<[i64; 4]>, Vec<UserRow>)> = p3
        .into_iter()
        .map(|w| {
            let mut c = vec![[0i64; 4]; ncampo * ndays];
            for d in 0..ndays {
                for k in 0..ncampo {
                    let a = w.camp[d * ncampo + k];
                    c[k * ndays + d] = [a.0 as i64, a.1 as i64, a.2 as i64, w.asst[d * ncampo + k] as i64];
                }
            }
            (c, w.rows)
        })
        .collect();
    write_outputs(&out, parts, &cats, &camp_ids, &camp_ch, ncampo, ndays, ws_day, nthreads);
    lap("written");
    // skip freeing everything
    std::process::exit(0);
}

#[allow(clippy::too_many_arguments)]
fn write_outputs(
    out: &Path,
    parts: Vec<(Vec<[i64; 4]>, Vec<UserRow>)>,
    _cats: &[Vec<u8>],
    camp_ids: &[Vec<u8>],
    camp_ch: &[Vec<u8>],
    ncampo: usize,
    ndays: usize,
    ws_day: i64,
    nthreads: usize,
) {
    use arrow_array::builder::{Date32Builder, Int64Builder, StringBuilder};
    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;

    let ud = out.join("user_daily");
    let cd = out.join("campaign_daily");
    std::fs::create_dir_all(&ud).unwrap();
    std::fs::create_dir_all(&cd).unwrap();
    let props = || {
        WriterProperties::builder()
            .set_compression(Compression::UNCOMPRESSED)
            .set_dictionary_enabled(false)
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build()
    };
    let s = |b: &[u8]| String::from_utf8_lossy(b).into_owned();

    let mut camp = vec![[0i64; 4]; ncampo * ndays];
    let mut all_rows: Vec<Vec<UserRow>> = Vec::new();
    for (c, r) in parts {
        for (a, b) in camp.iter_mut().zip(c.iter()) {
            for k in 0..4 {
                a[k] += b[k];
            }
        }
        all_rows.push(r);
    }

    {
        let mut cid = StringBuilder::new();
        let mut ch = StringBuilder::new();
        let mut day = Date32Builder::new();
        let mut cols: [Int64Builder; 4] = std::array::from_fn(|_| Int64Builder::new());
        for c in 0..ncampo {
            for d in 0..ndays {
                let a = camp[c * ndays + d];
                if a[0] > 0 || a[3] > 0 {
                    cid.append_value(s(&camp_ids[c]));
                    ch.append_value(s(&camp_ch[c]));
                    day.append_value((ws_day + d as i64) as i32);
                    for k in 0..4 {
                        cols[k].append_value(a[k]);
                    }
                }
            }
        }
        let [mut a0, mut a1, mut a2, mut a3] = cols;
        let batch = RecordBatch::try_from_iter(vec![
            ("campaign_id", Arc::new(cid.finish()) as ArrayRef),
            ("channel", Arc::new(ch.finish()) as ArrayRef),
            ("day", Arc::new(day.finish()) as ArrayRef),
            ("sessions", Arc::new(a0.finish()) as ArrayRef),
            ("purchasing_sessions", Arc::new(a1.finish()) as ArrayRef),
            ("revenue_cents", Arc::new(a2.finish()) as ArrayRef),
            ("assisted_sessions", Arc::new(a3.finish()) as ArrayRef),
        ])
        .unwrap();
        let f = File::create(cd.join("part-0.parquet")).unwrap();
        let mut wr = ArrowWriter::try_new(f, batch.schema(), Some(props())).unwrap();
        wr.write(&batch).unwrap();
        wr.close().unwrap();
    }

    drop(all_rows);
}
