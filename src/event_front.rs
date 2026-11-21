//! Single-pass guarded AVX2+PCLMULQDQ frontends for events (96B, 7 fields) and
//! clicks (64B, 4 fields).
//!
//! Scans each row directly into bitmasks (commas, newlines, quotes) without
//! intermediate batch offset arrays. Every fast-path assumption is checked before
//! emitting; any other physical row delegates to exact `csv::scan` so semantics
//! are 100% identical on all inputs.

use crate::csv;
use crate::key::{EKey, Key8};
use crate::links::SEED_A;
use crate::ts::{parse_price, DateCache};
use crate::SEED_ID;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event<'a> {
    pub ek: EKey,
    pub eh: u64,
    pub uid_k: Key8,
    pub uh: u64,
    pub product: &'a [u8],
    pub timestamp: Option<i64>,
    pub ing_day_ms: i64,
    pub ing_bcd: u32,
    pub kind: u8, // 0 invalid; 1 view/cart; 2 purchase
    pub price: u32,
}

pub trait EventSink<'a> {
    #[inline(always)]
    fn prefetch_keys(&self, _eh: u64, _uh: u64) {}
    fn on_event(&mut self, ev: Event<'a>);
}

impl<'a, F: FnMut(Event<'a>)> EventSink<'a> for F {
    #[inline(always)]
    fn on_event(&mut self, ev: Event<'a>) {
        self(ev);
    }
}

pub trait ClickSink<'a> {
    fn on_click(&mut self, fields: &[&'a [u8]; 4]);
}

impl<'a, F: FnMut(&[&'a [u8]; 4])> ClickSink<'a> for F {
    #[inline(always)]
    fn on_click(&mut self, fields: &[&'a [u8]; 4]) {
        self(fields);
    }
}

pub trait LinkSink<'a> {
    fn on_link(&mut self, a: &'a [u8], b: &'a [u8]);
}

impl<'a, F: FnMut(&'a [u8], &'a [u8])> LinkSink<'a> for F {
    #[inline(always)]
    fn on_link(&mut self, a: &'a [u8], b: &'a [u8]) {
        self(a, b);
    }
}

#[cold]
#[inline(never)]
fn fallback<'a, S: EventSink<'a>>(data: &'a [u8], dates: &DateCache, f: &mut S) {
    csv::scan::<7, _>(data, |fields| {
        let kind = match fields[4] {
            b"view" | b"cart" => 1,
            b"purchase" => 2,
            _ => 0,
        };
        let (timestamp, ing_day_ms, ing_bcd) = dates.parse_event_ingest_bcd(fields[2], fields[3]);
        let ek = EKey::new(fields[0]);
        let eh = ek.hash(SEED_ID);
        let uid_k = Key8::new(fields[1]);
        let uh = uid_k.hash(SEED_A);
        f.on_event(Event {
            ek,
            eh,
            uid_k,
            uh,
            product: fields[5],
            timestamp,
            ing_day_ms,
            ing_bcd,
            kind,
            price: if kind == 2 { parse_price(fields[6]) } else { 0 },
        });
    });
}

#[inline(always)]
fn scalar_masks96(data: &[u8; 96]) -> (u64, u32, u64, u32, u64, u32) {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    #[inline(always)]
    fn eq_mask(word: u64, value: u64) -> u64 {
        let y = word ^ value;
        let high = !(((y & LO7).wrapping_add(LO7)) | y | LO7);
        (high >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
    }
    let (mut c_lo, mut l_lo, mut q_lo) = (0u64, 0u64, 0u64);
    let (mut c_hi, mut l_hi, mut q_hi) = (0u32, 0u32, 0u32);
    let p = data.as_ptr();
    for i in 0..8 {
        let w = unsafe { (p.add(i * 8) as *const u64).read_unaligned() };
        c_lo |= eq_mask(w, 0x2c2c_2c2c_2c2c_2c2c) << (i * 8);
        l_lo |= eq_mask(w, 0x0a0a_0a0a_0a0a_0a0a) << (i * 8);
        q_lo |= eq_mask(w, 0x2222_2222_2222_2222) << (i * 8);
    }
    for i in 0..4 {
        let w = unsafe { (p.add(64 + i * 8) as *const u64).read_unaligned() };
        c_hi |= (eq_mask(w, 0x2c2c_2c2c_2c2c_2c2c) as u32) << (i * 8);
        l_hi |= (eq_mask(w, 0x0a0a_0a0a_0a0a_0a0a) as u32) << (i * 8);
        q_hi |= (eq_mask(w, 0x2222_2222_2222_2222) as u32) << (i * 8);
    }
    (c_lo, c_hi, l_lo, l_hi, q_lo, q_hi)
}

#[inline(always)]
fn parse_row_tail<'a, S: EventSink<'a>>(
    data: &'a [u8; 96],
    end: usize,
    c0: usize,
    c1: usize,
    c2: usize,
    c5: usize,
    q_lo: u64,
    dates: &DateCache,
    sink: &S,
) -> Option<Event<'a>> {
    let p = data.as_ptr();
    unsafe {
        let tq = ((q_lo >> (c1 + 1)) & 1) as usize;
        let ta = c1 + 1 + tq;
        let tlen = (c2 - c1 - 1) - (tq << 1);
        let iq = ((q_lo >> (c2 + 1)) & 1) as usize;
        let ia = c2 + 1 + iq;
        let c3 = ia + 20 + iq;
        if c3 >= end || *p.add(c3) != b',' {
            return None;
        }
        let q0 = (q_lo & 1) as usize;
        let q1 = ((q_lo >> (c0 + 1)) & 1) as usize;
        let eid = data.get_unchecked(q0..c0 - q0);
        let uid = data.get_unchecked(c0 + 1 + q1..c1 - q1);
        let ek = EKey::new(eid);
        let eh = ek.hash(SEED_ID);
        let uid_k = Key8::new(uid);
        let uh = uid_k.hash(SEED_A);
        sink.prefetch_keys(eh, uh);

        let (timestamp, ing_day_ms, ing_bcd) = if tlen == 20 {
            dates.parse_event_ingest_bcd(data.get_unchecked(ta..ta + 20), data.get_unchecked(ia..ia + 20))
        } else if tlen == 13 {
            let (id, ib) = dates.parse_f1_ingest_bcd(data.get_unchecked(ia..ia + 20));
            (DateCache::parse_ts_13(data.get_unchecked(ta..ta + 13)), id, ib)
        } else if tlen == 25 {
            dates.parse_f2_ingest_bcd(data.get_unchecked(ta..ta + 25), data.get_unchecked(ia..ia + 20))
        } else {
            return None;
        };
        let kq = (*p.add(c3 + 1) == b'"') as usize;
        let ka = c3 + 1 + kq;
        let k4 = (p.add(ka) as *const u32).read_unaligned();
        let is_vc = ((k4 == u32::from_ne_bytes(*b"view")) | (k4 == u32::from_ne_bytes(*b"cart")))
            & (*p.add(ka + 4 + kq) == b',');
        let (kind, c4) = if is_vc {
            (1u8, ka + 4 + kq)
        } else if (p.add(ka) as *const u64).read_unaligned() == u64::from_ne_bytes(*b"purchase")
            && *p.add(ka + 8 + kq) == b','
        {
            (2u8, ka + 8 + kq)
        } else {
            return None;
        };
        if c4 >= c5 {
            return None;
        }
        let pq = (*p.add(c4 + 1) == b'"') as usize;
        let price = if kind == 2 {
            let prq = (*p.add(c5 + 1) == b'"') as usize;
            parse_price(data.get_unchecked(c5 + 1 + prq..end - prq))
        } else {
            0
        };
        Some(Event {
            ek,
            eh,
            uid_k,
            uh,
            product: data.get_unchecked(c4 + 1 + pq..c5 - pq),
            timestamp,
            ing_day_ms,
            ing_bcd,
            kind,
            price,
        })
    }
}

#[inline(always)]
fn parse_row_hi<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8; 96],
    end_hi: usize,
    c_lo: u64,
    cm2: u32,
    q_lo: u64,
    qt2: u32,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit_hi = 1u32 << end_hi;
    let below_hi = end_bit_hi - 1;
    let c_hi = cm2 & below_hi;
    if c_lo.count_ones() + c_hi.count_ones() != 6 {
        return None;
    }
    let q_hi = qt2 & below_hi;
    let p_lo = parity64(q_lo);
    let p_hi_raw = parity64(q_hi as u64) as u32;
    let c0 = c_lo.trailing_zeros() as usize;
    let low1 = c_lo & c_lo.wrapping_sub(1);
    let c1 = low1.trailing_zeros() as usize;
    if c1 > 32 {
        return None;
    }
    let low2 = low1 & low1.wrapping_sub(1);
    let c2 = low2.trailing_zeros() as usize;
    if c2 >= 63 {
        return None;
    }
    let seps_lo = c_lo;
    let seps_hi = c_hi | end_bit_hi;
    let edges_lo = (seps_lo << 1) | 1 | (seps_lo >> 1) | ((seps_hi as u64) << 63);
    let edges_hi = (seps_hi << 1) | ((seps_lo >> 63) as u32) | (seps_hi >> 1);
    let p_hi = p_hi_raw ^ 0u32.wrapping_sub((p_lo >> 63) as u32);
    if ((q_lo & !edges_lo) | (p_lo & seps_lo)) != 0 || ((q_hi & !edges_hi) | (p_hi & seps_hi)) != 0 {
        return None;
    }
    let c5 = if c_hi != 0 {
        64 + (31 - c_hi.leading_zeros() as usize)
    } else {
        63 - c_lo.leading_zeros() as usize
    };
    parse_row_tail(data, 64 + end_hi, c0, c1, c2, c5, q_lo, dates, sink)
}

#[inline(always)]
fn parse_row_lo<'a, S: EventSink<'a>, P: Fn(u64) -> u64>(
    data: &'a [u8; 96],
    end: usize,
    c_lo: u64,
    q_lo: u64,
    dates: &DateCache,
    parity64: &P,
    sink: &S,
) -> Option<Event<'a>> {
    let end_bit = 1u64 << end;
    let below = end_bit - 1;
    let c = c_lo & below;
    if c.count_ones() != 6 {
        return None;
    }
    let c0 = c.trailing_zeros() as usize;
    let low1 = c & c.wrapping_sub(1);
    let c1 = low1.trailing_zeros() as usize;
    if c1 > 32 {
        return None;
    }
    let low2 = low1 & low1.wrapping_sub(1);
    let c2 = low2.trailing_zeros() as usize;
    if c2 >= 63 {
        return None;
    }
    let q = q_lo & below;
    let seps = c | end_bit;
    let edges = (seps << 1) | 1 | (seps >> 1);
    if ((q & !edges) | (parity64(q) & seps)) != 0 {
        return None;
    }
    let c5 = 63 - c.leading_zeros() as usize;
    parse_row_tail(data, end, c0, c1, c2, c5, q, dates, sink)
}

#[inline(always)]
fn scan_with<'a, S: EventSink<'a>, M: Fn(&[u8; 96]) -> (u64, u32, u64, u32, u64, u32), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    dates: &DateCache,
    mut f: S,
    masks: M,
    parity64: P,
) {
    while data.len() >= 96 {
        let bytes: &'a [u8; 96] = unsafe { &*data.as_ptr().cast::<[u8; 96]>() };
        let (c_lo, cm2, l_lo, nl2, q_lo, qt2) = masks(bytes);
        if l_lo == 0 {
            if nl2 != 0 {
                let end_hi = nl2.trailing_zeros() as usize;
                let end = 64 + end_hi;
                if let Some(ev) = parse_row_hi(bytes, end_hi, c_lo, cm2, q_lo, qt2, dates, &parity64, &f) {
                    f.on_event(ev);
                } else {
                    fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
                }
                data = unsafe { data.get_unchecked(end + 1..) };
            } else {
                let end = data[96..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(data.len(), |p| 96 + p + 1);
                fallback(&data[..end], dates, &mut f);
                data = &data[end..];
            }
        } else {
            let end = l_lo.trailing_zeros() as usize;
            if let Some(ev) = parse_row_lo(bytes, end, c_lo, q_lo, dates, &parity64, &f) {
                f.on_event(ev);
            } else {
                fallback(unsafe { data.get_unchecked(..end + 1) }, dates, &mut f);
            }
            data = unsafe { data.get_unchecked(end + 1..) };
        }
    }
    if !data.is_empty() {
        fallback(data, dates, &mut f);
    }
}

#[inline(always)]
pub fn scan<'a, S: EventSink<'a>>(data: &'a [u8], dates: &DateCache, f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_avx2(data, dates, f) };
        return;
    }
    scan_with(data, dates, f, scalar_masks96, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn handle_row_lo_avx2<'a, S: EventSink<'a>>(
    data: &'a [u8],
    bytes: &'a [u8; 96],
    c_lo: u64,
    q_lo: u64,
    dates: &DateCache,
    f: &mut S,
) -> usize {
    use core::arch::x86_64::*;
    let p = bytes.as_ptr();
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let b0 = _mm256_loadu_si256(p.cast());
    let b1 = _mm256_loadu_si256(p.add(32).cast());
    let l_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
        | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
    let end = l_lo.trailing_zeros() as usize;
    let v_ones = _mm_set1_epi64x(-1);
    let parity64 = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    if let Some(ev) = parse_row_lo(bytes, end, c_lo, q_lo, dates, &parity64, f) {
        f.on_event(ev);
    } else {
        fallback(data.get_unchecked(..end + 1), dates, f);
    }
    end + 1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_avx2<'a, S: EventSink<'a>>(mut data: &'a [u8], dates: &DateCache, mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity64 = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    while data.len() >= 96 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(512) as *const i8, _MM_HINT_T0);
        _mm_prefetch(p.add(576) as *const i8, _MM_HINT_T0);
        let bytes: &'a [u8; 96] = &*p.cast::<[u8; 96]>();
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let b2 = _mm256_loadu_si256(p.add(64).cast());
        let c_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let cm2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_cm)) as u32;
        let l_lo_any = _mm256_movemask_epi8(_mm256_or_si256(
            _mm256_cmpeq_epi8(b0, v_nl),
            _mm256_cmpeq_epi8(b1, v_nl),
        ));
        let nl2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_nl)) as u32;
        let q_lo = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        let qt2 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(b2, v_qt)) as u32;
        if l_lo_any == 0 {
            if nl2 != 0 {
                let end_hi = nl2.trailing_zeros() as usize;
                let end = 64 + end_hi;
                if let Some(ev) = parse_row_hi(bytes, end_hi, c_lo, cm2, q_lo, qt2, dates, &parity64, &f) {
                    f.on_event(ev);
                } else {
                    fallback(data.get_unchecked(..end + 1), dates, &mut f);
                }
                data = data.get_unchecked(end + 1..);
            } else {
                let end = data[96..]
                    .iter()
                    .position(|&b| b == b'\n')
                    .map_or(data.len(), |pos| 96 + pos + 1);
                fallback(&data[..end], dates, &mut f);
                data = &data[end..];
            }
        } else {
            let adv = handle_row_lo_avx2(data, bytes, c_lo, q_lo, dates, &mut f);
            data = data.get_unchecked(adv..);
        }
    }
    if !data.is_empty() {
        fallback(data, dates, &mut f);
    }
}

// ---------------------------------------------------------------------------
// Single-pass 64B u64 AVX2+PCLMULQDQ click frontend (4 fields, zero Tok arrays)

#[cold]
#[inline(never)]
fn fallback_clicks<'a, S: ClickSink<'a>>(data: &'a [u8], f: &mut S) {
    csv::scan::<4, _>(data, |fields| f.on_click(fields));
}

#[inline(always)]
fn prefix_xor64(mut q: u64) -> u64 {
    q ^= q << 1;
    q ^= q << 2;
    q ^= q << 4;
    q ^= q << 8;
    q ^= q << 16;
    q ^= q << 32;
    q
}

#[inline(always)]
fn scalar_masks64(data: &[u8; 64]) -> (u64, u64, u64) {
    const LO7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
    #[inline(always)]
    fn eq_mask(word: u64, value: u64) -> u64 {
        let y = word ^ value;
        let high = !(((y & LO7).wrapping_add(LO7)) | y | LO7);
        (high >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
    }
    let (mut c, mut l, mut q) = (0u64, 0u64, 0u64);
    let p = data.as_ptr();
    for i in 0..8 {
        let w = unsafe { (p.add(i * 8) as *const u64).read_unaligned() };
        c |= eq_mask(w, 0x2c2c_2c2c_2c2c_2c2c) << (i * 8);
        l |= eq_mask(w, 0x0a0a_0a0a_0a0a_0a0a) << (i * 8);
        q |= eq_mask(w, 0x2222_2222_2222_2222) << (i * 8);
    }
    (c, l, q)
}

#[inline(always)]
fn parse_click_row<'a, P: Fn(u64) -> u64>(
    data: &'a [u8; 64],
    end: usize,
    commas: u64,
    quotes: u64,
    parity: &P,
) -> Option<[&'a [u8]; 4]> {
    let below = (1u64 << end) - 1;
    let c = commas & below;
    if c.count_ones() != 3 {
        return None;
    }
    let q = quotes & below;
    let seps = c | (1u64 << end);
    let edges = (seps << 1) | 1 | (seps >> 1);
    if ((q & !edges) | (parity(q) & seps)) != 0 {
        return None;
    }
    let c0 = c.trailing_zeros() as usize;
    let c_1 = c & c.wrapping_sub(1);
    let c1 = c_1.trailing_zeros() as usize;
    let c2 = (c_1 & c_1.wrapping_sub(1)).trailing_zeros() as usize;
    let q0 = (q & 1) as usize;
    let q1 = ((q >> (c0 + 1)) & 1) as usize;
    let q2 = ((q >> (c1 + 1)) & 1) as usize;
    let q3 = ((q >> (c2 + 1)) & 1) as usize;
    unsafe {
        Some([
            data.get_unchecked(q0..c0 - q0),
            data.get_unchecked(c0 + 1 + q1..c1 - q1),
            data.get_unchecked(c1 + 1 + q2..c2 - q2),
            data.get_unchecked(c2 + 1 + q3..end - q3),
        ])
    }
}

#[inline(always)]
fn scan_clicks_with<'a, S: ClickSink<'a>, M: Fn(&[u8; 64]) -> (u64, u64, u64), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    mut f: S,
    masks: M,
    parity: P,
) {
    while data.len() >= 64 {
        let bytes: &'a [u8; 64] = unsafe { &*data.as_ptr().cast::<[u8; 64]>() };
        let (commas, lines, quotes) = masks(bytes);
        let end = lines.trailing_zeros() as usize;
        if end < 64 {
            if let Some(fields) = parse_click_row(bytes, end, commas, quotes, &parity) {
                f.on_click(&fields);
            } else {
                fallback_clicks(unsafe { data.get_unchecked(..end + 1) }, &mut f);
            }
            data = unsafe { data.get_unchecked(end + 1..) };
        } else {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_clicks(&data[..end], &mut f);
            data = &data[end..];
        }
    }
    if !data.is_empty() {
        fallback_clicks(data, &mut f);
    }
}

#[inline(always)]
pub fn scan_clicks<'a, S: ClickSink<'a>>(data: &'a [u8], f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_clicks_avx2(data, f) };
        return;
    }
    scan_clicks_with(data, f, scalar_masks64, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_clicks_avx2<'a, S: ClickSink<'a>>(mut data: &'a [u8], mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    while data.len() >= 64 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(384) as *const i8, _MM_HINT_T0);
        let bytes: &'a [u8; 64] = &*p.cast::<[u8; 64]>();
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let commas = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let lines = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
        let quotes = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        let end = lines.trailing_zeros() as usize;
        if end < 64 {
            if let Some(fields) = parse_click_row(bytes, end, commas, quotes, &parity) {
                f.on_click(&fields);
            } else {
                fallback_clicks(data.get_unchecked(..end + 1), &mut f);
            }
            data = data.get_unchecked(end + 1..);
        } else {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_clicks(&data[..end], &mut f);
            data = &data[end..];
        }
    }
    if !data.is_empty() {
        fallback_clicks(data, &mut f);
    }
}

// ---------------------------------------------------------------------------
// Multi-row-per-64B-load AVX2+PCLMULQDQ link frontend (2 fields, ~3 rows/load)

#[cold]
#[inline(never)]
fn fallback_links<'a, S: LinkSink<'a>>(data: &'a [u8], f: &mut S) {
    csv::scan::<2, _>(data, |fields| f.on_link(fields[0], fields[1]));
}

#[inline(always)]
fn scan_links_with<'a, S: LinkSink<'a>, M: Fn(&[u8; 64]) -> (u64, u64, u64), P: Fn(u64) -> u64>(
    mut data: &'a [u8],
    mut f: S,
    masks: M,
    parity: P,
) {
    while data.len() >= 64 {
        let bytes: &'a [u8; 64] = unsafe { &*data.as_ptr().cast::<[u8; 64]>() };
        let (mut commas, mut lines, mut quotes) = masks(bytes);
        if lines == 0 {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_links(&data[..end], &mut f);
            data = &data[end..];
            continue;
        }
        let mut off = 0usize;
        while lines != 0 {
            let end = lines.trailing_zeros() as usize;
            let end_bit = 1u64 << end;
            let below = end_bit - 1;
            let c = commas & below;
            let q = quotes & below;
            let seps = c | end_bit;
            let edges = (seps << 1) | 1 | (seps >> 1);
            if c.count_ones() == 1 && ((q & !edges) | (parity(q) & seps)) == 0 {
                let c0 = c.trailing_zeros() as usize;
                let q0 = (q & 1) as usize;
                let q1 = ((q >> (c0 + 1)) & 1) as usize;
                unsafe {
                    f.on_link(
                        data.get_unchecked(off + q0..off + c0 - q0),
                        data.get_unchecked(off + c0 + 1 + q1..off + end - q1),
                    );
                }
            } else {
                fallback_links(unsafe { data.get_unchecked(off..off + end + 1) }, &mut f);
            }
            let adv = end + 1;
            off += adv;
            if adv >= 64 {
                break;
            }
            commas >>= adv;
            lines >>= adv;
            quotes >>= adv;
        }
        data = unsafe { data.get_unchecked(off..) };
    }
    if !data.is_empty() {
        fallback_links(data, &mut f);
    }
}

#[inline(always)]
pub fn scan_links<'a, S: LinkSink<'a>>(data: &'a [u8], f: S) {
    #[cfg(target_arch = "x86_64")]
    if csv::simd_backend() == "avx2"
        && is_x86_feature_detected!("pclmulqdq")
        && is_x86_feature_detected!("popcnt")
    {
        unsafe { scan_links_avx2(data, f) };
        return;
    }
    scan_links_with(data, f, scalar_masks64, prefix_xor64);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi1,bmi2,popcnt,pclmulqdq")]
unsafe fn scan_links_avx2<'a, S: LinkSink<'a>>(mut data: &'a [u8], mut f: S) {
    use core::arch::x86_64::*;
    let v_cm = _mm256_set1_epi8(b',' as i8);
    let v_nl = _mm256_set1_epi8(b'\n' as i8);
    let v_qt = _mm256_set1_epi8(b'"' as i8);
    let v_ones = _mm_set1_epi64x(-1);
    let parity = |q: u64| -> u64 {
        _mm_cvtsi128_si64(_mm_clmulepi64_si128::<0>(_mm_cvtsi64_si128(q as i64), v_ones)) as u64
    };
    while data.len() >= 64 {
        let p = data.as_ptr();
        _mm_prefetch(p.add(384) as *const i8, _MM_HINT_T0);
        let b0 = _mm256_loadu_si256(p.cast());
        let b1 = _mm256_loadu_si256(p.add(32).cast());
        let mut commas = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_cm)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_cm)) as u32 as u64) << 32);
        let mut lines = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_nl)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_nl)) as u32 as u64) << 32);
        let mut quotes = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(b0, v_qt)) as u32 as u64)
            | ((_mm256_movemask_epi8(_mm256_cmpeq_epi8(b1, v_qt)) as u32 as u64) << 32);
        if lines == 0 {
            let end = data[64..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(data.len(), |pos| 64 + pos + 1);
            fallback_links(&data[..end], &mut f);
            data = &data[end..];
            continue;
        }
        let mut off = 0usize;
        while lines != 0 {
            let end = lines.trailing_zeros() as usize;
            let end_bit = 1u64 << end;
            let below = end_bit - 1;
            let c = commas & below;
            let q = quotes & below;
            let seps = c | end_bit;
            let edges = (seps << 1) | 1 | (seps >> 1);
            if c.count_ones() == 1 && ((q & !edges) | (parity(q) & seps)) == 0 {
                let c0 = c.trailing_zeros() as usize;
                let q0 = (q & 1) as usize;
                let q1 = ((q >> (c0 + 1)) & 1) as usize;
                f.on_link(
                    data.get_unchecked(off + q0..off + c0 - q0),
                    data.get_unchecked(off + c0 + 1 + q1..off + end - q1),
                );
            } else {
                fallback_links(data.get_unchecked(off..off + end + 1), &mut f);
            }
            let adv = end + 1;
            off += adv;
            if adv >= 64 {
                break;
            }
            commas >>= adv;
            lines >>= adv;
            quotes >>= adv;
        }
        data = data.get_unchecked(off..);
    }
    if !data.is_empty() {
        fallback_links(data, &mut f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dates() -> DateCache {
        DateCache::new(1_791_417_600_000, 1_792_627_200_000)
    }

    fn expected<'a>(input: &'a [u8], dates: &DateCache) -> Vec<Event<'a>> {
        let mut out = Vec::new();
        csv::scan::<7, _>(input, |f| {
            let kind = match f[4] {
                b"view" | b"cart" => 1,
                b"purchase" => 2,
                _ => 0,
            };
            let (ing_day_ms, ing_bcd) = dates.parse_f1_ingest_bcd(f[3]);
            let ek = EKey::new(f[0]);
            let eh = ek.hash(SEED_ID);
            let uid_k = Key8::new(f[1]);
            let uh = uid_k.hash(SEED_A);
            out.push(Event {
                ek,
                eh,
                uid_k,
                uh,
                product: f[5],
                timestamp: dates.parse_ts(f[2]),
                ing_day_ms,
                ing_bcd,
                kind,
                price: if kind == 2 { parse_price(f[6]) } else { 0 },
            });
        });
        out
    }

    fn check(input: &[u8]) {
        let d = dates();
        let want = expected(input, &d);
        let mut got = Vec::new();
        scan_with(input, &d, |e| got.push(e), scalar_masks96, prefix_xor64);
        assert_eq!(got, want, "scalar input={input:?}");
        got.clear();
        scan(input, &d, |e| got.push(e));
        assert_eq!(got, want, "selected input={input:?}");
    }

    #[test]
    fn bit_masks_every_lane_and_byte() {
        for lane in 0..96 {
            for byte in 0..=255u8 {
                let mut block = [0u8; 96];
                block[lane] = byte;
                let (c_lo, c_hi, l_lo, l_hi, q_lo, q_hi) = scalar_masks96(&block);
                let got = (
                    (c_lo as u128) | ((c_hi as u128) << 64),
                    (l_lo as u128) | ((l_hi as u128) << 64),
                    (q_lo as u128) | ((q_hi as u128) << 64),
                );
                let want = (
                    if byte == b',' { 1u128 << lane } else { 0 },
                    if byte == b'\n' { 1u128 << lane } else { 0 },
                    if byte == b'"' { 1u128 << lane } else { 0 },
                );
                assert_eq!(got, want);
            }
        }
        for lane in 0..64 {
            assert_eq!(prefix_xor64(1u64 << lane), u64::MAX << lane);
        }
    }

    #[test]
    fn timestamps_quotes_and_truncations() {
        for ts in [
            "2026-10-08T12:34:56Z",
            "2026-10-08T12:34:56-04:00",
            "1791462896000",
            "0001791462896000",
            "86400000",
            "2026-02-30T12:34:56Z",
            "2026-10-08T24:00:00Z",
            "2026-10-08T12:34:56+24:00",
            "2026-10-08T12:34:56Z\r",
            "999999999999999999999999",
            "",
        ] {
            for qmask in 0..128 {
                let vals = ["e1", "u1", ts, "2026-10-08T13:00:00Z", "purchase", "p1", "00250"];
                let row = vals
                    .iter()
                    .enumerate()
                    .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                    .collect::<Vec<_>>()
                    .join(",")
                    + "\n";
                let input = row.repeat(3).into_bytes();
                check(&input);
                for end in 0..row.len() {
                    check(&input[..row.len() + end]);
                }
            }
        }
    }

    #[test]
    fn mutations_offsets_long_ids_and_seams() {
        let valid = b"e1,u1,2026-10-08T12:34:56Z,2026-10-08T13:00:00Z,purchase,p1,250\n";
        for offset in 0..64 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    check(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 33, 63, 64, 65, 95, 96, 97, 128, 16_385] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(&valid[2..]);
            input.extend_from_slice(valid);
            check(&input);
        }
    }

    #[test]
    fn clicks_quotes_mutations_and_long_lines() {
        let check_clicks = |input: &[u8]| {
            let mut want: Vec<[Vec<u8>; 4]> = Vec::new();
            csv::scan_byte_oracle::<4, _>(input, |f| {
                want.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]);
            });
            let mut got: Vec<[Vec<u8>; 4]> = Vec::new();
            scan_clicks_with(
                input,
                |f: &[&[u8]; 4]| got.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]),
                scalar_masks64,
                prefix_xor64,
            );
            assert_eq!(got, want, "scalar clicks input={input:?}");
            got.clear();
            scan_clicks(input, |f: &[&[u8]; 4]| got.push([f[0].to_vec(), f[1].to_vec(), f[2].to_vec(), f[3].to_vec()]));
            assert_eq!(got, want, "selected clicks input={input:?}");
        };
        for ts in ["2026-10-08T12:34:56Z", "2026-10-08T12:34:56-04:00", "1791462896000", ""] {
            for qmask in 0..16 {
                let vals = ["c12345678", "u12345678", "c_1958", ts];
                let row = vals
                    .iter()
                    .enumerate()
                    .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                    .collect::<Vec<_>>()
                    .join(",")
                    + "\n";
                let input = row.repeat(3).into_bytes();
                check_clicks(&input);
                for end in 0..row.len() {
                    check_clicks(&input[..row.len() + end]);
                }
            }
        }
        let valid = b"c12345678,u12345678,c_1234,2026-10-08T12:34:56Z\n";
        for offset in 0..32 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    check_clicks(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 63, 64, 65, 128] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(b",u1,c_1,2026-10-08T12:34:56Z\n");
            input.extend_from_slice(valid);
            check_clicks(&input);
        }
    }

    #[test]
    fn links_quotes_mutations_and_long_lines() {
        let check_links = |input: &[u8]| {
            let mut want: Vec<[Vec<u8>; 2]> = Vec::new();
            csv::scan_byte_oracle::<2, _>(input, |f| {
                want.push([f[0].to_vec(), f[1].to_vec()]);
            });
            let mut got: Vec<[Vec<u8>; 2]> = Vec::new();
            scan_links_with(
                input,
                |a: &[u8], b: &[u8]| got.push([a.to_vec(), b.to_vec()]),
                scalar_masks64,
                prefix_xor64,
            );
            assert_eq!(got, want, "scalar links input={input:?}");
            got.clear();
            scan_links(input, |a: &[u8], b: &[u8]| got.push([a.to_vec(), b.to_vec()]));
            assert_eq!(got, want, "selected links input={input:?}");
        };
        for qmask in 0..4 {
            let vals = ["u12345678", "u87654321"];
            let row = vals
                .iter()
                .enumerate()
                .map(|(i, s)| if qmask & (1 << i) != 0 { format!("\"{s}\"") } else { (*s).to_owned() })
                .collect::<Vec<_>>()
                .join(",")
                + "\n";
            let input = row.repeat(6).into_bytes();
            check_links(&input);
            for end in 0..row.len() {
                check_links(&input[..row.len() * 3 + end]);
            }
        }
        let valid = b"u12345678,u87654321\n";
        for offset in 0..32 {
            for pos in 0..valid.len() {
                for byte in [0, b'"', b',', b'\n', b'\r', b' ', 255] {
                    let mut input = vec![b'!'; offset];
                    input.extend_from_slice(valid);
                    input[offset + pos] = byte;
                    input.extend_from_slice(valid);
                    input.extend_from_slice(valid);
                    input.extend_from_slice(valid);
                    check_links(&input[offset..]);
                }
            }
        }
        for size in [0, 31, 32, 62, 63, 64, 65, 128] {
            let mut input = vec![b'x'; size];
            input.extend_from_slice(b",u1\n");
            input.extend_from_slice(valid);
            input.extend_from_slice(valid);
            check_links(&input);
        }
    }
}
