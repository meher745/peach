//! Timestamp and purchase-price parsing for the three accepted forms (F1, F2, F3).
//!
//! Uses a 620-byte L1-resident compile-time calendar table (zero divisions/modulos
//! for years 1970..=2100, with exact fallback outside that range), single-branch
//! bitwise ASCII digit validation, and an unrolled 13-digit F3 fast path.

pub const DAY_MS: i64 = 86_400_000;
pub const INGEST_OOB_BASE: u32 = 0xF000_0000;
pub const INGEST_EXACT_I64: u32 = u32::MAX - 1;
pub const INGEST_NONE: u32 = u32::MAX;

#[inline(always)]
unsafe fn bcd_hms(p: *const u8) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    {
        let w = u64::from_be((p as *const u64).read_unaligned());
        core::arch::x86_64::_pext_u64(w, 0x0F0F_000F_0F00_0F0F) as u32
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    {
        let h0 = (*p & 0x0F) as u32;
        let h1 = (*p.add(1) & 0x0F) as u32;
        let m0 = (*p.add(3) & 0x0F) as u32;
        let m1 = (*p.add(4) & 0x0F) as u32;
        let s0 = (*p.add(6) & 0x0F) as u32;
        let s1 = (*p.add(7) & 0x0F) as u32;
        (h0 << 20) | (h1 << 16) | (m0 << 12) | (m1 << 8) | (s0 << 4) | s1
    }
}

#[inline(always)]
pub fn bcd_hms_to_ms(bcd: u32) -> i64 {
    let h = ((bcd >> 20) & 0xF) * 10 + ((bcd >> 16) & 0xF);
    let m = ((bcd >> 12) & 0xF) * 10 + ((bcd >> 8) & 0xF);
    let s = ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
    (h as i64 * 3600 + m as i64 * 60 + s as i64) * 1000
}

const MIN_LUT_YEAR: u32 = 1970;
const MAX_LUT_YEAR: u32 = 2100;
const LUT_YEARS: usize = (MAX_LUT_YEAR - MIN_LUT_YEAR + 1) as usize;

#[inline(always)]
const fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[inline(always)]
pub const fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

const DAYS_BEFORE_YEAR: [i32; LUT_YEARS] = {
    let mut t = [0i32; LUT_YEARS];
    let mut i = 0;
    while i < LUT_YEARS {
        t[i] = days_from_civil((MIN_LUT_YEAR as usize + i) as i64, 1, 1) as i32;
        i += 1;
    }
    t
};

const DAYS_BEFORE_MONTH: [[i32; 12]; 2] = [
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
    [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
];

const MONTH_DAYS: [[u8; 12]; 2] = [
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
    [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
];

/// Returns non-zero in high nibble (0xF0) iff `d >= 10` for any u8 `d`.
#[inline(always)]
fn non_digit_bits(d: u8) -> u8 {
    d | d.wrapping_add(6)
}

#[cold]
fn calendar_out_of_lut(y: u32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Option<i64> {
    if y == 0 {
        return None;
    }
    let leap = is_leap(y) as usize;
    if d > MONTH_DAYS[leap][(mo - 1) as usize] as u32 {
        return None;
    }
    let days = days_from_civil(y as i64, mo, d);
    Some(days * DAY_MS + (h as i64 * 3600 + mi as i64 * 60 + s as i64) * 1000)
}

/// Parses `YYYY-MM-DDTHH:MM:SS` (first 19 bytes; caller ensures `b.len() >= 19`).
#[inline(always)]
fn calendar(b: &[u8]) -> Option<i64> {
    let p = b.as_ptr();
    unsafe {
        if ((*p.add(4) ^ b'-')
            | (*p.add(7) ^ b'-')
            | (*p.add(10) ^ b'T')
            | (*p.add(13) ^ b':')
            | (*p.add(16) ^ b':'))
            != 0
        {
            return None;
        }
        let y0 = (*p).wrapping_sub(b'0');
        let y1 = (*p.add(1)).wrapping_sub(b'0');
        let y2 = (*p.add(2)).wrapping_sub(b'0');
        let y3 = (*p.add(3)).wrapping_sub(b'0');
        let m0 = (*p.add(5)).wrapping_sub(b'0');
        let m1 = (*p.add(6)).wrapping_sub(b'0');
        let d0 = (*p.add(8)).wrapping_sub(b'0');
        let d1 = (*p.add(9)).wrapping_sub(b'0');
        let h0 = (*p.add(11)).wrapping_sub(b'0');
        let h1 = (*p.add(12)).wrapping_sub(b'0');
        let n0 = (*p.add(14)).wrapping_sub(b'0');
        let n1 = (*p.add(15)).wrapping_sub(b'0');
        let s0 = (*p.add(17)).wrapping_sub(b'0');
        let s1 = (*p.add(18)).wrapping_sub(b'0');

        let bad_digits = non_digit_bits(y0)
            | non_digit_bits(y1)
            | non_digit_bits(y2)
            | non_digit_bits(y3)
            | non_digit_bits(m0)
            | non_digit_bits(m1)
            | non_digit_bits(d0)
            | non_digit_bits(d1)
            | non_digit_bits(h0)
            | non_digit_bits(h1)
            | non_digit_bits(n0)
            | non_digit_bits(n1)
            | non_digit_bits(s0)
            | non_digit_bits(s1);
        if (bad_digits & 0xF0) != 0 {
            return None;
        }

        let y = y0 as u32 * 1000 + y1 as u32 * 100 + y2 as u32 * 10 + y3 as u32;
        let mo = m0 as u32 * 10 + m1 as u32;
        let d = d0 as u32 * 10 + d1 as u32;
        let h = h0 as u32 * 10 + h1 as u32;
        let mi = n0 as u32 * 10 + n1 as u32;
        let s = s0 as u32 * 10 + s1 as u32;

        if mo == 0 || mo > 12 || d == 0 || h > 23 || mi > 59 || s > 59 {
            return None;
        }
        let yi = y.wrapping_sub(MIN_LUT_YEAR);
        if yi < LUT_YEARS as u32 {
            let leap = ((y & 3 == 0) && y != 2100) as usize;
            let mi_idx = (mo - 1) as usize;
            if d > *MONTH_DAYS.get_unchecked(leap).get_unchecked(mi_idx) as u32 {
                return None;
            }
            let days = (*DAYS_BEFORE_YEAR.get_unchecked(yi as usize)
                + *DAYS_BEFORE_MONTH.get_unchecked(leap).get_unchecked(mi_idx)
                + (d - 1) as i32) as i64;
            Some(days * DAY_MS + (h as i64 * 3600 + mi as i64 * 60 + s as i64) * 1000)
        } else {
            calendar_out_of_lut(y, mo, d, h, mi, s)
        }
    }
}

/// Ultra-fast path for `ingest_ts`, which rules.md guarantees is always valid F1
/// (`YYYY-MM-DDTHH:MM:SSZ`). Falls back to strict `parse_ts` if out of LUT year range.
#[inline(always)]
pub fn parse_f1_fast(b: &[u8]) -> Option<i64> {
    if b.len() == 20 {
        let p = b.as_ptr();
        unsafe {
            let y = (*p).wrapping_sub(b'0') as u32 * 1000
                + (*p.add(1)).wrapping_sub(b'0') as u32 * 100
                + (*p.add(2)).wrapping_sub(b'0') as u32 * 10
                + (*p.add(3)).wrapping_sub(b'0') as u32;
            let yi = y.wrapping_sub(MIN_LUT_YEAR);
            let mo = (*p.add(5)).wrapping_sub(b'0') as u32 * 10 + (*p.add(6)).wrapping_sub(b'0') as u32;
            let d = (*p.add(8)).wrapping_sub(b'0') as u32 * 10 + (*p.add(9)).wrapping_sub(b'0') as u32;
            if yi < LUT_YEARS as u32 && mo >= 1 && mo <= 12 && d >= 1 && d <= 31 {
                let h = (*p.add(11)).wrapping_sub(b'0') as i64 * 10 + (*p.add(12)).wrapping_sub(b'0') as i64;
                let mi = (*p.add(14)).wrapping_sub(b'0') as i64 * 10 + (*p.add(15)).wrapping_sub(b'0') as i64;
                let s = (*p.add(17)).wrapping_sub(b'0') as i64 * 10 + (*p.add(18)).wrapping_sub(b'0') as i64;
                let leap = ((y & 3 == 0) && y != 2100) as usize;
                let days = (*DAYS_BEFORE_YEAR.get_unchecked(yi as usize)
                    + *DAYS_BEFORE_MONTH.get_unchecked(leap).get_unchecked((mo - 1) as usize)
                    + (d - 1) as i32) as i64;
                return Some(days * DAY_MS + (h * 3600 + mi * 60 + s) * 1000);
            }
        }
    }
    parse_ts(b)
}

/// Direct-mapped 64-slot L1 date-prefix cache indexed directly by raw ASCII `DD` bits
/// `((dd & 3) << 4) | ((dd >> 8) & 15)` (zero multiply/range checks). Empty slots store
/// an impossible sentinel `dd` whose slot index never equals its own slot, so a single
/// `(pref ^ slot.prefix) | (dd ^ slot.dd)` check validates both occupancy and exact date.
#[derive(Clone, Copy)]
#[repr(C)]
struct DateSlot {
    prefix: u64, // little-endian 8 bytes "YYYY-MM-"
    day_ms: i64,
    day_tag: u32,
    dd: u16, // little-endian 2 bytes "DD"
    _pad: u16,
}

#[inline(always)]
fn dd_slot(dd: u16) -> usize {
    (((dd & 3) << 4) | ((dd >> 8) & 15)) as usize
}

pub struct DateCache {
    slots: [DateSlot; 64],
    pub base_day_ms: i64,
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (y + i64::from(m <= 2), m as u32, d as u32)
}

impl DateCache {
    pub fn new(ws: i64, we: i64) -> DateCache {
        let mut slots = [DateSlot { prefix: 0, day_ms: 0, day_tag: 0, dd: 0xFFFF, _pad: 0 }; 64];
        slots[63].dd = 0x0000;
        let d_start = (ws - 8 * DAY_MS).div_euclid(DAY_MS);
        let d_end = (we + 6 * DAY_MS).div_euclid(DAY_MS);
        let count = (d_end - d_start + 1).clamp(0, 28);
        for off in 0..count {
            let day = d_start + off;
            let (y, m, d) = civil_from_days(day);
            if y >= 1 && y <= 9999 && d >= 1 && d <= 31 {
                let s = format!("{y:04}-{m:02}-{d:02}");
                let sb = s.as_bytes();
                let prefix = u64::from_le_bytes(sb[0..8].try_into().unwrap());
                let dd = u16::from_le_bytes([sb[8], sb[9]]);
                slots[dd_slot(dd)] =
                    DateSlot { prefix, day_ms: day * DAY_MS, day_tag: (off as u32) << 24, dd, _pad: 0 };
            }
        }
        DateCache { slots, base_day_ms: d_start * DAY_MS }
    }

    #[inline(always)]
    fn cached_day(&self, b: &[u8]) -> Option<i64> {
        let p = b.as_ptr();
        unsafe {
            let pref = (p as *const u64).read_unaligned();
            let dd = (p.add(8) as *const u16).read_unaligned();
            let slot = *self.slots.get_unchecked(dd_slot(dd));
            if ((pref ^ slot.prefix) | ((dd ^ slot.dd) as u64)) == 0 {
                return Some(slot.day_ms);
            }
        }
        None
    }

    #[inline(always)]
    fn clock_strict(p: *const u8) -> Option<i64> {
        unsafe {
            let w = (p as *const u64).read_unaligned();
            let x = w ^ u64::from_le_bytes(*b"00:00:00");
            let bad = (x & 0xF0F0_FFF0_F0FF_F0F0)
                | (x.wrapping_add(0x0606_0006_0600_0606) & 0xF0F0_00F0_F000_F0F0);
            if bad == 0 {
                let hi = x & 0x000F_0000_0F00_000F;
                let lo = (x >> 8) & 0x000F_0000_0F00_000F;
                let hms = hi * 10 + lo;
                // Branchless SWAR range check: h <= 23 (h+104 < 128), mi <= 59 (mi+68 < 128), s <= 59 (s+68 < 128).
                if ((hms + 0x0044_0000_4400_0068) & 0x0080_0000_8000_0080) == 0 {
                    let h = (hms & 0xFF) as i64;
                    let mi = ((hms >> 24) & 0xFF) as i64;
                    let s = ((hms >> 48) & 0xFF) as i64;
                    return Some((h * 3600 + mi * 60 + s) * 1000);
                }
            }
        }
        None
    }

    #[inline(always)]
    fn calendar_cached(&self, b: &[u8]) -> Option<i64> {
        if let Some(day_ms) = self.cached_day(b) {
            let p = b.as_ptr();
            if unsafe { *p.add(10) } != b'T' {
                return None;
            }
            return Self::clock_strict(unsafe { p.add(11) }).map(|ms| day_ms + ms);
        }
        calendar(b)
    }

    #[cold]
    #[inline(never)]
    fn parse_event_ingest_bcd_slow(&self, event: &[u8], ingest: &[u8]) -> (Option<i64>, i64, u32) {
        let (ing_day_ms, ing_bcd) = self.parse_f1_ingest_bcd(ingest);
        (self.parse_ts(event), ing_day_ms, ing_bcd)
    }

    #[inline(always)]
    pub fn parse_event_ingest_bcd(&self, event: &[u8], ingest: &[u8]) -> (Option<i64>, i64, u32) {
        if event.len() == 20 && ingest.len() == 20 {
            let ep = event.as_ptr();
            let ip = ingest.as_ptr();
            unsafe {
                let ep_pref = (ep as *const u64).read_unaligned();
                let ep_dd = (ep.add(8) as *const u16).read_unaligned();
                if ((*ep.add(19) ^ b'Z') | (*ip.add(19) ^ b'Z')) == 0
                    && ep_pref == (ip as *const u64).read_unaligned()
                    && ep_dd == (ip.add(8) as *const u16).read_unaligned()
                {
                    let slot = *self.slots.get_unchecked(dd_slot(ep_dd));
                    if ((ep_pref ^ slot.prefix) | ((ep_dd ^ slot.dd) as u64)) == 0 {
                        let ev_ms = if *ep.add(10) == b'T' {
                            Self::clock_strict(ep.add(11)).map(|ms| slot.day_ms + ms)
                        } else {
                            None
                        };
                        let ing_bcd = slot.day_tag | bcd_hms(ip.add(11));
                        return (ev_ms, slot.day_ms, ing_bcd);
                    }
                }
            }
        }
        self.parse_event_ingest_bcd_slow(event, ingest)
    }

    #[cold]
    #[inline(never)]
    fn parse_f2_ingest_bcd_slow(&self, event: &[u8], ingest: &[u8]) -> (Option<i64>, i64, u32) {
        let (ing_day_ms, ing_bcd) = self.parse_f1_ingest_bcd(ingest);
        (self.parse_ts_25(event), ing_day_ms, ing_bcd)
    }

    #[inline(always)]
    pub fn parse_f2_ingest_bcd(&self, event: &[u8], ingest: &[u8]) -> (Option<i64>, i64, u32) {
        if event.len() == 25 && ingest.len() == 20 {
            let ep = event.as_ptr();
            let ip = ingest.as_ptr();
            unsafe {
                let sign = *ep.add(19);
                if *ep.add(22) == b':' && (sign == b'+' || sign == b'-') && *ip.add(19) == b'Z' {
                    let ep_pref = (ep as *const u64).read_unaligned();
                    let ep_dd = (ep.add(8) as *const u16).read_unaligned();
                    if ep_pref == (ip as *const u64).read_unaligned()
                        && ep_dd == (ip.add(8) as *const u16).read_unaligned()
                    {
                        let slot = *self.slots.get_unchecked(dd_slot(ep_dd));
                        if ((ep_pref ^ slot.prefix) | ((ep_dd ^ slot.dd) as u64)) == 0 {
                            let ing_bcd = slot.day_tag | bcd_hms(ip.add(11));
                            let ev_ms = if *ep.add(10) == b'T' {
                                Self::clock_strict(ep.add(11)).and_then(|ms| {
                                    let h0 = (*ep.add(20)).wrapping_sub(b'0');
                                    let h1 = (*ep.add(21)).wrapping_sub(b'0');
                                    let m0 = (*ep.add(23)).wrapping_sub(b'0');
                                    let m1 = (*ep.add(24)).wrapping_sub(b'0');
                                    if ((non_digit_bits(h0)
                                        | non_digit_bits(h1)
                                        | non_digit_bits(m0)
                                        | non_digit_bits(m1))
                                        & 0xF0)
                                        != 0
                                    {
                                        return None;
                                    }
                                    let oh = h0 as u32 * 10 + h1 as u32;
                                    let om = m0 as u32 * 10 + m1 as u32;
                                    if oh > 23 || om > 59 {
                                        return None;
                                    }
                                    let off = (oh as i64 * 60 + om as i64) * 60_000;
                                    let base = slot.day_ms + ms;
                                    Some(if sign == b'+' { base - off } else { base + off })
                                })
                            } else {
                                None
                            };
                            return (ev_ms, slot.day_ms, ing_bcd);
                        }
                    }
                }
            }
        }
        self.parse_f2_ingest_bcd_slow(event, ingest)
    }

    #[inline(always)]
    pub fn parse_f1_ingest_bcd(&self, b: &[u8]) -> (i64, u32) {
        if b.len() == 20 {
            let p = b.as_ptr();
            unsafe {
                if *p.add(19) == b'Z' {
                    let pref = (p as *const u64).read_unaligned();
                    let dd = (p.add(8) as *const u16).read_unaligned();
                    let slot = *self.slots.get_unchecked(dd_slot(dd));
                    if pref == slot.prefix && dd == slot.dd {
                        return (slot.day_ms, slot.day_tag | bcd_hms(p.add(11)));
                    }
                }
            }
        }
        Self::parse_f1_ingest_slow(b)
    }

    #[cold]
    #[inline(never)]
    fn parse_f1_ingest_slow(b: &[u8]) -> (i64, u32) {
        match parse_f1_fast(b) {
            Some(ms) => (ms, INGEST_EXACT_I64),
            None => (0, INGEST_NONE),
        }
    }

    /// Parse a known 13-byte F3 timestamp.
    #[inline(always)]
    pub fn parse_ts_13(b: &[u8]) -> Option<i64> {
        parse_f3_13(b)
    }

    /// Parse a known 25-byte F2 timestamp.
    #[inline(always)]
    pub fn parse_ts_25(&self, b: &[u8]) -> Option<i64> {
        let p = b.as_ptr();
        unsafe {
            let sign = *p.add(19);
            if *p.add(22) == b':' && (sign == b'+' || sign == b'-') {
                let base = self.calendar_cached(b)?;
                let h0 = (*p.add(20)).wrapping_sub(b'0');
                let h1 = (*p.add(21)).wrapping_sub(b'0');
                let m0 = (*p.add(23)).wrapping_sub(b'0');
                let m1 = (*p.add(24)).wrapping_sub(b'0');
                if ((non_digit_bits(h0) | non_digit_bits(h1) | non_digit_bits(m0) | non_digit_bits(m1)) & 0xF0) != 0 {
                    return None;
                }
                let oh = h0 as u32 * 10 + h1 as u32;
                let om = m0 as u32 * 10 + m1 as u32;
                if oh > 23 || om > 59 {
                    return None;
                }
                let off = (oh as i64 * 60 + om as i64) * 60_000;
                return Some(if sign == b'+' { base - off } else { base + off });
            }
        }
        parse_f3_general(b)
    }

    #[inline(always)]
    pub fn parse_ts(&self, b: &[u8]) -> Option<i64> {
        let n = b.len();
        if n == 20 && unsafe { *b.get_unchecked(19) } == b'Z' {
            return self.calendar_cached(b);
        }
        if n == 13 {
            return parse_f3_13(b);
        }
        if n == 25 {
            return self.parse_ts_25(b);
        }
        parse_f3_general(b)
    }
}

#[inline(always)]
pub(crate) fn swar_dec8(x: u64) -> u64 {
    let a = (x & 0x00FF_00FF_00FF_00FF) * 10 + ((x >> 8) & 0x00FF_00FF_00FF_00FF);
    let b = (a & 0x0000_FFFF_0000_FFFF) * 100 + ((a >> 16) & 0x0000_FFFF_0000_FFFF);
    (b & 0xFFFF_FFFF) * 10_000 + (b >> 32)
}

#[inline(always)]
fn parse_f3_13(b: &[u8]) -> Option<i64> {
    let p = b.as_ptr();
    unsafe {
        let x0 = (p as *const u64).read_unaligned() ^ 0x3030_3030_3030_3030;
        let x1 = (p.add(5) as *const u64).read_unaligned() ^ 0x3030_3030_3030_3030;
        let bad = ((x0 | x1) & 0xF0F0_F0F0_F0F0_F0F0)
            | ((x0.wrapping_add(0x0606_0606_0606_0606) | x1.wrapping_add(0x0606_0606_0606_0606))
                & 0xF0F0_F0F0_F0F0_F0F0);
        if bad != 0 {
            return None;
        }
        let hi8 = swar_dec8(x0);
        let lo5 = swar_dec8(x1 & 0xFFFF_FFFF_FF00_0000);
        Some((hi8 * 100_000 + lo5) as i64)
    }
}

/// Returns epoch ms, or None for a bad timestamp. A digits-only value too large
/// for i64 is returned as i64::MAX (valid form, outside every window).
#[inline(always)]
pub fn parse_ts(b: &[u8]) -> Option<i64> {
    let n = b.len();
    if n == 20 && unsafe { *b.get_unchecked(19) } == b'Z' {
        return calendar(b);
    }
    if n == 13 {
        return parse_f3_13(b);
    }
    if n == 25 {
        let p = b.as_ptr();
        unsafe {
            let sign = *p.add(19);
            if *p.add(22) == b':' && (sign == b'+' || sign == b'-') {
                let base = calendar(b)?;
                let h0 = (*p.add(20)).wrapping_sub(b'0');
                let h1 = (*p.add(21)).wrapping_sub(b'0');
                let m0 = (*p.add(23)).wrapping_sub(b'0');
                let m1 = (*p.add(24)).wrapping_sub(b'0');
                if ((non_digit_bits(h0) | non_digit_bits(h1) | non_digit_bits(m0) | non_digit_bits(m1)) & 0xF0) != 0 {
                    return None;
                }
                let oh = h0 as u32 * 10 + h1 as u32;
                let om = m0 as u32 * 10 + m1 as u32;
                if oh > 23 || om > 59 {
                    return None;
                }
                let off = (oh as i64 * 60 + om as i64) * 60_000;
                return Some(if sign == b'+' { base - off } else { base + off });
            }
        }
    }
    parse_f3_general(b)
}

#[cold]
fn parse_f3_general(b: &[u8]) -> Option<i64> {
    if b.is_empty() {
        return None;
    }
    let mut v: u64 = 0;
    let mut sig = 0usize;
    for &c in b {
        let x = c.wrapping_sub(b'0');
        if x >= 10 {
            return None;
        }
        if sig > 0 || x != 0 {
            sig += 1;
            if sig <= 18 {
                v = v * 10 + x as u64;
            }
        }
    }
    if sig > 18 {
        return Some(i64::MAX);
    }
    Some(v as i64)
}

/// Purchase price: digits only, value 1..=10_000_000. Returns 0 when invalid.
#[inline(always)]
pub fn parse_price(b: &[u8]) -> u32 {
    let n = b.len();
    if n == 0 {
        return 0;
    }
    if n <= 8 && b[0] != b'0' {
        let mut v: u32 = 0;
        for &c in b {
            let x = c.wrapping_sub(b'0');
            if x >= 10 {
                return 0;
            }
            v = v * 10 + x as u32;
        }
        return if v >= 1 && v <= 10_000_000 { v } else { 0 };
    }
    parse_price_slow(b)
}

#[cold]
fn parse_price_slow(b: &[u8]) -> u32 {
    let mut v: u64 = 0;
    let mut sig = 0usize;
    for &c in b {
        let x = c.wrapping_sub(b'0');
        if x >= 10 {
            return 0;
        }
        if sig > 0 || x != 0 {
            sig += 1;
            if sig > 9 {
                return 0;
            }
            v = v * 10 + x as u64;
        }
    }
    if (1..=10_000_000).contains(&v) {
        v as u32
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swar_hms_and_f3_reject_all_non_digits() {
        let cache = DateCache::new(0, 14 * DAY_MS);
        // Test every byte 0..=255 at every position of HH:MM:SS
        for pos in [11usize, 12, 13, 14, 15, 16, 17, 18] {
            for b in 0u8..=255 {
                let mut s = *b"1970-01-02T12:34:56Z";
                s[pos] = b;
                let got = cache.parse_ts(&s);
                let exp = parse_ts(&s);
                assert_eq!(got, exp, "hms pos={pos} b={b:#x}");
            }
        }
        // Test every byte 0..=255 at every position of 13-digit F3
        for pos in 0..13 {
            for b in 0u8..=255 {
                let mut s = *b"1700000000123";
                s[pos] = b;
                let got = parse_f3_13(&s);
                let is_dig = b.is_ascii_digit();
                assert_eq!(got.is_some(), is_dig, "f3 pos={pos} b={b:#x}");
                if is_dig {
                    let expected: i64 = std::str::from_utf8(&s).unwrap().parse().unwrap();
                    assert_eq!(got, Some(expected), "f3 val pos={pos} b={b:#x}");
                }
            }
        }
    }

    #[test]
    fn empty_date_slots_reject_nul_prefixes() {
        // Unpopulated slots have prefix 0; eight NUL bytes must not match them.
        let cache = DateCache::new(0, 14 * DAY_MS);
        for day in 0..=31 {
            for zone in ["Z", "+00:00"] {
                let text = format!("{}{day:02}T00:00:00{zone}", "\0".repeat(8));
                assert_eq!(cache.parse_ts(text.as_bytes()), None, "day={day} zone={zone}");
            }
        }
        assert_eq!(cache.parse_ts(b"1970-01-01T00:00:00Z"), Some(0));
    }

    #[test]
    fn forms_and_lut_edges() {
        // Exhaustively verify non_digit_bits for all 256 u8 values
        for x in 0u8..=255 {
            assert_eq!((non_digit_bits(x) & 0xF0) == 0, x < 10, "x={x}");
        }
        let f1 = parse_ts(b"2026-10-03T23:58:10Z").unwrap();
        assert_eq!(f1, 1_791_071_890_000);
        assert_eq!(parse_f1_fast(b"2026-10-03T23:58:10Z"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(f1));
        assert_eq!(parse_ts(b"1791071890000"), Some(f1));
        assert_eq!(parse_ts(b"0001791071890000"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-03T23:58:10-00:00"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-03T23:58:10z"), None);
        assert_eq!(parse_ts(b"2026-10-03t23:58:10Z"), None);
        assert_eq!(parse_ts(b"2026-10-03 23:58:10Z"), None);
        assert_eq!(parse_ts(b"2026-02-29T00:00:00Z"), None);
        assert_eq!(parse_ts(b"2027-02-29T00:00:00Z"), None);
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"2100-02-29T00:00:00Z"), None);
        assert!(parse_ts(b"2000-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"0000-01-01T00:00:00Z"), None);
        assert!(parse_ts(b"0001-01-01T00:00:00Z").is_some());
        assert!(parse_ts(b"9999-12-31T23:59:59Z").is_some());
        assert_eq!(parse_ts(b"1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_ts(b"2026-10-03T24:00:00Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:60Z"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10+24:00"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10+05:60"), None);
        assert_eq!(parse_ts(b"2026-10-03T23:58:10+0200"), None);
        assert_eq!(parse_ts(b"1791071890000.0"), None);
        assert_eq!(parse_ts(b"-5"), None);
        assert_eq!(parse_ts(b""), None);
        assert_eq!(parse_ts(b"9223372036854775807"), Some(i64::MAX));
        assert_eq!(parse_ts(b"9223372036854775808"), Some(i64::MAX));
        assert_eq!(parse_ts(b"99999999999999999999999"), Some(i64::MAX));
        assert_eq!(parse_ts(b"86400000"), Some(86_400_000));
        assert_eq!(parse_price(b"0250"), 250);
        assert_eq!(parse_price(b"10000000"), 10_000_000);
        assert_eq!(parse_price(b"10000001"), 0);
        assert_eq!(parse_price(b"0"), 0);
        assert_eq!(parse_price(b"000000000000000000001"), 1);
        assert_eq!(parse_price(b"1000\r"), 0);
    }
}
