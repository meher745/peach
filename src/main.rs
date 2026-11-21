//! Shopping Sessions pipeline.
//!
//! Stages: init (manifest, products, campaigns, mmaps) -> links (exact Key8
//! interning, lock-free union-find, LinkTable + Bloom) -> clicks (parse, validate,
//! resolve person, scatter to person shards) -> events (parse, validate, scatter a
//! compact RawEv to event_id shards) -> dedup (per event_id shard: exact dedup,
//! resolve person, scatter PEv to person shards) -> solve (per person shard:
//! sessionize, attribute, assist, aggregate) -> write Parquet.
//!
//! All shard data lives in pooled 2 MB blocks (`arena`), written through
//! write-combining staging lines, and is returned to the pool as soon as each
//! shard is consumed; the pool is released in parallel before exit.

mod arena;
mod csv;
mod event_front;
mod fixed;
#[allow(dead_code)]
mod hash;
mod key;
mod links;
mod parquet;
#[allow(dead_code)]
mod timebins;
mod ts;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::Instant;

use arena::{BlockCarver, HugeBuf, Scatter, Segs, ShardSet};
use key::{CKey, EKey, Key8};
use links::{prefetch, LinkTable, SEED_A};
use parquet::{Column, Kind, UserDailyBuf};
use ts::{bcd_hms_to_ms, parse_ts, DateCache, DAY_MS, INGEST_NONE, INGEST_OOB_BASE};

const GAP_MS: i64 = 1_800_000;
const SEVEN_DAYS_MS: i64 = 7 * DAY_MS;
const LATE_MS: i64 = DAY_MS;

pub(crate) const SEED_ID: u64 = 0x4b33_a62e_d433_d4a3;
const SEED_MAP: u64 = 0x8bb8_4b93_962e_acc9;

const MAX_ROWS_PER_FILE: usize = 2_000_000;

pub(crate) struct SyncPtr(pub *mut u64);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

// ---------------------------------------------------------------------------
// mmap

struct Mmap {
    ptr: *const u8,
    len: usize,
}

impl Mmap {
    fn open(path: &Path) -> std::io::Result<Mmap> {
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            return Ok(Mmap { ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(), len: 0 });
        }
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_PRIVATE, f.as_raw_fd(), 0)
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Mmap { ptr: p as *const u8, len })
    }
    fn data(&self) -> &'static [u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

fn list_csv(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".csv") && p.is_file() {
                v.push(p);
            }
        }
    }
    v.sort();
    v
}

fn map_dir(dir: &Path) -> Vec<&'static [u8]> {
    list_csv(dir)
        .iter()
        .map(|p| {
            // Mappings live for the whole process (no Drop / munmap).
            Mmap::open(p).unwrap_or_else(|e| panic!("mmap {}: {e}", p.display())).data()
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) struct Chunk {
    pub data: &'static [u8],
}

/// Splits every file's body (after its header line) into newline-aligned chunks.
pub(crate) fn make_chunks(files: &[&'static [u8]], target: usize) -> Vec<Chunk> {
    let mut out = Vec::new();
    for &d in files {
        let start = match d.iter().position(|&c| c == b'\n') {
            Some(i) => i + 1,
            None => continue,
        };
        let mut s = start;
        while s < d.len() {
            let mut e = (s + target).min(d.len());
            if e < d.len() {
                e = match d[e..].iter().position(|&c| c == b'\n') {
                    Some(i) => e + i + 1,
                    None => d.len(),
                };
            }
            out.push(Chunk { data: &d[s..e] });
            s = e;
        }
    }
    // Largest first for better balance.
    out.sort_by_key(|c| std::cmp::Reverse(c.data.len()));
    out
}

pub(crate) fn chunk_target(files: &[&'static [u8]], threads: usize) -> usize {
    let total: usize = files.iter().map(|d| d.len()).sum();
    (total / (threads * 24)).clamp(256 << 10, 8 << 20)
}

pub(crate) fn par<R: Send>(threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = (0..threads).map(|t| s.spawn(move || f(t))).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

#[inline(always)]
pub(crate) fn populate(d: &[u8]) {
    #[cfg(target_os = "linux")]
    unsafe {
        // MADV_POPULATE_READ (Linux 5.14+); harmless EINVAL elsewhere.
        let page = 4096usize;
        let a = (d.as_ptr() as usize) & !(page - 1);
        let e = d.as_ptr() as usize + d.len();
        libc::madvise(a as *mut libc::c_void, e - a, 22);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = d;
}

/// Drops the page-table entries of a fully parsed input chunk (whole pages only),
/// spreading the unmap work over the parse threads instead of process exit. The
/// mapping stays valid: a later access just re-reads the page cache.
#[inline(always)]
pub(crate) fn release_chunk(d: &[u8]) {
    #[cfg(all(target_os = "linux", not(test)))]
    unsafe {
        let page = 4096usize;
        let a = (d.as_ptr() as usize + page - 1) & !(page - 1);
        let e = (d.as_ptr() as usize + d.len()) & !(page - 1);
        if e > a {
            libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_DONTNEED);
        }
    }
    #[cfg(any(not(target_os = "linux"), test))]
    let _ = d;
}

// ---------------------------------------------------------------------------
// inline Key8 string maps (products, campaigns) — zero pointer chase to CSV

struct KeyMap {
    slots: Vec<(Key8, u32)>,
    len: usize,
    mask: usize,
    fast_prefix: &'static [u8],
    fast_pw: u64,
    fast_pmask: u64,
    fast: Vec<u16>,
}

/// Branch-light SWAR twin of `parse_canon_idx` for prefixes of <= 2 bytes (ids <= 8 bytes):
/// one 8-byte load (only when it stays inside the slice's first 4 KB page), masked prefix compare,
/// SWAR digit validation and `swar_dec8` conversion. Returns exactly what `parse_canon_idx` returns.
#[inline(always)]
fn parse_canon_idx_swar(s: &[u8], prefix: &[u8], pw: u64, pmask: u64, limit: usize) -> Option<usize> {
    let plen = prefix.len();
    let n = s.len();
    if n <= plen || n > plen + 6 {
        return None;
    }
    let p = s.as_ptr();
    if plen > 2 || (p as usize & 4095) > 4096 - 8 {
        return parse_canon_idx(s, prefix, limit);
    }
    let w = unsafe { (p as *const u64).read_unaligned() };
    let keep = u64::MAX >> (8 * (8 - n));
    let x = (w ^ 0x3030_3030_3030_3030) & keep & !pmask;
    let bad = ((w ^ pw) & pmask)
        | (x & 0xF0F0_F0F0_F0F0_F0F0)
        | (x.wrapping_add(0x0606_0606_0606_0606 & keep & !pmask) & 0xF0F0_F0F0_F0F0_F0F0);
    // Reject leading zero on multi-digit ids so "p01" / "c_01" always falls back to exact KeyMap.
    let lead0 = ((x >> (8 * plen)) & 0xFF) == 0 && n > plen + 1;
    if bad != 0 || lead0 {
        return None;
    }
    let acc = crate::ts::swar_dec8(x << (8 * (8 - n))) as usize;
    if acc < limit {
        Some(acc)
    } else {
        None
    }
}

#[inline(always)]
fn parse_canon_idx(s: &[u8], prefix: &[u8], limit: usize) -> Option<usize> {
    let plen = prefix.len();
    let n = s.len();
    if n <= plen || n > plen + 6 {
        return None;
    }
    let p = s.as_ptr();
    unsafe {
        for i in 0..plen {
            if *p.add(i) != *prefix.get_unchecked(i) {
                return None;
            }
        }
        let d0 = (*p.add(plen)).wrapping_sub(b'0');
        // Reject leading zero on multi-digit ids so "p01" / "c_01" always falls back to exact KeyMap.
        if d0 > 9 || (d0 == 0 && n > plen + 1) {
            return None;
        }
        let mut acc = d0 as usize;
        let mut i = plen + 1;
        while i < n {
            let d = (*p.add(i)).wrapping_sub(b'0');
            if d > 9 {
                return None;
            }
            acc = acc * 10 + d as usize;
            i += 1;
        }
        if acc < limit {
            Some(acc)
        } else {
            None
        }
    }
}

impl KeyMap {
    fn new(n: usize) -> KeyMap {
        let cap = (n * 2).next_power_of_two().max(16);
        KeyMap {
            slots: vec![(Key8::EMPTY, u32::MAX); cap],
            len: 0,
            mask: cap - 1,
            fast_prefix: b"",
            fast_pw: 0,
            fast_pmask: 0,
            fast: Vec::new(),
        }
    }
    fn new_with_fast(_n: usize, fast_prefix: &'static [u8], fast_cap: usize) -> KeyMap {
        // Canonical IDs with values < u16::MAX live in compact L1/L2 `fast`; `slots` grows lazily otherwise.
        let cap = 16usize;
        KeyMap {
            slots: vec![(Key8::EMPTY, u32::MAX); cap],
            len: 0,
            mask: cap - 1,
            fast_prefix,
            fast_pw: fast_prefix.iter().take(8).enumerate().fold(0u64, |a, (i, &b)| a | ((b as u64) << (8 * i))),
            fast_pmask: if fast_prefix.len() >= 8 { u64::MAX } else { (1u64 << (8 * fast_prefix.len())) - 1 },
            fast: vec![u16::MAX; fast_cap],
        }
    }
    #[cold]
    fn grow_slots(&mut self) {
        let new_cap = (self.slots.len() * 2).max(32);
        let old = std::mem::replace(&mut self.slots, vec![(Key8::EMPTY, u32::MAX); new_cap]);
        self.mask = new_cap - 1;
        for (k, v) in old {
            if v != u32::MAX {
                let mut i = k.hash(SEED_MAP) as usize & self.mask;
                while self.slots[i].1 != u32::MAX {
                    i = (i + 1) & self.mask;
                }
                self.slots[i] = (k, v);
            }
        }
    }
    #[inline(always)]
    fn fast_idx(&self, k: &[u8]) -> Option<usize> {
        parse_canon_idx_swar(k, self.fast_prefix, self.fast_pw, self.fast_pmask, self.fast.len())
    }
    fn insert(&mut self, k: &[u8], v: u32) {
        if !self.fast.is_empty() && v < u16::MAX as u32 {
            if let Some(idx) = parse_canon_idx(k, self.fast_prefix, self.fast.len()) {
                if self.fast[idx] == u16::MAX {
                    self.fast[idx] = v as u16;
                }
                return;
            }
        }
        if self.len * 2 >= self.slots.len() {
            self.grow_slots();
        }
        let key = Key8::new(k);
        let mut i = key.hash(SEED_MAP) as usize & self.mask;
        loop {
            let s = &mut self.slots[i];
            if s.1 == u32::MAX {
                *s = (key, v);
                self.len += 1;
                return;
            }
            if s.0 == key {
                return; // first wins (keys are unique anyway)
            }
            i = (i + 1) & self.mask;
        }
    }
    #[cold]
    #[inline(never)]
    fn get_hash(&self, k: &[u8]) -> Option<u32> {
        let key = Key8::new(k);
        let mut i = key.hash(SEED_MAP) as usize & self.mask;
        loop {
            let s = unsafe { self.slots.get_unchecked(i) };
            if s.1 == u32::MAX {
                return None;
            }
            if s.0 == key {
                return Some(s.1);
            }
            i = (i + 1) & self.mask;
        }
    }
    #[inline(always)]
    fn get(&self, k: &[u8]) -> Option<u32> {
        if !self.fast.is_empty() {
            if let Some(idx) = parse_canon_idx(k, self.fast_prefix, self.fast.len()) {
                let v = unsafe { *self.fast.get_unchecked(idx) };
                if v != u16::MAX {
                    return Some(v as u32);
                }
                if self.len == 0 {
                    return None;
                }
            }
        }
        self.get_hash(k)
    }
    /// `get` with the SWAR single-load fast index (used by the click frontend only, keeping
    /// `scan_avx2`'s purchase-path code layout unchanged).
    #[inline(always)]
    fn get_swar(&self, k: &[u8]) -> Option<u32> {
        if !self.fast.is_empty() {
            if let Some(idx) = self.fast_idx(k) {
                let v = unsafe { *self.fast.get_unchecked(idx) };
                if v != u16::MAX {
                    return Some(v as u32);
                }
                if self.len == 0 {
                    return None;
                }
            }
        }
        self.get_hash(k)
    }
}

fn read_static(path: &Path) -> &'static [u8] {
    match std::fs::read(path) {
        Ok(v) => Box::leak(v.into_boxed_slice()),
        Err(_) => &[],
    }
}

fn body(d: &'static [u8]) -> &'static [u8] {
    match d.iter().position(|&c| c == b'\n') {
        Some(i) => &d[i + 1..],
        None => &[],
    }
}

// ---------------------------------------------------------------------------
// records

/// One 16-byte event_id dedup log record, sharded by event_id.
/// `eid` uses fast exact `EKey`. `ingest` is BCD `(day_idx << 24) | 0xHHMMSS`
/// (< INGEST_OOB_BASE), or `INGEST_OOB_BASE + idx` into `oob_ingest`, or `INGEST_NONE`.
/// `loc` is the exact `arena` locator of the valid `PEv` in `ev_set`, or `LOC_INVALID`
/// for an invalid event (which can still win dedup and suppress a later valid copy).
#[derive(Clone, Copy)]
#[repr(C)]
struct DedupRec {
    eid: EKey,
    ingest: u32,
    loc: u32,
}

#[cold]
#[inline(never)]
fn encode_special_ingest(ing_bcd: u32, exact_ms: i64, oob: &Mutex<Vec<i64>>) -> u32 {
    if ing_bcd == INGEST_NONE {
        return INGEST_NONE;
    }
    let mut g = oob.lock().unwrap();
    let idx = g.len() as u32;
    g.push(exact_ms);
    INGEST_OOB_BASE + idx
}

#[cold]
#[inline(never)]
fn is_late_slow(ing_bcd: u32, ing_day_ms: i64, ts: i64) -> bool {
    let ing_ms = if ing_bcd < INGEST_OOB_BASE {
        ing_day_ms + bcd_hms_to_ms(ing_bcd)
    } else {
        ing_day_ms
    };
    ing_ms.saturating_sub(ts) > LATE_MS
}

#[cold]
#[inline(never)]
fn push_wide_price(wide_ref: &Mutex<Vec<(u32, u16)>>, price: u32, cat: u32) -> u32 {
    let mut g = wide_ref.lock().unwrap();
    g.push((price, cat as u16));
    g.len() as u32
}

#[inline(always)]
fn ingest_less(a: u32, b: u32, base_day_ms: i64, oob: &[i64]) -> bool {
    if (a | b) < INGEST_OOB_BASE {
        a < b
    } else {
        ingest_less_slow(a, b, base_day_ms, oob)
    }
}

#[cold]
#[inline(never)]
fn ingest_less_slow(a: u32, b: u32, base_day_ms: i64, oob: &[i64]) -> bool {
    let dec = |v: u32| -> (u8, i64) {
        if v == INGEST_NONE {
            (1, i64::MAX)
        } else if v >= INGEST_OOB_BASE {
            (0, oob[(v - INGEST_OOB_BASE) as usize])
        } else {
            (0, base_day_ms + ((v >> 24) as i64) * DAY_MS + bcd_hms_to_ms(v))
        }
    };
    dec(a) < dec(b)
}

/// Compact 12-byte non-purchase event (~97% of events), sharded by raw user_id.
#[derive(Clone, Copy)]
#[repr(C, packed(4))]
struct PEv12 {
    person: Key8,
    ts: u32,
}

/// 16-byte purchase event (~3% of events), sharded by raw user_id in a separate ShardSet.
/// `pc` is `price << 8 | category`, or (with >256 categories) `1 + idx` into `wide_store`.
#[derive(Clone, Copy)]
#[repr(C)]
struct PPurch {
    person: Key8,
    ts: u32,
    pc: u32,
}

#[inline(always)]
unsafe fn prefetch_ev_loc(loc: u32) {
    if (loc & arena::LOC_PURCH_BIT) == 0 {
        prefetch(arena::loc_ptr12::<PEv12>(loc) as *const u8);
    } else {
        prefetch(arena::loc_ptr::<PPurch>(loc & !arena::LOC_PURCH_BIT) as *const u8);
    }
}

#[inline(always)]
unsafe fn tombstone_ev_loc(loc: u32) {
    if (loc & arena::LOC_PURCH_BIT) == 0 {
        let pev = arena::loc_ptr12::<PEv12>(loc);
        std::ptr::addr_of_mut!((*pev).person).write_unaligned(Key8::TOMBSTONE);
    } else {
        let pp = arena::loc_ptr::<PPurch>(loc & !arena::LOC_PURCH_BIT);
        std::ptr::addr_of_mut!((*pp).person).write(Key8::TOMBSTONE);
    }
}

/// A valid click, sharded by raw uid / canonical person (24 bytes).
#[derive(Clone, Copy)]
#[repr(C)]
struct Click {
    person: Key8,
    cid: CKey,
    ts: u32,
    camp: u32,
}

/// Compact 8-byte event inside solve packed as `(ts << 32) | pc` for single-register sorting.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
struct Ev8(u64);

impl Ev8 {
    #[inline(always)]
    fn new(ts: u32, pc: u32) -> Ev8 {
        Ev8(((ts as u64) << 32) | pc as u64)
    }
    #[inline(always)]
    fn ts(self) -> u32 {
        (self.0 >> 32) as u32
    }
    #[inline(always)]
    fn pc(self) -> u32 {
        self.0 as u32
    }
}

/// Compact 16-byte click inside solve.
#[derive(Clone, Copy)]
#[repr(C)]
struct Cl24 {
    cid: CKey,
    ts: u32,
    camp: u32,
}

const _: () = assert!(std::mem::size_of::<DedupRec>() == 16);
const _: () = assert!(std::mem::size_of::<PEv12>() == 12);
const _: () = assert!(std::mem::size_of::<PPurch>() == 16);
const _: () = assert!(std::mem::size_of::<Click>() == 24);
const _: () = assert!(std::mem::size_of::<Ev8>() == 8);
const _: () = assert!(std::mem::size_of::<Cl24>() == 16);
const _: () = assert!(arena::SEG / std::mem::size_of::<DedupRec>() <= 1024);

struct Ctx {
    ws: i64,
    base: i64,
    we: i64,
    day0: i64,
    ndays: usize,
    dates: DateCache,
    products: KeyMap,
    unknown_cat: u32,
    cat_names: Vec<&'static [u8]>,
    wide: bool,
    camps: KeyMap,
    camp_ids: Vec<&'static [u8]>,
    camp_channels: Vec<&'static [u8]>,
}

/// Single-cache-line 16-way dedup bucket for shards with `< 65,536` records (100% of 20M & 60M shards):
/// 16 x 16-bit odd tags `(h as u16) | 1` (0 = empty) colocated with 16 x 16-bit sequential record indices
/// in ONE 64-byte cache line; combined with Lemire fast-range `nbuckets` (`248 KB` @ 20M, `368 KB` @ 60M)
/// and precomputed `(bi, tag)` ring for ~61.5% load factor (~4.2% overflow vs 7.4% for 15-way).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DedupBucket16 {
    tags: [u16; 16],
    pos: [u16; 16],
}

const _: () = assert!(std::mem::size_of::<DedupBucket16>() == 64);

#[inline(always)]
unsafe fn bucket16_masks(tags_ptr: *const u16, tag: u16) -> (u32, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let v = _mm256_load_si256(tags_ptr as *const __m256i);
        let eq = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v, _mm256_set1_epi16(tag as i16))) as u32;
        let em = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v, _mm256_setzero_si256())) as u32;
        (eq, em)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (mut eq, mut em) = (0u32, 0u32);
        for i in 0..16 {
            let t = *tags_ptr.add(i);
            eq |= (if t == tag { 3u32 } else { 0 }) << (i * 2);
            em |= (if t == 0 { 3u32 } else { 0 }) << (i * 2);
        }
        (eq, em)
    }
}

#[inline(always)]
fn fast_bucket(h: u64, nbuckets: usize) -> usize {
    // Use bits 47..16 (32 independent entropy bits: disjoint from top ID_BITS shard bits and bottom 16-bit tag!).
    ((((h >> 16) & 0xFFFF_FFFF).wrapping_mul(nbuckets as u64)) >> 32) as usize
}

/// Fallback 8-way 32-bit dedup bucket for oversized shards (`>= 65,536` records).
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct DedupBucket {
    tags: [u32; 8],
    pos: [u32; 8],
}

const DEDUP_BUCKET_EMPTY: DedupBucket = DedupBucket { tags: [0; 8], pos: [0; 8] };
const _: () = assert!(std::mem::size_of::<DedupBucket>() == 64);

#[inline(always)]
unsafe fn bucket_masks(tags_ptr: *const u32, tag: u32) -> (u32, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::*;
        let v = _mm256_load_si256(tags_ptr as *const __m256i);
        let eq = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
            v,
            _mm256_set1_epi32(tag as i32),
        ))) as u32;
        let em = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(
            v,
            _mm256_setzero_si256(),
        ))) as u32;
        (eq, em)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (mut eq, mut em) = (0u32, 0u32);
        for i in 0..8 {
            let t = *tags_ptr.add(i);
            eq |= ((t == tag) as u32) << i;
            em |= ((t == 0) as u32) << i;
        }
        (eq, em)
    }
}

#[cold]
#[inline(never)]
fn dedup_shard_wide(
    segs: &Segs<DedupRec>,
    buckets: &mut Vec<DedupBucket>,
    base: i64,
    oob: &[i64],
) {
    let n = segs.len();
    let nseg = segs.segs.len();
    assert!(nseg < (1 << 22), "event_id shard too large");
    let nbuckets = ((n / 5) + 1).next_power_of_two().max(4);
    if buckets.len() < nbuckets {
        buckets.resize(nbuckets, DEDUP_BUCKET_EMPTY);
    }
    unsafe { std::ptr::write_bytes(buckets.as_mut_ptr(), 0, nbuckets) };
    let bmask = nbuckets - 1;
    let bptr = buckets.as_mut_ptr();
    let seg_meta = segs.segs.as_ptr();

    const PF: usize = 16;
    let mut hring = [0u64; PF];
    for si in 0..nseg {
        let seg = segs.seg(si);
        let slen = seg.len();
        let pre = slen.min(PF);
        for ri in 0..pre {
            let h = unsafe { seg.get_unchecked(ri).eid.hash(SEED_ID) };
            hring[ri] = h;
            prefetch(unsafe { bptr.add(((h >> 32) as usize) & bmask) } as *const u8);
        }
        for ri in 0..slen {
            let h = hring[ri & (PF - 1)];
            if ri + PF < slen {
                let fh = unsafe { seg.get_unchecked(ri + PF).eid.hash(SEED_ID) };
                hring[(ri + PF) & (PF - 1)] = fh;
                prefetch(unsafe { bptr.add(((fh >> 32) as usize) & bmask) } as *const u8);
            }
            let tag = (h as u32) | 1;
            let me = ((si << 10) | ri) as u32;
            let mut bi = ((h >> 32) as usize) & bmask;
            'probe: loop {
                unsafe {
                    let bkt = bptr.add(bi);
                    let (mut eq, em) = bucket_masks((*bkt).tags.as_ptr(), tag);
                    while eq != 0 {
                        let lane = eq.trailing_zeros() as usize;
                        eq &= eq - 1;
                        let oi = *(*bkt).pos.get_unchecked(lane);
                        let os = *seg_meta.add((oi >> 10) as usize);
                        let o = &*(os.ptr as *const DedupRec).add((oi & 1023) as usize);
                        let r = seg.get_unchecked(ri);
                        if o.eid == r.eid {
                            let loser_loc = if ingest_less(r.ingest, o.ingest, base, oob) {
                                *(*bkt).pos.get_unchecked_mut(lane) = me;
                                o.loc
                            } else {
                                r.loc
                            };
                            if loser_loc != arena::LOC_INVALID {
                                tombstone_ev_loc(loser_loc);
                            }
                            break 'probe;
                        }
                    }
                    if em != 0 {
                        let lane = em.trailing_zeros() as usize;
                        *(*bkt).tags.get_unchecked_mut(lane) = tag;
                        *(*bkt).pos.get_unchecked_mut(lane) = me;
                        break;
                    }
                    bi = (bi + 1) & bmask;
                }
            }
        }
    }
}

/// Single-pass L2-resident exact `event_id` dedup over 16 B `DedupRec` shards using
/// 64-byte 16-way AVX2 colocated `(tag16[16], pos16[16])` buckets (256 KB @ 20M, 512 KB @ 60M).
fn dedup_shard(
    segs: &mut Segs<DedupRec>,
    buckets16: &mut HugeBuf<DedupBucket16>,
    buckets32: &mut Vec<DedupBucket>,
    seg_starts: &mut Vec<u32>,
    loser_locs: &mut Vec<u32>,
    base: i64,
    oob: &[i64],
) {
    let n = segs.len();
    if n < 2 {
        return;
    }
    if n > u16::MAX as usize {
        dedup_shard_wide(segs, buckets32, base, oob);
        return;
    }
    const DEDUP_SEG_CAP: u32 = (arena::SEG / std::mem::size_of::<DedupRec>()) as u32;
    let nseg = segs.segs.len();
    // Partition full segments (n == 1008) first so 85-95% of duplicate/collision lookups
    // resolve `(osi, ori)` in 2 branchless ops instead of a 5-step binary search on `seg_starts`.
    let mut n_full = 0usize;
    for i in 0..nseg {
        if segs.segs[i].n == DEDUP_SEG_CAP {
            if i != n_full {
                segs.segs.swap(i, n_full);
            }
            n_full += 1;
        }
    }
    let full_end = n_full as u32 * DEDUP_SEG_CAP;
    seg_starts.clear();
    let mut off = full_end;
    for s in &segs.segs[n_full..] {
        seg_starts.push(off);
        off += s.n;
    }
    // Non-power-of-2 Lemire fast-range sizing (~61.5% load factor on 16-way buckets, zero 2x cliff):
    // ~3968 buckets (248 KB) @ 20M, ~5888 buckets (368 KB vs 512 KB, -28%!) @ 60M, backed by 1 pooled 2MB THP block!
    let nbuckets = ((n / 10) + 64) & !63;
    buckets16.reset_zeroed(nbuckets);
    let bptr = buckets16.as_mut_ptr();
    let seg_meta = segs.segs.as_ptr();
    loser_locs.clear();

    const PF: usize = 16;
    let mut bring = [(0u32, 0u16); PF];
    let mut me = 0u16;
    for si in 0..nseg {
        let seg = segs.seg(si);
        let slen = seg.len();
        let pre = slen.min(PF);
        for ri in 0..pre {
            let h = unsafe { seg.get_unchecked(ri).eid.hash(SEED_ID) };
            let bi = fast_bucket(h, nbuckets);
            bring[ri] = (bi as u32, (h as u16) | 1);
            prefetch(unsafe { bptr.add(bi) } as *const u8);
        }
        let main_end = slen.saturating_sub(PF);
        for ri in 0..main_end {
            let (bi0, tag) = bring[ri & (PF - 1)];
            let fh = unsafe { seg.get_unchecked(ri + PF).eid.hash(SEED_ID) };
            let fbi = fast_bucket(fh, nbuckets);
            bring[(ri + PF) & (PF - 1)] = (fbi as u32, (fh as u16) | 1);
            prefetch(unsafe { bptr.add(fbi) } as *const u8);
            let mut bi = bi0 as usize;
            'probe: loop {
                unsafe {
                    let bkt = bptr.add(bi);
                    let (mut eq, em) = bucket16_masks((*bkt).tags.as_ptr(), tag);
                    while eq != 0 {
                        let lane = (eq.trailing_zeros() >> 1) as usize;
                        eq &= eq - 1;
                        eq &= eq - 1;
                        let oi = *(*bkt).pos.get_unchecked(lane) as u32;
                        let (osi, ori) = if oi < full_end {
                            let s = oi / DEDUP_SEG_CAP;
                            (s as usize, (oi - s * DEDUP_SEG_CAP) as usize)
                        } else {
                            let t = seg_starts.partition_point(|&st| st <= oi) - 1;
                            (n_full + t, (oi - *seg_starts.get_unchecked(t)) as usize)
                        };
                        let os = *seg_meta.add(osi);
                        let o = &*(os.ptr as *const DedupRec).add(ori);
                        let r = seg.get_unchecked(ri);
                        if o.eid == r.eid {
                            let loser_loc = if ingest_less(r.ingest, o.ingest, base, oob) {
                                *(*bkt).pos.get_unchecked_mut(lane) = me;
                                o.loc
                            } else {
                                r.loc
                            };
                            if loser_loc != arena::LOC_INVALID {
                                loser_locs.push(loser_loc);
                            }
                            break 'probe;
                        }
                    }
                    if em != 0 {
                        let lane = (em.trailing_zeros() >> 1) as usize;
                        *(*bkt).tags.get_unchecked_mut(lane) = tag;
                        *(*bkt).pos.get_unchecked_mut(lane) = me;
                        break;
                    }
                    bi += 1;
                    if bi == nbuckets {
                        bi = 0;
                    }
                }
            }
            me = me.wrapping_add(1);
        }
        for ri in main_end..slen {
            let (bi0, tag) = bring[ri & (PF - 1)];
            let mut bi = bi0 as usize;
            'probe: loop {
                unsafe {
                    let bkt = bptr.add(bi);
                    let (mut eq, em) = bucket16_masks((*bkt).tags.as_ptr(), tag);
                    while eq != 0 {
                        let lane = (eq.trailing_zeros() >> 1) as usize;
                        eq &= eq - 1;
                        eq &= eq - 1;
                        let oi = *(*bkt).pos.get_unchecked(lane) as u32;
                        let (osi, ori) = if oi < full_end {
                            let s = oi / DEDUP_SEG_CAP;
                            (s as usize, (oi - s * DEDUP_SEG_CAP) as usize)
                        } else {
                            let t = seg_starts.partition_point(|&st| st <= oi) - 1;
                            (n_full + t, (oi - *seg_starts.get_unchecked(t)) as usize)
                        };
                        let os = *seg_meta.add(osi);
                        let o = &*(os.ptr as *const DedupRec).add(ori);
                        let r = seg.get_unchecked(ri);
                        if o.eid == r.eid {
                            let loser_loc = if ingest_less(r.ingest, o.ingest, base, oob) {
                                *(*bkt).pos.get_unchecked_mut(lane) = me;
                                o.loc
                            } else {
                                r.loc
                            };
                            if loser_loc != arena::LOC_INVALID {
                                loser_locs.push(loser_loc);
                            }
                            break 'probe;
                        }
                    }
                    if em != 0 {
                        let lane = (em.trailing_zeros() >> 1) as usize;
                        *(*bkt).tags.get_unchecked_mut(lane) = tag;
                        *(*bkt).pos.get_unchecked_mut(lane) = me;
                        break;
                    }
                    bi += 1;
                    if bi == nbuckets {
                        bi = 0;
                    }
                }
            }
            me = me.wrapping_add(1);
        }
    }
    // Deferred 16-ahead prefetched tombstone pass: overlaps cold DRAM RFO writes into `ev_set`/`purch_set`.
    let nl = loser_locs.len();
    if nl > 0 {
        let lp = loser_locs.as_ptr();
        let pre = nl.min(PF);
        for i in 0..pre {
            unsafe { prefetch_ev_loc(*lp.add(i)) };
        }
        let main_end = nl.saturating_sub(PF);
        for i in 0..main_end {
            unsafe {
                prefetch_ev_loc(*lp.add(i + PF));
                tombstone_ev_loc(*lp.add(i));
            }
        }
        for i in main_end..nl {
            unsafe { tombstone_ev_loc(*lp.add(i)) };
        }
    }
}

// ---------------------------------------------------------------------------
// solve

#[derive(Clone, Copy, Default)]
struct CampAcc {
    sessions: u32,
    purchasing: u32,
    assisted: u32,
    revenue: u64,
}

/// L1-tiny 2-byte per-(campaign, day) counter pair, laid out as `[camp * ndays + day]`.
/// All 14 days of a campaign sit in 28 contiguous bytes (half a cache line), so
/// multi-day `close()` and multi-day attributed sessions stay in L1/L2 cache.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Cell8 {
    sess: u8,
    assisted: u8,
}

#[derive(Clone, Copy, Default)]
struct CellHi {
    sess: u32,
    assisted: u32,
}

#[derive(Clone, Copy, Default)]
struct PurchRev {
    purchasing: u32,
    revenue: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct Sess {
    rev: u64,
    start: u32,
    day: u16,
    purch: bool,
}
const _: () = assert!(std::mem::size_of::<Sess>() == 16);

fn new_user_cols_heap(cap: usize) -> [Column; 6] {
    [
        Column::new("user_id", Kind::Str, cap),
        Column::new("day", Kind::Date, cap),
        Column::new("sessions", Kind::I32, cap),
        Column::new("events", Kind::I32, cap),
        Column::new("revenue_cents", Kind::I64, cap),
        Column::new("top_category", Kind::OptStr, cap),
    ]
}

struct UserDailyPart {
    buf: UserDailyBuf,
    blocks: Vec<(usize, u32)>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct AggBin {
    min_ts: u32,
    max_ts: u32,
    cnt: u32,
    head: u32, // head index into `bin_purch`, or u32::MAX if no purchases in this bin
}

const AGG_BIN_EMPTY: AggBin = AggBin { min_ts: u32::MAX, max_ts: 0, cnt: 0, head: u32::MAX };
const BIN_AGG_MIN_EVENTS: usize = 9;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CampState {
    cnt: u16,
    since: u16,
}
const _: () = assert!(std::mem::size_of::<CampState>() == 4);

struct Worker<'a> {
    ctx: &'a Ctx,
    wide_store: &'a [(u32, u16)],
    k1: usize,
    ndays: usize,
    /// Co-located `(cnt, since)` per campaign (1 cacheline instead of 2 in `attribute_clicks`).
    camp_st: Vec<CampState>,
    /// Pooled 2 MB THP block shared by `cells` + `cells_hi` + `purch_rev` (1 L1 dTLB entry!).
    acc_block: Option<(usize, u32)>,
    /// Transposed `[camp * ndays + day]` L1-tiny u8 counters (~56 KB for 2001x14), 2MB THP backed.
    cells: HugeBuf<Cell8>,
    /// Overflow carries for `cells` (rarely touched, only on 256-wrap), 2MB THP backed.
    cells_hi: HugeBuf<CellHi>,
    /// Purchasing sessions & revenue `[camp * ndays + day]` (touched on ~3% of sessions), 2MB THP backed.
    purch_rev: HugeBuf<PurchRev>,
    cat_rev: Vec<u64>,
    touched: Vec<u32>,
    sess: Vec<Sess>,
    day_end: Vec<u32>,
    /// L1-resident half-hour bins for fused 1-pass sessionization (672*16B = 10.75 KB).
    agg_bins: Vec<AggBin>,
    /// 88-byte non-empty bin bitmask so Pass 2 jumps directly between occupied bins via `tzcnt`.
    bin_mask: Vec<u64>,
    /// Intrusive per-bin linked list of purchases `(pc, next_idx)` (touched on only ~3% of events).
    bin_purch: Vec<(u32, u32)>,
    ud: UserDailyBuf,
    carver: BlockCarver,
    done_cols: Vec<UserDailyPart>,
}

impl<'a> Worker<'a> {
    fn new(ctx: &'a Ctx, wide_store: &'a [(u32, u16)]) -> Worker<'a> {
        let k1 = ctx.camp_ids.len();
        let ndays = ctx.ndays;
        let total = k1 * ndays;
        let nbins = ndays * 48;
        let b_cells = (total * std::mem::size_of::<Cell8>() + 63) & !63;
        let b_hi = (total * std::mem::size_of::<CellHi>() + 63) & !63;
        let b_pr = (total * std::mem::size_of::<PurchRev>() + 63) & !63;
        let (acc_block, cells, cells_hi, purch_rev) = if b_cells + b_hi + b_pr <= arena::BLOCK {
            let (addr, id) = arena::get_block();
            unsafe {
                std::ptr::write_bytes(addr as *mut u8, 0, b_cells + b_hi + b_pr);
                let mut c = HugeBuf::from_borrowed(addr as *mut Cell8, total);
                c.set_len(total);
                let mut hi = HugeBuf::from_borrowed((addr + b_cells) as *mut CellHi, total);
                hi.set_len(total);
                let mut pr = HugeBuf::from_borrowed((addr + b_cells + b_hi) as *mut PurchRev, total);
                pr.set_len(total);
                (Some((addr, id)), c, hi, pr)
            }
        } else {
            (
                None,
                HugeBuf::new_zeroed(total),
                HugeBuf::new_zeroed(total),
                HugeBuf::new_zeroed(total),
            )
        };
        Worker {
            ctx,
            wide_store,
            k1,
            ndays,
            camp_st: vec![CampState::default(); k1],
            acc_block,
            cells,
            cells_hi,
            purch_rev,
            cat_rev: vec![0; ctx.cat_names.len()],
            touched: Vec::new(),
            sess: Vec::new(),
            day_end: vec![0; ndays],
            agg_bins: vec![AGG_BIN_EMPTY; nbins],
            bin_mask: vec![0u64; nbins.div_ceil(64)],
            bin_purch: Vec::new(),
            ud: UserDailyBuf::new(),
            carver: BlockCarver::new(),
            done_cols: Vec::new(),
        }
    }

    fn into_acc(self) -> Vec<CampAcc> {
        let k1 = self.k1;
        let ndays = self.ndays;
        let mut out = vec![CampAcc::default(); k1 * ndays];
        for k in 0..k1 {
            let base = k * ndays;
            for d in 0..ndays {
                let src = base + d;
                let c = self.cells[src];
                let h = self.cells_hi[src];
                let pr = self.purch_rev[src];
                let sessions = h.sess + c.sess as u32;
                let assisted = h.assisted + c.assisted as u32;
                if sessions != 0 || assisted != 0 {
                    out[d * k1 + k] = CampAcc {
                        sessions,
                        purchasing: pr.purchasing,
                        assisted,
                        revenue: pr.revenue,
                    };
                }
            }
        }
        out
    }

    #[inline(always)]
    fn add_session(&mut self, camp: usize, day: usize, purch: bool, rev: u64) {
        let idx = camp * self.ndays + day;
        let c = unsafe { self.cells.get_unchecked_mut(idx) };
        let (nv, ov) = c.sess.overflowing_add(1);
        c.sess = nv;
        if ov {
            unsafe { self.cells_hi.get_unchecked_mut(idx).sess += 256 };
        }
        if purch {
            let pr = unsafe { self.purch_rev.get_unchecked_mut(idx) };
            pr.purchasing += 1;
            pr.revenue += rev;
        }
    }

    #[inline(always)]
    fn add_organic_sessions(&mut self, day: usize, add: u32) {
        let c = unsafe { self.cells.get_unchecked_mut(day) };
        let sum = c.sess as u32 + add;
        c.sess = sum as u8;
        if sum >= 256 {
            unsafe { self.cells_hi.get_unchecked_mut(day).sess += sum & !255 };
        }
    }

    #[inline(always)]
    fn add_assisted(&mut self, idx: usize, add: u32) {
        let c = unsafe { self.cells.get_unchecked_mut(idx) };
        let sum = c.assisted as u32 + add;
        c.assisted = sum as u8;
        if sum >= 256 {
            unsafe { self.cells_hi.get_unchecked_mut(idx).assisted += sum & !255 };
        }
    }

    #[inline(always)]
    fn price_cat(&self, pc: u32) -> (u64, usize) {
        if !self.ctx.wide {
            ((pc >> 8) as u64, (pc & 0xFF) as usize)
        } else {
            let (p, c) = unsafe { *self.wide_store.get_unchecked((pc - 1) as usize) };
            (p as u64, c as usize)
        }
    }

    #[cold]
    #[inline(never)]
    fn roll_user_file(&mut self) {
        let prev = std::mem::replace(&mut self.ud, UserDailyBuf::new());
        let blocks = self.carver.take_blocks();
        self.done_cols.push(UserDailyPart { buf: prev, blocks });
    }

    #[inline(always)]
    fn begin_person(&mut self, person: Key8) {
        if self.ud.len() >= MAX_ROWS_PER_FILE {
            self.roll_user_file();
        }
        self.ud.begin_person(person, &mut self.carver);
    }

    #[inline(always)]
    fn end_person(&mut self) {
        self.ud.end_person(&mut self.carver);
    }

    #[inline]
    fn flush_row(&mut self, day: u32, ns: u32, ne: u32, rev: u64) {
        if rev == 0 {
            self.ud.push_zero_rev(day, ns, ne, &mut self.carver);
        } else {
            let mut top = -1i32;
            let mut best = 0u64;
            for &c in &self.touched {
                let v = unsafe { *self.cat_rev.get_unchecked(c as usize) };
                if v > best || (v == best && (c as i32) < top) {
                    best = v;
                    top = c as i32;
                }
                unsafe { *self.cat_rev.get_unchecked_mut(c as usize) = 0 };
            }
            self.touched.clear();
            self.ud.push_with_rev(day, ns, ne, rev, top as usize, &mut self.carver);
        }
    }

    #[inline(always)]
    fn close(&mut self, camp: usize, a: u32, b: u32) {
        if a >= b {
            return;
        }
        let base = camp * self.ndays;
        let mut i = a;
        while i < b {
            let day = unsafe { self.sess.get_unchecked(i as usize).day as usize };
            let end = b.min(unsafe { *self.day_end.get_unchecked(day) });
            self.add_assisted(base + day, end - i);
            i = end;
        }
    }

    /// One person: events in session-preserving order, clicks unsorted (sorted lazily when nsess >= 2).
    fn person(&mut self, evs: &[Ev8], clicks: &mut [Cl24], ctmp: &mut Vec<Cl24>) {
        let day0ms = self.ctx.day0 * DAY_MS - self.ctx.base;
        if evs.len() == 1 && clicks.is_empty() {
            let e = unsafe { *evs.get_unchecked(0) };
            let day = ((e.ts() as i64 - day0ms) / DAY_MS) as u32;
            let pc = e.pc();
            if pc == 0 {
                self.ud.push_zero_rev(day, 1, 1, &mut self.carver);
                self.add_session(0, day as usize, false, 0);
            } else {
                let (p, c) = self.price_cat(pc);
                self.ud.push_with_rev(day, 1, 1, p, c, &mut self.carver);
                self.add_session(0, day as usize, true, p);
            }
            return;
        }
        let e0 = unsafe { *evs.get_unchecked(0) };
        let ts0 = e0.ts() as i64;
        let mut day_idx = ((ts0 - day0ms) / DAY_MS) as u32;
        let mut next_day_ms = day0ms + (day_idx as i64 + 1) * DAY_MS;
        let mut cur_day = day_idx;
        let (mut ns, mut ne, mut rev) = (1u32, 1u32, 0u64);
        let mut prev = ts0;
        let mut s_start = ts0;
        let mut s_rev = 0u64;
        let mut s_purch = false;
        let pc0 = e0.pc();
        if pc0 != 0 {
            let (p, c) = self.price_cat(pc0);
            s_rev = p;
            s_purch = true;
            rev = p;
            self.touched.push(c as u32);
            unsafe { *self.cat_rev.get_unchecked_mut(c) = p };
        }
        if clicks.is_empty() {
            // Fast organic-only path: zero `self.sess` writes or second pass!
            for &e in &evs[1..] {
                let ts = e.ts() as i64;
                if ts - prev > GAP_MS {
                    self.add_session(0, cur_day as usize, s_purch, s_rev);
                    s_rev = 0;
                    s_purch = false;
                    if ts >= next_day_ms {
                        while ts >= next_day_ms {
                            day_idx += 1;
                            next_day_ms += DAY_MS;
                        }
                        self.flush_row(cur_day, ns, ne, rev);
                        cur_day = day_idx;
                        ns = 0;
                        ne = 0;
                        rev = 0;
                    }
                    ns += 1;
                }
                prev = ts;
                ne += 1;
                let pc = e.pc();
                if pc != 0 {
                    let (p, c) = self.price_cat(pc);
                    s_rev += p;
                    s_purch = true;
                    rev += p;
                    let cr = unsafe { self.cat_rev.get_unchecked_mut(c) };
                    if *cr == 0 {
                        self.touched.push(c as u32);
                    }
                    *cr += p;
                }
            }
            self.add_session(0, cur_day as usize, s_purch, s_rev);
            self.flush_row(cur_day, ns, ne, rev);
            return;
        }
        self.sess.clear();
        for &e in &evs[1..] {
            let ts = e.ts() as i64;
            if ts - prev > GAP_MS {
                self.sess.push(Sess { rev: s_rev, start: s_start as u32, day: cur_day as u16, purch: s_purch });
                s_start = ts;
                s_rev = 0;
                s_purch = false;
                if ts >= next_day_ms {
                    while ts >= next_day_ms {
                        day_idx += 1;
                        next_day_ms += DAY_MS;
                    }
                    self.flush_row(cur_day, ns, ne, rev);
                    cur_day = day_idx;
                    ns = 0;
                    ne = 0;
                    rev = 0;
                }
                ns += 1;
            }
            prev = ts;
            ne += 1;
            let pc = e.pc();
            if pc != 0 {
                let (p, c) = self.price_cat(pc);
                s_rev += p;
                s_purch = true;
                rev += p;
                let cr = unsafe { self.cat_rev.get_unchecked_mut(c) };
                if *cr == 0 {
                    self.touched.push(c as u32);
                }
                *cr += p;
            }
        }
        self.sess.push(Sess { rev: s_rev, start: s_start as u32, day: cur_day as u16, purch: s_purch });
        self.flush_row(cur_day, ns, ne, rev);
        self.attribute_clicks(clicks, ctmp);
    }

    #[inline(always)]
    fn flush_row_zero_rev(&mut self, day: u32, ns: u32, ne: u32) {
        self.ud.push_zero_rev(day, ns, ne, &mut self.carver);
    }

    /// Fast path for small persons (< BIN_AGG_MIN_EVENTS) with zero purchases:
    /// operates on 4-byte sorted `u32` timestamps with zero price/category work.
    fn person_no_purch(&mut self, tss: &[u32], clicks: &mut [Cl24], ctmp: &mut Vec<Cl24>) {
        let day0ms = self.ctx.day0 * DAY_MS - self.ctx.base;
        let ts0 = unsafe { *tss.get_unchecked(0) } as i64;
        let mut day_idx = ((ts0 - day0ms) / DAY_MS) as u32;
        if tss.len() == 1 && clicks.is_empty() {
            self.flush_row_zero_rev(day_idx, 1, 1);
            self.add_session(0, day_idx as usize, false, 0);
            return;
        }
        let mut next_day_ms = day0ms + (day_idx as i64 + 1) * DAY_MS;
        let mut cur_day = day_idx;
        let (mut ns, mut ne) = (1u32, 1u32);
        let mut prev = ts0;
        if clicks.is_empty() {
            for &t in &tss[1..] {
                let ts = t as i64;
                if ts - prev > GAP_MS {
                    if ts >= next_day_ms {
                        while ts >= next_day_ms {
                            day_idx += 1;
                            next_day_ms += DAY_MS;
                        }
                        self.add_organic_sessions(cur_day as usize, ns);
                        self.flush_row_zero_rev(cur_day, ns, ne);
                        cur_day = day_idx;
                        ns = 0;
                        ne = 0;
                    }
                    ns += 1;
                }
                prev = ts;
                ne += 1;
            }
            self.add_organic_sessions(cur_day as usize, ns);
            self.flush_row_zero_rev(cur_day, ns, ne);
            return;
        }
        self.sess.clear();
        let mut s_start = ts0;
        for &t in &tss[1..] {
            let ts = t as i64;
            if ts - prev > GAP_MS {
                self.sess.push(Sess { rev: 0, start: s_start as u32, day: cur_day as u16, purch: false });
                s_start = ts;
                if ts >= next_day_ms {
                    while ts >= next_day_ms {
                        day_idx += 1;
                        next_day_ms += DAY_MS;
                    }
                    self.flush_row_zero_rev(cur_day, ns, ne);
                    cur_day = day_idx;
                    ns = 0;
                    ne = 0;
                }
                ns += 1;
            }
            prev = ts;
            ne += 1;
        }
        self.sess.push(Sess { rev: 0, start: s_start as u32, day: cur_day as u16, purch: false });
        self.flush_row_zero_rev(cur_day, ns, ne);
        self.attribute_clicks(clicks, ctmp);
    }

    /// Fused 1-pass L1 half-hour bin sessionizer with 2-level bitmask (`word_mask` + `bin_mask`)
    /// for all persons with >= BIN_AGG_MIN_EVENTS (10) unsorted events: zero sorting, zero empty
    /// word scans, branchless 4-byte non-purchase binning, separate purchase loop, and `tzcnt` walk!
    fn person_binned(
        &mut self,
        tss: &[u32],
        purch: &[(u32, u32)],
        clicks: &mut [Cl24],
        ctmp: &mut Vec<Cl24>,
    ) {
        let day0_off = (self.ctx.day0 * DAY_MS - self.ctx.base) as u32;
        const GAP_U32: u32 = GAP_MS as u32;
        let bptr = self.agg_bins.as_mut_ptr();
        let mptr = self.bin_mask.as_mut_ptr();
        let nwords = self.bin_mask.len();
        let use_wmask = nwords <= 64;
        let mut word_mask = 0u64;
#[inline(always)]
#[allow(unused_mut)]
fn min_u32_cmov(mut a: u32, b: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "cmp {a:e}, {b:e}",
            "cmova {a:e}, {b:e}",
            a = inout(reg) a,
            b = in(reg) b,
            options(pure, nomem, nostack),
        );
        a
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.min(b)
    }
}

#[inline(always)]
#[allow(unused_mut)]
fn max_u32_cmov(mut a: u32, b: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "cmp {a:e}, {b:e}",
            "cmovb {a:e}, {b:e}",
            a = inout(reg) a,
            b = in(reg) b,
            options(pure, nomem, nostack),
        );
        a
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.max(b)
    }
}

        // Single branchless sequential pass over 4-byte non-purchase timestamps (97% of events).
        for &ts in tss {
            let b = ((ts - day0_off) / GAP_U32) as usize;
            let bin = unsafe { &mut *bptr.add(b) };
            bin.min_ts = min_u32_cmov(bin.min_ts, ts);
            bin.max_ts = max_u32_cmov(bin.max_ts, ts);
            bin.cnt += 1;
            let w = b >> 6;
            unsafe { *mptr.add(w) |= 1u64 << (b & 63) };
            word_mask |= 1u64 << (w & 63);
        }
        let organic_only = clicks.is_empty();
        if !organic_only {
            self.sess.clear();
        }
        if purch.is_empty() {
            let mut have_sess = false;
            let mut prev_max = i64::MIN;
            let mut cur_day = 0u32;
            let (mut ns, mut ne) = (0u32, 0u32);
            let mut s_start = 0u32;
            macro_rules! walk_word_np {
                ($w_idx:expr) => {{
                    let w_idx: usize = $w_idx;
                    let mut m = unsafe { *mptr.add(w_idx) };
                    unsafe { *mptr.add(w_idx) = 0 };
                    let base_b = w_idx << 6;
                    while m != 0 {
                        let b = base_b + m.trailing_zeros() as usize;
                        m &= m - 1;
                        let bin = unsafe { *bptr.add(b) };
                        unsafe { *bptr.add(b) = AGG_BIN_EMPTY };
                        let min_ts = bin.min_ts as i64;
                        if !have_sess || min_ts - prev_max > GAP_MS {
                            let new_day = (b / 48) as u32;
                            if have_sess {
                                if !organic_only {
                                    self.sess.push(Sess { rev: 0, start: s_start, day: cur_day as u16, purch: false });
                                }
                                if new_day != cur_day {
                                    if organic_only {
                                        self.add_organic_sessions(cur_day as usize, ns);
                                    }
                                    self.flush_row_zero_rev(cur_day, ns, ne);
                                    cur_day = new_day;
                                    ns = 0;
                                    ne = 0;
                                }
                            } else {
                                have_sess = true;
                                cur_day = new_day;
                            }
                            s_start = bin.min_ts;
                            ns += 1;
                        }
                        prev_max = bin.max_ts as i64;
                        ne += bin.cnt;
                    }
                }};
            }
            if use_wmask {
                let mut wm = word_mask;
                while wm != 0 {
                    let w_idx = wm.trailing_zeros() as usize;
                    wm &= wm - 1;
                    walk_word_np!(w_idx);
                }
            } else {
                for w_idx in 0..nwords {
                    if unsafe { *mptr.add(w_idx) } != 0 {
                        walk_word_np!(w_idx);
                    }
                }
            }
            if organic_only {
                self.add_organic_sessions(cur_day as usize, ns);
                self.flush_row_zero_rev(cur_day, ns, ne);
                return;
            }
            self.sess.push(Sess { rev: 0, start: s_start, day: cur_day as u16, purch: false });
            self.flush_row_zero_rev(cur_day, ns, ne);
            self.attribute_clicks(clicks, ctmp);
            return;
        }
        // Separate tiny pass over purchases (~3% of events).
        self.bin_purch.clear();
        for &(ts, pc) in purch {
            let b = ((ts - day0_off) / GAP_U32) as usize;
            let bin = unsafe { &mut *bptr.add(b) };
            bin.min_ts = bin.min_ts.min(ts);
            bin.max_ts = bin.max_ts.max(ts);
            bin.cnt += 1;
            let w = b >> 6;
            unsafe { *mptr.add(w) |= 1u64 << (b & 63) };
            word_mask |= 1u64 << (w & 63);
            let idx = self.bin_purch.len() as u32;
            self.bin_purch.push((pc, bin.head));
            bin.head = idx;
        }

        // Walk ONLY occupied bins in chronological order via 2-level `tzcnt`, resetting in place.
        let mut have_sess = false;
        let mut prev_max = i64::MIN;
        let mut cur_day = 0u32;
        let (mut ns, mut ne, mut rev) = (0u32, 0u32, 0u64);
        let (mut s_start, mut s_rev, mut s_purch) = (0u32, 0u64, false);

        macro_rules! walk_word_p {
            ($w_idx:expr) => {{
                let w_idx: usize = $w_idx;
                let mut m = unsafe { *mptr.add(w_idx) };
                unsafe { *mptr.add(w_idx) = 0 };
                let base_b = w_idx << 6;
                while m != 0 {
                    let b = base_b + m.trailing_zeros() as usize;
                    m &= m - 1;
                    let bin = unsafe { *bptr.add(b) };
                    unsafe { *bptr.add(b) = AGG_BIN_EMPTY };
                    let min_ts = bin.min_ts as i64;
                    if !have_sess || min_ts - prev_max > GAP_MS {
                        let new_day = (b / 48) as u32;
                        if have_sess {
                            if organic_only {
                                self.add_session(0, cur_day as usize, s_purch, s_rev);
                            } else {
                                self.sess.push(Sess { rev: s_rev, start: s_start, day: cur_day as u16, purch: s_purch });
                            }
                            if new_day != cur_day {
                                self.flush_row(cur_day, ns, ne, rev);
                                cur_day = new_day;
                                ns = 0;
                                ne = 0;
                                rev = 0;
                            }
                        } else {
                            have_sess = true;
                            cur_day = new_day;
                        }
                        s_start = bin.min_ts;
                        s_rev = 0;
                        s_purch = false;
                        ns += 1;
                    }
                    prev_max = bin.max_ts as i64;
                    ne += bin.cnt;
                    let mut pi = bin.head;
                    while pi != u32::MAX {
                        let (pc, next_pi) = unsafe { *self.bin_purch.get_unchecked(pi as usize) };
                        let (p, c) = self.price_cat(pc);
                        s_rev += p;
                        s_purch = true;
                        rev += p;
                        let cr = unsafe { self.cat_rev.get_unchecked_mut(c) };
                        if *cr == 0 {
                            self.touched.push(c as u32);
                        }
                        *cr += p;
                        pi = next_pi;
                    }
                }
            }};
        }
        if use_wmask {
            let mut wm = word_mask;
            while wm != 0 {
                let w_idx = wm.trailing_zeros() as usize;
                wm &= wm - 1;
                walk_word_p!(w_idx);
            }
        } else {
            for w_idx in 0..nwords {
                if unsafe { *mptr.add(w_idx) } != 0 {
                    walk_word_p!(w_idx);
                }
            }
        }
        if organic_only {
            self.add_session(0, cur_day as usize, s_purch, s_rev);
            self.flush_row(cur_day, ns, ne, rev);
            return;
        }
        self.sess.push(Sess { rev: s_rev, start: s_start, day: cur_day as u16, purch: s_purch });
        self.flush_row(cur_day, ns, ne, rev);
        self.attribute_clicks(clicks, ctmp);
    }

    #[inline(never)]
    fn attribute_clicks(&mut self, clicks: &mut [Cl24], ctmp: &mut Vec<Cl24>) {
        const SEVEN_DAYS_U32: u32 = SEVEN_DAYS_MS as u32;
        let nsess = self.sess.len();
        if nsess == 1 {
            // Single-session O(n) fast path: zero click sorting, zero day_end, zero close().
            let s = self.sess[0];
            let lo = s.start - SEVEN_DAYS_U32;
            let hi = s.start;
            let sday = s.day as usize;
            let ndays = self.ndays;
            let mut best_ts = i64::MIN;
            let mut best_cid = CKey::EMPTY;
            let mut credited = 0usize;
            for c in clicks.iter() {
                if c.ts >= lo && c.ts <= hi {
                    let cts = c.ts as i64;
                    if cts > best_ts || (cts == best_ts && c.cid.cmp_bytes(&best_cid) == std::cmp::Ordering::Less) {
                        best_ts = cts;
                        best_cid = c.cid;
                        credited = c.camp as usize;
                    }
                    let k = c.camp as usize;
                    let st = unsafe { self.camp_st.get_unchecked_mut(k) };
                    if st.cnt == 0 {
                        st.cnt = 1;
                        self.add_assisted(k * ndays + sday, 1);
                        self.touched.push(k as u32);
                    }
                }
            }
            for &k in &self.touched {
                unsafe { self.camp_st.get_unchecked_mut(k as usize).cnt = 0 };
            }
            self.touched.clear();
            self.add_session(credited, sday, s.purch, s.rev);
            return;
        }
        let mut clicks = clicks;
        if clicks.len() >= 3 {
            // In-place zero-alloc dead-click compaction: drop clicks outside [first_sess - 7d, last_sess]
            // before sorting so 0/1/2-surviving-click users hit O(n) fast paths and >=3 sorts less!
            let first_lo = unsafe { self.sess.get_unchecked(0).start } - SEVEN_DAYS_U32;
            let last_hi = unsafe { self.sess.get_unchecked(nsess - 1).start };
            let mut w = 0usize;
            let n = clicks.len();
            for i in 0..n {
                let c = unsafe { *clicks.get_unchecked(i) };
                if c.ts >= first_lo && c.ts <= last_hi {
                    if w != i {
                        unsafe { *clicks.get_unchecked_mut(w) = c };
                    }
                    w += 1;
                }
            }
            if w == 0 {
                for j in 0..nsess {
                    let s = unsafe { *self.sess.get_unchecked(j) };
                    self.add_session(0, s.day as usize, s.purch, s.rev);
                }
                return;
            }
            clicks = unsafe { clicks.get_unchecked_mut(..w) };
        }
        if clicks.len() == 1 {
            let c = unsafe { *clicks.get_unchecked(0) };
            let cts = c.ts;
            let hi = cts.saturating_add(SEVEN_DAYS_U32);
            let k = c.camp as usize;
            let base = k * self.ndays;
            for j in 0..nsess {
                let s = unsafe { *self.sess.get_unchecked(j) };
                let sday = s.day as usize;
                if s.start >= cts && s.start <= hi {
                    self.add_session(k, sday, s.purch, s.rev);
                    self.add_assisted(base + sday, 1);
                } else {
                    self.add_session(0, sday, s.purch, s.rev);
                }
            }
            return;
        }
        if clicks.len() == 2 {
            let (mut c0, mut c1) = unsafe { (*clicks.get_unchecked(0), *clicks.get_unchecked(1)) };
            if click_gt(&c0, &c1) {
                std::mem::swap(&mut c0, &mut c1);
            }
            let hi0 = c0.ts.saturating_add(SEVEN_DAYS_U32);
            let hi1 = c1.ts.saturating_add(SEVEN_DAYS_U32);
            let same_ts = c0.ts == c1.ts;
            let k0 = c0.camp as usize;
            let k1 = c1.camp as usize;
            let base0 = k0 * self.ndays;
            let base1 = k1 * self.ndays;
            if k0 == k1 {
                for j in 0..nsess {
                    let s = unsafe { *self.sess.get_unchecked(j) };
                    let sday = s.day as usize;
                    let hit = (s.start >= c0.ts && s.start <= hi0) || (s.start >= c1.ts && s.start <= hi1);
                    if hit {
                        self.add_session(k0, sday, s.purch, s.rev);
                        self.add_assisted(base0 + sday, 1);
                    } else {
                        self.add_session(0, sday, s.purch, s.rev);
                    }
                }
            } else {
                for j in 0..nsess {
                    let s = unsafe { *self.sess.get_unchecked(j) };
                    let sday = s.day as usize;
                    let in0 = s.start >= c0.ts && s.start <= hi0;
                    let in1 = s.start >= c1.ts && s.start <= hi1;
                    let credited = if in1 && !same_ts {
                        k1
                    } else if in0 {
                        k0
                    } else {
                        0
                    };
                    self.add_session(credited, sday, s.purch, s.rev);
                    if in0 {
                        self.add_assisted(base0 + sday, 1);
                    }
                    if in1 {
                        self.add_assisted(base1 + sday, 1);
                    }
                }
            }
            return;
        }
        if clicks.len() >= u16::MAX as usize || nsess >= u16::MAX as usize {
            self.attribute_clicks_wide(clicks, ctmp);
            return;
        }
        sort_person_clicks_by_ts(clicks, ctmp);
        // Session days are monotone non-decreasing. Overwrite only occupied days (O(1) if all share 1 day).
        let first_day = unsafe { self.sess.get_unchecked(0).day as usize };
        let last_day = unsafe { self.sess.get_unchecked(nsess - 1).day as usize };
        if first_day == last_day {
            unsafe { *self.day_end.get_unchecked_mut(first_day) = nsess as u32 };
        } else {
            for (i, s) in self.sess.iter().enumerate() {
                unsafe { *self.day_end.get_unchecked_mut(s.day as usize) = i as u32 + 1 };
            }
        }
        let n = clicks.len();
        let (mut cl, mut cr) = (0usize, 0usize);
        let mut latest_ts = u32::MAX;
        let mut latest_cid = CKey::EMPTY;
        let mut latest_camp = 0usize;
        for j in 0..nsess {
            let s = unsafe { *self.sess.get_unchecked(j) };
            while cr < n && unsafe { clicks.get_unchecked(cr).ts } <= s.start {
                let c = unsafe { clicks.get_unchecked(cr) };
                if c.ts != latest_ts {
                    latest_ts = c.ts;
                    latest_cid = c.cid;
                    latest_camp = c.camp as usize;
                } else if c.cid.cmp_bytes(&latest_cid) == std::cmp::Ordering::Less {
                    latest_cid = c.cid;
                    latest_camp = c.camp as usize;
                }
                let k = c.camp as usize;
                let st = unsafe { self.camp_st.get_unchecked_mut(k) };
                st.cnt += 1;
                if st.cnt == 1 {
                    st.since = j as u16;
                }
                cr += 1;
            }
            let lo = s.start - SEVEN_DAYS_U32;
            while cl < cr && unsafe { clicks.get_unchecked(cl).ts } < lo {
                let k = unsafe { clicks.get_unchecked(cl).camp as usize };
                let st = unsafe { self.camp_st.get_unchecked_mut(k) };
                st.cnt -= 1;
                if st.cnt == 0 {
                    let a = st.since as u32;
                    self.close(k, a, j as u32);
                }
                cl += 1;
            }
            let credited = if cl < cr { latest_camp } else { 0 };
            self.add_session(credited, s.day as usize, s.purch, s.rev);
        }
        for i in cl..cr {
            let k = unsafe { clicks.get_unchecked(i).camp as usize };
            let st = unsafe { self.camp_st.get_unchecked_mut(k) };
            if st.cnt > 0 {
                st.cnt = 0;
                let a = st.since as u32;
                self.close(k, a, nsess as u32);
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn attribute_clicks_wide(&mut self, clicks: &mut [Cl24], ctmp: &mut Vec<Cl24>) {
        const SEVEN_DAYS_U32: u32 = SEVEN_DAYS_MS as u32;
        let nsess = self.sess.len();
        sort_person_clicks_by_ts(clicks, ctmp);
        for (i, s) in self.sess.iter().enumerate() {
            unsafe { *self.day_end.get_unchecked_mut(s.day as usize) = i as u32 + 1 };
        }
        let mut st_wide = vec![(0u32, 0u32); self.k1];
        let n = clicks.len();
        let (mut cl, mut cr) = (0usize, 0usize);
        let mut latest_ts = u32::MAX;
        let mut latest_cid = CKey::EMPTY;
        let mut latest_camp = 0usize;
        for j in 0..nsess {
            let s = unsafe { *self.sess.get_unchecked(j) };
            while cr < n && unsafe { clicks.get_unchecked(cr).ts } <= s.start {
                let c = unsafe { clicks.get_unchecked(cr) };
                if c.ts != latest_ts {
                    latest_ts = c.ts;
                    latest_cid = c.cid;
                    latest_camp = c.camp as usize;
                } else if c.cid.cmp_bytes(&latest_cid) == std::cmp::Ordering::Less {
                    latest_cid = c.cid;
                    latest_camp = c.camp as usize;
                }
                let k = c.camp as usize;
                st_wide[k].0 += 1;
                if st_wide[k].0 == 1 {
                    st_wide[k].1 = j as u32;
                }
                cr += 1;
            }
            let lo = s.start - SEVEN_DAYS_U32;
            while cl < cr && unsafe { clicks.get_unchecked(cl).ts } < lo {
                let k = unsafe { clicks.get_unchecked(cl).camp as usize };
                st_wide[k].0 -= 1;
                if st_wide[k].0 == 0 {
                    let a = st_wide[k].1;
                    self.close(k, a, j as u32);
                }
                cl += 1;
            }
            let credited = if cl < cr { latest_camp } else { 0 };
            self.add_session(credited, s.day as usize, s.purch, s.rev);
        }
        for i in cl..cr {
            let k = unsafe { clicks.get_unchecked(i).camp as usize };
            if st_wide[k].0 > 0 {
                st_wide[k].0 = 0;
                let a = st_wide[k].1;
                self.close(k, a, nsess as u32);
            }
        }
    }

}

// ---------------------------------------------------------------------------
// per-shard direct person interning + compact L2 solve (8B Ev8 + 24B Cl24)

/// Stable L1-resident LSD radix sort of `v` by a 32-bit key (4 passes of 8 bits,
/// only 4 KB of stack histograms; passes in which every key shares the same byte are skipped).
fn radix_by_u32<T: Copy>(v: &mut [T], tmp: &mut Vec<T>, key: impl Fn(&T) -> u32) {
    let n = v.len();
    tmp.clear();
    tmp.reserve(n);
    unsafe { tmp.set_len(n) };
    let mut cnt = [[0u32; 256]; 4];
    for x in v.iter() {
        let k = key(x);
        unsafe {
            *cnt[0].get_unchecked_mut((k & 255) as usize) += 1;
            *cnt[1].get_unchecked_mut(((k >> 8) & 255) as usize) += 1;
            *cnt[2].get_unchecked_mut(((k >> 16) & 255) as usize) += 1;
            *cnt[3].get_unchecked_mut((k >> 24) as usize) += 1;
        }
    }
    let mut in_v = true;
    for (pass, c) in cnt.iter_mut().enumerate() {
        if c.iter().any(|&x| x as usize == n) {
            continue;
        }
        let mut sum = 0u32;
        for x in c.iter_mut() {
            let t = *x;
            *x = sum;
            sum += t;
        }
        let sh = pass as u32 * 8;
        let (src, dst): (&[T], &mut [T]) = if in_v { (&*v, &mut tmp[..]) } else { (&tmp[..], &mut *v) };
        let dst = dst.as_mut_ptr();
        for x in src {
            let d = ((key(x) >> sh) & 255) as usize;
            unsafe {
                let slot = c.get_unchecked_mut(d);
                dst.add(*slot as usize).write(*x);
                *slot += 1;
            }
        }
        in_v = !in_v;
    }
    if !in_v {
        v.copy_from_slice(&tmp[..n]);
    }
}

#[inline(always)]
fn cswap32(a: &mut u32, b: &mut u32) {
    let (x, y) = (*a, *b);
    *a = x.min(y);
    *b = x.max(y);
}

#[inline(always)]
fn sort_person_ts(sl: &mut [u32]) {
    let n = sl.len();
    if n < 2 {
        return;
    }
    unsafe {
        if n == 2 {
            let mut a = *sl.get_unchecked(0);
            let mut b = *sl.get_unchecked(1);
            cswap32(&mut a, &mut b);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            return;
        }
        if n == 3 {
            let (mut a, mut b, mut c) = (*sl.get_unchecked(0), *sl.get_unchecked(1), *sl.get_unchecked(2));
            cswap32(&mut a, &mut b);
            cswap32(&mut b, &mut c);
            cswap32(&mut a, &mut b);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            return;
        }
        if n == 4 {
            let (mut a, mut b, mut c, mut d) = (
                *sl.get_unchecked(0),
                *sl.get_unchecked(1),
                *sl.get_unchecked(2),
                *sl.get_unchecked(3),
            );
            cswap32(&mut a, &mut b);
            cswap32(&mut c, &mut d);
            cswap32(&mut a, &mut c);
            cswap32(&mut b, &mut d);
            cswap32(&mut b, &mut c);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            *sl.get_unchecked_mut(3) = d;
            return;
        }
        if n == 5 {
            let (mut a, mut b, mut c, mut d, mut e) = (
                *sl.get_unchecked(0),
                *sl.get_unchecked(1),
                *sl.get_unchecked(2),
                *sl.get_unchecked(3),
                *sl.get_unchecked(4),
            );
            cswap32(&mut a, &mut b);
            cswap32(&mut c, &mut d);
            cswap32(&mut a, &mut c);
            cswap32(&mut b, &mut d);
            cswap32(&mut b, &mut c);
            cswap32(&mut d, &mut e);
            cswap32(&mut c, &mut d);
            cswap32(&mut b, &mut c);
            cswap32(&mut a, &mut b);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            *sl.get_unchecked_mut(3) = d;
            *sl.get_unchecked_mut(4) = e;
            return;
        }
        if n == 6 {
            let (mut a, mut b, mut c, mut d, mut e, mut f) = (
                *sl.get_unchecked(0),
                *sl.get_unchecked(1),
                *sl.get_unchecked(2),
                *sl.get_unchecked(3),
                *sl.get_unchecked(4),
                *sl.get_unchecked(5),
            );
            cswap32(&mut a, &mut b);
            cswap32(&mut b, &mut c);
            cswap32(&mut a, &mut b);
            cswap32(&mut d, &mut e);
            cswap32(&mut e, &mut f);
            cswap32(&mut d, &mut e);
            cswap32(&mut a, &mut d);
            cswap32(&mut b, &mut e);
            cswap32(&mut c, &mut f);
            cswap32(&mut b, &mut d);
            cswap32(&mut c, &mut e);
            cswap32(&mut c, &mut d);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            *sl.get_unchecked_mut(3) = d;
            *sl.get_unchecked_mut(4) = e;
            *sl.get_unchecked_mut(5) = f;
            return;
        }
        if n == 7 {
            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g) = (
                *sl.get_unchecked(0),
                *sl.get_unchecked(1),
                *sl.get_unchecked(2),
                *sl.get_unchecked(3),
                *sl.get_unchecked(4),
                *sl.get_unchecked(5),
                *sl.get_unchecked(6),
            );
            cswap32(&mut a, &mut g);
            cswap32(&mut c, &mut d);
            cswap32(&mut e, &mut f);
            cswap32(&mut a, &mut c);
            cswap32(&mut b, &mut e);
            cswap32(&mut d, &mut g);
            cswap32(&mut a, &mut b);
            cswap32(&mut c, &mut f);
            cswap32(&mut d, &mut e);
            cswap32(&mut b, &mut c);
            cswap32(&mut e, &mut g);
            cswap32(&mut c, &mut d);
            cswap32(&mut e, &mut f);
            cswap32(&mut b, &mut c);
            cswap32(&mut d, &mut e);
            cswap32(&mut f, &mut g);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            *sl.get_unchecked_mut(3) = d;
            *sl.get_unchecked_mut(4) = e;
            *sl.get_unchecked_mut(5) = f;
            *sl.get_unchecked_mut(6) = g;
            return;
        }
        if n == 8 {
            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h) = (
                *sl.get_unchecked(0),
                *sl.get_unchecked(1),
                *sl.get_unchecked(2),
                *sl.get_unchecked(3),
                *sl.get_unchecked(4),
                *sl.get_unchecked(5),
                *sl.get_unchecked(6),
                *sl.get_unchecked(7),
            );
            cswap32(&mut a, &mut b);
            cswap32(&mut c, &mut d);
            cswap32(&mut e, &mut f);
            cswap32(&mut g, &mut h);
            cswap32(&mut a, &mut c);
            cswap32(&mut b, &mut d);
            cswap32(&mut e, &mut g);
            cswap32(&mut f, &mut h);
            cswap32(&mut b, &mut c);
            cswap32(&mut f, &mut g);
            cswap32(&mut a, &mut e);
            cswap32(&mut d, &mut h);
            cswap32(&mut b, &mut f);
            cswap32(&mut c, &mut g);
            cswap32(&mut b, &mut e);
            cswap32(&mut d, &mut g);
            cswap32(&mut c, &mut e);
            cswap32(&mut d, &mut f);
            cswap32(&mut d, &mut e);
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
            *sl.get_unchecked_mut(3) = d;
            *sl.get_unchecked_mut(4) = e;
            *sl.get_unchecked_mut(5) = f;
            *sl.get_unchecked_mut(6) = g;
            *sl.get_unchecked_mut(7) = h;
            return;
        }
    }
    for i in 1..n {
        let x = unsafe { *sl.get_unchecked(i) };
        let mut j = i;
        while j > 0 {
            let prev = unsafe { *sl.get_unchecked(j - 1) };
            if prev <= x {
                break;
            }
            unsafe { *sl.get_unchecked_mut(j) = prev };
            j -= 1;
        }
        unsafe { *sl.get_unchecked_mut(j) = x };
    }
}

#[inline(always)]
fn sort_person_events(sl: &mut [Ev8]) {
    let n = sl.len();
    if n < 2 {
        return;
    }
    if n == 2 {
        unsafe {
            let a = *sl.get_unchecked(0);
            let b = *sl.get_unchecked(1);
            if a.0 > b.0 {
                *sl.get_unchecked_mut(0) = b;
                *sl.get_unchecked_mut(1) = a;
            }
        }
        return;
    }
    if n == 3 {
        unsafe {
            let mut a = *sl.get_unchecked(0);
            let mut b = *sl.get_unchecked(1);
            let mut c = *sl.get_unchecked(2);
            if a.0 > b.0 {
                std::mem::swap(&mut a, &mut b);
            }
            if b.0 > c.0 {
                std::mem::swap(&mut b, &mut c);
            }
            if a.0 > b.0 {
                std::mem::swap(&mut a, &mut b);
            }
            *sl.get_unchecked_mut(0) = a;
            *sl.get_unchecked_mut(1) = b;
            *sl.get_unchecked_mut(2) = c;
        }
        return;
    }
    for i in 1..n {
        let x = unsafe { *sl.get_unchecked(i) };
        let mut j = i;
        while j > 0 {
            let prev = unsafe { *sl.get_unchecked(j - 1) };
            if prev.0 <= x.0 {
                break;
            }
            unsafe { *sl.get_unchecked_mut(j) = prev };
            j -= 1;
        }
        unsafe { *sl.get_unchecked_mut(j) = x };
    }
}

#[inline(always)]
fn click_gt(a: &Cl24, b: &Cl24) -> bool {
    a.ts > b.ts || (a.ts == b.ts && a.cid.cmp_bytes(&b.cid) == std::cmp::Ordering::Greater)
}

#[inline(always)]
fn sort_person_clicks_by_ts(sl: &mut [Cl24], tmp: &mut Vec<Cl24>) {
    let n = sl.len();
    if n <= 20 {
        for i in 1..n {
            let x = unsafe { *sl.get_unchecked(i) };
            let xts = x.ts;
            let mut j = i;
            while j > 0 {
                let prev = unsafe { *sl.get_unchecked(j - 1) };
                if prev.ts <= xts {
                    break;
                }
                unsafe { *sl.get_unchecked_mut(j) = prev };
                j -= 1;
            }
            unsafe { *sl.get_unchecked_mut(j) = x };
        }
        return;
    }
    if n > 128 {
        radix_by_u32(sl, tmp, |c| c.ts);
    } else {
        sl.sort_unstable_by_key(|c| c.ts);
    }
}

/// 16-byte inline person-intern slot: single cacheline access gives key, pid, and count.
#[derive(Clone, Copy)]
#[repr(C)]
struct HtEnt {
    key: Key8,
    pid: u32,
    cnt: u32,
}

const _: () = assert!(std::mem::size_of::<HtEnt>() == 16);

/// 8-byte linked non-purchase event for hash-free Tier-2 solve.
#[derive(Clone, Copy)]
#[repr(C)]
struct LEv8 {
    lpid: u32,
    ts: u32,
}
const _: () = assert!(std::mem::size_of::<LEv8>() == 8);

/// 16-byte linked purchase event for hash-free Tier-2 solve (~3% of linked events).
#[derive(Clone, Copy)]
#[repr(C)]
struct LPurch {
    lpid: u32,
    ts: u32,
    pc: u32,
    _pad: u32,
}
const _: () = assert!(std::mem::size_of::<LPurch>() == 16);

/// 16-byte linked click for hash-free Tier-2 solve.
/// Bit 31 of `packed` == 0: `lpid = packed >> 16`, `camp = packed & 0xFFFF`.
/// Bit 31 of `packed` == 1: `(lpid, camp) = wide_cl[(packed & 0x7FFF_FFFF) as usize]`.
#[derive(Clone, Copy)]
#[repr(C)]
struct LClick {
    cid: CKey,
    ts: u32,
    packed: u32,
}
const _: () = assert!(std::mem::size_of::<LClick>() == 16);

#[cold]
#[inline(never)]
fn push_wide_cl(wide: &Mutex<Vec<(u32, u32)>>, lpid: u32, camp: u32) -> u32 {
    let mut g = wide.lock().unwrap();
    let idx = g.len() as u32;
    g.push((lpid, camp));
    0x8000_0000 | idx
}

#[inline(always)]
fn pack_lclick(lpid: u32, camp: u32, wide: &Mutex<Vec<(u32, u32)>>) -> u32 {
    if lpid < 32_768 && camp < 65_536 {
        (lpid << 16) | camp
    } else {
        push_wide_cl(wide, lpid, camp)
    }
}

#[inline(always)]
fn unpack_lclick(packed: u32, wide: &[(u32, u32)]) -> (u32, u32) {
    if packed & 0x8000_0000 == 0 {
        (packed >> 16, packed & 0xFFFF)
    } else {
        unsafe { *wide.get_unchecked((packed & 0x7FFF_FFFF) as usize) }
    }
}

/// Map dense canonical id `cid` -> `(tier2_shard, local_pid)` with 8-key (64B cacheline) striping
/// so `links.canon_key(cid)` reads sequential 64B lines in Tier 2 while balancing every shard ±8.
#[inline(always)]
fn cid_to_shard_lpid(cid: u32, l_bits: u32, l_mask: usize) -> (usize, u32) {
    let sh = ((cid >> 3) as usize) & l_mask;
    let lpid = ((cid >> (3 + l_bits)) << 3) | (cid & 7);
    (sh, lpid)
}

#[inline(always)]
fn shard_lpid_to_cid(sh: usize, lpid: u32, l_bits: u32) -> u32 {
    ((lpid & !7) << l_bits) | ((sh as u32) << 3) | (lpid & 7)
}

#[inline(always)]
fn shard_local_np(n_canon: usize, sh: usize, l_bits: u32) -> usize {
    let super_bits = l_bits + 3;
    let full = n_canon >> super_bits;
    let rem = n_canon & ((1usize << super_bits) - 1);
    (full << 3) + rem.saturating_sub(sh << 3).min(8)
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawPurch {
    pid: u32,
    ts: u32,
    pc: u32,
}

/// Scratch buffers reused across all shards handled by one solve worker.
impl Drop for Worker<'_> {
    fn drop(&mut self) {
        if let Some(blk) = self.acc_block.take() {
            arena::put_blocks(&[blk]);
        }
    }
}

struct SolveScratch {
    /// Pooled 2 MB THP block shared by `ht` (512 KB) + `ev_ts` (640 KB) + `ev_pids16` (256 KB) + `cls` (640 KB).
    /// Keeps all 4 hot per-shard tables in ONE L1 dTLB entry and cuts solve THP blocks 7->2 per worker!
    scratch_block: Option<(usize, u32)>,
    ht: HugeBuf<HtEnt>,
    ht_mask: usize,
    pkeys: Vec<(Key8, u64)>,
    canon: Vec<u32>,
    link_probe: Vec<(u32, u64)>,
    miss_clicks: Vec<(Click, u64)>,
    miss_probe: Vec<u32>,
    ev_off: Vec<u32>,
    ev_cur: Vec<u32>,
    ev_pids: Vec<u32>,
    ev_pids16: HugeBuf<u16>,
    ev_ts: HugeBuf<u32>,
    raw_purch: Vec<RawPurch>,
    p_off: Vec<u32>,
    p_cur: Vec<u32>,
    purch: Vec<(u32, u32)>,
    cl_off: Vec<u32>,
    cl_cur: Vec<u32>,
    cl_pids: Vec<u32>,
    cl_pids16: Vec<u16>,
    cls: HugeBuf<Cl24>,
    ctmp: Vec<Cl24>,
}

impl Drop for SolveScratch {
    fn drop(&mut self) {
        if let Some(blk) = self.scratch_block.take() {
            arena::put_blocks(&[blk]);
        }
    }
}

impl SolveScratch {
    fn new() -> Self {
        let (addr, id) = arena::get_block();
        // Layout inside 2 MB block (512 + 640 + 256 + 640 = 2048 KB == 2 MB):
        // [0..512KB): ht (32,768 HtEnt); [512..1152KB): ev_ts (163,840 u32);
        // [1152..1408KB): ev_pids16 (131,072 u16); [1408..2048KB): cls (40,960 Cl24).
        let ht = unsafe { HugeBuf::from_borrowed(addr as *mut HtEnt, 32_768) };
        let ev_ts = unsafe { HugeBuf::from_borrowed((addr + (512 << 10)) as *mut u32, 163_840) };
        let ev_pids16 = unsafe { HugeBuf::from_borrowed((addr + (1152 << 10)) as *mut u16, 131_072) };
        let cls = unsafe { HugeBuf::from_borrowed((addr + (1408 << 10)) as *mut Cl24, 40_960) };
        SolveScratch {
            scratch_block: Some((addr, id)),
            ht,
            ht_mask: 0,
            pkeys: Vec::new(),
            canon: Vec::new(),
            link_probe: Vec::new(),
            miss_clicks: Vec::new(),
            miss_probe: Vec::new(),
            ev_off: Vec::new(),
            ev_cur: Vec::new(),
            ev_pids: Vec::new(),
            ev_pids16,
            ev_ts,
            raw_purch: Vec::new(),
            p_off: Vec::new(),
            p_cur: Vec::new(),
            purch: Vec::new(),
            cl_off: Vec::new(),
            cl_cur: Vec::new(),
            cl_pids: Vec::new(),
            cl_pids16: Vec::new(),
            cls,
            ctmp: Vec::new(),
        }
    }

    #[inline(always)]
    fn group_purchases(&mut self, np: usize) -> usize {
        let npurch = self.raw_purch.len();
        if npurch == 0 {
            return 0;
        }
        self.p_off.clear();
        self.p_off.resize(np + 1, 0);
        for rp in &self.raw_purch {
            unsafe { *self.p_off.get_unchecked_mut(rp.pid as usize) += 1 };
        }
        let mut psum = 0u32;
        self.p_cur.clear();
        self.p_cur.reserve(np);
        let pcur_out = self.p_cur.as_mut_ptr();
        for i in 0..np {
            let cnt = unsafe { *self.p_off.get_unchecked(i) };
            unsafe {
                *self.p_off.get_unchecked_mut(i) = psum;
                pcur_out.add(i).write(psum);
            }
            psum += cnt;
        }
        unsafe {
            self.p_cur.set_len(np);
            *self.p_off.get_unchecked_mut(np) = psum;
        }
        self.purch.clear();
        self.purch.reserve(npurch);
        let po = self.purch.as_mut_ptr();
        for rp in &self.raw_purch {
            unsafe {
                let dst = self.p_cur.get_unchecked_mut(rp.pid as usize);
                po.add(*dst as usize).write((rp.ts, rp.pc));
                *dst += 1;
            }
        }
        unsafe { self.purch.set_len(npurch) };
        npurch
    }

    #[inline(always)]
    fn solve_person(
        pkey: Key8,
        ts_slice: &mut [u32],
        purch_slice: &[(u32, u32)],
        cls_slice: &mut [Cl24],
        w: &mut Worker<'_>,
        ctmp: &mut Vec<Cl24>,
    ) {
        let tot = ts_slice.len() + purch_slice.len();
        w.begin_person(pkey);
        if tot >= BIN_AGG_MIN_EVENTS {
            w.person_binned(ts_slice, purch_slice, cls_slice, ctmp);
        } else if purch_slice.is_empty() {
            sort_person_ts(ts_slice);
            w.person_no_purch(ts_slice, cls_slice, ctmp);
        } else {
            let mut tmp = [Ev8(0); BIN_AGG_MIN_EVENTS];
            let mut k = 0usize;
            for &ts in ts_slice.iter() {
                unsafe { *tmp.get_unchecked_mut(k) = Ev8::new(ts, 0) };
                k += 1;
            }
            for &(ts, pc) in purch_slice {
                unsafe { *tmp.get_unchecked_mut(k) = Ev8::new(ts, pc) };
                k += 1;
            }
            let sl = unsafe { tmp.get_unchecked_mut(..tot) };
            sort_person_events(sl);
            w.person(sl, cls_slice, ctmp);
        }
        w.end_person();
    }

    fn reset_ht(&mut self, cap: usize) {
        self.ht.reset_zeroed(cap);
        self.ht_mask = cap - 1;
        // Pre-reserve pid=0 for `Key8::TOMBSTONE` so deduplicated-away `PEv`s route
        // branchlessly into slot 0 in both Pass 1 and Pass 2 (zero unpredictable branches!).
        let th = Key8::TOMBSTONE.hash(SEED_A) as usize & self.ht_mask;
        self.ht[th] = HtEnt { key: Key8::TOMBSTONE, pid: 0, cnt: 0 };
    }

    #[inline(always)]
    fn intern(&mut self, person: Key8, h: u64) -> u32 {
        let mask = self.ht_mask;
        let mut i = h as usize & mask;
        loop {
            let e = unsafe { self.ht.get_unchecked_mut(i) };
            if e.key == person {
                e.cnt += 1;
                return e.pid;
            }
            if e.key.is_empty() {
                let pid = self.pkeys.len() as u32;
                self.pkeys.push((person, h));
                *e = HtEnt { key: person, pid, cnt: 1 };
                if self.pkeys.len() * 2 >= self.ht.len() {
                    self.grow_ht();
                }
                return pid;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline(always)]
    fn intern_no_inc(&mut self, person: Key8, h: u64) -> u32 {
        let mask = self.ht_mask;
        let mut i = h as usize & mask;
        loop {
            let e = unsafe { self.ht.get_unchecked_mut(i) };
            if e.key == person {
                return e.pid;
            }
            if e.key.is_empty() {
                let pid = self.pkeys.len() as u32;
                self.pkeys.push((person, h));
                *e = HtEnt { key: person, pid, cnt: 0 };
                if self.pkeys.len() * 2 >= self.ht.len() {
                    self.grow_ht();
                }
                return pid;
            }
            i = (i + 1) & mask;
        }
    }

    #[inline(always)]
    fn ingest_purch_segs(&mut self, ps: Segs<PPurch>) {
        self.raw_purch.clear();
        for seg in ps.iter() {
            for p in seg {
                if p.person == Key8::TOMBSTONE {
                    continue;
                }
                let h = p.person.hash(SEED_A);
                let pid = self.intern_no_inc(p.person, h);
                self.raw_purch.push(RawPurch { pid, ts: p.ts, pc: p.pc });
            }
        }
        ps.free();
    }

    #[cold]
    #[inline(never)]
    fn grow_ht(&mut self) {
        let new_cap = (self.ht.len() * 2).max(256);
        let old = std::mem::replace(&mut self.ht, HugeBuf::new_zeroed(new_cap));
        self.ht_mask = new_cap - 1;
        let mask = self.ht_mask;
        for &e in old.iter() {
            if !e.key.is_empty() {
                let mut i = e.key.hash(SEED_A) as usize & mask;
                while !self.ht[i].key.is_empty() {
                    i = (i + 1) & mask;
                }
                self.ht[i] = e;
            }
        }
    }

    #[inline(always)]
    fn lookup(&self, person: Key8, phash: u64) -> Option<u32> {
        let mask = self.ht_mask;
        let mut i = phash as usize & mask;
        loop {
            let e = unsafe { self.ht.get_unchecked(i) };
            if e.key == person {
                return Some(e.pid);
            }
            if e.key.is_empty() {
                return None;
            }
            i = (i + 1) & mask;
        }
    }

    fn run_shard(&mut self, es: Segs<PEv12>, ps: Segs<PPurch>, cs: Segs<Click>, w: &mut Worker<'_>) {
        let nev = es.len();
        let npr = ps.len();
        if nev == 0 && npr == 0 {
            es.free();
            ps.free();
            cs.free();
            return;
        }
        let init_cap = (((nev + npr) / 8).next_power_of_two()).clamp(256, 8_192);
        self.reset_ht(init_cap);
        self.pkeys.clear();
        self.pkeys.push((Key8::TOMBSTONE, 0));
        self.ev_pids.clear();
        self.ev_pids.reserve(nev);

        // Pass 1 over non-purchase events: intern exact Key8 persons in inline 16B HtEnt with 8-ahead prefetch
        const PF: usize = 8;
        let mut hring = [0u64; PF];
        let pid_out = self.ev_pids.as_mut_ptr();
        let mut pk = 0usize;
        for seg in es.iter() {
            let slen = seg.len();
            let pre = slen.min(PF);
            for ri in 0..pre {
                let person = unsafe { seg.get_unchecked(ri).person };
                let h = person.hash(SEED_A);
                hring[ri] = h;
                prefetch(unsafe { self.ht.as_ptr().add(h as usize & self.ht_mask) } as *const u8);
            }
            for ri in 0..slen {
                let h = hring[ri & (PF - 1)];
                if ri + PF < slen {
                    let fperson = unsafe { seg.get_unchecked(ri + PF).person };
                    let fh = fperson.hash(SEED_A);
                    hring[(ri + PF) & (PF - 1)] = fh;
                    prefetch(unsafe { self.ht.as_ptr().add(fh as usize & self.ht_mask) } as *const u8);
                }
                let person = unsafe { seg.get_unchecked(ri).person };
                let pid = self.intern(person, h);
                unsafe { pid_out.add(pk).write(pid) };
                pk += 1;
            }
        }
        unsafe { self.ev_pids.set_len(nev) };
        self.ingest_purch_segs(ps);
        let np = self.pkeys.len();
        if np <= 1 {
            es.free();
            cs.free();
            return;
        }

        // Populate per-person counts branchlessly (HT_EMPTY has pid=0, cnt=0 -> no-op!)
        self.ev_off.clear();
        self.ev_off.resize(np + 1, 0);
        let eoff_ptr = self.ev_off.as_mut_ptr();
        for e in self.ht.iter() {
            unsafe { *eoff_ptr.add(e.pid as usize) += e.cnt };
        }
        let mut sum = 0u32;
        self.ev_cur.clear();
        self.ev_cur.reserve(np);
        let cur_out = self.ev_cur.as_mut_ptr();
        for i in 0..np {
            let cnt = unsafe { *self.ev_off.get_unchecked(i) };
            unsafe {
                *self.ev_off.get_unchecked_mut(i) = sum;
                cur_out.add(i).write(sum);
            }
            sum += cnt;
        }
        unsafe {
            self.ev_cur.set_len(np);
            *self.ev_off.get_unchecked_mut(np) = sum;
        }

        // Pass 2 over non-purchase events: 100% branchless scatter of 4-byte ts into L2 buffer!
        self.ev_ts.clear();
        self.ev_ts.reserve_uninit(nev);
        let eo = self.ev_ts.as_mut_ptr();
        let pid_ptr = self.ev_pids.as_ptr();
        let cur_ptr = self.ev_cur.as_mut_ptr();
        let mut k = 0usize;
        for seg in es.iter() {
            for e in seg {
                unsafe {
                    if k + PF < nev {
                        let fpid = *pid_ptr.add(k + PF) as usize;
                        let fdst = *cur_ptr.add(fpid) as usize;
                        prefetch(eo.add(fdst) as *const u8);
                    }
                    let pid = *pid_ptr.add(k) as usize;
                    let dst = cur_ptr.add(pid);
                    eo.add(*dst as usize).write(e.ts);
                    *dst += 1;
                }
                k += 1;
            }
        }
        es.free();
        let npurch = self.group_purchases(np);

        // Clicks: lookup in the same inline HtEnt table with 8-ahead prefetch; drop clicks for 0-event persons!
        let ncl_raw = cs.len();
        self.cl_off.clear();
        self.cl_off.resize(np + 1, 0);
        self.cl_pids.clear();
        self.cl_pids.reserve(ncl_raw);
        let mut ncl_kept = 0usize;
        if ncl_raw > 0 {
            let ht_ptr = self.ht.as_ptr();
            let ht_mask = self.ht_mask;
            let clp_out = self.cl_pids.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                let slen = seg.len();
                let pre = slen.min(PF);
                for ri in 0..pre {
                    let h = unsafe { seg.get_unchecked(ri).person.hash(SEED_A) };
                    hring[ri] = h;
                    prefetch(unsafe { ht_ptr.add(h as usize & ht_mask) } as *const u8);
                }
                for ri in 0..slen {
                    let h = hring[ri & (PF - 1)];
                    if ri + PF < slen {
                        let fh = unsafe { seg.get_unchecked(ri + PF).person.hash(SEED_A) };
                        hring[(ri + PF) & (PF - 1)] = fh;
                        prefetch(unsafe { ht_ptr.add(fh as usize & ht_mask) } as *const u8);
                    }
                    let c = unsafe { seg.get_unchecked(ri) };
                    if let Some(pid) = self.lookup(c.person, h) {
                        unsafe {
                            clp_out.add(ck).write(pid);
                            *self.cl_off.get_unchecked_mut(pid as usize) += 1;
                        }
                        ncl_kept += 1;
                    } else {
                        unsafe { clp_out.add(ck).write(u32::MAX) };
                    }
                    ck += 1;
                }
            }
            unsafe { self.cl_pids.set_len(ncl_raw) };
        }

        self.cls.clear();
        if ncl_kept > 0 {
            let mut csum = 0u32;
            self.cl_cur.clear();
            self.cl_cur.reserve(np);
            let ccur_out = self.cl_cur.as_mut_ptr();
            for i in 0..np {
                let cnt = unsafe { *self.cl_off.get_unchecked(i) };
                unsafe {
                    *self.cl_off.get_unchecked_mut(i) = csum;
                    ccur_out.add(i).write(csum);
                }
                csum += cnt;
            }
            unsafe {
                self.cl_cur.set_len(np);
                *self.cl_off.get_unchecked_mut(np) = csum;
            }
            self.cls.reserve_uninit(ncl_kept);
            let co = self.cls.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                for c in seg {
                    let pid = unsafe { *self.cl_pids.get_unchecked(ck) };
                    ck += 1;
                    if pid != u32::MAX {
                        unsafe {
                            let dst = self.cl_cur.get_unchecked_mut(pid as usize);
                            co.add(*dst as usize).write(Cl24 { cid: c.cid, ts: c.ts, camp: c.camp });
                            *dst += 1;
                        }
                    }
                }
            }
            unsafe { self.cls.set_len(ncl_kept) };
        }
        cs.free();

        // Process each person cleanly in L1/L2 cache (skip pid=0 tombstone bucket)
        let eo_base = self.ev_ts.as_mut_ptr();
        for pid in 1..np {
            let es0 = unsafe { *self.ev_off.get_unchecked(pid) } as usize;
            let es1 = unsafe { *self.ev_cur.get_unchecked(pid) } as usize;
            let purch_slice: &[(u32, u32)] = if npurch > 0 {
                let ps0 = unsafe { *self.p_off.get_unchecked(pid) } as usize;
                let ps1 = unsafe { *self.p_off.get_unchecked(pid + 1) } as usize;
                unsafe { self.purch.get_unchecked(ps0..ps1) }
            } else {
                &[]
            };
            let cls_slice: &mut [Cl24] = if ncl_kept > 0 {
                let cs0 = unsafe { *self.cl_off.get_unchecked(pid) } as usize;
                let cs1 = unsafe { *self.cl_off.get_unchecked(pid + 1) } as usize;
                unsafe { self.cls.get_unchecked_mut(cs0..cs1) }
            } else {
                &mut []
            };
            let pkey = unsafe { self.pkeys.get_unchecked(pid).0 };
            let ts_slice = unsafe { std::slice::from_raw_parts_mut(eo_base.add(es0), es1 - es0) };
            Self::solve_person(pkey, ts_slice, purch_slice, cls_slice, w, &mut self.ctmp);
        }
    }

    fn flush_miss_clicks<const STAGE: usize>(
        &mut self,
        links: &LinkTable,
        lcl_w: &mut Scatter<'_, LClick, STAGE>,
        l_bits: u32,
        l_mask: usize,
        wide_camps: bool,
        wide_cl: &Mutex<Vec<(u32, u32)>>,
    ) {
        let nm = self.miss_clicks.len();
        if nm == 0 {
            return;
        }
        const PF: usize = 16;
        self.miss_probe.clear();
        self.miss_probe.reserve(nm);
        let mp_ptr = self.miss_probe.as_mut_ptr();
        let mut npr = 0usize;
        for i in 0..nm.min(PF) {
            links.prefetch_bloom(unsafe { self.miss_clicks.get_unchecked(i).1 });
        }
        for i in 0..nm {
            if i + PF < nm {
                links.prefetch_bloom(unsafe { self.miss_clicks.get_unchecked(i + PF).1 });
            }
            unsafe {
                mp_ptr.add(npr).write(i as u32);
                npr += usize::from(links.maybe(self.miss_clicks.get_unchecked(i).1));
            }
        }
        unsafe { self.miss_probe.set_len(npr) };
        for j in 0..npr.min(PF) {
            let idx = unsafe { *self.miss_probe.get_unchecked(j) as usize };
            links.prefetch(unsafe { self.miss_clicks.get_unchecked(idx).1 });
        }
        for j in 0..npr {
            if j + PF < npr {
                let fidx = unsafe { *self.miss_probe.get_unchecked(j + PF) as usize };
                links.prefetch(unsafe { self.miss_clicks.get_unchecked(fidx).1 });
            }
            let idx = unsafe { *self.miss_probe.get_unchecked(j) as usize };
            let (c, h) = unsafe { *self.miss_clicks.get_unchecked(idx) };
            if let Some(cid) = links.lookup(h, c.person) {
                let (sh, lpid) = cid_to_shard_lpid(cid, l_bits, l_mask);
                let packed = if lpid < 32_768 && !wide_camps {
                    (lpid << 16) | c.camp
                } else {
                    pack_lclick(lpid, c.camp, wide_cl)
                };
                lcl_w.push(sh, LClick { cid: c.cid, ts: c.ts, packed });
            }
        }
        self.miss_clicks.clear();
    }

    /// Two-tier solve Tier 1: raw-uid shard. Resolves each distinct uid ONCE in `links`,
    /// solves truly unlinked users immediately in place, and forwards linked users'
    /// non-purchases (`LEv8`), purchases (`LPurch`), and clicks (`LClick`) by dense `cid` to Tier 2.
    #[allow(clippy::too_many_arguments)]
    fn run_tier1_shard<const EV_STAGE: usize, const P_STAGE: usize, const CL_STAGE: usize>(
        &mut self,
        es: Segs<PEv12>,
        ps: Segs<PPurch>,
        cs: Segs<Click>,
        links: &LinkTable,
        w: &mut Worker<'_>,
        lev_w: &mut Scatter<'_, LEv8, EV_STAGE>,
        lp_w: &mut Scatter<'_, LPurch, P_STAGE>,
        lcl_w: &mut Scatter<'_, LClick, CL_STAGE>,
        l_bits: u32,
        l_mask: usize,
        wide_camps: bool,
        wide_cl: &Mutex<Vec<(u32, u32)>>,
    ) {
        if links.is_empty() {
            self.run_shard(es, ps, cs, w);
            return;
        }
        let nev = es.len();
        let npr = ps.len();
        if nev == 0 && npr == 0 {
            if cs.len() > 0 {
                self.miss_clicks.clear();
                for seg in cs.iter() {
                    for &c in seg {
                        self.miss_clicks.push((c, c.person.hash(SEED_A)));
                    }
                }
                self.flush_miss_clicks(links, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
            }
            es.free();
            ps.free();
            cs.free();
            return;
        }

        if nev + npr >= u16::MAX as usize {
            self.run_tier1_shard_wide(
                es, ps, cs, links, w, lev_w, lp_w, lcl_w, l_bits, l_mask, wide_camps, wide_cl,
            );
            return;
        }

        let init_cap = (((nev + npr) / 8).next_power_of_two()).clamp(256, 8_192);
        self.reset_ht(init_cap);
        self.pkeys.clear();
        self.pkeys.push((Key8::TOMBSTONE, 0));
        self.ev_pids16.clear();
        self.ev_pids16.reserve_uninit(nev);

        // Pass 1 over 12B non-purchase events: intern exact raw Key8 uids in inline 16B HtEnt
        const PF: usize = 8;
        let mut hring = [0u64; PF];
        let pid_out = self.ev_pids16.as_mut_ptr();
        let mut pk = 0usize;
        for seg in es.iter() {
            let slen = seg.len();
            let pre = slen.min(PF);
            for ri in 0..pre {
                let person = unsafe { seg.get_unchecked(ri).person };
                let h = person.hash(SEED_A);
                hring[ri] = h;
                prefetch(unsafe { self.ht.as_ptr().add(h as usize & self.ht_mask) } as *const u8);
            }
            let main_end = slen.saturating_sub(PF);
            for ri in 0..main_end {
                let h = hring[ri & (PF - 1)];
                let fperson = unsafe { seg.get_unchecked(ri + PF).person };
                let fh = fperson.hash(SEED_A);
                hring[(ri + PF) & (PF - 1)] = fh;
                prefetch(unsafe { self.ht.as_ptr().add(fh as usize & self.ht_mask) } as *const u8);
                let person = unsafe { seg.get_unchecked(ri).person };
                let pid = self.intern(person, h);
                unsafe { pid_out.add(pk).write(pid as u16) };
                pk += 1;
            }
            for ri in main_end..slen {
                let h = hring[ri & (PF - 1)];
                let person = unsafe { seg.get_unchecked(ri).person };
                let pid = self.intern(person, h);
                unsafe { pid_out.add(pk).write(pid as u16) };
                pk += 1;
            }
        }
        unsafe { self.ev_pids16.set_len(nev) };
        self.ingest_purch_segs(ps);
        let np = self.pkeys.len();
        if np <= 1 {
            if cs.len() > 0 {
                self.miss_clicks.clear();
                for seg in cs.iter() {
                    for &c in seg {
                        self.miss_clicks.push((c, c.person.hash(SEED_A)));
                    }
                }
                self.flush_miss_clicks(links, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
            }
            es.free();
            cs.free();
            return;
        }

        // Prefix sum counts branchlessly for ALL users while `es`, `ev_pids16`, and `ht` are L2-hot!
        self.ev_off.clear();
        self.ev_off.resize(np + 1, 0);
        let eoff_ptr = self.ev_off.as_mut_ptr();
        for e in self.ht.iter() {
            unsafe { *eoff_ptr.add(e.pid as usize) += e.cnt };
        }
        let mut sum = 0u32;
        self.ev_cur.clear();
        self.ev_cur.reserve(np);
        let cur_out = self.ev_cur.as_mut_ptr();
        for i in 0..np {
            let cnt = unsafe { *self.ev_off.get_unchecked(i) };
            unsafe {
                *self.ev_off.get_unchecked_mut(i) = sum;
                cur_out.add(i).write(sum);
            }
            sum += cnt;
        }
        unsafe {
            self.ev_cur.set_len(np);
            *self.ev_off.get_unchecked_mut(np) = sum;
        }

        // Pass 2 over 12B non-purchase events: 100% branchless scatter of 4-byte ts into L2 buffer!
        self.ev_ts.clear();
        self.ev_ts.reserve_uninit(nev);
        let eo = self.ev_ts.as_mut_ptr();
        let pid_ptr = self.ev_pids16.as_ptr();
        let cur_ptr = self.ev_cur.as_mut_ptr();
        let mut k = 0usize;
        for seg in es.iter() {
            let slen = seg.len();
            let main_end = slen.saturating_sub(PF);
            for ri in 0..main_end {
                unsafe {
                    let fpid = *pid_ptr.add(k + PF) as usize;
                    let fdst = *cur_ptr.add(fpid) as usize;
                    prefetch(eo.add(fdst) as *const u8);
                    let ts = seg.get_unchecked(ri).ts;
                    let pid = *pid_ptr.add(k) as usize;
                    let dst = cur_ptr.add(pid);
                    eo.add(*dst as usize).write(ts);
                    *dst += 1;
                }
                k += 1;
            }
            for ri in main_end..slen {
                unsafe {
                    let ts = seg.get_unchecked(ri).ts;
                    let pid = *pid_ptr.add(k) as usize;
                    let dst = cur_ptr.add(pid);
                    eo.add(*dst as usize).write(ts);
                    *dst += 1;
                }
                k += 1;
            }
        }
        es.free();
        let npurch = self.group_purchases(np);

        // Clicks: group all in-ht clicks by u16 pid while `ht` is STILL L2-hot!
        let ncl_raw = cs.len();
        self.cl_off.clear();
        self.cl_off.resize(np + 1, 0);
        self.cl_pids16.clear();
        self.cl_pids16.reserve(ncl_raw);
        self.miss_clicks.clear();
        let mut ncl_kept = 0usize;
        if ncl_raw > 0 {
            let ht_ptr = self.ht.as_ptr();
            let ht_mask = self.ht_mask;
            let clp_out = self.cl_pids16.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                let slen = seg.len();
                let pre = slen.min(PF);
                for ri in 0..pre {
                    let h = unsafe { seg.get_unchecked(ri).person.hash(SEED_A) };
                    hring[ri] = h;
                    prefetch(unsafe { ht_ptr.add(h as usize & ht_mask) } as *const u8);
                }
                let main_end = slen.saturating_sub(PF);
                for ri in 0..main_end {
                    let h = hring[ri & (PF - 1)];
                    let fh = unsafe { seg.get_unchecked(ri + PF).person.hash(SEED_A) };
                    hring[(ri + PF) & (PF - 1)] = fh;
                    prefetch(unsafe { ht_ptr.add(fh as usize & ht_mask) } as *const u8);
                    let c = unsafe { seg.get_unchecked(ri) };
                    if let Some(pid) = self.lookup(c.person, h) {
                        unsafe {
                            clp_out.add(ck).write(pid as u16);
                            *self.cl_off.get_unchecked_mut(pid as usize) += 1;
                        }
                        ncl_kept += 1;
                    } else {
                        unsafe { clp_out.add(ck).write(u16::MAX) };
                        self.miss_clicks.push((*c, h));
                    }
                    ck += 1;
                }
                for ri in main_end..slen {
                    let h = hring[ri & (PF - 1)];
                    let c = unsafe { seg.get_unchecked(ri) };
                    if let Some(pid) = self.lookup(c.person, h) {
                        unsafe {
                            clp_out.add(ck).write(pid as u16);
                            *self.cl_off.get_unchecked_mut(pid as usize) += 1;
                        }
                        ncl_kept += 1;
                    } else {
                        unsafe { clp_out.add(ck).write(u16::MAX) };
                        self.miss_clicks.push((*c, h));
                    }
                    ck += 1;
                }
            }
            unsafe { self.cl_pids16.set_len(ncl_raw) };
        }

        self.cls.clear();
        if ncl_kept > 0 {
            let mut csum = 0u32;
            self.cl_cur.clear();
            self.cl_cur.reserve(np);
            let ccur_out = self.cl_cur.as_mut_ptr();
            for i in 0..np {
                let cnt = unsafe { *self.cl_off.get_unchecked(i) };
                unsafe {
                    *self.cl_off.get_unchecked_mut(i) = csum;
                    ccur_out.add(i).write(csum);
                }
                csum += cnt;
            }
            unsafe {
                self.cl_cur.set_len(np);
                *self.cl_off.get_unchecked_mut(np) = csum;
            }
            self.cls.reserve_uninit(ncl_kept);
            let co = self.cls.as_mut_ptr();
            let clp_ptr = self.cl_pids16.as_ptr();
            let ccur_ptr = self.cl_cur.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                let slen = seg.len();
                let main_end = slen.saturating_sub(PF);
                for ri in 0..main_end {
                    unsafe {
                        let fpid = *clp_ptr.add(ck + PF);
                        if fpid != u16::MAX {
                            let fdst = *ccur_ptr.add(fpid as usize) as usize;
                            prefetch(co.add(fdst) as *const u8);
                        }
                        let pid = *clp_ptr.add(ck);
                        if pid != u16::MAX {
                            let c = seg.get_unchecked(ri);
                            let dst = ccur_ptr.add(pid as usize);
                            co.add(*dst as usize).write(Cl24 { cid: c.cid, ts: c.ts, camp: c.camp });
                            *dst += 1;
                        }
                    }
                    ck += 1;
                }
                for ri in main_end..slen {
                    unsafe {
                        let pid = *clp_ptr.add(ck);
                        if pid != u16::MAX {
                            let c = seg.get_unchecked(ri);
                            let dst = ccur_ptr.add(pid as usize);
                            co.add(*dst as usize).write(Cl24 { cid: c.cid, ts: c.ts, camp: c.camp });
                            *dst += 1;
                        }
                    }
                    ck += 1;
                }
            }
            unsafe { self.cls.set_len(ncl_kept) };
        }
        cs.free();
        self.finish_tier1_shard(np, npurch, ncl_kept, links, w, lev_w, lp_w, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
    }

    #[cold]
    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn run_tier1_shard_wide<const EV_STAGE: usize, const P_STAGE: usize, const CL_STAGE: usize>(
        &mut self,
        es: Segs<PEv12>,
        ps: Segs<PPurch>,
        cs: Segs<Click>,
        links: &LinkTable,
        w: &mut Worker<'_>,
        lev_w: &mut Scatter<'_, LEv8, EV_STAGE>,
        lp_w: &mut Scatter<'_, LPurch, P_STAGE>,
        lcl_w: &mut Scatter<'_, LClick, CL_STAGE>,
        l_bits: u32,
        l_mask: usize,
        wide_camps: bool,
        wide_cl: &Mutex<Vec<(u32, u32)>>,
    ) {
        let nev = es.len();
        let npr = ps.len();
        let init_cap = (((nev + npr) / 8).next_power_of_two()).clamp(256, 8_192);
        self.reset_ht(init_cap);
        self.pkeys.clear();
        self.pkeys.push((Key8::TOMBSTONE, 0));
        self.ev_pids.clear();
        self.ev_pids.reserve(nev);

        let pid_out = self.ev_pids.as_mut_ptr();
        let mut pk = 0usize;
        for seg in es.iter() {
            for e in seg {
                let person = e.person;
                let h = person.hash(SEED_A);
                let pid = self.intern(person, h);
                unsafe { pid_out.add(pk).write(pid) };
                pk += 1;
            }
        }
        unsafe { self.ev_pids.set_len(nev) };
        self.ingest_purch_segs(ps);
        let np = self.pkeys.len();
        if np <= 1 {
            if cs.len() > 0 {
                self.miss_clicks.clear();
                for seg in cs.iter() {
                    for &c in seg {
                        self.miss_clicks.push((c, c.person.hash(SEED_A)));
                    }
                }
                self.flush_miss_clicks(links, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
            }
            es.free();
            cs.free();
            return;
        }
        self.ev_off.clear();
        self.ev_off.resize(np + 1, 0);
        let eoff_ptr = self.ev_off.as_mut_ptr();
        for e in self.ht.iter() {
            unsafe { *eoff_ptr.add(e.pid as usize) += e.cnt };
        }
        let mut sum = 0u32;
        self.ev_cur.clear();
        self.ev_cur.reserve(np);
        let cur_out = self.ev_cur.as_mut_ptr();
        for i in 0..np {
            let cnt = unsafe { *self.ev_off.get_unchecked(i) };
            unsafe {
                *self.ev_off.get_unchecked_mut(i) = sum;
                cur_out.add(i).write(sum);
            }
            sum += cnt;
        }
        unsafe {
            self.ev_cur.set_len(np);
            *self.ev_off.get_unchecked_mut(np) = sum;
        }
        self.ev_ts.clear();
        self.ev_ts.reserve_uninit(nev);
        let eo = self.ev_ts.as_mut_ptr();
        let pid_ptr = self.ev_pids.as_ptr();
        let cur_ptr = self.ev_cur.as_mut_ptr();
        let mut k = 0usize;
        for seg in es.iter() {
            for e in seg {
                unsafe {
                    let pid = *pid_ptr.add(k) as usize;
                    let dst = cur_ptr.add(pid);
                    eo.add(*dst as usize).write(e.ts);
                    *dst += 1;
                }
                k += 1;
            }
        }
        es.free();
        let npurch = self.group_purchases(np);

        let ncl_raw = cs.len();
        self.cl_off.clear();
        self.cl_off.resize(np + 1, 0);
        self.cl_pids.clear();
        self.cl_pids.reserve(ncl_raw);
        self.miss_clicks.clear();
        let mut ncl_kept = 0usize;
        if ncl_raw > 0 {
            let clp_out = self.cl_pids.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                for c in seg {
                    let h = c.person.hash(SEED_A);
                    if let Some(pid) = self.lookup(c.person, h) {
                        unsafe {
                            clp_out.add(ck).write(pid);
                            *self.cl_off.get_unchecked_mut(pid as usize) += 1;
                        }
                        ncl_kept += 1;
                    } else {
                        unsafe { clp_out.add(ck).write(u32::MAX) };
                        self.miss_clicks.push((*c, h));
                    }
                    ck += 1;
                }
            }
            unsafe { self.cl_pids.set_len(ncl_raw) };
        }
        self.cls.clear();
        if ncl_kept > 0 {
            let mut csum = 0u32;
            self.cl_cur.clear();
            self.cl_cur.reserve(np);
            let ccur_out = self.cl_cur.as_mut_ptr();
            for i in 0..np {
                let cnt = unsafe { *self.cl_off.get_unchecked(i) };
                unsafe {
                    *self.cl_off.get_unchecked_mut(i) = csum;
                    ccur_out.add(i).write(csum);
                }
                csum += cnt;
            }
            unsafe {
                self.cl_cur.set_len(np);
                *self.cl_off.get_unchecked_mut(np) = csum;
            }
            self.cls.reserve_uninit(ncl_kept);
            let co = self.cls.as_mut_ptr();
            let mut ck = 0usize;
            for seg in cs.iter() {
                for c in seg {
                    let pid = unsafe { *self.cl_pids.get_unchecked(ck) };
                    ck += 1;
                    if pid != u32::MAX {
                        unsafe {
                            let dst = self.cl_cur.get_unchecked_mut(pid as usize);
                            co.add(*dst as usize).write(Cl24 { cid: c.cid, ts: c.ts, camp: c.camp });
                            *dst += 1;
                        }
                    }
                }
            }
            unsafe { self.cls.set_len(ncl_kept) };
        }
        cs.free();
        self.finish_tier1_shard(np, npurch, ncl_kept, links, w, lev_w, lp_w, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn finish_tier1_shard<const EV_STAGE: usize, const P_STAGE: usize, const CL_STAGE: usize>(
        &mut self,
        np: usize,
        npurch: usize,
        ncl_kept: usize,
        links: &LinkTable,
        w: &mut Worker<'_>,
        lev_w: &mut Scatter<'_, LEv8, EV_STAGE>,
        lp_w: &mut Scatter<'_, LPurch, P_STAGE>,
        lcl_w: &mut Scatter<'_, LClick, CL_STAGE>,
        l_bits: u32,
        l_mask: usize,
        wide_camps: bool,
        wide_cl: &Mutex<Vec<(u32, u32)>>,
    ) {
        const PF: usize = 16;
        // Batch all LinkTable DRAM probes AFTER es/ht/cs are done (zero L2 eviction of es/ht!).
        if !self.miss_clicks.is_empty() {
            self.flush_miss_clicks(links, lcl_w, l_bits, l_mask, wide_camps, wide_cl);
        }
        self.canon.clear();
        self.canon.resize(np, u32::MAX);
        self.link_probe.clear();
        self.link_probe.reserve(np);
        let lp_ptr = self.link_probe.as_mut_ptr();
        let mut npr = 0usize;
        let pre = np.min(1 + PF);
        for pid in 1..pre {
            links.prefetch_bloom(unsafe { self.pkeys.get_unchecked(pid).1 });
        }
        for pid in 1..np {
            if pid + PF < np {
                links.prefetch_bloom(unsafe { self.pkeys.get_unchecked(pid + PF).1 });
            }
            let h = unsafe { self.pkeys.get_unchecked(pid).1 };
            unsafe {
                lp_ptr.add(npr).write((pid as u32, h));
                npr += usize::from(links.maybe(h));
            }
        }
        unsafe { self.link_probe.set_len(npr) };
        for j in 0..npr.min(PF) {
            links.prefetch(unsafe { self.link_probe.get_unchecked(j).1 });
        }
        for j in 0..npr {
            if j + PF < npr {
                links.prefetch(unsafe { self.link_probe.get_unchecked(j + PF).1 });
            }
            let (pid, h) = unsafe { *self.link_probe.get_unchecked(j) };
            let uid = unsafe { self.pkeys.get_unchecked(pid as usize).0 };
            if let Some(cid) = links.lookup(h, uid) {
                unsafe { *self.canon.get_unchecked_mut(pid as usize) = cid };
            }
        }

        // Per distinct user (skip pid=0 tombstone bucket): solve unlinked in place, or forward to Tier 2!
        let eo_base = self.ev_ts.as_mut_ptr();
        for pid in 1..np {
            let es0 = unsafe { *self.ev_off.get_unchecked(pid) } as usize;
            let es1 = unsafe { *self.ev_cur.get_unchecked(pid) } as usize;
            let purch_slice: &[(u32, u32)] = if npurch > 0 {
                let ps0 = unsafe { *self.p_off.get_unchecked(pid) } as usize;
                let ps1 = unsafe { *self.p_off.get_unchecked(pid + 1) } as usize;
                unsafe { self.purch.get_unchecked(ps0..ps1) }
            } else {
                &[]
            };
            let cid = unsafe { *self.canon.get_unchecked(pid) };
            if cid == u32::MAX {
                let cls_slice: &mut [Cl24] = if ncl_kept > 0 {
                    let cs0 = unsafe { *self.cl_off.get_unchecked(pid) } as usize;
                    let cs1 = unsafe { *self.cl_off.get_unchecked(pid + 1) } as usize;
                    unsafe { self.cls.get_unchecked_mut(cs0..cs1) }
                } else {
                    &mut []
                };
                let pkey = unsafe { self.pkeys.get_unchecked(pid).0 };
                let ts_slice = unsafe { std::slice::from_raw_parts_mut(eo_base.add(es0), es1 - es0) };
                Self::solve_person(pkey, ts_slice, purch_slice, cls_slice, w, &mut self.ctmp);
            } else {
                let (sh, lpid) = cid_to_shard_lpid(cid, l_bits, l_mask);
                let ts_slice = unsafe { std::slice::from_raw_parts(eo_base.add(es0), es1 - es0) };
                if !ts_slice.is_empty() {
                    lev_w.push_mapped(sh, ts_slice, |ts| LEv8 { lpid, ts });
                }
                if !purch_slice.is_empty() {
                    lp_w.push_mapped(sh, purch_slice, |(ts, pc)| LPurch { lpid, ts, pc, _pad: 0 });
                }
                if ncl_kept > 0 {
                    let cs0 = unsafe { *self.cl_off.get_unchecked(pid) } as usize;
                    let cs1 = unsafe { *self.cl_off.get_unchecked(pid + 1) } as usize;
                    if cs0 < cs1 {
                        let cls_s = unsafe { self.cls.get_unchecked(cs0..cs1) };
                        if lpid < 32_768 && !wide_camps {
                            let lpid_hi = lpid << 16;
                            lcl_w.push_mapped(sh, cls_s, |c| LClick {
                                cid: c.cid,
                                ts: c.ts,
                                packed: lpid_hi | c.camp,
                            });
                        } else {
                            lcl_w.push_mapped(sh, cls_s, |c| LClick {
                                cid: c.cid,
                                ts: c.ts,
                                packed: pack_lclick(lpid, c.camp, wide_cl),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Two-tier solve Tier 2: 100% hash-free, `ht`-free, `ev_pids`-free dense-`lpid` solve!
    #[allow(clippy::too_many_arguments)]
    fn run_tier2_shard(
        &mut self,
        sh: usize,
        l_bits: u32,
        links: &LinkTable,
        es: Segs<LEv8>,
        ps: Segs<LPurch>,
        cs: Segs<LClick>,
        wide_cl: &[(u32, u32)],
        w: &mut Worker<'_>,
    ) {
        let nev = es.len();
        let npurch = ps.len();
        if nev == 0 && npurch == 0 {
            es.free();
            ps.free();
            cs.free();
            return;
        }
        let local_np = shard_local_np(links.num_canon(), sh, l_bits);
        if local_np == 0 {
            es.free();
            ps.free();
            cs.free();
            return;
        }

        // Step 1: Count & scatter 8-byte non-purchase events directly by L1D-resident `lpid`.
        self.ev_off.clear();
        self.ev_off.resize(local_np + 1, 0);
        if nev > 0 {
            let off_ptr = self.ev_off.as_mut_ptr();
            for seg in es.iter() {
                for e in seg {
                    unsafe { *off_ptr.add(e.lpid as usize) += 1 };
                }
            }
            let mut sum = 0u32;
            self.ev_cur.clear();
            self.ev_cur.reserve(local_np);
            let cur_out = self.ev_cur.as_mut_ptr();
            for i in 0..local_np {
                let cnt = unsafe { *off_ptr.add(i) };
                unsafe {
                    *off_ptr.add(i) = sum;
                    cur_out.add(i).write(sum);
                }
                sum += cnt;
            }
            unsafe {
                self.ev_cur.set_len(local_np);
                *off_ptr.add(local_np) = sum;
            }
            self.ev_ts.clear();
            self.ev_ts.reserve_uninit(nev);
            let eo = self.ev_ts.as_mut_ptr();
            let cur_ptr = self.ev_cur.as_mut_ptr();
            const PF: usize = 8;
            for seg in es.iter() {
                let slen = seg.len();
                let main_end = slen.saturating_sub(PF);
                for ri in 0..main_end {
                    unsafe {
                        let flpid = seg.get_unchecked(ri + PF).lpid as usize;
                        let fdst = *cur_ptr.add(flpid) as usize;
                        prefetch(eo.add(fdst) as *const u8);
                        let e = seg.get_unchecked(ri);
                        let dst = cur_ptr.add(e.lpid as usize);
                        eo.add(*dst as usize).write(e.ts);
                        *dst += 1;
                    }
                }
                for ri in main_end..slen {
                    unsafe {
                        let e = seg.get_unchecked(ri);
                        let dst = cur_ptr.add(e.lpid as usize);
                        eo.add(*dst as usize).write(e.ts);
                        *dst += 1;
                    }
                }
            }
        }
        es.free();

        // Step 2: Count & scatter ~3% purchases directly by `lpid`.
        if npurch > 0 {
            self.p_off.clear();
            self.p_off.resize(local_np + 1, 0);
            let poff_ptr = self.p_off.as_mut_ptr();
            for seg in ps.iter() {
                for p in seg {
                    unsafe { *poff_ptr.add(p.lpid as usize) += 1 };
                }
            }
            let mut psum = 0u32;
            self.p_cur.clear();
            self.p_cur.reserve(local_np);
            let pcur_out = self.p_cur.as_mut_ptr();
            for i in 0..local_np {
                let cnt = unsafe { *poff_ptr.add(i) };
                unsafe {
                    *poff_ptr.add(i) = psum;
                    pcur_out.add(i).write(psum);
                }
                psum += cnt;
            }
            unsafe {
                self.p_cur.set_len(local_np);
                *poff_ptr.add(local_np) = psum;
            }
            self.purch.clear();
            self.purch.reserve(npurch);
            let po = self.purch.as_mut_ptr();
            let pcur_ptr = self.p_cur.as_mut_ptr();
            for seg in ps.iter() {
                for p in seg {
                    unsafe {
                        let dst = pcur_ptr.add(p.lpid as usize);
                        po.add(*dst as usize).write((p.ts, p.pc));
                        *dst += 1;
                    }
                }
            }
            unsafe { self.purch.set_len(npurch) };
        }
        ps.free();

        // Step 3: Count & scatter clicks directly by `lpid`, skipping 0-event persons via L1D `ev_off`/`p_off`!
        let ncl_raw = cs.len();
        self.cls.clear();
        let mut ncl_kept = 0usize;
        if ncl_raw > 0 {
            self.cl_off.clear();
            self.cl_off.resize(local_np + 1, 0);
            let coff_ptr = self.cl_off.as_mut_ptr();
            let eoff_ptr = self.ev_off.as_ptr();
            let poff_ptr = self.p_off.as_ptr();
            let has_wide = !wide_cl.is_empty();
            if !has_wide {
                for seg in cs.iter() {
                    for c in seg {
                        let lpid = (c.packed >> 16) as usize;
                        let has_ev = unsafe {
                            *eoff_ptr.add(lpid) != *eoff_ptr.add(lpid + 1)
                                || (npurch > 0 && *poff_ptr.add(lpid) != *poff_ptr.add(lpid + 1))
                        };
                        if has_ev {
                            unsafe { *coff_ptr.add(lpid) += 1 };
                            ncl_kept += 1;
                        }
                    }
                }
            } else {
                for seg in cs.iter() {
                    for c in seg {
                        let (lpid_u, _) = unpack_lclick(c.packed, wide_cl);
                        let lpid = lpid_u as usize;
                        let has_ev = unsafe {
                            *eoff_ptr.add(lpid) != *eoff_ptr.add(lpid + 1)
                                || (npurch > 0 && *poff_ptr.add(lpid) != *poff_ptr.add(lpid + 1))
                        };
                        if has_ev {
                            unsafe { *coff_ptr.add(lpid) += 1 };
                            ncl_kept += 1;
                        }
                    }
                }
            }
            if ncl_kept > 0 {
                let mut csum = 0u32;
                self.cl_cur.clear();
                self.cl_cur.reserve(local_np);
                let ccur_out = self.cl_cur.as_mut_ptr();
                for i in 0..local_np {
                    let cnt = unsafe { *coff_ptr.add(i) };
                    unsafe {
                        *coff_ptr.add(i) = csum;
                        ccur_out.add(i).write(csum);
                    }
                    csum += cnt;
                }
                unsafe {
                    self.cl_cur.set_len(local_np);
                    *coff_ptr.add(local_np) = csum;
                }
                self.cls.reserve_uninit(ncl_kept);
                let co = self.cls.as_mut_ptr();
                let ccur_ptr = self.cl_cur.as_mut_ptr();
                if !has_wide {
                    for seg in cs.iter() {
                        for c in seg {
                            let lpid = (c.packed >> 16) as usize;
                            if unsafe { *coff_ptr.add(lpid) != *coff_ptr.add(lpid + 1) } {
                                unsafe {
                                    let dst = ccur_ptr.add(lpid);
                                    co.add(*dst as usize).write(Cl24 {
                                        cid: c.cid,
                                        ts: c.ts,
                                        camp: c.packed & 0xFFFF,
                                    });
                                    *dst += 1;
                                }
                            }
                        }
                    }
                } else {
                    for seg in cs.iter() {
                        for c in seg {
                            let (lpid_u, camp) = unpack_lclick(c.packed, wide_cl);
                            let lpid = lpid_u as usize;
                            if unsafe { *coff_ptr.add(lpid) != *coff_ptr.add(lpid + 1) } {
                                unsafe {
                                    let dst = ccur_ptr.add(lpid);
                                    co.add(*dst as usize).write(Cl24 { cid: c.cid, ts: c.ts, camp });
                                    *dst += 1;
                                }
                            }
                        }
                    }
                }
                unsafe { self.cls.set_len(ncl_kept) };
            }
        }
        cs.free();

        // Step 4: Solve each active canonical person in dense L1/L2 order!
        let eo_base = self.ev_ts.as_mut_ptr();
        for lpid in 0..local_np {
            let es0 = unsafe { *self.ev_off.get_unchecked(lpid) } as usize;
            let es1 = unsafe { *self.ev_off.get_unchecked(lpid + 1) } as usize;
            let purch_slice: &[(u32, u32)] = if npurch > 0 {
                let ps0 = unsafe { *self.p_off.get_unchecked(lpid) } as usize;
                let ps1 = unsafe { *self.p_off.get_unchecked(lpid + 1) } as usize;
                unsafe { self.purch.get_unchecked(ps0..ps1) }
            } else {
                &[]
            };
            if es0 == es1 && purch_slice.is_empty() {
                continue;
            }
            let cls_slice: &mut [Cl24] = if ncl_kept > 0 {
                let cs0 = unsafe { *self.cl_off.get_unchecked(lpid) } as usize;
                let cs1 = unsafe { *self.cl_off.get_unchecked(lpid + 1) } as usize;
                unsafe { self.cls.get_unchecked_mut(cs0..cs1) }
            } else {
                &mut []
            };
            let cid = shard_lpid_to_cid(sh, lpid as u32, l_bits);
            let pkey = links.canon_key(cid);
            let ts_slice = unsafe { std::slice::from_raw_parts_mut(eo_base.add(es0), es1 - es0) };
            Self::solve_person(pkey, ts_slice, purch_slice, cls_slice, w, &mut self.ctmp);
        }
    }
}

// ---------------------------------------------------------------------------

fn peak_rss_mb() -> f64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let v = ru.ru_maxrss as f64;
    if cfg!(target_os = "macos") {
        v / (1024.0 * 1024.0)
    } else {
        v / 1024.0
    }
}

fn manifest_value(m: &str, key: &str) -> i64 {
    let pat = format!("\"{key}\"");
    let i = m.find(&pat).unwrap_or_else(|| panic!("manifest missing {key}")) + pat.len();
    let rest = &m[i..];
    let rest = &rest[rest.find(':').expect("manifest colon") + 1..];
    let q = rest.find('"').expect("manifest quote") + 1;
    let e = rest[q..].find('"').expect("manifest quote") + q;
    parse_ts(rest[q..e].as_bytes()).unwrap_or_else(|| panic!("bad manifest {key}"))
}

const PURCH_SEG: usize = 2048;

struct EventHandler<'a, 'b, const ID_BITS: u32, const P_BITS: u32, const EV_STAGE: usize, const P_STAGE: usize> {
    ctx: &'a Ctx,
    ws: i64,
    base: i64,
    wlen: u64,
    oob_ref: &'b Mutex<Vec<i64>>,
    wide_ref: &'b Mutex<Vec<(u32, u16)>>,
    id_w: Scatter<'b, DedupRec, EV_STAGE>,
    ev_w: Scatter<'b, PEv12, P_STAGE>,
    purch_w: Scatter<'b, PPurch, 64, PURCH_SEG>,
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn push_purchase(
    ctx: &Ctx,
    wide_ref: &Mutex<Vec<(u32, u16)>>,
    purch_w: &mut Scatter<'_, PPurch, 64, PURCH_SEG>,
    p_sh: usize,
    uid_k: Key8,
    ts_rel: u32,
    product: &[u8],
    price: u32,
) -> u32 {
    if price == 0 {
        return arena::LOC_INVALID;
    }
    let cat = ctx.products.get_swar(product).unwrap_or(ctx.unknown_cat);
    let pc = if !ctx.wide {
        (price << 8) | cat
    } else {
        push_wide_price(wide_ref, price, cat)
    };
    arena::LOC_PURCH_BIT | purch_w.push_with_loc(p_sh, PPurch { person: uid_k, ts: ts_rel, pc })
}

impl<'a, 'b, const ID_BITS: u32, const P_BITS: u32, const EV_STAGE: usize, const P_STAGE: usize>
    event_front::EventSink<'_> for &mut EventHandler<'a, 'b, ID_BITS, P_BITS, EV_STAGE, P_STAGE>
{
    #[inline(always)]
    fn prefetch_keys(&self, eh: u64, uh: u64) {
        self.id_w.prefetch_shard((eh >> (64 - ID_BITS)) as usize);
        self.ev_w.prefetch_located_shard((uh >> (64 - P_BITS)) as usize);
    }

    #[inline(always)]
    fn on_event(&mut self, ev: event_front::Event<'_>) {
        if ev.ek.0 == 0 {
            return;
        }
        let ek = ev.ek;
        let id_sh = (ev.eh >> (64 - ID_BITS)) as usize;
        let ing_u32 = if ev.ing_bcd < INGEST_OOB_BASE {
            ev.ing_bcd
        } else {
            encode_special_ingest(ev.ing_bcd, ev.ing_day_ms, self.oob_ref)
        };
        let mut loc = arena::LOC_INVALID;
        'v: {
            if ev.ing_bcd == INGEST_NONE
                || ev.kind == 0
                || ev.uid_k.is_empty()
                || ev.product.is_empty()
            {
                break 'v;
            }
            let ts = match ev.timestamp {
                Some(t) if (t.wrapping_sub(self.ws) as u64) < self.wlen => t,
                _ => break 'v,
            };
            if ev.ing_day_ms.saturating_sub(ts) > LATE_MS - DAY_MS
                && is_late_slow(ev.ing_bcd, ev.ing_day_ms, ts)
            {
                break 'v;
            }
            let p_sh = (ev.uh >> (64 - P_BITS)) as usize;
            let ts_rel = (ts - self.base) as u32;
            if ev.kind == 1 {
                loc = self.ev_w.push_with_loc12(p_sh, PEv12 { person: ev.uid_k, ts: ts_rel });
            } else {
                loc = push_purchase(
                    self.ctx,
                    self.wide_ref,
                    &mut self.purch_w,
                    p_sh,
                    ev.uid_k,
                    ts_rel,
                    ev.product,
                    ev.price,
                );
            }
        }
        self.id_w.push(id_sh, DedupRec { eid: ek, ingest: ing_u32, loc });
    }
}

struct ClickHandler<'a, 'b, const P_BITS: u32, const CL_STAGE: usize> {
    ctx: &'a Ctx,
    cws: i64,
    cwlen: u64,
    base: i64,
    w: Scatter<'b, Click, CL_STAGE>,
}

impl<'a, 'b, const P_BITS: u32, const CL_STAGE: usize> event_front::ClickSink<'_>
    for &mut ClickHandler<'a, 'b, P_BITS, CL_STAGE>
{
    #[inline(always)]
    fn on_click(&mut self, f: &[&[u8]; 4]) {
        let (cid, uid) = (f[0], f[1]);
        if cid.is_empty() || uid.is_empty() {
            return;
        }
        let uid_k = Key8::new(uid);
        let uh = uid_k.hash(SEED_A);
        let sh = (uh >> (64 - P_BITS)) as usize;
        self.w.prefetch_shard(sh);
        let Some(camp) = self.ctx.camps.get_swar(f[2]) else { return };
        let ts = match self.ctx.dates.parse_ts(f[3]) {
            Some(t) if (t.wrapping_sub(self.cws) as u64) < self.cwlen => t,
            _ => return,
        };
        self.w.push(
            sh,
            Click { person: uid_k, cid: CKey::new(cid), ts: (ts - self.base) as u32, camp },
        );
    }
}

fn run_pipeline<const ID_BITS: u32, const P_BITS: u32, const EV_STAGE: usize, const P_STAGE: usize, const CL_STAGE: usize>(
    ctx: &Ctx,
    link_files: &[&'static [u8]],
    event_files: &[&'static [u8]],
    click_files: &[&'static [u8]],
    threads: usize,
    mark: &mut impl FnMut(&'static str),
) -> (Vec<(Vec<UserDailyPart>, Vec<CampAcc>)>, LinkTable) {
    let id_shards: usize = 1 << ID_BITS;
    let p_shards: usize = 1 << P_BITS;
    let wlen: u64 = (ctx.we - ctx.ws).max(0) as u64;

    // ---- events + dedup + clicks with exact `threads`-on-`threads` scheduling:
    // When links is much smaller than events (20M/60M), thread 0 runs `build_links` on its
    // dedicated core (zero CFS over-subscription!) and immediately joins `events` once done,
    // leaving all `threads` cores 100% free of links interference during `dedup` and `clicks`.
    let link_bytes: usize = link_files.iter().map(|d| d.len()).sum();
    let event_bytes: usize = event_files.iter().map(|d| d.len()).sum();
    let inline_links = threads >= 2 && link_bytes * 4 < event_bytes;
    let chunks = make_chunks(event_files, chunk_target(event_files, threads));
    let id_set: ShardSet<DedupRec> = ShardSet::new_shared(id_shards);
    let ev_set: ShardSet<PEv12> = ShardSet::new_shared(p_shards);
    let purch_set: ShardSet<PPurch, PURCH_SEG> = ShardSet::new_shared(p_shards);
    let oob_ingest: Mutex<Vec<i64>> = Mutex::new(Vec::new());
    let wide_store: Mutex<Vec<(u32, u16)>> = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    let links_sync = if threads <= 1 {
        let lt = links::build_links(link_files, 1);
        mark("links");
        Some(lt)
    } else {
        None
    };
    let links_slot: Mutex<Option<LinkTable>> = Mutex::new(None);
    let (links_owned, click_set, wide_store) = std::thread::scope(|s| {
        let links_job = if links_sync.is_none() && !inline_links {
            Some(s.spawn(move || links::build_links(link_files, 1)))
        } else {
            None
        };
        par(threads, |t| {
            if inline_links && t == 0 {
                let lt = links::build_links(link_files, 1);
                *links_slot.lock().unwrap() = Some(lt);
                if next.load(Ordering::Relaxed) >= chunks.len() {
                    return;
                }
            }
            let mut handler = EventHandler::<ID_BITS, P_BITS, EV_STAGE, P_STAGE> {
                ctx,
                ws: ctx.ws,
                base: ctx.base,
                wlen,
                oob_ref: &oob_ingest,
                wide_ref: &wide_store,
                id_w: Scatter::<DedupRec, EV_STAGE>::new(&id_set),
                ev_w: Scatter::<PEv12, P_STAGE>::new_located12(&ev_set),
                purch_w: Scatter::<PPurch, 64, PURCH_SEG>::new_located(&purch_set),
            };
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= chunks.len() {
                    break;
                }
                let d = chunks[i].data;
                populate(d);
                event_front::scan(d, &ctx.dates, &mut handler);
            }
            handler.id_w.finish();
            handler.ev_w.finish();
            handler.purch_w.finish();
        });
        mark("events");

        let oob_ingest = oob_ingest.into_inner().unwrap();
        let wide_store = wide_store.into_inner().unwrap();

        // ---- dedup (exact event_id) via 1-pass 64B-line AVX2 16-way tag16/pos16 buckets, marking losing PEvs in place!
        let next = AtomicUsize::new(0);
        let oob_slice = &oob_ingest[..];
        let base_day_ms = ctx.dates.base_day_ms;
        par(threads, |_| {
            let mut buckets16: HugeBuf<DedupBucket16> = HugeBuf::empty();
            let mut buckets32: Vec<DedupBucket> = Vec::new();
            let mut seg_starts: Vec<u32> = Vec::new();
            let mut loser_locs: Vec<u32> = Vec::new();
            loop {
                let s = next.fetch_add(1, Ordering::Relaxed);
                if s >= id_shards {
                    break;
                }
                let mut segs = id_set.take(s);
                dedup_shard(
                    &mut segs,
                    &mut buckets16,
                    &mut buckets32,
                    &mut seg_starts,
                    &mut loser_locs,
                    base_day_ms,
                    oob_slice,
                );
                segs.free();
            }
        });
        id_set.free_all();
        mark("dedup");

        // ---- clicks: parse, validate, scatter directly by raw user_id (ZERO LinkTable lookups!)
        let cws = ctx.ws - SEVEN_DAYS_MS;
        let cwlen: u64 = (ctx.we - cws).max(0) as u64;
        let chunks = make_chunks(click_files, chunk_target(click_files, threads));
        let click_set: ShardSet<Click> = ShardSet::new_shared(p_shards);
        let next = AtomicUsize::new(0);
        par(threads, |_| {
            let mut handler = ClickHandler::<P_BITS, CL_STAGE> {
                ctx,
                cws,
                cwlen,
                base: ctx.base,
                w: Scatter::<Click, CL_STAGE>::new(&click_set),
            };
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= chunks.len() {
                    break;
                }
                let d = chunks[i].data;
                populate(d);
                event_front::scan_clicks(d, &mut handler);
            }
            handler.w.finish();
        });
        mark("clicks");

        let lt = if inline_links {
            mark("links_wait");
            links_slot.into_inner().unwrap().expect("inline links build")
        } else {
            match links_job {
                Some(j) => {
                    let lt = j.join().expect("links build");
                    mark("links_wait");
                    lt
                }
                None => links_sync.unwrap(),
            }
        };
        (lt, click_set, wide_store)
    });
    let links = &links_owned;

    // ---- Two-tier solve: Tier 1 resolves distinct uids once, solves unlinked in place,
    // forwards linked by dense cid (8B LEv8 + 16B LPurch + 16B LClick) to hash-free Tier 2!
    let mut order: Vec<(usize, usize)> = (0..p_shards)
        .map(|p| (p, (ev_set.count(p) + purch_set.count(p)) * 2 + click_set.count(p)))
        .collect();
    order.sort_by_key(|x| std::cmp::Reverse(x.1));
    let l_bits = P_BITS - 1;
    let l_shards = 1usize << l_bits;
    let l_mask = l_shards - 1;
    let wide_camps = ctx.camp_ids.len() >= 65_536;
    let linked_ev_set: ShardSet<LEv8> = ShardSet::new_shared(l_shards);
    let linked_p_set: ShardSet<LPurch> = ShardSet::new_shared(l_shards);
    let linked_cl_set: ShardSet<LClick> = ShardSet::new_shared(l_shards);
    let wide_cl_mu: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    let wide_cl_ro: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());
    let order2: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let wide_slice = &wide_store[..];
    let next1 = AtomicUsize::new(0);
    let next2 = AtomicUsize::new(0);
    let bar = Barrier::new(threads);
    let results: Vec<(Vec<UserDailyPart>, Vec<CampAcc>)> = par(threads, |_| {
        let mut w = Worker::new(ctx, wide_slice);
        let mut sc = SolveScratch::new();
        {
            let mut lev_w = Scatter::<LEv8, 128>::new(&linked_ev_set);
            let mut lp_w = Scatter::<LPurch, 64>::new(&linked_p_set);
            let mut lcl_w = Scatter::<LClick, 128>::new(&linked_cl_set);
            loop {
                let k = next1.fetch_add(1, Ordering::Relaxed);
                if k >= p_shards {
                    break;
                }
                let p = order[k].0;
                let es = ev_set.take(p);
                let ps = purch_set.take(p);
                let cs = click_set.take(p);
                sc.run_tier1_shard(
                    es,
                    ps,
                    cs,
                    links,
                    &mut w,
                    &mut lev_w,
                    &mut lp_w,
                    &mut lcl_w,
                    l_bits,
                    l_mask,
                    wide_camps,
                    &wide_cl_mu,
                );
            }
            lev_w.finish();
            lp_w.finish();
            lcl_w.finish();
        }
        if bar.wait().is_leader() {
            ev_set.release_blocks();
            purch_set.release_blocks();
            click_set.release_blocks();
            if !links.is_empty() {
                *wide_cl_ro.lock().unwrap() = std::mem::take(&mut *wide_cl_mu.lock().unwrap());
                let mut o2: Vec<(usize, usize)> = (0..l_shards)
                    .map(|p| {
                        (
                            p,
                            (linked_ev_set.count(p) + linked_p_set.count(p)) * 2 + linked_cl_set.count(p),
                        )
                    })
                    .filter(|x| x.1 > 0)
                    .collect();
                o2.sort_by_key(|x| std::cmp::Reverse(x.1));
                *order2.lock().unwrap() = o2;
            }
        }
        bar.wait();
        if !links.is_empty() {
            let o2 = order2.lock().unwrap().clone();
            let wide_cl_local = wide_cl_ro.lock().unwrap().clone();
            loop {
                let k = next2.fetch_add(1, Ordering::Relaxed);
                if k >= o2.len() {
                    break;
                }
                let p = o2[k].0;
                let es = linked_ev_set.take(p);
                let ps = linked_p_set.take(p);
                let cs = linked_cl_set.take(p);
                sc.run_tier2_shard(p, l_bits, links, es, ps, cs, &wide_cl_local, &mut w);
            }
        }
        if w.ud.len() > 0 {
            w.roll_user_file();
        }
        let done_cols = std::mem::take(&mut w.done_cols);
        (done_cols, w.into_acc())
    });
    ev_set.free_all();
    purch_set.free_all();
    click_set.free_all();
    linked_ev_set.free_all();
    linked_p_set.free_all();
    linked_cl_set.free_all();
    mark("solve");
    (results, links_owned)
}

fn main() {
    let t0 = Instant::now();
    let mut marks: Vec<(&str, f64)> = Vec::new();
    let mut last = t0;
    let mut mark = |name: &'static str, marks: &mut Vec<(&str, f64)>| {
        let now = Instant::now();
        marks.push((name, (now - last).as_secs_f64() * 1000.0));
        last = now;
    };

    let args: Vec<String> = std::env::args().collect();
    let env = |k: &str| std::env::var(k).ok().filter(|s| !s.is_empty());
    let data_dir = PathBuf::from(args.get(1).cloned().or_else(|| env("DATA_DIR")).unwrap_or_else(|| "/data".into()));
    let out_dir =
        PathBuf::from(args.get(2).cloned().or_else(|| env("OUTPUT_DIR")).unwrap_or_else(|| "/output".into()));
    let threads = env("NCPU")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8))
        .clamp(1, 128);

    // ---- init
    let manifest = std::fs::read_to_string(data_dir.join("manifest.json")).expect("read manifest.json");
    let ws = manifest_value(&manifest, "window_start");
    let we = manifest_value(&manifest, "window_end");
    assert!(we - ws + SEVEN_DAYS_MS < u32::MAX as i64, "event window too long");
    let day0 = ws.div_euclid(DAY_MS);
    let ndays = if we > ws { ((we - 1).div_euclid(DAY_MS) - day0 + 1) as usize } else { 1 };

    let pdata = read_static(&data_dir.join("products.csv"));
    let mut prod_rows: Vec<(&'static [u8], &'static [u8])> = Vec::new();
    csv::scan::<2, _>(body(pdata), |f| prod_rows.push((f[0], f[1])));
    let mut cat_names: Vec<&'static [u8]> = prod_rows.iter().map(|r| r.1).collect();
    cat_names.push(b"unknown");
    cat_names.sort_unstable();
    cat_names.dedup();
    assert!(cat_names.len() < u16::MAX as usize);
    let mut catmap = KeyMap::new(cat_names.len());
    for (i, c) in cat_names.iter().enumerate() {
        catmap.insert(c, i as u32);
    }
    let unknown_cat = catmap.get(b"unknown").unwrap();
    let mut products = KeyMap::new_with_fast(prod_rows.len(), b"p", 65_536);
    for (p, c) in &prod_rows {
        products.insert(p, catmap.get(c).unwrap());
    }

    let cdata = read_static(&data_dir.join("campaigns.csv"));
    let mut camp_ids: Vec<&'static [u8]> = vec![b"organic"];
    let mut camp_channels: Vec<&'static [u8]> = vec![b"organic"];
    let mut camp_rows: Vec<(&'static [u8], &'static [u8])> = Vec::new();
    csv::scan::<2, _>(body(cdata), |f| camp_rows.push((f[0], f[1])));
    let mut camps = KeyMap::new_with_fast(camp_rows.len(), b"c_", 8_192);
    for (id, ch) in camp_rows {
        if camps.get(id).is_none() {
            camps.insert(id, camp_ids.len() as u32);
            camp_ids.push(id);
            camp_channels.push(ch);
        }
    }

    let link_files = map_dir(&data_dir.join("links"));
    let click_files = map_dir(&data_dir.join("clicks"));
    let event_files = map_dir(&data_dir.join("events"));
    let wide = cat_names.len() > 256;
    let ctx = Ctx {
        ws,
        base: ws - SEVEN_DAYS_MS,
        we,
        day0,
        ndays,
        dates: DateCache::new(ws, we),
        products,
        unknown_cat,
        cat_names,
        wide,
        camps,
        camp_ids,
        camp_channels,
    };
    let ctx = &ctx;
    mark("init", &mut marks);

    let event_bytes: usize = event_files.iter().map(|d| d.len()).sum();
    let force_10 = env("FORCE_SHARD_BITS").as_deref() == Some("10");
    let (results, links_owned) = if force_10 || event_bytes > (2500 << 20) {
        let mut m = |name: &'static str| mark(name, &mut marks);
        run_pipeline::<10, 10, 64, 192, 192>(ctx, &link_files, &event_files, &click_files, threads, &mut m)
    } else {
        let mut m = |name: &'static str| mark(name, &mut marks);
        run_pipeline::<9, 9, 128, 192, 192>(ctx, &link_files, &event_files, &click_files, threads, &mut m)
    };

    // ---- write (pooled blocks are released in parallel meanwhile)
    let ud = out_dir.join("user_daily");
    let cd = out_dir.join("campaign_daily");
    std::fs::create_dir_all(&ud).expect("create user_daily");
    std::fs::create_dir_all(&cd).expect("create campaign_daily");
    let k1 = ctx.camp_ids.len();
    let mut jobs: Vec<UserDailyPart> = Vec::new();
    let mut accs: Vec<Vec<CampAcc>> = Vec::with_capacity(results.len());
    for (chunks, a) in results {
        jobs.extend(chunks);
        accs.push(a);
    }
    let jobs = &jobs;
    let accs = &accs;
    let links_ref = &links_owned;
    let free_blocks = arena::take_free_blocks();
    let free_ref = &free_blocks;
    // Defer read-only input mmap PTE release out of `events`/`clicks`/`links` (zero cross-core
    // TLB shootdown IPIs during AVX2 parsing!) into batched 64 MB slices released here in parallel.
    let mut mmap_slices: Vec<&'static [u8]> = Vec::new();
    const MMAP_SLICE: usize = 64 << 20;
    for files in [&event_files, &click_files, &link_files] {
        for &d in files {
            if d.len() < (1 << 20) {
                continue;
            }
            let mut off = 0usize;
            while off < d.len() {
                let end = (off + MMAP_SLICE).min(d.len());
                mmap_slices.push(&d[off..end]);
                off = end;
            }
        }
    }
    mmap_slices.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let mmap_ref = &mmap_slices;
    let next_job = AtomicUsize::new(0);
    let next_free = AtomicUsize::new(0);
    let next_mmap = AtomicUsize::new(0);
    let ud_ref = &ud;
    let cd_ref = &cd;
    par(threads, |t| {
        if t == 0 {
            // campaign_daily reduce + write + release LinkTable pages on worker 0,
            // then steal remaining user_daily writes & free arena block releases!
            let mut tot = vec![CampAcc::default(); k1 * ctx.ndays];
            for a in accs {
                for (dst, x) in tot.iter_mut().zip(a) {
                    dst.sessions += x.sessions;
                    dst.purchasing += x.purchasing;
                    dst.assisted += x.assisted;
                    dst.revenue += x.revenue;
                }
            }
            let n = tot.iter().filter(|x| x.sessions > 0 || x.assisted > 0).count();
            let mut c_id = Column::new("campaign_id", Kind::Str, n);
            let mut c_ch = Column::new("channel", Kind::Str, n);
            let mut c_day = Column::new("day", Kind::Date, n);
            let mut c_s = Column::new("sessions", Kind::I64, n);
            let mut c_p = Column::new("purchasing_sessions", Kind::I64, n);
            let mut c_r = Column::new("revenue_cents", Kind::I64, n);
            let mut c_a = Column::new("assisted_sessions", Kind::I64, n);
            for d in 0..ctx.ndays {
                for k in 0..k1 {
                    let x = &tot[d * k1 + k];
                    if x.sessions == 0 && x.assisted == 0 {
                        continue;
                    }
                    c_id.push_str(ctx.camp_ids[k]);
                    c_ch.push_str(ctx.camp_channels[k]);
                    c_day.push_date((ctx.day0 + d as i64) as i32);
                    c_s.push_i64(x.sessions as i64);
                    c_p.push_i64(x.purchasing as i64);
                    c_r.push_i64(x.revenue as i64);
                    c_a.push_i64(x.assisted as i64);
                }
            }
            parquet::write_file(&cd_ref.join("part-00000.parquet"), &[c_id, c_ch, c_day, c_s, c_p, c_r, c_a])
                .expect("write campaign_daily");
            links_ref.release_pages();
        }
        let mut wsc = parquet::WriteScratch::new();
        loop {
            let i = next_job.fetch_add(1, Ordering::Relaxed);
            if i >= jobs.len() {
                break;
            }
            parquet::write_user_daily_packed(
                &ud_ref.join(format!("part-{i:05}.parquet")),
                &jobs[i].buf,
                ctx.day0,
                ctx.ndays,
                &ctx.cat_names,
                &mut wsc,
            )
            .expect("write user_daily");
            arena::release_owned_blocks(&jobs[i].blocks);
            jobs[i].buf.release_heap_pages();
        }
        const BATCH: usize = 4;
        loop {
            let start = next_free.fetch_add(BATCH, Ordering::Relaxed);
            if start >= free_ref.len() {
                break;
            }
            let end = (start + BATCH).min(free_ref.len());
            for &(addr, _) in &free_ref[start..end] {
                arena::release_block(addr);
            }
        }
        loop {
            let i = next_mmap.fetch_add(1, Ordering::Relaxed);
            if i >= mmap_ref.len() {
                break;
            }
            release_chunk(mmap_ref[i]);
        }
    });
    if jobs.is_empty() {
        let empty = new_user_cols_heap(0);
        parquet::write_file(&ud.join("part-00000.parquet"), &empty).expect("write user_daily");
    }
    mark("write", &mut marks);

    let total = (Instant::now() - t0).as_secs_f64() * 1000.0;
    let mut line = format!("[timing] total={total:.1}ms");
    for (k, v) in &marks {
        line.push_str(&format!(" {k}={v:.1}ms"));
    }
    line.push_str(&format!(" threads={threads} simd={} peak_rss_mb={:.1}", csv::simd_backend(), peak_rss_mb()));
    eprintln!("{line}");
    let _ = std::io::stderr().flush();
    if env("PGO_TRAIN").is_some() {
        std::process::exit(0);
    }
    unsafe { libc::_exit(0) };
}

#[cfg(test)]
mod canon_idx_tests {
    use super::{parse_canon_idx, parse_canon_idx_swar};

    #[test]
    fn swar_matches_scalar() {
        let alphabet: &[u8] = b"0123456789pc_P:/ \xff";
        let mut buf = vec![0u8; 3 * 4096];
        for (prefix, limit) in [(&b"p"[..], 65_536usize), (&b"c_"[..], 8_192), (&b""[..], 1000)] {
            let pw = prefix.iter().enumerate().fold(0u64, |a, (i, &b)| a | ((b as u64) << (8 * i)));
            let pmask = if prefix.is_empty() { 0 } else { (1u64 << (8 * prefix.len())) - 1 };
            let mut seed = 0x1234_5678_9abc_def0u64;
            for it in 0..400_000 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let n = (seed % 10) as usize;
                let off = if it % 3 == 0 { 4096 - (seed >> 8) as usize % 9 } else { 100 + (seed >> 8) as usize % 64 };
                let mut r = seed >> 16;
                for i in 0..n {
                    let pick = if i < prefix.len() && r & 7 != 0 { prefix[i] } else { alphabet[(r % alphabet.len() as u64) as usize] };
                    buf[off + i] = pick;
                    r = r.rotate_left(7) ^ 0x9e37_79b9;
                }
                for j in n..n + 8 {
                    buf[off + j] = b"9_0p,\n\"c"[j % 8];
                }
                let s = &buf[off..off + n];
                assert_eq!(
                    parse_canon_idx_swar(s, prefix, pw, pmask, limit),
                    parse_canon_idx(s, prefix, limit),
                    "prefix={prefix:?} s={s:?}"
                );
            }
            for v in 0..200_000usize {
                for lead in ["", "0", "00"] {
                    let txt = format!("{}{}{}", std::str::from_utf8(prefix).unwrap(), lead, v);
                    let b = txt.as_bytes();
                    if b.len() > 9 { continue; }
                    buf[200..200 + b.len()].copy_from_slice(b);
                    let s = &buf[200..200 + b.len()];
                    assert_eq!(parse_canon_idx_swar(s, prefix, pw, pmask, limit), parse_canon_idx(s, prefix, limit), "{txt}");
                }
            }
        }
    }
}

#[cfg(test)]
mod sort_tests {
    use super::{cid_to_shard_lpid, shard_local_np, shard_lpid_to_cid, sort_person_ts};

    #[test]
    fn sort_person_ts_networks_zero_one_exhaustive() {
        for n in 0..=8 {
            for mask in 0u32..(1u32 << n) {
                let mut got: Vec<u32> = (0..n).map(|i| (mask >> i) & 1).collect();
                let mut want = got.clone();
                sort_person_ts(&mut got);
                want.sort_unstable();
                assert_eq!(got, want, "n={n} mask={mask:#b}");
            }
        }
    }

    #[test]
    fn cmov_min_max_exact() {
        let vals = [0u32, 1, 2, 100, 0x7FFF_FFFF, 0x8000_0000, u32::MAX - 1, u32::MAX];
        for &a in &vals {
            for &b in &vals {
                #[cfg(target_arch = "x86_64")]
                let (mn, mx) = unsafe {
                    let mut mn = a;
                    let mut mx = a;
                    core::arch::asm!("cmp {x:e}, {y:e}", "cmova {x:e}, {y:e}", x = inout(reg) mn, y = in(reg) b, options(pure, nomem, nostack));
                    core::arch::asm!("cmp {x:e}, {y:e}", "cmovb {x:e}, {y:e}", x = inout(reg) mx, y = in(reg) b, options(pure, nomem, nostack));
                    (mn, mx)
                };
                #[cfg(not(target_arch = "x86_64"))]
                let (mn, mx) = (a.min(b), a.max(b));
                assert_eq!(mn, a.min(b), "min({a}, {b})");
                assert_eq!(mx, a.max(b), "max({a}, {b})");
            }
        }
    }

    #[test]
    fn fast_bucket_uniform_within_shard() {
        let nbuckets = 5888usize;
        let mut counts = vec![0u32; nbuckets];
        // Fix top 10 shard bits to a constant shard id (simulating shard 0x2AB at ID_BITS=10).
        let shard_prefix = 0x2ABu64 << 54;
        for i in 0..60_000u64 {
            let h = (crate::key::mum(i ^ crate::SEED_ID, crate::key::P0) & ((1u64 << 54) - 1)) | shard_prefix;
            counts[super::fast_bucket(h, nbuckets)] += 1;
        }
        let max_c = *counts.iter().max().unwrap();
        let non_zero = counts.iter().filter(|&&c| c > 0).count();
        assert!(non_zero > nbuckets * 99 / 100, "non_zero={non_zero}/{nbuckets}");
        assert!(max_c < 35, "max_c={max_c}");
    }

    #[test]
    fn cid_shard_lpid_bijection() {
        for l_bits in 0..=5u32 {
            let l_shards = 1usize << l_bits;
            let l_mask = l_shards - 1;
            for n_canon in [0usize, 1, 7, 8, 9, 15, 16, 31, 32, 33, 100, 255, 256, 257, 513] {
                let mut total = 0usize;
                let mut seen = vec![false; n_canon];
                for sh in 0..l_shards {
                    let local_np = shard_local_np(n_canon, sh, l_bits);
                    total += local_np;
                    for lpid in 0..local_np {
                        let cid = shard_lpid_to_cid(sh, lpid as u32, l_bits);
                        assert!((cid as usize) < n_canon);
                        let (sh2, lpid2) = cid_to_shard_lpid(cid, l_bits, l_mask);
                        assert_eq!((sh2, lpid2), (sh, lpid as u32));
                        assert!(!seen[cid as usize]);
                        seen[cid as usize] = true;
                    }
                }
                assert_eq!(total, n_canon);
                assert!(seen.into_iter().all(|b| b));
            }
        }
    }
}
