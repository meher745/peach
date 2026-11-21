//! Linked-ID resolution.
//!
//! `build_links` interns every linked ID exactly (`Key8`), joins components with a
//! lock-free union-find whose roots are always the byte-wise smallest ID, and stores
//! only the IDs whose person differs from themselves in a compact `LinkTable`
//! (exact `Key8` -> person `Key8`). A 2 MB blocked Bloom filter over those IDs
//! lets unlinked/root IDs skip the DRAM table probe, and `Resolver` adds a 64-entry
//! direct-mapped per-thread cache (whale IDs) plus a prefetch delay ring.

use std::sync::atomic::{AtomicU32, AtomicU8, AtomicUsize, Ordering};

use crate::arena::{HugeBuf, Scatter, ShardSet};
use crate::key::Key8;
use crate::{csv, fixed, make_chunks, chunk_target, par, populate, SyncPtr};

pub const SEED_L: u64 = 0x9e37_79b9_7f4a_7c15;
pub const SEED_A: u64 = SEED_L;

const REGION_BITS: u32 = 8;
const REGIONS: usize = 1 << REGION_BITS;
/// Link-side shards for interning (256 L2 shards -> 32 KB L1D staging, 256 KB 60M ltab slice).
const SIDE_BITS: u32 = 8;
const SIDE_SHARDS: usize = 1 << SIDE_BITS;

/// Zero-padding 5-way 64-byte cache-line bucket for `LinkTable`:
/// 5 keys (40B, first 32B aligned for single AVX2 `vpcmpeqq`), 5 cids (20B), `len: 0..=5` (4B).
/// Halves 60M `LinkTable` footprint (128 MB -> 64 MB) and keeps lookups in 1 cache line!
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct LinkBucket {
    keys: [Key8; 5],
    cids: [u32; 5],
    len: u32,
}

#[allow(dead_code)]
const LBUCKET_EMPTY: LinkBucket = LinkBucket { keys: [Key8::EMPTY; 5], cids: [0; 5], len: 0 };
const _: () = assert!(std::mem::size_of::<LinkBucket>() == 64);

#[inline(always)]
fn link_nbuckets(total_nr: usize) -> usize {
    // Load factor <= 4/5 (80%): 97%+ of lookups hit in a single 64-byte cache line.
    ((total_nr / 4) + 1).next_power_of_two().max(4)
}

pub struct LinkTable {
    tab: HugeBuf<LinkBucket>,
    mask: usize,
    tab_shift: u32,
    bloom: HugeBuf<u64>,
    bloom_shift: u32,
    canon_keys: HugeBuf<Key8>,
    scratch_mmaps: Vec<(usize, usize)>,
    empty: bool,
}

impl Drop for LinkTable {
    fn drop(&mut self) {
        #[cfg(any(not(target_os = "linux"), test))]
        for &(addr, bytes) in &self.scratch_mmaps {
            unsafe { libc::munmap(addr as *mut libc::c_void, bytes) };
        }
    }
}

#[inline(always)]
fn bloom_word(h: u64, shift: u32) -> usize {
    (h >> shift) as usize
}
#[inline(always)]
fn bloom_bits(h: u64) -> u64 {
    (1u64 << (h & 63)) | (1u64 << ((h >> 6) & 63)) | (1u64 << ((h >> 12) & 63))
}

#[inline(always)]
pub fn prefetch(p: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(p as *const i8, core::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
}

impl LinkTable {
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.empty
    }

    #[inline(always)]
    pub fn num_canon(&self) -> usize {
        self.canon_keys.len()
    }

    #[inline(always)]
    pub fn canon_key(&self, cid: u32) -> Key8 {
        unsafe { *self.canon_keys.get_unchecked(cid as usize) }
    }

    #[inline(always)]
    pub fn prefetch_bloom(&self, h: u64) {
        prefetch(unsafe { self.bloom.as_ptr().add(bloom_word(h, self.bloom_shift)) } as *const u8);
    }

    /// Release LinkTable backing pages (and deferred build-time scratch mmaps) back to the OS once solve finishes.
    pub fn release_pages(&self) {
        self.tab.release_pages();
        self.bloom.release_pages();
        self.canon_keys.release_pages();
        #[cfg(all(target_os = "linux", not(test)))]
        for &(addr, bytes) in &self.scratch_mmaps {
            unsafe { libc::madvise(addr as *mut libc::c_void, bytes, libc::MADV_DONTNEED) };
        }
    }

    #[inline(always)]
    fn slot(&self, h: u64) -> usize {
        (h >> self.tab_shift) as usize
    }

    #[inline(always)]
    pub fn maybe(&self, h: u64) -> bool {
        let b = bloom_bits(h);
        unsafe { *self.bloom.get_unchecked(bloom_word(h, self.bloom_shift)) & b == b }
    }

    #[inline(always)]
    pub fn prefetch(&self, h: u64) {
        prefetch(unsafe { self.tab.as_ptr().add(self.slot(h)) } as *const u8);
    }

    /// Dense canonical component id `cid` (`0..num_canon()`) if `uid` belongs to a linked component of size >= 2.
    #[inline(always)]
    pub fn lookup(&self, h: u64, uid: Key8) -> Option<u32> {
        let mask = self.mask;
        let mut i = (h >> self.tab_shift) as usize;
        loop {
            let b = unsafe { self.tab.get_unchecked(i) };
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use core::arch::x86_64::*;
                let v = _mm256_load_si256(b.keys.as_ptr() as *const __m256i);
                let eq = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(
                    v,
                    _mm256_set1_epi64x(uid.0 as i64),
                ))) as u32;
                if eq != 0 {
                    return Some(*b.cids.get_unchecked(eq.trailing_zeros() as usize));
                }
                if b.keys[4] == uid {
                    return Some(b.cids[4]);
                }
                if b.len < 5 {
                    return None;
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                let n = b.len as usize;
                for k in 0..n {
                    if unsafe { *b.keys.get_unchecked(k) } == uid {
                        return Some(unsafe { *b.cids.get_unchecked(k) });
                    }
                }
                if n < 5 {
                    return None;
                }
            }
            i = (i + 1) & mask;
        }
    }

    #[cfg(test)]
    pub fn resolve(&self, uid: Key8) -> Key8 {
        let h = uid.hash(SEED_L);
        if !self.maybe(h) {
            return uid;
        }
        self.lookup(h, uid).map(|cid| self.canon_key(cid)).unwrap_or(uid)
    }
}

// ---------------------------------------------------------------------------
// per-thread resolver: L1 direct-mapped cache + Bloom + prefetch ring

#[allow(dead_code)]
#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct CEnt {
    uid: Key8,
    person: Key8,
    phash: u64,
}

#[allow(dead_code)]
const CACHE: usize = 512;
#[allow(dead_code)]
const RA: usize = 8;
#[allow(dead_code)]
const RB: usize = 16;

#[allow(dead_code)]
type Item<R> = (R, Key8, u64);

#[allow(dead_code)]
pub struct Resolver<'a, R: Copy> {
    lt: &'a LinkTable,
    cache: [CEnt; CACHE],
    a: [Item<R>; RA],
    ah: usize,
    at: usize,
    b: [Item<R>; RB],
    bh: usize,
    bt: usize,
}

#[allow(dead_code)]
impl<'a, R: Copy> Resolver<'a, R> {
    pub fn new(lt: &'a LinkTable, zero: R) -> Self {
        Resolver {
            lt,
            cache: [CEnt { uid: Key8::EMPTY, person: Key8::EMPTY, phash: 0 }; CACHE],
            a: [(zero, Key8::EMPTY, 0); RA],
            ah: 0,
            at: 0,
            b: [(zero, Key8::EMPTY, 0); RB],
            bh: 0,
            bt: 0,
        }
    }

    /// Resolves `uid` to (person, person hash) and eventually calls `emit` for it.
    #[inline(always)]
    pub fn push(&mut self, item: R, uid: Key8, emit: &mut impl FnMut(R, Key8, u64)) {
        let h = uid.hash(SEED_L);
        let c = unsafe { self.cache.get_unchecked((h as usize) & (CACHE - 1)) };
        if c.uid == uid {
            emit(item, c.person, c.phash);
            return;
        }
        prefetch(unsafe { self.lt.bloom.as_ptr().add(bloom_word(h, self.lt.bloom_shift)) } as *const u8);
        if self.at - self.ah == RA {
            let x = unsafe { *self.a.get_unchecked(self.ah & (RA - 1)) };
            self.ah += 1;
            self.stage_b(x, emit);
        }
        unsafe { *self.a.get_unchecked_mut(self.at & (RA - 1)) = (item, uid, h) };
        self.at += 1;
    }

    #[inline(always)]
    fn stage_b(&mut self, x: Item<R>, emit: &mut impl FnMut(R, Key8, u64)) {
        let (item, uid, h) = x;
        if !self.lt.maybe(h) {
            unsafe { *self.cache.get_unchecked_mut((h as usize) & (CACHE - 1)) = CEnt { uid, person: uid, phash: h } };
            emit(item, uid, h);
            return;
        }
        self.lt.prefetch(h);
        if self.bt - self.bh == RB {
            let y = unsafe { *self.b.get_unchecked(self.bh & (RB - 1)) };
            self.bh += 1;
            self.finish_one(y, emit);
        }
        unsafe { *self.b.get_unchecked_mut(self.bt & (RB - 1)) = x };
        self.bt += 1;
    }

    #[inline(always)]
    fn finish_one(&mut self, y: Item<R>, emit: &mut impl FnMut(R, Key8, u64)) {
        let (item, uid, h) = y;
        let (person, phash) = match self.lt.lookup(h, uid) {
            Some(cid) => {
                let p = self.lt.canon_key(cid);
                (p, p.hash(SEED_A))
            }
            None => (uid, h),
        };
        unsafe { *self.cache.get_unchecked_mut((h as usize) & (CACHE - 1)) = CEnt { uid, person, phash } };
        emit(item, person, phash);
    }

    pub fn drain(&mut self, emit: &mut impl FnMut(R, Key8, u64)) {
        while self.ah < self.at {
            let x = unsafe { *self.a.get_unchecked(self.ah & (RA - 1)) };
            self.ah += 1;
            self.stage_b(x, emit);
        }
        while self.bh < self.bt {
            let y = unsafe { *self.b.get_unchecked(self.bh & (RB - 1)) };
            self.bh += 1;
            self.finish_one(y, emit);
        }
        (self.ah, self.at, self.bh, self.bt) = (0, 0, 0, 0);
    }
}

// ---------------------------------------------------------------------------
// build

#[derive(Clone, Copy)]
#[repr(C)]
struct LSide {
    key: Key8,
    ep: u64, // thread << 40 | endpoint slot
}

pub fn build_links(files: &[&'static [u8]], threads: usize) -> LinkTable {
    if threads <= 1 {
        return build_links_seq_sharded(files);
    }
    let chunks = make_chunks(files, chunk_target(files, threads).min(1 << 20));
    let next = AtomicUsize::new(0);
    let sides: ShardSet<LSide> = ShardSet::new_shared(SIDE_SHARDS);
    // Per thread: number of edges.
    let edges: Vec<usize> = par(threads, |t| {
        let mut w = Scatter::<LSide, 128>::new(&sides);
        let mut tok = fixed::Tok::default();
        let mut ne = 0u64;
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= chunks.len() {
                break;
            }
            let d = chunks[i].data;
            populate(d);
            fixed::scan::<2, _>(&mut tok, d, |f| {
                let (a, b) = (f[0], f[1]);
                if a.is_empty() || b.is_empty() {
                    return;
                }
                for (side, k) in [a, b].into_iter().enumerate() {
                    let key = Key8::new(k);
                    let s = (key.hash(SEED_L) >> (64 - SIDE_BITS)) as usize;
                    w.push(s, LSide { key, ep: ((t as u64) << 40) | (ne * 2 + side as u64) });
                }
                ne += 1;
            });
        }
        w.finish();
        ne as usize
    });

    // Intern per shard; record each endpoint's (shard, local node).
    // Every slot in `eps` is written once before being read, so skip single-threaded zero-fill.
    let mut eps: Vec<Vec<u64>> = edges
        .iter()
        .map(|&e| {
            let mut v = Vec::with_capacity(e * 2);
            unsafe { v.set_len(e * 2) };
            v
        })
        .collect();
    let ep_ptrs: Vec<SyncPtr> = eps.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
    let next = AtomicUsize::new(0);
    let shard_keys: Vec<Vec<(usize, Vec<Key8>)>> = par(threads, |_| {
        let mut res = Vec::new();
        let mut tab: Vec<u32> = Vec::new();
        loop {
            let s = next.fetch_add(1, Ordering::Relaxed);
            if s >= SIDE_SHARDS {
                break;
            }
            let segs = sides.take(s);
            let n = segs.len();
            let cap = (n * 2).next_power_of_two().max(16);
            let mask = cap - 1;
            tab.clear();
            tab.resize(cap, 0);
            let mut keys: Vec<Key8> = Vec::with_capacity(n / 2 + 16);
            let tptr = tab.as_mut_ptr();
            const PF: usize = 8;
            for seg in segs.iter() {
                let slen = seg.len();
                for ri in 0..slen {
                    if ri + PF < slen {
                        let fh = unsafe { seg.get_unchecked(ri + PF).key.hash(SEED_L) };
                        prefetch(unsafe { tptr.add(fh as usize & mask) } as *const u8);
                    }
                    let side = unsafe { seg.get_unchecked(ri) };
                    let mut i = side.key.hash(SEED_L) as usize & mask;
                    let node;
                    loop {
                        let e = unsafe { *tptr.add(i) };
                        if e == 0 {
                            node = keys.len() as u32;
                            keys.push(side.key);
                            unsafe { *tptr.add(i) = node + 1 };
                            break;
                        }
                        if unsafe { *keys.get_unchecked((e - 1) as usize) } == side.key {
                            node = e - 1;
                            break;
                        }
                        i = (i + 1) & mask;
                    }
                    let ep = unsafe { ep_ptrs.get_unchecked((side.ep >> 40) as usize).0 };
                    unsafe { *ep.add((side.ep & ((1 << 40) - 1)) as usize) = ((s as u64) << 32) | node as u64 };
                }
            }
            segs.free();
            res.push((s, keys));
        }
        res
    });
    sides.free_all();
    let mut per_shard: Vec<Vec<Key8>> = (0..SIDE_SHARDS).map(|_| Vec::new()).collect();
    for (s, k) in shard_keys.into_iter().flatten() {
        per_shard[s] = k;
    }
    let mut base = vec![0u32; SIDE_SHARDS + 1];
    for s in 0..SIDE_SHARDS {
        base[s + 1] = base[s] + per_shard[s].len() as u32;
    }
    let n = base[SIDE_SHARDS] as usize;
    let mut keys: Vec<Key8> = Vec::with_capacity(n);
    unsafe { keys.set_len(n) };
    let mut parent: Vec<AtomicU32> = Vec::with_capacity(n);
    unsafe { parent.set_len(n) };
    {
        let kp = SyncPtr(keys.as_mut_ptr() as *mut u64);
        let pp = SyncPtr(parent.as_mut_ptr() as *mut u64);
        let kp = &kp;
        let pp = &pp;
        let per_shard = &per_shard;
        let base = &base;
        let step_s = SIDE_SHARDS.div_ceil(threads).max(1);
        par(threads, |t| {
            let kdst = kp.0 as *mut Key8;
            let pdst = pp.0 as *mut AtomicU32;
            for s in (t * step_s).min(SIDE_SHARDS)..((t + 1) * step_s).min(SIDE_SHARDS) {
                let b = base[s] as usize;
                let src = &per_shard[s];
                if !src.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), kdst.add(b), src.len());
                    }
                }
                for i in b..b + src.len() {
                    unsafe { pdst.add(i).write(AtomicU32::new(i as u32)) };
                }
            }
        });
    }
    drop(per_shard);
    let keys = &keys;
    let parent = &parent;
    let less = |a: u32, b: u32| keys[a as usize].cmp_bytes(&keys[b as usize]) == std::cmp::Ordering::Less;
    let find = |mut x: u32| -> u32 {
        loop {
            let p = parent[x as usize].load(Ordering::Relaxed);
            if p == x {
                return x;
            }
            let g = parent[p as usize].load(Ordering::Relaxed);
            if g != p {
                let _ = parent[x as usize].compare_exchange_weak(p, g, Ordering::Relaxed, Ordering::Relaxed);
            }
            x = p;
        }
    };
    {
        const JOB: usize = 1 << 16;
        let jobs: Vec<(usize, usize)> =
            eps.iter().enumerate().flat_map(|(t, v)| (0..v.len()).step_by(JOB).map(move |o| (t, o))).collect();
        let next = AtomicUsize::new(0);
        let gid = |v: u64| base[(v >> 32) as usize] + (v as u32);
        par(threads, |_| loop {
            let j = next.fetch_add(1, Ordering::Relaxed);
            if j >= jobs.len() {
                break;
            }
            let (t, o) = jobs[j];
            let v = &eps[t][o..(o + JOB).min(eps[t].len())];
            for e in v.chunks_exact(2) {
                let (mut a, mut b) = (gid(e[0]), gid(e[1]));
                loop {
                    a = find(a);
                    b = find(b);
                    if a == b {
                        break;
                    }
                    let (lo, hi) = if less(a, b) { (a, b) } else { (b, a) };
                    if parent[hi as usize]
                        .compare_exchange(hi, lo, Ordering::Relaxed, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                }
            }
        });
    }
    drop(eps);

    // Every node belonging to a component of size >= 2 (both aliases AND their canonical root)
    // is stored in LinkTable so Tier-1 solve can tell truly unlinked users apart from linked roots.
    let step = n.div_ceil(threads).max(1);
    let has_alias: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(0)).collect();
    par(threads, |t| {
        for x in (t * step).min(n)..((t + 1) * step).min(n) {
            let r = find(x as u32);
            if r != x as u32 {
                has_alias[r as usize].store(1, Ordering::Relaxed);
                parent[x].store(r, Ordering::Relaxed);
            }
        }
    });
    let mut root_cid = vec![u32::MAX; n];
    let mut canon_keys: Vec<Key8> = Vec::new();
    for x in 0..n {
        if has_alias[x].load(Ordering::Relaxed) != 0 {
            root_cid[x] = canon_keys.len() as u32;
            canon_keys.push(keys[x]);
        }
    }
    drop(has_alias);
    let root_cid_ref = &root_cid;
    let groups: Vec<Vec<Vec<u64>>> = par(threads, |t| {
        let mut g: Vec<Vec<u64>> = (0..REGIONS).map(|_| Vec::new()).collect();
        for x in (t * step).min(n)..((t + 1) * step).min(n) {
            let r = parent[x].load(Ordering::Relaxed);
            let cid = root_cid_ref[r as usize];
            if cid != u32::MAX {
                let h = keys[x].hash(SEED_L);
                g[(h >> (64 - REGION_BITS)) as usize].push(((cid as u64) << 32) | x as u64);
            }
        }
        g
    });
    drop(root_cid);
    let mut counts = vec![0usize; REGIONS];
    for g in &groups {
        for (r, v) in g.iter().enumerate() {
            counts[r] += v.len();
        }
    }
    let total_nr: usize = counts.iter().sum();
    let bloom_words = ((total_nr * 12 / 64).next_power_of_two()).clamp(1 << 10, 1 << 20);
    let bloom_shift = 64 - bloom_words.trailing_zeros();
    let nbuckets = link_nbuckets(total_nr);
    let tab_shift = 64 - nbuckets.trailing_zeros();
    let mask = nbuckets - 1;
    let mut tab = HugeBuf::<LinkBucket>::new_zeroed(nbuckets);
    let mut bloom = HugeBuf::<u64>::new_zeroed(bloom_words);
    for g in &groups {
        for r in g {
            for &v in r {
                let (cid, x) = ((v >> 32) as u32, v as u32 as usize);
                let key = keys[x];
                let h = key.hash(SEED_L);
                bloom[bloom_word(h, bloom_shift)] |= bloom_bits(h);
                let mut i = (h >> tab_shift) as usize;
                while tab[i].len == 5 {
                    i = (i + 1) & mask;
                }
                let b = &mut tab[i];
                let k = b.len as usize;
                b.keys[k] = key;
                b.cids[k] = cid;
                b.len = (k + 1) as u32;
            }
        }
    }
    LinkTable {
        tab,
        mask,
        tab_shift,
        bloom,
        bloom_shift,
        canon_keys: HugeBuf::from_slice(&canon_keys),
        scratch_mmaps: Vec::new(),
        empty: total_nr == 0,
    }
}

/// Fast zero-atomic single-threaded sharded link-table builder: 128 L2-resident
/// intern shards + multi-row AVX2 link scanner + cached hash + flat non-atomic union-find.
fn build_links_seq_sharded(files: &[&'static [u8]]) -> LinkTable {
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct LSideSeq {
        key: Key8,
        ep: u32,
        h32: u32,
    }
    const _: () = assert!(std::mem::size_of::<LSideSeq>() == 16);

    struct LinkHandler<'a> {
        w: Scatter<'a, LSideSeq, 128>,
        ne: u32,
    }
    impl crate::event_front::LinkSink<'_> for &mut LinkHandler<'_> {
        #[inline(always)]
        fn on_link(&mut self, a: &[u8], b: &[u8]) {
            if a.is_empty() || b.is_empty() {
                return;
            }
            let ka = Key8::new(a);
            let kb = Key8::new(b);
            let ha = ka.hash(SEED_L);
            let hb = kb.hash(SEED_L);
            let ep = self.ne * 2;
            self.w.push((ha >> (64 - SIDE_BITS)) as usize, LSideSeq { key: ka, ep, h32: ha as u32 });
            self.w.push((hb >> (64 - SIDE_BITS)) as usize, LSideSeq { key: kb, ep: ep + 1, h32: hb as u32 });
            self.ne += 1;
        }
    }

    let sides: ShardSet<LSideSeq> = ShardSet::new_shared(SIDE_SHARDS);
    let ne = {
        let mut handler = LinkHandler { w: Scatter::<LSideSeq, 128>::new(&sides), ne: 0 };
        for &d in files {
            let start = match d.iter().position(|&c| c == b'\n') {
                Some(i) => i + 1,
                None => continue,
            };
            let body = &d[start..];
            if body.is_empty() {
                continue;
            }
            populate(body);
            crate::event_front::scan_links(body, &mut handler);
        }
        let ne = handler.ne;
        handler.w.finish();
        ne
    };

    let total_eps = (ne as usize) * 2;
    if total_eps == 0 {
        sides.free_all();
        return LinkTable {
            tab: HugeBuf::new_zeroed(4),
            mask: 3,
            tab_shift: 62,
            bloom: HugeBuf::new_zeroed(1 << 10),
            bloom_shift: 54,
            canon_keys: HugeBuf::empty(),
            scratch_mmaps: Vec::new(),
            empty: true,
        };
    }

    // Back all build-time arrays with 2 MB THP HugeBufs and NEVER munmap them during `events`
    // (munmap takes mmap_lock for WRITE and fires cross-core TLB shootdowns to all parsing threads!).
    let mut scratch_mmaps: Vec<(usize, usize)> = Vec::new();
    let mut eps = HugeBuf::<u32>::empty();
    eps.reserve_uninit(total_eps);
    unsafe { eps.set_len(total_eps) };
    let eptr = eps.as_mut_ptr();
    let mut keys = HugeBuf::<Key8>::empty();
    keys.reserve_uninit(total_eps);
    let kptr_mut = keys.as_mut_ptr();
    let mut n = 0usize;
    let mut tab = HugeBuf::<(Key8, u32)>::empty();
    const PF: usize = 16;
    for s in 0..SIDE_SHARDS {
        let segs = sides.take(s);
        let sn = segs.len();
        if sn == 0 {
            segs.free();
            continue;
        }
        let cap = (sn + sn / 2).next_power_of_two().max(16);
        let mask = cap - 1;
        tab.reset_zeroed(cap);
        let tptr = tab.as_mut_ptr();
        for seg in segs.iter() {
            let slen = seg.len();
            for ri in 0..slen {
                if ri + PF < slen {
                    let fs = unsafe { seg.get_unchecked(ri + PF) };
                    prefetch(unsafe { tptr.add(fs.h32 as usize & mask) } as *const u8);
                    prefetch(unsafe { eptr.add(fs.ep as usize) } as *const u8);
                }
                let side = unsafe { seg.get_unchecked(ri) };
                let mut i = side.h32 as usize & mask;
                let node;
                loop {
                    let e = unsafe { &mut *tptr.add(i) };
                    if e.0 == side.key {
                        node = e.1;
                        break;
                    }
                    if e.0.is_empty() {
                        node = n as u32;
                        unsafe { kptr_mut.add(n).write(side.key) };
                        n += 1;
                        *e = (side.key, node);
                        break;
                    }
                    i = (i + 1) & mask;
                }
                unsafe { *eptr.add(side.ep as usize) = node };
            }
        }
        segs.free();
    }
    sides.free_all();
    unsafe { keys.set_len(n) };
    if let Some(m) = tab.detach_mmap() {
        scratch_mmaps.push(m);
    }

    let mut parent = HugeBuf::<u32>::empty();
    parent.reserve_uninit(n);
    unsafe {
        let p = parent.as_mut_ptr();
        for i in 0..n {
            p.add(i).write(i as u32);
        }
        parent.set_len(n);
    }

    #[inline(always)]
    fn find_seq(parent: &mut [u32], mut x: u32) -> u32 {
        loop {
            let p = unsafe { *parent.get_unchecked(x as usize) };
            if p == x {
                return x;
            }
            let g = unsafe { *parent.get_unchecked(p as usize) };
            unsafe { *parent.get_unchecked_mut(x as usize) = g };
            x = g;
        }
    }

    let ep_pairs = eps.len() >> 1;
    let eptr_ro = eps.as_ptr();
    let pptr = parent.as_ptr();
    let kptr = keys.as_ptr();
    const UPF: usize = 16;
    for ei in 0..ep_pairs {
        if ei + UPF < ep_pairs {
            unsafe {
                let fa = *eptr_ro.add((ei + UPF) * 2) as usize;
                let fb = *eptr_ro.add((ei + UPF) * 2 + 1) as usize;
                prefetch(pptr.add(fa) as *const u8);
                prefetch(pptr.add(fb) as *const u8);
                prefetch(kptr.add(fa) as *const u8);
                prefetch(kptr.add(fb) as *const u8);
            }
        }
        let ea = unsafe { *eptr_ro.add(ei * 2) };
        let eb = unsafe { *eptr_ro.add(ei * 2 + 1) };
        let a = find_seq(&mut parent, ea);
        let b = find_seq(&mut parent, eb);
        if a != b {
            if unsafe { keys.get_unchecked(a as usize).cmp_bytes(keys.get_unchecked(b as usize)) }
                == std::cmp::Ordering::Less
            {
                unsafe { *parent.get_unchecked_mut(b as usize) = a };
            } else {
                unsafe { *parent.get_unchecked_mut(a as usize) = b };
            }
        }
    }
    if let Some(m) = eps.detach_mmap() {
        scratch_mmaps.push(m);
    }

    // Flatten union-find & tag each component root in place in `parent[r] = 0x8000_0000 | cid`
    // (zero extra `root_cid` / `has_alias` allocation!).
    #[inline(always)]
    fn find_seq_tagged(parent: &mut [u32], mut x: u32) -> u32 {
        loop {
            let p = unsafe { *parent.get_unchecked(x as usize) };
            if p == x || (p & 0x8000_0000) != 0 {
                return x;
            }
            let g = unsafe { *parent.get_unchecked(p as usize) };
            if (g & 0x8000_0000) != 0 {
                return p;
            }
            unsafe { *parent.get_unchecked_mut(x as usize) = g };
            x = g;
        }
    }

    assert!(n < 0x8000_0000, "too many link nodes");
    let mut canon_keys = HugeBuf::<Key8>::empty();
    canon_keys.reserve_uninit(n / 2 + 16);
    let ck_ptr = canon_keys.as_mut_ptr();
    let mut n_canon = 0usize;
    for x in 0..n {
        let p0 = unsafe { *parent.get_unchecked(x) };
        if (p0 & 0x8000_0000) != 0 {
            continue;
        }
        let r = find_seq_tagged(&mut parent, x as u32);
        if r != x as u32 {
            unsafe {
                *parent.get_unchecked_mut(x) = r;
                let pr = parent.get_unchecked_mut(r as usize);
                if (*pr & 0x8000_0000) == 0 {
                    *pr = 0x8000_0000 | (n_canon as u32);
                    ck_ptr.add(n_canon).write(*keys.get_unchecked(r as usize));
                    n_canon += 1;
                }
            }
        }
    }
    unsafe { canon_keys.set_len(n_canon) };
    // Resolve every node's cid tag in place in `parent` (zero `linked` Vec allocation!)
    // and count `total_nr`. Because `keys[0..n]` is already grouped by top `SIDE_BITS` (8 bits)
    // of `key.hash(SEED_L)`, top-bit indexing into `bloom` and `ltab` keeps every 1/256th
    // slice L1/L2-resident during insertion AND aligns with `ev_set`/`click_set` P_BITS in Tier 1!
    let mut total_nr = 0usize;
    for x in 0..n {
        let p = unsafe { *parent.get_unchecked(x) };
        let cid_tag = if (p & 0x8000_0000) != 0 {
            p
        } else {
            unsafe { *parent.get_unchecked(p as usize) }
        };
        if (cid_tag & 0x8000_0000) != 0 {
            unsafe { *parent.get_unchecked_mut(x) = cid_tag };
            total_nr += 1;
        } else {
            unsafe { *parent.get_unchecked_mut(x) = 0 };
        }
    }

    let bloom_words = ((total_nr * 12 / 64).next_power_of_two()).clamp(1 << 10, 1 << 20);
    let bloom_shift = 64 - bloom_words.trailing_zeros();
    let nbuckets = link_nbuckets(total_nr);
    let tab_shift = 64 - nbuckets.trailing_zeros();
    let mask = nbuckets - 1;
    let mut ltab = HugeBuf::<LinkBucket>::new_zeroed(nbuckets);
    let mut bloom = HugeBuf::<u64>::new_zeroed(bloom_words);
    let lptr = ltab.as_mut_ptr();
    let bptr = bloom.as_mut_ptr();
    let kptr = keys.as_ptr();
    let pptr = parent.as_ptr();
    for x in 0..n {
        let tag = unsafe { *pptr.add(x) };
        if (tag & 0x8000_0000) != 0 {
            let key = unsafe { *kptr.add(x) };
            let h = key.hash(SEED_L);
            let cid = tag & 0x7FFF_FFFF;
            unsafe {
                *bptr.add(bloom_word(h, bloom_shift)) |= bloom_bits(h);
                let mut i = (h >> tab_shift) as usize;
                while (*lptr.add(i)).len == 5 {
                    i = (i + 1) & mask;
                }
                let b = &mut *lptr.add(i);
                let k = b.len as usize;
                *b.keys.get_unchecked_mut(k) = key;
                *b.cids.get_unchecked_mut(k) = cid;
                b.len = (k + 1) as u32;
            }
        }
    }
    if let Some(m) = keys.detach_mmap() {
        scratch_mmaps.push(m);
    }
    if let Some(m) = parent.detach_mmap() {
        scratch_mmaps.push(m);
    }
    LinkTable {
        tab: ltab,
        mask,
        tab_shift,
        bloom,
        bloom_shift,
        canon_keys,
        scratch_mmaps,
        empty: total_nr == 0,
    }
}

/// Fast zero-sync single-threaded link-table builder (retained for reference/tests).
#[allow(dead_code)]
pub fn build_links_single(files: &[&'static [u8]]) -> LinkTable {
    let total_bytes: usize = files.iter().map(|d| d.len()).sum();
    let est = (total_bytes / 18).max(64);
    let mut cap = (est * 2).next_power_of_two().max(256);
    let mut tab: Vec<u32> = vec![0; cap];
    let mut mask = cap - 1;
    let mut keys: Vec<Key8> = Vec::with_capacity(est);
    let mut parent: Vec<u32> = Vec::with_capacity(est);

    #[inline(always)]
    fn find_seq(parent: &mut [u32], mut x: u32) -> u32 {
        loop {
            let p = unsafe { *parent.get_unchecked(x as usize) };
            if p == x {
                return x;
            }
            let g = unsafe { *parent.get_unchecked(p as usize) };
            unsafe { *parent.get_unchecked_mut(x as usize) = g };
            x = g;
        }
    }

    #[cold]
    fn grow(tab: &mut Vec<u32>, cap: &mut usize, mask: &mut usize, keys: &[Key8]) {
        *cap *= 2;
        *mask = *cap - 1;
        tab.clear();
        tab.resize(*cap, 0);
        let m = *mask;
        for (id, k) in keys.iter().enumerate() {
            let mut i = k.hash(SEED_L) as usize & m;
            while unsafe { *tab.get_unchecked(i) } != 0 {
                i = (i + 1) & m;
            }
            unsafe { *tab.get_unchecked_mut(i) = id as u32 + 1 };
        }
    }

    for &d in files {
        let start = match d.iter().position(|&c| c == b'\n') {
            Some(i) => i + 1,
            None => continue,
        };
        let body = &d[start..];
        if body.is_empty() {
            continue;
        }
        populate(body);
        csv::scan::<2, _>(body, |f| {
            let (fa, fb) = (f[0], f[1]);
            if fa.is_empty() || fb.is_empty() {
                return;
            }
            let mut intern = |k: Key8| -> u32 {
                if keys.len() * 2 >= cap {
                    grow(&mut tab, &mut cap, &mut mask, &keys);
                }
                let mut i = k.hash(SEED_L) as usize & mask;
                loop {
                    let e = unsafe { *tab.get_unchecked(i) };
                    if e == 0 {
                        let id = keys.len() as u32;
                        keys.push(k);
                        parent.push(id);
                        unsafe { *tab.get_unchecked_mut(i) = id + 1 };
                        return id;
                    }
                    let id = e - 1;
                    if unsafe { *keys.get_unchecked(id as usize) } == k {
                        return id;
                    }
                    i = (i + 1) & mask;
                }
            };
            let ia = intern(Key8::new(fa));
            let ib = intern(Key8::new(fb));
            let a = find_seq(&mut parent, ia);
            let b = find_seq(&mut parent, ib);
            if a != b {
                if keys[a as usize].cmp_bytes(&keys[b as usize]) == std::cmp::Ordering::Less {
                    unsafe { *parent.get_unchecked_mut(b as usize) = a };
                } else {
                    unsafe { *parent.get_unchecked_mut(a as usize) = b };
                }
            }
        });
    }
    drop(tab);

    let n = keys.len();
    let mut root_cid = vec![u32::MAX; n];
    let mut canon_keys: Vec<Key8> = Vec::with_capacity(n / 2 + 16);
    for x in 0..n {
        let r = find_seq(&mut parent, x as u32);
        if r != x as u32 {
            parent[x] = r;
            if root_cid[r as usize] == u32::MAX {
                root_cid[r as usize] = canon_keys.len() as u32;
                canon_keys.push(keys[r as usize]);
            }
        }
    }
    let mut linked: Vec<(u32, u32, u64)> = Vec::with_capacity(n + 16);
    for x in 0..n {
        let r = parent[x];
        let cid = root_cid[r as usize];
        if cid != u32::MAX {
            let h = unsafe { keys.get_unchecked(x) }.hash(SEED_L);
            linked.push((x as u32, cid, h));
        }
    }
    drop(parent);
    drop(root_cid);

    let total_nr = linked.len();
    let bloom_words = ((total_nr * 12 / 64).next_power_of_two()).clamp(1 << 10, 1 << 20);
    let bloom_shift = 64 - bloom_words.trailing_zeros();
    let nbuckets = link_nbuckets(total_nr);
    let tab_shift = 64 - nbuckets.trailing_zeros();
    let mask = nbuckets - 1;
    let mut ltab = HugeBuf::<LinkBucket>::new_zeroed(nbuckets);
    let mut bloom = HugeBuf::<u64>::new_zeroed(bloom_words);
    let lptr = ltab.as_mut_ptr();
    let bptr = bloom.as_mut_ptr();
    let nlink = linked.len();
    const LPF: usize = 16;
    for idx in 0..nlink {
        if idx + LPF < nlink {
            let (_, _, fh) = unsafe { *linked.get_unchecked(idx + LPF) };
            prefetch(unsafe { bptr.add(bloom_word(fh, bloom_shift)) } as *const u8);
            prefetch(unsafe { lptr.add((fh >> tab_shift) as usize) } as *const u8);
        }
        let (x, cid, h) = unsafe { *linked.get_unchecked(idx) };
        unsafe {
            *bptr.add(bloom_word(h, bloom_shift)) |= bloom_bits(h);
            let mut i = (h >> tab_shift) as usize;
            while (*lptr.add(i)).len == 5 {
                i = (i + 1) & mask;
            }
            let b = &mut *lptr.add(i);
            let k = b.len as usize;
            *b.keys.get_unchecked_mut(k) = *keys.get_unchecked(x as usize);
            *b.cids.get_unchecked_mut(k) = cid;
            b.len = (k + 1) as u32;
        }
    }
    LinkTable {
        tab: ltab,
        mask,
        tab_shift,
        bloom,
        bloom_shift,
        canon_keys: HugeBuf::from_slice(&canon_keys),
        scratch_mmaps: Vec::new(),
        empty: total_nr == 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_min_name_resolution() {
        // Chain of 1000 hops in shuffled order, NUL/length-only differences, long IDs.
        let mut lines = String::from("user_id_a,user_id_b\n");
        let mut order: Vec<usize> = (0..999).collect();
        let mut x = 0x9e37_79b9u64;
        for i in (1..order.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            order.swap(i, (x % (i as u64 + 1)) as usize);
        }
        for &i in &order {
            lines.push_str(&format!("n{:04},n{:04}\n", i + 1, i));
        }
        lines.push_str("a\0,a\0b\nb,a\0\0\n");
        let long_a = "L".repeat(15) + "zz";
        let long_b = "L".repeat(15) + "ab";
        lines.push_str(&format!("{long_a},{long_b}\nM,{long_a}\nq,q\n"));
        let data: &'static [u8] = Box::leak(lines.into_bytes().into_boxed_slice());
        for mode in [0usize, 1, 4] {
            let lt = if mode == 0 { build_links_single(&[data]) } else { build_links(&[data], mode) };
            for i in 0..1000 {
                assert_eq!(lt.resolve(Key8::new(format!("n{i:04}").as_bytes())), Key8::new(b"n0000"));
            }
            assert_eq!(lt.resolve(Key8::new(b"a\0b")), Key8::new(b"a\0"));
            assert_eq!(lt.resolve(Key8::new(b"b")), Key8::new(b"a\0\0"));
            assert_eq!(lt.resolve(Key8::new(b"a")), Key8::new(b"a"));
            assert_eq!(lt.resolve(Key8::new(long_a.as_bytes())), Key8::new(long_b.as_bytes()));
            assert_eq!(lt.resolve(Key8::new(b"M")), Key8::new(long_b.as_bytes()));
            assert_eq!(lt.resolve(Key8::new(b"q")), Key8::new(b"q"));
            assert_eq!(lt.resolve(Key8::new(b"unlinked")), Key8::new(b"unlinked"));
            // Two-tier invariant: component roots of size >= 2 MUST be present in LinkTable,
            // while self-links and unlinked IDs must NOT be present.
            let k_root = Key8::new(b"n0000");
            let root_cid = lt.lookup(k_root.hash(SEED_L), k_root).expect("root in LinkTable");
            assert_eq!(lt.canon_key(root_cid), k_root);
            assert_eq!(lt.num_canon(), 4);
            let k_self = Key8::new(b"q");
            assert_eq!(lt.lookup(k_self.hash(SEED_L), k_self), None);
            let k_un = Key8::new(b"unlinked");
            assert_eq!(lt.lookup(k_un.hash(SEED_L), k_un), None);
        }
    }
}
