//! 2 MB pooled block arena, chained per-shard segment storage, and a
//! write-combining scatter writer.
//!
//! * `pool`: process-wide free list of 2 MB-aligned anonymous blocks (THP hinted on
//!   Linux). Blocks are reused across phases and released in parallel before exit.
//! * `ShardSet<T>`: `S` shards of fixed-size records. Each shard owns a chain of
//!   blocks carved into `SEG`-byte segments; a writer thread claims one segment of
//!   a shard at a time, so no shard can ever overflow (it just chains more blocks).
//! * `Scatter<T>`: per-thread writer with one 64-byte-aligned staging area per shard.
//!   When a staging area fills (64 B, or 192 B for 24/48-byte records) it is copied
//!   to the shard's segment with non-temporal stores (AVX2 on x86_64).

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub const BLOCK: usize = 2 << 20;
/// Segment size: a multiple of 256, 192, 128, and 64 (so 16/24/32/48-byte records
/// and 128B/192B staging flushes never straddle a segment).
pub const SEG: usize = 84 * 192; // 16128 = 63 * 256 = 126 * 128 = 252 * 64
const SLAB_BLOCKS: usize = 16;

pub const MAX_BLOCKS: usize = 8_190;
#[cfg(test)]
pub const LOC_EMPTY: u32 = u32::MAX - 1;
pub const LOC_INVALID: u32 = u32::MAX;
pub const LOC_PURCH_BIT: u32 = 0x8000_0000;

static BLOCK_ADDRS: [AtomicUsize; MAX_BLOCKS] = [const { AtomicUsize::new(0) }; MAX_BLOCKS];

/// Resolves a `loc` returned by `Scatter::push_with_loc` into a raw pointer to the 16-byte record.
#[inline(always)]
pub unsafe fn loc_ptr<T>(loc: u32) -> *mut T {
    const {
        assert!(std::mem::size_of::<T>() == 16);
    }
    let bid = (loc >> 17) as usize;
    let off = (loc & 0x1_FFFF) as usize;
    let base = BLOCK_ADDRS.get_unchecked(bid).load(Ordering::Relaxed);
    (base as *mut T).add(off)
}

/// Resolves a `loc` returned by `Scatter::push_with_loc12` into a raw pointer to the 12-byte record.
#[inline(always)]
pub unsafe fn loc_ptr12<T>(loc: u32) -> *mut T {
    const {
        assert!(std::mem::size_of::<T>() == 12);
    }
    let bid = (loc >> 18) as usize;
    let off = (loc & 0x3_FFFF) as usize;
    let base = BLOCK_ADDRS.get_unchecked(bid).load(Ordering::Relaxed);
    (base as *mut T).add(off)
}

// ---------------------------------------------------------------------------
// block pool

struct PoolInner {
    free: Vec<(usize, u32)>,
    slabs: Vec<(usize, usize)>,
    next_id: u32,
}

static POOL: Mutex<PoolInner> = Mutex::new(PoolInner { free: Vec::new(), slabs: Vec::new(), next_id: 0 });

fn map_slab(inner: &mut PoolInner) {
    let want = SLAB_BLOCKS * BLOCK;
    let len = want + BLOCK;
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert!(p != libc::MAP_FAILED, "mmap block slab failed");
    let start = p as usize;
    let aligned = (start + BLOCK - 1) & !(BLOCK - 1);
    unsafe {
        if aligned > start {
            libc::munmap(start as *mut libc::c_void, aligned - start);
        }
        let end = aligned + want;
        if start + len > end {
            libc::munmap(end as *mut libc::c_void, start + len - end);
        }
        #[cfg(target_os = "linux")]
        libc::madvise(aligned as *mut libc::c_void, want, libc::MADV_HUGEPAGE);
    }
    inner.slabs.push((aligned, want));
    for i in (0..SLAB_BLOCKS).rev() {
        let addr = aligned + i * BLOCK;
        let id = inner.next_id;
        assert!((id as usize) < MAX_BLOCKS, "arena block id overflow");
        inner.next_id += 1;
        BLOCK_ADDRS[id as usize].store(addr, Ordering::Relaxed);
        inner.free.push((addr, id));
    }
}

pub fn get_block() -> (usize, u32) {
    let mut g = POOL.lock().unwrap();
    if g.free.is_empty() {
        map_slab(&mut g);
    }
    g.free.pop().unwrap()
}

pub fn put_blocks(blocks: &[(usize, u32)]) {
    if blocks.is_empty() {
        return;
    }
    POOL.lock().unwrap().free.extend_from_slice(blocks);
}

/// Bytes of address space mapped by the pool.
#[cfg(test)]
pub fn mapped_bytes() -> usize {
    POOL.lock().unwrap().slabs.iter().map(|s| s.1).sum()
}

/// Removes and returns every block currently in `POOL.free` so caller threads can
/// release them concurrently alongside file writes without spawning extra threads.
pub fn take_free_blocks() -> Vec<(usize, u32)> {
    let mut g = POOL.lock().unwrap();
    std::mem::take(&mut g.free)
}

/// Releases one 2 MB block back to the kernel via `MADV_DONTNEED`.
#[inline]
pub fn release_block(addr: usize) {
    #[cfg(all(target_os = "linux", not(test)))]
    unsafe {
        libc::madvise(addr as *mut libc::c_void, BLOCK, libc::MADV_DONTNEED);
    }
    #[cfg(any(not(target_os = "linux"), test))]
    let _ = addr;
}

/// Returns every block currently in `POOL.free` to the kernel, split across `threads` threads,
/// so process exit does not free them on one core. Caller-owned blocks (e.g. output Parquet
/// columns still being written to disk) are untouched and freed via `release_owned_blocks`.
#[allow(dead_code)]
pub fn release_free(threads: usize) {
    let blocks: Vec<(usize, u32)> = {
        let mut g = POOL.lock().unwrap();
        std::mem::take(&mut g.free)
    };
    if blocks.is_empty() {
        return;
    }
    let threads = threads.clamp(1, blocks.len());
    std::thread::scope(|s| {
        for t in 0..threads {
            let blocks = &blocks;
            s.spawn(move || {
                for &(a, _) in blocks.iter().skip(t).step_by(threads) {
                    // MADV_DONTNEED only takes the mmap lock for reading, so the
                    // threads zap their blocks concurrently.
                    #[cfg(target_os = "linux")]
                    unsafe {
                        libc::madvise(a as *mut libc::c_void, BLOCK, libc::MADV_DONTNEED)
                    };
                    #[cfg(not(target_os = "linux"))]
                    let _ = a;
                }
            });
        }
    });
}

/// Releases caller-owned 2 MB arena blocks back to the kernel (or returns them to `POOL` in tests).
pub fn release_owned_blocks(blocks: &[(usize, u32)]) {
    #[cfg(all(target_os = "linux", not(test)))]
    for &(a, _) in blocks {
        unsafe {
            libc::madvise(a as *mut libc::c_void, BLOCK, libc::MADV_DONTNEED);
        }
    }
    #[cfg(any(not(target_os = "linux"), test))]
    put_blocks(blocks);
}

/// 2 MB-THP-backed contiguous buffer for hot random-access tables (`Scatter` staging,
/// `dedup` buckets, `LinkTable`, `SolveScratch.ht/ev_ts/cls`, `Worker` campaign tables).
/// - `bytes <= BLOCK` (2 MB): borrows one pooled 2 MB THP block from `POOL` (zero syscalls,
///   reuses already-faulted pages across phases, returned to `POOL` on `Drop`).
/// - `bytes > BLOCK`: allocates a 2 MB-aligned `MAP_ANON` slab with `MADV_HUGEPAGE` set
///   *before* first touch (zero upfront memset for OS-zeroed pages, `munmap` on `Drop`).
pub struct HugeBuf<T: Copy> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    backing: HugeBacking,
}

enum HugeBacking {
    Empty,
    Borrowed,
    PoolBlock(usize, u32),
    Mmap(usize, usize),
}

unsafe impl<T: Copy + Send> Send for HugeBuf<T> {}
unsafe impl<T: Copy + Sync> Sync for HugeBuf<T> {}

fn alloc_aligned_mmap(want: usize) -> (usize, usize) {
    let aligned_want = (want + BLOCK - 1) & !(BLOCK - 1);
    let len = aligned_want + BLOCK;
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert!(p != libc::MAP_FAILED, "HugeBuf mmap failed");
    let start = p as usize;
    let aligned = (start + BLOCK - 1) & !(BLOCK - 1);
    unsafe {
        if aligned > start {
            libc::munmap(start as *mut libc::c_void, aligned - start);
        }
        let end = aligned + aligned_want;
        if start + len > end {
            libc::munmap(end as *mut libc::c_void, start + len - end);
        }
        #[cfg(target_os = "linux")]
        libc::madvise(aligned as *mut libc::c_void, aligned_want, libc::MADV_HUGEPAGE);
    }
    (aligned, aligned_want)
}

#[allow(dead_code)]
impl<T: Copy> HugeBuf<T> {
    pub fn empty() -> Self {
        HugeBuf {
            ptr: std::ptr::NonNull::<T>::dangling().as_ptr(),
            len: 0,
            cap: 0,
            backing: HugeBacking::Empty,
        }
    }

    /// Creates a `HugeBuf` backed by a caller-owned slice inside a 2 MB THP block.
    /// If `reset_zeroed` / `reserve_uninit` ever exceeds `cap`, it transparently
    /// upgrades to its own `PoolBlock` or `Mmap` backing.
    pub unsafe fn from_borrowed(ptr: *mut T, cap: usize) -> Self {
        HugeBuf { ptr, len: 0, cap, backing: HugeBacking::Borrowed }
    }

    /// Creates a `HugeBuf` from a slice of `T` (used by cold/parallel paths).
    pub fn from_slice(src: &[T]) -> Self {
        let mut b = Self::empty();
        if !src.is_empty() {
            b.reserve_uninit(src.len());
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), b.ptr, src.len());
                b.set_len(src.len());
            }
        }
        b
    }

    /// Detaches any OS `Mmap` backing so the caller can release its pages later via
    /// `MADV_DONTNEED` instead of calling `munmap()` concurrently with worker threads.
    /// If backed by a `PoolBlock`, returns the block to `POOL` immediately (zero syscalls).
    pub fn detach_mmap(mut self) -> Option<(usize, usize)> {
        let res = match std::mem::replace(&mut self.backing, HugeBacking::Empty) {
            HugeBacking::Empty | HugeBacking::Borrowed => None,
            HugeBacking::PoolBlock(addr, id) => {
                put_blocks(&[(addr, id)]);
                None
            }
            HugeBacking::Mmap(addr, bytes) => Some((addr, bytes)),
        };
        self.ptr = std::ptr::NonNull::<T>::dangling().as_ptr();
        self.len = 0;
        self.cap = 0;
        res
    }

    pub fn new_zeroed(n: usize) -> Self {
        let mut b = Self::empty();
        b.reset_zeroed(n);
        b
    }

    /// Ensures capacity for `n` elements and sets `len = n` with all `n` elements zero-bytes.
    pub fn reset_zeroed(&mut self, n: usize) {
        if n == 0 {
            self.len = 0;
            return;
        }
        let need_bytes = n * std::mem::size_of::<T>();
        if n <= self.cap {
            unsafe { std::ptr::write_bytes(self.ptr, 0, n) };
            self.len = n;
            return;
        }
        self.release_backing();
        if need_bytes <= BLOCK {
            let (addr, id) = get_block();
            self.ptr = addr as *mut T;
            self.cap = BLOCK / std::mem::size_of::<T>();
            self.backing = HugeBacking::PoolBlock(addr, id);
            unsafe { std::ptr::write_bytes(self.ptr, 0, n) };
        } else {
            let (addr, mapped) = alloc_aligned_mmap(need_bytes);
            self.ptr = addr as *mut T;
            self.cap = mapped / std::mem::size_of::<T>();
            self.backing = HugeBacking::Mmap(addr, mapped);
            // Fresh anonymous mmap is already OS-zeroed!
        }
        self.len = n;
    }

    /// Ensures capacity for at least `n` elements without zeroing (caller initializes).
    pub fn reserve_uninit(&mut self, n: usize) {
        if n <= self.cap {
            return;
        }
        let need_bytes = n * std::mem::size_of::<T>();
        self.release_backing();
        if need_bytes <= BLOCK {
            let (addr, id) = get_block();
            self.ptr = addr as *mut T;
            self.cap = BLOCK / std::mem::size_of::<T>();
            self.backing = HugeBacking::PoolBlock(addr, id);
        } else {
            let (addr, mapped) = alloc_aligned_mmap(need_bytes);
            self.ptr = addr as *mut T;
            self.cap = mapped / std::mem::size_of::<T>();
            self.backing = HugeBacking::Mmap(addr, mapped);
        }
        self.len = 0;
    }

    #[inline(always)]
    pub unsafe fn set_len(&mut self, n: usize) {
        debug_assert!(n <= self.cap);
        self.len = n;
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    pub fn release_pages(&self) {
        #[cfg(all(target_os = "linux", not(test)))]
        match self.backing {
            HugeBacking::Empty | HugeBacking::Borrowed => {}
            HugeBacking::PoolBlock(addr, _) => {
                unsafe { libc::madvise(addr as *mut libc::c_void, BLOCK, libc::MADV_DONTNEED) };
            }
            HugeBacking::Mmap(addr, bytes) => {
                unsafe { libc::madvise(addr as *mut libc::c_void, bytes, libc::MADV_DONTNEED) };
            }
        }
    }

    fn release_backing(&mut self) {
        match std::mem::replace(&mut self.backing, HugeBacking::Empty) {
            HugeBacking::Empty | HugeBacking::Borrowed => {}
            HugeBacking::PoolBlock(addr, id) => {
                put_blocks(&[(addr, id)]);
            }
            HugeBacking::Mmap(addr, bytes) => unsafe {
                libc::munmap(addr as *mut libc::c_void, bytes);
            },
        }
        self.ptr = std::ptr::NonNull::<T>::dangling().as_ptr();
        self.len = 0;
        self.cap = 0;
    }
}

impl<T: Copy> std::ops::Deref for HugeBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Copy> std::ops::DerefMut for HugeBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: Copy> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        self.release_backing();
    }
}

/// Per-worker bump allocator carving 64-byte-aligned chunks out of pooled 2 MB THP blocks.
/// Used to back `user_daily` Parquet columns with already-faulted arena pages left over from
/// `id_set`/`ev_set`, avoiding fresh jemalloc zero-page faults during `solve`.
pub struct BlockCarver {
    cur_ptr: usize,
    cur_rem: usize,
    blocks: Vec<(usize, u32)>,
}

impl BlockCarver {
    pub fn new() -> BlockCarver {
        BlockCarver { cur_ptr: 0, cur_rem: 0, blocks: Vec::new() }
    }

    #[inline]
    pub fn alloc(&mut self, bytes: usize) -> *mut u8 {
        let aligned = (bytes + 63) & !63;
        debug_assert!(aligned <= BLOCK);
        if self.cur_rem < aligned {
            let (b, id) = get_block();
            self.blocks.push((b, id));
            self.cur_ptr = b;
            self.cur_rem = BLOCK;
        }
        let p = self.cur_ptr as *mut u8;
        self.cur_ptr += aligned;
        self.cur_rem -= aligned;
        p
    }

    pub fn take_blocks(&mut self) -> Vec<(usize, u32)> {
        self.cur_ptr = 0;
        self.cur_rem = 0;
        std::mem::take(&mut self.blocks)
    }
}

/// Returns every pooled page to the kernel, split across `threads` threads, so
/// process exit does not free them on one core. All blocks must be back in the pool.
#[allow(dead_code)]
pub fn release_all(threads: usize) {
    let slabs: Vec<(usize, usize)> = {
        let mut g = POOL.lock().unwrap();
        g.free.clear();
        std::mem::take(&mut g.slabs)
    };
    let pieces: Vec<(usize, usize)> =
        slabs.iter().flat_map(|&(a, len)| (0..len / BLOCK).map(move |i| (a + i * BLOCK, BLOCK))).collect();
    if pieces.is_empty() {
        return;
    }
    let threads = threads.clamp(1, pieces.len());
    std::thread::scope(|s| {
        for t in 0..threads {
            let pieces = &pieces;
            s.spawn(move || {
                for &(a, len) in pieces.iter().skip(t).step_by(threads) {
                    // MADV_DONTNEED only takes the mmap lock for reading, so the
                    // threads zap their blocks concurrently.
                    #[cfg(target_os = "linux")]
                    unsafe {
                        libc::madvise(a as *mut libc::c_void, len, libc::MADV_DONTNEED)
                    };
                    #[cfg(not(target_os = "linux"))]
                    unsafe {
                        libc::munmap(a as *mut libc::c_void, len)
                    };
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// shard set

#[derive(Clone, Copy)]
pub struct Seg {
    pub ptr: usize,
    pub n: u32,
}

struct ShardState {
    blocks: Vec<(usize, u32)>,
    next_seg: usize, // segments used in the last block
    segs: Vec<Seg>,
}

struct GroupPool {
    #[allow(dead_code)]
    remaining: AtomicUsize,
    blocks: Mutex<Vec<(usize, u32)>>,
}

pub struct ShardSet<T, const SEG_SZ: usize = SEG> {
    shards: Vec<Mutex<ShardState>>,
    groups: Vec<GroupPool>,
    group_shift: u32,
    _t: PhantomData<T>,
}

unsafe impl<T, const SEG_SZ: usize> Sync for ShardSet<T, SEG_SZ> {}
unsafe impl<T, const SEG_SZ: usize> Send for ShardSet<T, SEG_SZ> {}

/// One shard's records, as returned by `ShardSet::take`.
pub struct Segs<T> {
    pub segs: Vec<Seg>,
    blocks: Vec<(usize, u32)>,
    _t: PhantomData<T>,
}

unsafe impl<T> Send for Segs<T> {}

impl<T: Copy> Segs<T> {
    pub fn len(&self) -> usize {
        self.segs.iter().map(|s| s.n as usize).sum()
    }
    #[inline(always)]
    pub fn seg(&self, i: usize) -> &[T] {
        let s = self.segs[i];
        unsafe { std::slice::from_raw_parts(s.ptr as *const T, s.n as usize) }
    }
    pub fn iter(&self) -> impl Iterator<Item = &[T]> + '_ {
        (0..self.segs.len()).map(move |i| self.seg(i))
    }
    /// Returns this shard's blocks to the pool.
    pub fn free(self) {
        put_blocks(&self.blocks);
    }
}

impl<T: Copy, const SEG_SZ: usize> ShardSet<T, SEG_SZ> {
    const SEGS_PER_BLK: usize = BLOCK / SEG_SZ;

    /// Thread-local 2 MB blocks carved lock-free into `SEG_SZ`-byte segments across `ngroups`
    /// contiguous shard groups (`ngroups` must be a power of 2 <= `n` and divide `n`).
    /// When `ngroups > 1`, calling `mark_shard_done(s)` after consuming shard `s`
    /// returns each group's blocks to `POOL` as soon as that group finishes, enabling
    /// incremental hugepage reuse by the next phase while keeping active TLB footprint tiny.
    pub fn new_grouped(n: usize, ngroups: usize) -> ShardSet<T, SEG_SZ> {
        assert!(SEG_SZ % std::mem::size_of::<T>() == 0);
        let ng = if n >= ngroups && n.is_power_of_two() && ngroups.is_power_of_two() {
            ngroups
        } else {
            1
        };
        let per_group = n / ng;
        let group_shift = if ng == 1 { 63 } else { per_group.trailing_zeros() };
        ShardSet {
            shards: (0..n)
                .map(|_| Mutex::new(ShardState { blocks: Vec::new(), next_seg: Self::SEGS_PER_BLK, segs: Vec::new() }))
                .collect(),
            groups: (0..ng)
                .map(|_| GroupPool { remaining: AtomicUsize::new(per_group), blocks: Mutex::new(Vec::new()) })
                .collect(),
            group_shift,
            _t: PhantomData,
        }
    }

    /// Single-group thread-local 2 MB block carving (`ngroups = 1`).
    pub fn new_shared(n: usize) -> ShardSet<T, SEG_SZ> {
        Self::new_grouped(n, 1)
    }

    /// Alias kept for tests.
    #[cfg(test)]
    pub fn new(n: usize) -> ShardSet<T, SEG_SZ> {
        Self::new_grouped(n, 1)
    }

    #[inline(always)]
    fn group_of(&self, s: usize) -> usize {
        if self.groups.len() == 1 {
            0
        } else {
            s >> self.group_shift
        }
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    fn push_seg(&self, s: usize, done: Seg) {
        if done.n > 0 {
            self.shards[s].lock().unwrap().segs.push(done);
        }
    }

    fn register_group_blocks(&self, g: usize, blocks: &[(usize, u32)]) {
        if !blocks.is_empty() {
            self.groups[g].blocks.lock().unwrap().extend_from_slice(blocks);
        }
    }

    /// Marks shard `s` completely consumed (after `Segs::free()`). Once every shard in
    /// its group finishes, that group's 2 MB blocks return to `POOL` immediately.
    #[cfg(test)]
    pub fn mark_shard_done(&self, s: usize) {
        if self.groups.len() <= 1 {
            return;
        }
        let g = self.group_of(s);
        if self.groups[g].remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let blocks = std::mem::take(&mut *self.groups[g].blocks.lock().unwrap());
            put_blocks(&blocks);
        }
    }

    /// Number of records currently stored in shard `s` (all writers finished).
    pub fn count(&self, s: usize) -> usize {
        self.shards[s].lock().unwrap().segs.iter().map(|x| x.n as usize).sum()
    }

    /// Removes shard `s`'s records (all writers must have finished).
    pub fn take(&self, s: usize) -> Segs<T> {
        let mut g = self.shards[s].lock().unwrap();
        g.next_seg = Self::SEGS_PER_BLK;
        Segs { segs: std::mem::take(&mut g.segs), blocks: std::mem::take(&mut g.blocks), _t: PhantomData }
    }

    /// Returns all remaining blocks to the pool (via `&self`, for mid-pipeline barriers).
    pub fn release_blocks(&self) {
        for gp in &self.groups {
            let blocks = std::mem::take(&mut *gp.blocks.lock().unwrap());
            put_blocks(&blocks);
        }
        for sh in &self.shards {
            let blocks = std::mem::take(&mut sh.lock().unwrap().blocks);
            put_blocks(&blocks);
        }
    }

    /// Returns all remaining blocks to the pool.
    pub fn free_all(self) {
        for gp in self.groups {
            let blocks = gp.blocks.into_inner().unwrap();
            put_blocks(&blocks);
        }
        for sh in self.shards {
            let blocks = sh.into_inner().unwrap().blocks;
            put_blocks(&blocks);
        }
    }
}

// ---------------------------------------------------------------------------
// scatter writer

#[derive(Clone, Copy)]
struct Dest {
    dst: usize, // next write address in current segment (0 = none)
    rem: u32,   // bytes remaining in current segment (0 = full/unallocated)
}

struct GroupCur {
    block: usize,
    block_id: u32,
    next_seg: usize,
    blocks: Vec<(usize, u32)>,
}

pub struct Scatter<'a, T: Copy, const STAGE: usize = 64, const SEG_SZ: usize = SEG> {
    set: &'a ShardSet<T, SEG_SZ>,
    /// Single 2 MB THP block holding `stage`, `fill`, `dest`, and `loc_next` (1 L1 dTLB entry!).
    meta_block: Option<(usize, u32)>,
    stage_ptr: *mut u8,
    fill_ptr: *mut u8,
    dest_ptr: *mut Dest,
    loc_ptr: *mut u32,
    nshards: usize,
    located: bool,
    nt: bool,
    gcur: Vec<GroupCur>,
}

impl<'a, T: Copy, const STAGE: usize, const SEG_SZ: usize> Drop for Scatter<'a, T, STAGE, SEG_SZ> {
    fn drop(&mut self) {
        if let Some(b) = self.meta_block.take() {
            put_blocks(&[b]);
        }
    }
}

impl<'a, T: Copy, const STAGE: usize, const SEG_SZ: usize> Scatter<'a, T, STAGE, SEG_SZ> {
    const SZ: usize = std::mem::size_of::<T>();
    const PER: u8 = (STAGE / std::mem::size_of::<T>()) as u8;
    const SEGS_PER_BLK: usize = BLOCK / SEG_SZ;

    pub fn new(set: &'a ShardSet<T, SEG_SZ>) -> Scatter<'a, T, STAGE, SEG_SZ> {
        const {
            assert!(STAGE >= 32 && STAGE % std::mem::size_of::<T>() == 0);
            assert!(STAGE / std::mem::size_of::<T>() <= 255);
            assert!(SEG_SZ % STAGE == 0);
        }
        let n = set.len();
        let ng = set.groups.len();
        let stage_bytes = (n * STAGE + 63) & !63;
        let fill_bytes = (n + 63) & !63;
        let dest_bytes = (n * std::mem::size_of::<Dest>() + 63) & !63;
        let loc_bytes = (n * std::mem::size_of::<u32>() + 63) & !63;
        assert!(stage_bytes + fill_bytes + dest_bytes + loc_bytes <= BLOCK, "Scatter meta exceeds 2MB");
        let (baddr, bid) = get_block();
        let stage_ptr = baddr as *mut u8;
        let fill_ptr = (baddr + stage_bytes) as *mut u8;
        let dest_ptr = (baddr + stage_bytes + fill_bytes) as *mut Dest;
        let loc_ptr = (baddr + stage_bytes + fill_bytes + dest_bytes) as *mut u32;
        unsafe {
            std::ptr::write_bytes(fill_ptr, 0, n);
            std::ptr::write_bytes(dest_ptr, 0, n);
        }
        Scatter {
            set,
            meta_block: Some((baddr, bid)),
            stage_ptr,
            fill_ptr,
            dest_ptr,
            loc_ptr,
            nshards: n,
            located: false,
            nt: nt_enabled() && (STAGE % 64 == 0),
            gcur: (0..ng)
                .map(|_| GroupCur { block: 0, block_id: 0, next_seg: Self::SEGS_PER_BLK, blocks: Vec::new() })
                .collect(),
        }
    }

    /// Creates a `Scatter` for 16-byte records supporting `push_with_loc`, which returns the
    /// exact stable `loc` where each staged record will land after flushing (NT or tail).
    /// `Self::PER` must be a power of 2 so `loc & (PER - 1)` gives the exact staging slot
    /// without touching `fill`.
    pub fn new_located(set: &'a ShardSet<T, SEG_SZ>) -> Scatter<'a, T, STAGE, SEG_SZ> {
        const {
            assert!(std::mem::size_of::<T>() == 16);
            assert!((STAGE / 16).is_power_of_two());
            assert!((SEG_SZ / 16) % (STAGE / 16) == 0);
            assert!((BLOCK / SEG_SZ) * (SEG_SZ / 16) <= (1 << 17));
        }
        let mut sc = Self::new(set);
        sc.located = true;
        unsafe { std::ptr::write_bytes(sc.loc_ptr, 0xFF, sc.nshards) }; // LOC_INVALID = u32::MAX
        sc
    }

    /// Creates a `Scatter` for 12-byte records supporting `push_with_loc12` (bit 31 always 0).
    pub fn new_located12(set: &'a ShardSet<T, SEG_SZ>) -> Scatter<'a, T, STAGE, SEG_SZ> {
        const {
            assert!(std::mem::size_of::<T>() == 12);
            assert!((STAGE / 12).is_power_of_two());
            assert!((SEG_SZ / 12) % (STAGE / 12) == 0);
            assert!((BLOCK / SEG_SZ) * (SEG_SZ / 12) < (1 << 18));
        }
        let mut sc = Self::new(set);
        sc.located = true;
        unsafe { std::ptr::write_bytes(sc.loc_ptr, 0xFF, sc.nshards) }; // LOC_INVALID = u32::MAX
        sc
    }

    #[inline(always)]
    fn alloc_local_seg(&mut self, s: usize) -> (usize, u32) {
        let g = self.set.group_of(s);
        let gc = unsafe { self.gcur.get_unchecked_mut(g) };
        if gc.next_seg == Self::SEGS_PER_BLK {
            let (b, bid) = get_block();
            gc.blocks.push((b, bid));
            gc.block = b;
            gc.block_id = bid;
            gc.next_seg = 0;
        }
        let off = gc.next_seg * SEG_SZ;
        let p = gc.block + off;
        let loc = (gc.block_id << 17) | ((off >> 4) as u32);
        gc.next_seg += 1;
        (p, loc)
    }

    #[inline(always)]
    fn alloc_local_seg12(&mut self, s: usize) -> (usize, u32) {
        let g = self.set.group_of(s);
        let gc = unsafe { self.gcur.get_unchecked_mut(g) };
        if gc.next_seg == Self::SEGS_PER_BLK {
            let (b, bid) = get_block();
            gc.blocks.push((b, bid));
            gc.block = b;
            gc.block_id = bid;
            gc.next_seg = 0;
        }
        let seg_idx = gc.next_seg;
        let p = gc.block + seg_idx * SEG_SZ;
        let loc = (gc.block_id << 18) | ((seg_idx * (SEG_SZ / 12)) as u32);
        gc.next_seg = seg_idx + 1;
        (p, loc)
    }

    #[cold]
    #[inline(never)]
    fn swap_seg(&mut self, s: usize, prev_end: usize) -> usize {
        if prev_end != 0 {
            self.set.push_seg(s, Seg { ptr: prev_end - SEG_SZ, n: (SEG_SZ / Self::SZ) as u32 });
        }
        self.alloc_local_seg(s).0
    }

    #[cold]
    #[inline(never)]
    fn ensure_seg_loc(&mut self, s: usize) -> u32 {
        let (p, loc) = self.alloc_local_seg(s);
        unsafe {
            *self.dest_ptr.add(s) = Dest { dst: p, rem: SEG_SZ as u32 };
        }
        loc
    }

    #[cold]
    #[inline(never)]
    fn ensure_seg_loc12(&mut self, s: usize) -> u32 {
        let (p, loc) = self.alloc_local_seg12(s);
        unsafe {
            *self.dest_ptr.add(s) = Dest { dst: p, rem: SEG_SZ as u32 };
        }
        loc
    }

    #[cold]
    #[inline(never)]
    fn finish_full_seg_loc(&mut self, s: usize, seg_end: usize) {
        self.set.push_seg(s, Seg { ptr: seg_end - SEG_SZ, n: (SEG_SZ / Self::SZ) as u32 });
        unsafe {
            *self.dest_ptr.add(s) = Dest { dst: 0, rem: 0 };
            *self.loc_ptr.add(s) = LOC_INVALID;
        }
    }

    #[inline(always)]
    pub fn prefetch_shard(&self, s: usize) {
        unsafe {
            crate::links::prefetch(self.stage_ptr.add(s * STAGE));
        }
    }

    #[inline(always)]
    pub fn prefetch_located_shard(&self, s: usize) {
        unsafe {
            crate::links::prefetch(self.stage_ptr.add(s * STAGE));
        }
    }

    #[inline(always)]
    pub fn push(&mut self, s: usize, x: T) {
        unsafe {
            let fp = self.fill_ptr.add(s);
            let f = *fp;
            let st = self.stage_ptr.add(s * STAGE) as *mut T;
            st.add(f as usize).write(x);
            let nf = f + 1;
            if nf == Self::PER {
                *fp = 0;
                self.flush(s);
            } else {
                *fp = nf;
            }
        }
    }

    #[inline(always)]
    pub fn push_with_loc(&mut self, s: usize, x: T) -> u32 {
        const {
            assert!(std::mem::size_of::<T>() == 16);
            assert!((STAGE / 16).is_power_of_two());
        }
        let per_mask = (Self::PER as u32) - 1;
        unsafe {
            let lp = self.loc_ptr.add(s);
            let mut loc = *lp;
            if loc == LOC_INVALID {
                loc = self.ensure_seg_loc(s);
            }
            let f = (loc & per_mask) as usize;
            let st = self.stage_ptr.add(s * STAGE) as *mut T;
            st.add(f).write(x);
            let nloc = loc + 1;
            *lp = nloc;
            if (nloc & per_mask) == 0 {
                self.flush_located(s);
            }
            loc
        }
    }

    #[inline(always)]
    pub fn push_with_loc12(&mut self, s: usize, x: T) -> u32 {
        const {
            assert!(std::mem::size_of::<T>() == 12);
            assert!((STAGE / 12).is_power_of_two());
        }
        let per_mask = (Self::PER as u32) - 1;
        unsafe {
            let lp = self.loc_ptr.add(s);
            let mut loc = *lp;
            if loc == LOC_INVALID {
                loc = self.ensure_seg_loc12(s);
            }
            let f = (loc & per_mask) as usize;
            let st = self.stage_ptr.add(s * STAGE) as *mut T;
            st.add(f).write(x);
            let nloc = loc + 1;
            *lp = nloc;
            if (nloc & per_mask) == 0 {
                self.flush_located(s);
            }
            loc
        }
    }

    /// Streams a contiguous slice of items into one fixed shard `s`, hoisting the
    /// shard metadata and staging pointer out of the loop (used for Tier-1 -> Tier-2 runs).
    #[inline(always)]
    pub fn push_mapped<U: Copy>(&mut self, s: usize, items: &[U], mut map: impl FnMut(U) -> T) {
        if items.is_empty() {
            return;
        }
        unsafe {
            let mut f = *self.fill_ptr.add(s);
            let mut d = *self.dest_ptr.add(s);
            let src = self.stage_ptr.add(s * STAGE);
            let st = src as *mut T;
            let nt = self.nt;
            for &item in items {
                st.add(f as usize).write(map(item));
                f += 1;
                if f == Self::PER {
                    let mut dst = d.dst;
                    let mut rem = d.rem;
                    if rem == 0 {
                        dst = self.swap_seg(s, dst);
                        rem = SEG_SZ as u32;
                    }
                    copy_stage::<STAGE>(nt, dst as *mut u8, src);
                    d = Dest { dst: dst + STAGE, rem: rem - STAGE as u32 };
                    f = 0;
                }
            }
            *self.fill_ptr.add(s) = f;
            *self.dest_ptr.add(s) = d;
        }
    }

    #[inline(always)]
    fn flush(&mut self, s: usize) {
        let d = unsafe { *self.dest_ptr.add(s) };
        let mut dst = d.dst;
        let mut rem = d.rem;
        if rem == 0 {
            dst = self.swap_seg(s, dst);
            rem = SEG_SZ as u32;
        }
        let src = unsafe { self.stage_ptr.add(s * STAGE) };
        unsafe { copy_stage::<STAGE>(self.nt, dst as *mut u8, src) };
        unsafe {
            *self.dest_ptr.add(s) = Dest { dst: dst + STAGE, rem: rem - STAGE as u32 };
        }
    }

    #[inline(always)]
    fn flush_located(&mut self, s: usize) {
        let d = unsafe { *self.dest_ptr.add(s) };
        let src = unsafe { self.stage_ptr.add(s * STAGE) };
        unsafe { copy_stage::<STAGE>(self.nt, d.dst as *mut u8, src) };
        let next_dst = d.dst + STAGE;
        let next_rem = d.rem - STAGE as u32;
        if next_rem == 0 {
            self.finish_full_seg_loc(s, next_dst);
        } else {
            unsafe {
                *self.dest_ptr.add(s) = Dest { dst: next_dst, rem: next_rem };
            }
        }
    }

    /// Writes out every partial staging area and publishes all segments.
    pub fn finish(mut self) {
        let located = self.located;
        let per_mask = (Self::PER as usize) - 1;
        for s in 0..self.nshards {
            let f = if located {
                let loc = unsafe { *self.loc_ptr.add(s) };
                if loc == LOC_INVALID { 0 } else { (loc as usize) & per_mask }
            } else {
                unsafe { *self.fill_ptr.add(s) as usize }
            };
            let d = unsafe { *self.dest_ptr.add(s) };
            let (mut dst, mut rem) = (d.dst, d.rem as usize);
            if f > 0 {
                if rem == 0 {
                    dst = self.swap_seg(s, dst);
                    rem = SEG_SZ;
                }
                let bytes = f * Self::SZ;
                unsafe {
                    let src = self.stage_ptr.add(s * STAGE);
                    std::ptr::copy_nonoverlapping(src, dst as *mut u8, bytes);
                }
                dst += bytes;
                rem -= bytes;
            }
            if dst != 0 {
                let used = SEG_SZ - rem;
                self.set.push_seg(s, Seg { ptr: dst - used, n: (used / Self::SZ) as u32 });
            }
        }
        sfence();
        for (g, gc) in self.gcur.iter().enumerate() {
            self.set.register_group_blocks(g, &gc.blocks);
        }
        if let Some(b) = self.meta_block.take() {
            put_blocks(&[b]);
        }
    }
}

fn nt_enabled() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        crate::csv::simd_backend() == "avx2"
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn sfence() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_sfence()
    }
}

/// Copies `BYTES` from `src` to `dst` (using AVX2 NT stores when `BYTES % 64 == 0` and `nt`).
#[inline(always)]
unsafe fn copy_stage<const BYTES: usize>(nt: bool, dst: *mut u8, src: *const u8) {
    #[cfg(target_arch = "x86_64")]
    {
        if BYTES % 64 == 0 && nt {
            stream_avx2(dst, src, BYTES);
            return;
        }
    }
    let _ = nt;
    std::ptr::copy_nonoverlapping(src, dst, BYTES);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2")]
#[inline]
unsafe fn stream_avx2(dst: *mut u8, src: *const u8, bytes: usize) {
    use core::arch::x86_64::*;
    if bytes == 64 {
        let a = _mm256_load_si256(src as *const __m256i);
        let b = _mm256_load_si256(src.add(32) as *const __m256i);
        _mm256_stream_si256(dst as *mut __m256i, a);
        _mm256_stream_si256(dst.add(32) as *mut __m256i, b);
        return;
    }
    if bytes == 128 {
        let a = _mm256_load_si256(src as *const __m256i);
        let b = _mm256_load_si256(src.add(32) as *const __m256i);
        let c = _mm256_load_si256(src.add(64) as *const __m256i);
        let d = _mm256_load_si256(src.add(96) as *const __m256i);
        _mm256_stream_si256(dst as *mut __m256i, a);
        _mm256_stream_si256(dst.add(32) as *mut __m256i, b);
        _mm256_stream_si256(dst.add(64) as *mut __m256i, c);
        _mm256_stream_si256(dst.add(96) as *mut __m256i, d);
        return;
    }
    if bytes == 192 {
        let a = _mm256_load_si256(src as *const __m256i);
        let b = _mm256_load_si256(src.add(32) as *const __m256i);
        let c = _mm256_load_si256(src.add(64) as *const __m256i);
        let d = _mm256_load_si256(src.add(96) as *const __m256i);
        let e = _mm256_load_si256(src.add(128) as *const __m256i);
        let f = _mm256_load_si256(src.add(160) as *const __m256i);
        _mm256_stream_si256(dst as *mut __m256i, a);
        _mm256_stream_si256(dst.add(32) as *mut __m256i, b);
        _mm256_stream_si256(dst.add(64) as *mut __m256i, c);
        _mm256_stream_si256(dst.add(96) as *mut __m256i, d);
        _mm256_stream_si256(dst.add(128) as *mut __m256i, e);
        _mm256_stream_si256(dst.add(160) as *mut __m256i, f);
        return;
    }
    let mut i = 0;
    while i < bytes {
        let a = _mm256_load_si256(src.add(i) as *const __m256i);
        let b = _mm256_load_si256(src.add(i + 32) as *const __m256i);
        _mm256_stream_si256(dst.add(i) as *mut __m256i, a);
        _mm256_stream_si256(dst.add(i + 32) as *mut __m256i, b);
        i += 64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    #[repr(C)]
    struct R48 {
        a: u64,
        b: u64,
        c: u64,
        d: u64,
        e: u64,
        f: u64,
    }
    #[derive(Clone, Copy, PartialEq, Debug)]
    #[repr(C)]
    struct R24 {
        a: u64,
        b: u64,
        c: u64,
    }

    #[test]
    fn chained_blocks_whale_shard() {
        // One shard receives ~10 MB (5+ blocks) from 4 threads; nothing is lost.
        let set: ShardSet<R48> = ShardSet::new_grouped(256, 8);
        let per_thread = 60_000u64;
        std::thread::scope(|s| {
            for t in 0..4u64 {
                let set = &set;
                s.spawn(move || {
                    let mut w = Scatter::<R48, 192>::new(set);
                    for i in 0..per_thread {
                        let shard = if i % 10 < 9 { 7 } else { (i % 256) as usize };
                        let v = t * 1_000_000 + i;
                        w.push(shard, R48 { a: v, b: v + 1, c: v + 2, d: v + 3, e: v + 4, f: shard as u64 });
                    }
                    w.finish();
                });
            }
        });
        assert!(
            set.groups[0].blocks.lock().unwrap().len() >= 5,
            "whale group chained {} blocks",
            set.groups[0].blocks.lock().unwrap().len()
        );
        let mut seen = Vec::new();
        for sh in 0..256 {
            let segs = set.take(sh);
            assert_eq!(segs.len(), segs.iter().map(|x| x.len()).sum::<usize>());
            for seg in segs.iter() {
                for r in seg {
                    assert_eq!(r.f, sh as u64);
                    assert_eq!((r.b, r.c, r.d, r.e), (r.a + 1, r.a + 2, r.a + 3, r.a + 4));
                    seen.push(r.a);
                }
            }
            segs.free();
            set.mark_shard_done(sh);
            if sh == 31 {
                assert!(set.groups[0].blocks.lock().unwrap().is_empty());
            }
        }
        set.free_all();
        seen.sort_unstable();
        let mut exp: Vec<u64> = (0..4u64).flat_map(|t| (0..per_thread).map(move |i| t * 1_000_000 + i)).collect();
        exp.sort_unstable();
        assert_eq!(seen, exp);

        // Freed blocks are reused by the next set.
        let before = mapped_bytes();
        let set2: ShardSet<R24> = ShardSet::new(4);
        let mut w = Scatter::<R24, 192>::new(&set2);
        for i in 0..100_000u64 {
            w.push((i % 3) as usize, R24 { a: i, b: !i, c: i ^ 5 });
        }
        w.finish();
        let mut n = 0;
        for sh in 0..4 {
            let segs = set2.take(sh);
            for seg in segs.iter() {
                for r in seg {
                    assert_eq!((r.b, r.c, r.a % 3), (!r.a, r.a ^ 5, sh as u64));
                    n += 1;
                }
            }
            segs.free();
        }
        set2.free_all();
        assert_eq!(n, 100_000);
        assert_eq!(mapped_bytes(), before);

        // Shared-block ShardSet reuses blocks and returns all of them on free_all().
        let set3: ShardSet<R24> = ShardSet::new_shared(64);
        let mut w3 = Scatter::<R24, 192>::new(&set3);
        for i in 0..50_000u64 {
            w3.push((i % 64) as usize, R24 { a: i, b: !i, c: i ^ 9 });
        }
        w3.finish();
        let mut n3 = 0;
        for sh in 0..64 {
            let segs = set3.take(sh);
            for seg in segs.iter() {
                for r in seg {
                    assert_eq!((r.b, r.c, r.a % 64), (!r.a, r.a ^ 9, sh as u64));
                    n3 += 1;
                }
            }
            segs.free();
        }
        set3.free_all();
        assert_eq!(n3, 50_000);
        assert_eq!(mapped_bytes(), before);
    }

    #[test]
    fn located_scatter_and_loc_ptr() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        #[repr(C)]
        struct R16 {
            id: u64,
            ts: u32,
            tag: u32,
        }
        let set: ShardSet<R16> = ShardSet::new_shared(64);
        let locs: Vec<Vec<(u32, u64)>> = std::thread::scope(|s| {
            (0..4u64)
                .map(|t| {
                    let set = &set;
                    s.spawn(move || {
                        let mut w = Scatter::<R16, 64>::new_located(set);
                        let mut out = Vec::with_capacity(40_000);
                        for i in 0..40_000u64 {
                            let sh = if i % 5 == 0 { 3 } else { (i % 64) as usize };
                            let id = t * 100_000 + i;
                            let loc = w.push_with_loc(sh, R16 { id, ts: i as u32, tag: sh as u32 });
                            assert_ne!(loc, LOC_INVALID);
                            assert_ne!(loc, LOC_EMPTY);
                            out.push((loc, id));
                        }
                        w.finish();
                        out
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        // Verify every single returned loc points to the exact record and supports in-place mutation.
        for batch in &locs {
            for &(loc, id) in batch {
                unsafe {
                    let p = loc_ptr::<R16>(loc);
                    assert_eq!((*p).id, id);
                    if id % 7 == 0 {
                        std::ptr::addr_of_mut!((*p).ts).write(u32::MAX);
                    }
                }
            }
        }
        let mut total = 0usize;
        let mut killed = 0usize;
        for sh in 0..64 {
            let segs = set.take(sh);
            for seg in segs.iter() {
                for r in seg {
                    assert_eq!(r.tag, sh as u32);
                    if r.id % 7 == 0 {
                        assert_eq!(r.ts, u32::MAX);
                        killed += 1;
                    } else {
                        assert_ne!(r.ts, u32::MAX);
                    }
                    total += 1;
                }
            }
            segs.free();
        }
        set.free_all();
        assert_eq!(total, 160_000);
        assert!(killed > 22_000);
    }

    #[test]
    fn located_scatter12_and_loc_ptr12() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        #[repr(C, packed(4))]
        struct R12 {
            id: u64,
            ts: u32,
        }
        assert_eq!(std::mem::size_of::<R12>(), 12);
        let set: ShardSet<R12> = ShardSet::new_shared(64);
        let locs: Vec<Vec<(u32, u64)>> = std::thread::scope(|s| {
            (0..4u64)
                .map(|t| {
                    let set = &set;
                    s.spawn(move || {
                        let mut w = Scatter::<R12, 192>::new_located12(set);
                        let mut out = Vec::with_capacity(50_000);
                        for i in 0..50_000u64 {
                            let sh = if i % 5 == 0 { 7 } else { (i % 64) as usize };
                            let id = t * 100_000 + i;
                            let loc = w.push_with_loc12(sh, R12 { id, ts: sh as u32 });
                            assert_eq!(loc & LOC_PURCH_BIT, 0);
                            assert_ne!(loc, LOC_INVALID);
                            out.push((loc, id));
                        }
                        w.finish();
                        out
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });
        for batch in &locs {
            for &(loc, id) in batch {
                unsafe {
                    let p = loc_ptr12::<R12>(loc);
                    let got_id = std::ptr::addr_of!((*p).id).read_unaligned();
                    assert_eq!(got_id, id);
                    if id % 11 == 0 {
                        std::ptr::addr_of_mut!((*p).id).write_unaligned(u64::MAX);
                    }
                }
            }
        }
        let mut total = 0usize;
        let mut killed = 0usize;
        for sh in 0..64 {
            let segs = set.take(sh);
            for seg in segs.iter() {
                for r in seg {
                    let id = r.id;
                    let ts = r.ts;
                    assert_eq!(ts, sh as u32);
                    if id == u64::MAX {
                        killed += 1;
                    }
                    total += 1;
                }
            }
            segs.free();
        }
        set.free_all();
        assert_eq!(total, 200_000);
        assert!(killed > 18_000);
    }
}
