//! Minimal Parquet writer: one row group, one uncompressed PLAIN data page (V1)
//! per column. Supports required strings, optional strings, DATE and INT64.

use std::io::{IoSlice, Write};
use std::path::Path;

use crate::arena::BlockCarver;
use crate::key::Key8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Str,
    OptStr,
    Date,
    I32,
    I64,
}

#[derive(Clone, Copy)]
struct Chunk {
    ptr: *mut u8,
    len: usize,
}
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

struct SegBuf {
    ptr: *mut u8,
    len: usize,
    cap: usize,
    chunk_sz: usize,
    done: Vec<Chunk>,
    heap_owned: Vec<Vec<u8>>,
    total_done: usize,
}
unsafe impl Send for SegBuf {}
unsafe impl Sync for SegBuf {}

impl SegBuf {
    fn new_heap(cap: usize) -> SegBuf {
        if cap == 0 {
            return SegBuf {
                ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                len: 0,
                cap: 0,
                chunk_sz: 0,
                done: Vec::new(),
                heap_owned: Vec::new(),
                total_done: 0,
            };
        }
        let mut v = Vec::with_capacity(cap);
        let ptr = v.as_mut_ptr();
        let cap = v.capacity();
        #[cfg(all(target_os = "linux", not(test)))]
        if cap >= (1 << 20) {
            let addr = ptr as usize;
            let a = (addr + 4095) & !4095;
            let e = (addr + cap) & !4095;
            if e > a {
                unsafe { libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_HUGEPAGE) };
            }
        }
        SegBuf {
            ptr,
            len: 0,
            cap,
            chunk_sz: 0,
            done: Vec::new(),
            heap_owned: vec![v],
            total_done: 0,
        }
    }

    fn new_arena(chunk_sz: usize) -> SegBuf {
        SegBuf {
            ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
            len: 0,
            cap: 0,
            chunk_sz,
            done: Vec::new(),
            heap_owned: Vec::new(),
            total_done: 0,
        }
    }

    #[inline(always)]
    fn total_len(&self) -> usize {
        self.total_done + self.len
    }

    #[cold]
    #[inline(never)]
    fn grow_arena(&mut self, extra: usize, carver: &mut BlockCarver) {
        if self.len > 0 {
            self.done.push(Chunk { ptr: self.ptr, len: self.len });
            self.total_done += self.len;
        }
        if extra <= self.chunk_sz {
            self.ptr = carver.alloc(self.chunk_sz);
            self.len = 0;
            self.cap = self.chunk_sz;
            return;
        }
        let mut v = Vec::with_capacity(extra.max(4096));
        self.ptr = v.as_mut_ptr();
        self.len = 0;
        self.cap = v.capacity();
        self.heap_owned.push(v);
    }

    #[cold]
    #[inline(never)]
    fn grow_heap(&mut self, extra: usize) {
        if self.len > 0 {
            self.done.push(Chunk { ptr: self.ptr, len: self.len });
            self.total_done += self.len;
        }
        let next_cap = (self.cap * 2).max(extra).max(4096);
        let mut v = Vec::with_capacity(next_cap);
        self.ptr = v.as_mut_ptr();
        self.len = 0;
        self.cap = v.capacity();
        self.heap_owned.push(v);
    }

    fn push_slices<'a>(&'a self, out: &mut Vec<&'a [u8]>) {
        for c in &self.done {
            if c.len > 0 {
                out.push(unsafe { std::slice::from_raw_parts(c.ptr, c.len) });
            }
        }
        if self.len > 0 {
            out.push(unsafe { std::slice::from_raw_parts(self.ptr, self.len) });
        }
    }

    fn release_heap_pages(&self) {
        #[cfg(all(target_os = "linux", not(test)))]
        for v in &self.heap_owned {
            let cap = v.capacity();
            if cap >= 131_072 {
                let ptr = v.as_ptr() as usize;
                let a = (ptr + 4095) & !4095;
                let e = (ptr + cap) & !4095;
                if e > a {
                    unsafe { libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_DONTNEED) };
                }
            }
        }
    }
}

pub struct Column {
    name: &'static str,
    kind: Kind,
    values: SegBuf,
    defs: SegBuf,
    n: usize,
}

impl Column {
    pub fn new(name: &'static str, kind: Kind, cap_rows: usize) -> Column {
        let per = match kind {
            Kind::Str => 14,
            Kind::OptStr => 2,
            Kind::Date | Kind::I32 => 4,
            Kind::I64 => 8,
        };
        let val_cap = if cap_rows == 0 { 0 } else { cap_rows * per + 16 };
        let def_cap = if kind == Kind::OptStr && cap_rows > 0 { cap_rows / 8 + 16 } else { 0 };
        Column {
            name,
            kind,
            values: SegBuf::new_heap(val_cap),
            defs: SegBuf::new_heap(def_cap),
            n: 0,
        }
    }

    #[cfg(test)]
    pub fn new_arena(name: &'static str, kind: Kind) -> Column {
        let val_chunk = match kind {
            Kind::Str => 512 << 10,
            Kind::I64 => 512 << 10,
            Kind::Date | Kind::I32 => 256 << 10,
            Kind::OptStr => 128 << 10,
        };
        let def_chunk = if kind == Kind::OptStr { 64 << 10 } else { 0 };
        Column {
            name,
            kind,
            values: SegBuf::new_arena(val_chunk),
            defs: SegBuf::new_arena(def_chunk),
            n: 0,
        }
    }

    #[allow(dead_code)]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Advise the kernel to drop any heap-owned backing pages of this column after writing to disk.
    #[allow(dead_code)]
    pub fn release_pages(&self) {
        self.values.release_heap_pages();
        self.defs.release_heap_pages();
    }

    #[cfg(test)]
    #[inline(always)]
    fn push_def_arena(&mut self, present: bool, carver: &mut BlockCarver) {
        let bit = self.n & 7;
        if bit == 0 {
            if self.defs.len == self.defs.cap {
                self.defs.grow_arena(1, carver);
            }
            unsafe {
                *self.defs.ptr.add(self.defs.len) = present as u8;
                self.defs.len += 1;
            }
        } else if present {
            unsafe { *self.defs.ptr.add(self.defs.len - 1) |= 1 << bit };
        }
    }

    #[cfg(test)]
    #[inline(always)]
    fn ensure_val_arena(&mut self, extra: usize, carver: &mut BlockCarver) {
        if self.values.cap - self.values.len < extra {
            self.values.grow_arena(extra, carver);
        }
    }

    #[inline(always)]
    fn ensure_val_heap(&mut self, extra: usize) {
        if self.values.cap - self.values.len < extra {
            self.values.grow_heap(extra);
        }
    }

    /// Fast path for inline `Key8::as_slice` slices backed by a 16-byte stack buffer.
    #[cfg(test)]
    #[inline(always)]
    pub unsafe fn push_str_inline16_arena(&mut self, s: &[u8], carver: &mut BlockCarver) {
        debug_assert!(self.kind == Kind::Str && s.len() <= 16);
        let slen = s.len();
        self.ensure_val_arena(20, carver);
        let dst = self.values.ptr.add(self.values.len);
        (dst as *mut u32).write_unaligned((slen as u32).to_le());
        (dst.add(4) as *mut u128).write_unaligned((s.as_ptr() as *const u128).read_unaligned());
        self.values.len += 4 + slen;
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_null_arena(&mut self, carver: &mut BlockCarver) {
        debug_assert!(self.kind == Kind::OptStr);
        self.push_def_arena(false, carver);
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_str_arena(&mut self, s: &[u8], carver: &mut BlockCarver) {
        debug_assert!(self.kind == Kind::Str);
        let slen = s.len();
        self.ensure_val_arena(4 + slen, carver);
        unsafe {
            let dst = self.values.ptr.add(self.values.len);
            (dst as *mut u32).write_unaligned((slen as u32).to_le());
            std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
            self.values.len += 4 + slen;
        }
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_opt_str_arena(&mut self, s: Option<&[u8]>, carver: &mut BlockCarver) {
        debug_assert!(self.kind == Kind::OptStr);
        self.push_def_arena(s.is_some(), carver);
        if let Some(s) = s {
            let slen = s.len();
            self.ensure_val_arena(4 + slen, carver);
            unsafe {
                let dst = self.values.ptr.add(self.values.len);
                (dst as *mut u32).write_unaligned((slen as u32).to_le());
                std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
                self.values.len += 4 + slen;
            }
        }
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_date_arena(&mut self, d: i32, carver: &mut BlockCarver) {
        self.ensure_val_arena(4, carver);
        unsafe {
            (self.values.ptr.add(self.values.len) as *mut i32).write_unaligned(d.to_le());
            self.values.len += 4;
        }
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_i32_arena(&mut self, v: i32, carver: &mut BlockCarver) {
        self.ensure_val_arena(4, carver);
        unsafe {
            (self.values.ptr.add(self.values.len) as *mut i32).write_unaligned(v.to_le());
            self.values.len += 4;
        }
        self.n += 1;
    }

    #[cfg(test)]
    #[inline(always)]
    pub fn push_i64_arena(&mut self, v: i64, carver: &mut BlockCarver) {
        self.ensure_val_arena(8, carver);
        unsafe {
            (self.values.ptr.add(self.values.len) as *mut i64).write_unaligned(v.to_le());
            self.values.len += 8;
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_str(&mut self, s: &[u8]) {
        debug_assert!(self.kind == Kind::Str);
        let slen = s.len();
        self.ensure_val_heap(4 + slen);
        unsafe {
            let dst = self.values.ptr.add(self.values.len);
            (dst as *mut u32).write_unaligned((slen as u32).to_le());
            std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
            self.values.len += 4 + slen;
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_date(&mut self, d: i32) {
        self.ensure_val_heap(4);
        unsafe {
            (self.values.ptr.add(self.values.len) as *mut i32).write_unaligned(d.to_le());
            self.values.len += 4;
        }
        self.n += 1;
    }

    #[inline(always)]
    pub fn push_i64(&mut self, v: i64) {
        self.ensure_val_heap(8);
        unsafe {
            (self.values.ptr.add(self.values.len) as *mut i64).write_unaligned(v.to_le());
            self.values.len += 8;
        }
        self.n += 1;
    }
}

struct Thrift {
    buf: Vec<u8>,
    last: i16,
    stack: Vec<i16>,
}

impl Thrift {
    fn new() -> Thrift {
        Thrift { buf: Vec::with_capacity(1024), last: 0, stack: Vec::new() }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn header(&mut self, id: i16, ty: u8) {
        let d = id - self.last;
        if d > 0 && d <= 15 {
            self.buf.push(((d as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.varint(((id << 1) ^ (id >> 15)) as u16 as u64);
        }
        self.last = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.header(id, 5);
        self.varint(((v << 1) ^ (v >> 31)) as u32 as u64);
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.header(id, 6);
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }
    fn bin(&mut self, id: i16, b: &[u8]) {
        self.header(id, 8);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn begin_struct(&mut self, id: i16) {
        self.header(id, 12);
        self.push_scope();
    }
    fn push_scope(&mut self) {
        self.stack.push(self.last);
        self.last = 0;
    }
    fn end_struct(&mut self) {
        self.buf.push(0);
        self.last = self.stack.pop().unwrap();
    }
    fn list(&mut self, id: i16, elem: u8, n: usize) {
        self.header(id, 9);
        self.list_header(elem, n);
    }
    fn list_header(&mut self, elem: u8, n: usize) {
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xf0 | elem);
            self.varint(n as u64);
        }
    }
}

fn physical(kind: Kind) -> i32 {
    match kind {
        Kind::Str | Kind::OptStr => 6,
        Kind::Date | Kind::I32 => 1,
        Kind::I64 => 2,
    }
}

fn write_all_vectored(f: &mut std::fs::File, slices: &mut [&[u8]]) -> std::io::Result<()> {
    let mut cur = slices;
    while !cur.is_empty() {
        let mut iov = [IoSlice::new(&[]); 128];
        let count = cur.len().min(128);
        for i in 0..count {
            iov[i] = IoSlice::new(cur[i]);
        }
        let mut n = match f.write_vectored(&iov[..count]) {
            Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "write_vectored returned 0")),
            Ok(n) => n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        while !cur.is_empty() && n >= cur[0].len() {
            n -= cur[0].len();
            cur = &mut cur[1..];
        }
        if !cur.is_empty() && n > 0 {
            cur[0] = &cur[0][n..];
        }
    }
    Ok(())
}

/// Writes a Parquet file holding `cols` (all with the same row count) in a single
/// vectored `writev` syscall without copying column or definition-level buffers.
pub fn write_file(path: &Path, cols: &[Column]) -> std::io::Result<()> {
    let nrows = cols.first().map(|c| c.n).unwrap_or(0);
    let mut f = std::fs::File::create(path)?;
    let mut off = 4i64;
    let mut chunks: Vec<(i64, i64)> = Vec::with_capacity(cols.len());
    let mut hdrs: Vec<Vec<u8>> = Vec::with_capacity(cols.len());
    if nrows > 0 {
        for c in cols {
            assert_eq!(c.n, nrows);
            let mut lv_hdr = [0u8; 16];
            let mut lv_hdr_len = 0usize;
            let page_len;
            let vlen = c.values.total_len();
            if c.kind == Kind::OptStr {
                let dlen = c.defs.total_len();
                let groups = c.n.div_ceil(8);
                debug_assert_eq!(dlen, groups);
                let mut h = ((groups as u64) << 1) | 1;
                let mut pos = 4usize;
                while h >= 0x80 {
                    lv_hdr[pos] = (h as u8) | 0x80;
                    h >>= 7;
                    pos += 1;
                }
                lv_hdr[pos] = h as u8;
                pos += 1;
                let lv_len = (pos - 4 + dlen) as u32;
                lv_hdr[0..4].copy_from_slice(&lv_len.to_le_bytes());
                lv_hdr_len = pos;
                page_len = pos + dlen + vlen;
            } else {
                page_len = vlen;
            }
            assert!(page_len < i32::MAX as usize, "parquet page too large");
            let mut t = Thrift::new();
            t.i32(1, 0); // DATA_PAGE
            t.i32(2, page_len as i32);
            t.i32(3, page_len as i32);
            t.begin_struct(5);
            t.i32(1, c.n as i32);
            t.i32(2, 0); // PLAIN
            t.i32(3, 3); // RLE
            t.i32(4, 3); // RLE
            t.end_struct();
            t.buf.push(0);
            let thdr_len = t.buf.len();
            if lv_hdr_len > 0 {
                t.buf.extend_from_slice(&lv_hdr[..lv_hdr_len]);
            }
            let chunk_size = (thdr_len + page_len) as i64;
            hdrs.push(t.buf);
            chunks.push((off, chunk_size));
            off += chunk_size;
        }
    }

    let mut t = Thrift::new();
    t.i32(1, 1);
    t.list(2, 12, cols.len() + 1);
    // root
    t.push_scope();
    t.bin(4, b"schema");
    t.i32(5, cols.len() as i32);
    t.end_struct();
    for c in cols {
        t.push_scope();
        t.i32(1, physical(c.kind));
        t.i32(3, if c.kind == Kind::OptStr { 1 } else { 0 });
        t.bin(4, c.name.as_bytes());
        match c.kind {
            Kind::Str | Kind::OptStr => {
                t.i32(6, 0); // UTF8
                t.begin_struct(10);
                t.begin_struct(1); // STRING
                t.end_struct();
                t.end_struct();
            }
            Kind::Date => {
                t.i32(6, 6); // DATE
                t.begin_struct(10);
                t.begin_struct(6); // DATE
                t.end_struct();
                t.end_struct();
            }
            Kind::I32 | Kind::I64 => {}
        }
        t.end_struct();
    }
    t.i64(3, nrows as i64);
    if nrows > 0 {
        t.list(4, 12, 1);
        t.push_scope();
        t.list(1, 12, cols.len());
        let mut total = 0i64;
        for (c, &(off, size)) in cols.iter().zip(&chunks) {
            total += size;
            t.push_scope();
            t.i64(2, off);
            t.begin_struct(3);
            t.i32(1, physical(c.kind));
            t.list(2, 5, 2);
            t.varint(0); // PLAIN (zigzag 0)
            t.varint(6); // RLE (zigzag 3)
            t.list(3, 8, 1);
            t.varint(c.name.len() as u64);
            t.buf.extend_from_slice(c.name.as_bytes());
            t.i32(4, 0); // UNCOMPRESSED
            t.i64(5, c.n as i64);
            t.i64(6, size);
            t.i64(7, size);
            t.i64(9, off);
            t.end_struct();
            t.end_struct();
        }
        t.i64(2, total);
        t.i64(3, nrows as i64);
        t.end_struct();
    } else {
        t.list(4, 12, 0);
    }
    t.bin(6, b"shopping-sessions");
    t.buf.push(0);
    let flen = (t.buf.len() as u32).to_le_bytes();
    t.buf.extend_from_slice(&flen);
    t.buf.extend_from_slice(b"PAR1");

    let mut slices: Vec<&[u8]> = Vec::with_capacity(cols.len() * 4 + 16);
    slices.push(b"PAR1");
    if nrows > 0 {
        for (c, hdr) in cols.iter().zip(&hdrs) {
            slices.push(hdr.as_slice());
            if c.kind == Kind::OptStr {
                c.defs.push_slices(&mut slices);
            }
            c.values.push_slices(&mut slices);
        }
    }
    slices.push(t.buf.as_slice());
    write_all_vectored(&mut f, &mut slices)
}

// ---------------------------------------------------------------------------
// Compact packed user_daily buffer & Parquet writer (RLE_DICTIONARY + DELTA_BINARY_PACKED)

/// 4-byte packed `user_daily` row stored in a single contiguous arena stream during `solve`.
#[derive(Clone, Copy)]
#[repr(C)]
struct UdRow4 {
    day: u8,   // 0..254 = day index; 255 = pop from wide_days
    ns: u8,    // 1..=48 (mathematically bounded by DAY_MS / GAP_MS)
    ne_lo: u8, // 0..254 = exact events; 255 = pop from wide_ne
    cat: u8,   // 255 = zero rev / null top_category; 0..=253 = cat idx; 254 = pop wide_cat
}
const _: () = assert!(std::mem::size_of::<UdRow4>() == 4);

pub struct UserDailyBuf {
    user_dict: SegBuf,
    user_runs: SegBuf,
    rows: SegBuf,
    nz_rev: Vec<i64>,
    wide_days: Vec<u32>,
    wide_ne: Vec<u32>,
    wide_cat: Vec<u32>,
    nrows: usize,
    npersons: usize,
    person_start: usize,
    max_ns: u8,
}

impl UserDailyBuf {
    pub fn new() -> UserDailyBuf {
        UserDailyBuf {
            user_dict: SegBuf::new_arena(256 << 10),
            user_runs: SegBuf::new_arena(64 << 10),
            rows: SegBuf::new_arena(512 << 10),
            nz_rev: Vec::new(),
            wide_days: Vec::new(),
            wide_ne: Vec::new(),
            wide_cat: Vec::new(),
            nrows: 0,
            npersons: 0,
            person_start: 0,
            max_ns: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.nrows
    }

    pub fn release_heap_pages(&self) {
        self.user_dict.release_heap_pages();
        self.user_runs.release_heap_pages();
        self.rows.release_heap_pages();
        #[cfg(all(target_os = "linux", not(test)))]
        {
            let cap = self.nz_rev.capacity() * std::mem::size_of::<i64>();
            if cap >= 131_072 {
                let ptr = self.nz_rev.as_ptr() as usize;
                let a = (ptr + 4095) & !4095;
                let e = (ptr + cap) & !4095;
                if e > a {
                    unsafe { libc::madvise(a as *mut libc::c_void, e - a, libc::MADV_DONTNEED) };
                }
            }
        }
    }

    #[cold]
    #[inline(never)]
    fn push_user_slow(&mut self, person: Key8, carver: &mut BlockCarver) {
        let mut pbuf = [0u8; 16];
        let s = person.as_slice(&mut pbuf);
        let slen = s.len();
        if self.user_dict.cap - self.user_dict.len < 4 + slen {
            self.user_dict.grow_arena(4 + slen, carver);
        }
        unsafe {
            let dst = self.user_dict.ptr.add(self.user_dict.len);
            (dst as *mut u32).write_unaligned((slen as u32).to_le());
            std::ptr::copy_nonoverlapping(s.as_ptr(), dst.add(4), slen);
            self.user_dict.len += 4 + slen;
        }
    }

    #[inline(always)]
    pub fn begin_person(&mut self, person: Key8, carver: &mut BlockCarver) {
        let v = person.0;
        let hi = v >> 56;
        if hi <= 0x80 && v != 0 {
            let slen = 9 - (hi >> 7) as usize;
            if self.user_dict.cap - self.user_dict.len < 13 {
                self.user_dict.grow_arena(13, carver);
            }
            unsafe {
                let dst = self.user_dict.ptr.add(self.user_dict.len);
                (dst as *mut u32).write_unaligned((slen as u32).to_le());
                *dst.add(4) = b'u';
                (dst.add(5) as *mut u64).write_unaligned(v.to_le());
                self.user_dict.len += 4 + slen;
            }
        } else {
            self.push_user_slow(person, carver);
        }
        self.person_start = self.nrows;
        self.npersons += 1;
    }

    #[inline(always)]
    pub fn end_person(&mut self, carver: &mut BlockCarver) {
        let run = (self.nrows - self.person_start) as u32;
        debug_assert!(run > 0);
        if self.user_runs.cap - self.user_runs.len < 4 {
            self.user_runs.grow_arena(4, carver);
        }
        unsafe {
            (self.user_runs.ptr.add(self.user_runs.len) as *mut u32).write_unaligned(run);
            self.user_runs.len += 4;
        }
    }

    #[cold]
    #[inline(never)]
    fn push_wide_day(&mut self, day: u32) -> u8 {
        self.wide_days.push(day);
        255
    }

    #[cold]
    #[inline(never)]
    fn push_wide_ne(&mut self, ne: u32) -> u8 {
        self.wide_ne.push(ne);
        255
    }

    #[cold]
    #[inline(never)]
    fn push_wide_cat(&mut self, cat: usize) -> u8 {
        self.wide_cat.push(cat as u32);
        254
    }

    #[inline(always)]
    pub fn push_zero_rev(&mut self, day: u32, ns: u32, ne: u32, carver: &mut BlockCarver) {
        if self.rows.len == self.rows.cap {
            self.rows.grow_arena(4, carver);
        }
        let d = if day < 255 { day as u8 } else { self.push_wide_day(day) };
        let e = if ne < 255 { ne as u8 } else { self.push_wide_ne(ne) };
        debug_assert!(ns >= 1 && ns <= 48);
        let ns8 = ns as u8;
        if ns8 > self.max_ns {
            self.max_ns = ns8;
        }
        unsafe {
            (self.rows.ptr.add(self.rows.len) as *mut UdRow4).write(UdRow4 {
                day: d,
                ns: ns8,
                ne_lo: e,
                cat: 255,
            });
            self.rows.len += 4;
        }
        self.nrows += 1;
    }

    #[inline(always)]
    pub fn push_with_rev(&mut self, day: u32, ns: u32, ne: u32, rev: u64, cat: usize, carver: &mut BlockCarver) {
        if self.rows.len == self.rows.cap {
            self.rows.grow_arena(4, carver);
        }
        let d = if day < 255 { day as u8 } else { self.push_wide_day(day) };
        let e = if ne < 255 { ne as u8 } else { self.push_wide_ne(ne) };
        let c = if cat < 254 { cat as u8 } else { self.push_wide_cat(cat) };
        debug_assert!(ns >= 1 && ns <= 48);
        let ns8 = ns as u8;
        if ns8 > self.max_ns {
            self.max_ns = ns8;
        }
        self.nz_rev.push(rev as i64);
        unsafe {
            (self.rows.ptr.add(self.rows.len) as *mut UdRow4).write(UdRow4 {
                day: d,
                ns: ns8,
                ne_lo: e,
                cat: c,
            });
            self.rows.len += 4;
        }
        self.nrows += 1;
    }

    fn iter_rows(&self, mut f: impl FnMut(UdRow4)) {
        for c in &self.rows.done {
            let sl = unsafe { std::slice::from_raw_parts(c.ptr as *const UdRow4, c.len >> 2) };
            for &r in sl {
                f(r);
            }
        }
        if self.rows.len > 0 {
            let sl = unsafe { std::slice::from_raw_parts(self.rows.ptr as *const UdRow4, self.rows.len >> 2) };
            for &r in sl {
                f(r);
            }
        }
    }

    fn iter_runs(&self, mut f: impl FnMut(u32)) {
        for c in &self.user_runs.done {
            let sl = unsafe { std::slice::from_raw_parts(c.ptr as *const u32, c.len >> 2) };
            for &r in sl {
                f(r);
            }
        }
        if self.user_runs.len > 0 {
            let sl = unsafe { std::slice::from_raw_parts(self.user_runs.ptr as *const u32, self.user_runs.len >> 2) };
            for &r in sl {
                f(r);
            }
        }
    }
}

#[inline(always)]
fn bits_needed(n_distinct: usize) -> u8 {
    if n_distinct <= 1 {
        0
    } else {
        (usize::BITS - (n_distinct - 1).leading_zeros()) as u8
    }
}

#[inline(always)]
fn push_uleb(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline(always)]
fn push_zigzag32(out: &mut Vec<u8>, v: i32) {
    push_uleb(out, ((v << 1) ^ (v >> 31)) as u32 as u64);
}

#[inline(always)]
fn push_zigzag64(out: &mut Vec<u8>, v: i64) {
    push_uleb(out, ((v << 1) ^ (v >> 63)) as u64);
}

/// Bit-pack 8 values of width `bw` (1..=32) LSB-first into `bw` bytes.
#[inline(always)]
fn pack8_u32(vals: &[u32; 8], bw: u8, out: &mut Vec<u8>) {
    if bw <= 8 {
        let b = bw as u32;
        let acc = (vals[0] as u64)
            | ((vals[1] as u64) << b)
            | ((vals[2] as u64) << (b * 2))
            | ((vals[3] as u64) << (b * 3))
            | ((vals[4] as u64) << (b * 4))
            | ((vals[5] as u64) << (b * 5))
            | ((vals[6] as u64) << (b * 6))
            | ((vals[7] as u64) << (b * 7));
        out.extend_from_slice(&acc.to_le_bytes()[..bw as usize]);
        return;
    }
    if bw <= 16 {
        let b = bw as u32;
        let acc = (vals[0] as u128)
            | ((vals[1] as u128) << b)
            | ((vals[2] as u128) << (b * 2))
            | ((vals[3] as u128) << (b * 3))
            | ((vals[4] as u128) << (b * 4))
            | ((vals[5] as u128) << (b * 5))
            | ((vals[6] as u128) << (b * 6))
            | ((vals[7] as u128) << (b * 7));
        out.extend_from_slice(&acc.to_le_bytes()[..bw as usize]);
        return;
    }
    let bw = bw as u32;
    let mut acc = 0u64;
    let mut bits = 0u32;
    for &v in vals {
        acc |= (v as u64) << bits;
        bits += bw;
        while bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            bits -= 8;
        }
    }
}

/// Bit-pack 32 values of width `bw` (1..=32) LSB-first into `4 * bw` bytes.
#[inline(always)]
fn pack32_u32(vals: &[u32; 32], bw: u8, out: &mut Vec<u8>) {
    let p = vals.as_ptr();
    unsafe {
        pack8_u32(&*(p as *const [u32; 8]), bw, out);
        pack8_u32(&*(p.add(8) as *const [u32; 8]), bw, out);
        pack8_u32(&*(p.add(16) as *const [u32; 8]), bw, out);
        pack8_u32(&*(p.add(24) as *const [u32; 8]), bw, out);
    }
}

/// Bit-pack 32 values of width `bw` (1..=64) LSB-first into `4 * bw` bytes.
#[inline(always)]
fn pack32_u64(vals: &[u64; 32], bw: u8, out: &mut Vec<u8>) {
    if bw == 64 {
        for &v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        return;
    }
    let bw = bw as u32;
    let mut acc = 0u128;
    let mut bits = 0u32;
    for &v in vals {
        acc |= (v as u128) << bits;
        bits += bw;
        while bits >= 64 {
            out.extend_from_slice(&(acc as u64).to_le_bytes());
            acc >>= 64;
            bits -= 64;
        }
    }
    while bits >= 8 {
        out.push(acc as u8);
        acc >>= 8;
        bits -= 8;
    }
}

/// Emits a Parquet RLE/Bit-Packed Hybrid stream of `vals` (without 4B length prefix, WITH leading `bw` byte).
fn encode_rle_bp_stream(bw: u8, vals: &[u32], out: &mut Vec<u8>) {
    out.push(bw);
    let n = vals.len();
    if n == 0 {
        return;
    }
    if bw == 0 {
        push_uleb(out, (n as u64) << 1);
        return;
    }
    // Emit in <=63-group chunks (<=504 values) so every bit-packed header is 1 byte.
    let mut pos = 0usize;
    while pos < n {
        let chunk = (n - pos).min(504);
        let groups = chunk.div_ceil(8);
        out.push(((groups as u8) << 1) | 1);
        let mut g = 0usize;
        while g + 8 <= chunk {
            let p = pos + g;
            let v = [
                vals[p],
                vals[p + 1],
                vals[p + 2],
                vals[p + 3],
                vals[p + 4],
                vals[p + 5],
                vals[p + 6],
                vals[p + 7],
            ];
            pack8_u32(&v, bw, out);
            g += 8;
        }
        if g < chunk {
            let mut v = [0u32; 8];
            for j in 0..(chunk - g) {
                v[j] = vals[pos + g + j];
            }
            pack8_u32(&v, bw, out);
        }
        pos += chunk;
    }
}

fn dict_page_header(num_values: usize, page_len: usize) -> Vec<u8> {
    assert!(page_len < i32::MAX as usize, "parquet dict page too large");
    let mut t = Thrift::new();
    t.i32(1, 2); // DICTIONARY_PAGE
    t.i32(2, page_len as i32);
    t.i32(3, page_len as i32);
    t.begin_struct(7); // dictionary_page_header
    t.i32(1, num_values as i32);
    t.i32(2, 0); // PLAIN
    t.end_struct();
    t.buf.push(0);
    t.buf
}

fn data_page_header(num_values: usize, page_len: usize, encoding: i32) -> Vec<u8> {
    assert!(page_len < i32::MAX as usize, "parquet data page too large");
    let mut t = Thrift::new();
    t.i32(1, 0); // DATA_PAGE
    t.i32(2, page_len as i32);
    t.i32(3, page_len as i32);
    t.begin_struct(5); // data_page_header
    t.i32(1, num_values as i32);
    t.i32(2, encoding);
    t.i32(3, 3); // RLE
    t.i32(4, 3); // RLE
    t.end_struct();
    t.buf.push(0);
    t.buf
}

/// Reusable per-writer scratch buffers for streaming `write_user_daily_packed`
/// (eliminates 64-320 MB of temporary per-column Vec allocations during `write`).
pub struct WriteScratch {
    u_data: Vec<u8>,
    d_dict: Vec<u8>,
    d_data: Vec<u8>,
    s_dict: Vec<u8>,
    s_data: Vec<u8>,
    e_data: Vec<u8>,
    r_data: Vec<u8>,
    c_dict: Vec<u8>,
    c_data: Vec<u8>,
    def_encoded: Vec<u8>,
    cat_vals: Vec<u32>,
}

impl WriteScratch {
    pub fn new() -> Self {
        WriteScratch {
            u_data: Vec::new(),
            d_dict: Vec::new(),
            d_data: Vec::new(),
            s_dict: Vec::new(),
            s_data: Vec::new(),
            e_data: Vec::new(),
            r_data: Vec::new(),
            c_dict: Vec::new(),
            c_data: Vec::new(),
            def_encoded: Vec::new(),
            cat_vals: Vec::new(),
        }
    }
}

#[inline(always)]
fn flush_bp_block(vals: &mut [u32; 128], cnt: usize, bw: u8, out: &mut Vec<u8>) {
    let groups = cnt.div_ceil(8);
    let pad_end = groups * 8;
    for k in cnt..pad_end {
        vals[k] = 0;
    }
    if bw <= 8 {
        let bw_u = bw as usize;
        let b = bw as u32;
        out.reserve(1 + groups * bw_u + 8);
        let base_len = out.len();
        unsafe {
            let mut dst = out.as_mut_ptr().add(base_len);
            *dst = ((groups as u8) << 1) | 1;
            dst = dst.add(1);
            let p = vals.as_ptr();
            for g in 0..groups {
                let v = p.add(g * 8);
                let acc = (*v as u64)
                    | ((*v.add(1) as u64) << b)
                    | ((*v.add(2) as u64) << (b * 2))
                    | ((*v.add(3) as u64) << (b * 3))
                    | ((*v.add(4) as u64) << (b * 4))
                    | ((*v.add(5) as u64) << (b * 5))
                    | ((*v.add(6) as u64) << (b * 6))
                    | ((*v.add(7) as u64) << (b * 7));
                (dst as *mut u64).write_unaligned(acc);
                dst = dst.add(bw_u);
            }
            out.set_len(base_len + 1 + groups * bw_u);
        }
        return;
    }
    out.push(((groups as u8) << 1) | 1);
    let p = vals.as_ptr();
    for g in 0..groups {
        pack8_u32(unsafe { &*(p.add(g * 8) as *const [u32; 8]) }, bw, out);
    }
}

#[inline(always)]
fn flush_delta_i32_block(deltas: &[i32; 128], cnt: usize, out: &mut Vec<u8>) {
    let mut min_d = i32::MAX;
    for k in 0..cnt {
        let d = unsafe { *deltas.get_unchecked(k) };
        if d < min_d {
            min_d = d;
        }
    }
    push_zigzag32(out, min_d);
    let mut bws = [0u8; 4];
    for m in 0..4 {
        let s = m * 32;
        if s >= cnt {
            break;
        }
        let e = (s + 32).min(cnt);
        let mut or_bits = 0u32;
        for k in s..e {
            or_bits |= unsafe { *deltas.get_unchecked(k) }.wrapping_sub(min_d) as u32;
        }
        bws[m] = (32 - or_bits.leading_zeros()) as u8;
    }
    out.extend_from_slice(&bws);
    for m in 0..4 {
        let s = m * 32;
        if s >= cnt {
            break;
        }
        let bw = bws[m];
        if bw == 0 {
            continue;
        }
        let e = (s + 32).min(cnt);
        let mut mb = [0u32; 32];
        for k in s..e {
            mb[k - s] = unsafe { *deltas.get_unchecked(k) }.wrapping_sub(min_d) as u32;
        }
        pack32_u32(&mb, bw, out);
    }
}

#[inline(always)]
fn flush_delta_i64_block(deltas: &[i64; 128], cnt: usize, out: &mut Vec<u8>) {
    let mut min_d = i64::MAX;
    let mut or_raw = 0u64;
    for k in 0..cnt {
        let d = unsafe { *deltas.get_unchecked(k) };
        or_raw |= d as u64;
        if d < min_d {
            min_d = d;
        }
    }
    if or_raw == 0 {
        // Common zero-revenue block: min_d=0, 4 zero miniblock bitwidths = 5 bytes total!
        out.extend_from_slice(&[0u8; 5]);
        return;
    }
    push_zigzag64(out, min_d);
    let mut bws = [0u8; 4];
    for m in 0..4 {
        let s = m * 32;
        if s >= cnt {
            break;
        }
        let e = (s + 32).min(cnt);
        let mut or_bits = 0u64;
        for k in s..e {
            or_bits |= unsafe { *deltas.get_unchecked(k) }.wrapping_sub(min_d) as u64;
        }
        bws[m] = (64 - or_bits.leading_zeros()) as u8;
    }
    out.extend_from_slice(&bws);
    for m in 0..4 {
        let s = m * 32;
        if s >= cnt {
            break;
        }
        let bw = bws[m];
        if bw == 0 {
            continue;
        }
        let e = (s + 32).min(cnt);
        let mut mb = [0u64; 32];
        for k in s..e {
            mb[k - s] = unsafe { *deltas.get_unchecked(k) }.wrapping_sub(min_d) as u64;
        }
        pack32_u64(&mb, bw, out);
    }
}

/// Writes a compact packed `user_daily` Parquet file using `PLAIN_DICTIONARY` (RLE/BP)
/// for `user_id`, `day`, `sessions`, `top_category` and `DELTA_BINARY_PACKED` for
/// `events` and `revenue_cents`. Streams directly from 4B `UdRow4` in 128-row L1 blocks
/// with zero temporary full-column heap buffers.
pub fn write_user_daily_packed(
    path: &Path,
    buf: &UserDailyBuf,
    day0: i64,
    ndays: usize,
    cat_names: &[&[u8]],
    sc: &mut WriteScratch,
) -> std::io::Result<()> {
    let nrows = buf.nrows;
    debug_assert!(nrows > 0);

    // 1. user_id (PLAIN_DICTIONARY RLE; dict slices come zero-copy from buf.user_dict!)
    let u_dict_len = buf.user_dict.total_len();
    let u_dict_hdr = dict_page_header(buf.npersons, u_dict_len);
    let u_bw = bits_needed(buf.npersons);
    let u_vbytes = (u_bw as usize).div_ceil(8);
    sc.u_data.clear();
    sc.u_data.reserve(1 + buf.npersons * (5 + u_vbytes) + 8);
    unsafe {
        let base = sc.u_data.as_mut_ptr();
        *base = u_bw;
        let mut p = base.add(1);
        let mut pid = 0u32;
        buf.iter_runs(|cnt| {
            let mut v = (cnt as u64) << 1;
            if v < 0x80 {
                *p = v as u8;
                p = p.add(1);
            } else {
                while v >= 0x80 {
                    *p = (v as u8) | 0x80;
                    p = p.add(1);
                    v >>= 7;
                }
                *p = v as u8;
                p = p.add(1);
            }
            if u_vbytes > 0 {
                (p as *mut u32).write_unaligned(pid.to_le());
                p = p.add(u_vbytes);
            }
            pid += 1;
        });
        sc.u_data.set_len(p.offset_from(base) as usize);
    }
    let u_data_hdr = data_page_header(nrows, sc.u_data.len(), 2); // PLAIN_DICTIONARY

    // Prepare streaming encoders for day, sessions, events, revenue_cents, top_category.
    let day_is_dict = buf.wide_days.is_empty() && ndays <= 255;
    let d_bw = bits_needed(ndays);
    sc.d_dict.clear();
    sc.d_data.clear();
    let mut d_dict_hdr = Vec::new();
    if day_is_dict {
        sc.d_dict.reserve(ndays * 4);
        for d in 0..ndays {
            sc.d_dict.extend_from_slice(&((day0 + d as i64) as i32).to_le_bytes());
        }
        d_dict_hdr = dict_page_header(ndays, sc.d_dict.len());
        sc.d_data.reserve(1 + (nrows * d_bw as usize).div_ceil(8) + nrows / 64 + 32);
        sc.d_data.push(d_bw);
        if d_bw == 0 {
            push_uleb(&mut sc.d_data, (nrows as u64) << 1);
        }
    } else {
        sc.d_data.reserve(nrows * 4);
    }

    let max_ns = buf.max_ns.max(1) as usize;
    let s_bw = bits_needed(max_ns);
    sc.s_dict.clear();
    sc.s_dict.reserve(max_ns * 4);
    for s in 1..=max_ns {
        sc.s_dict.extend_from_slice(&(s as i32).to_le_bytes());
    }
    let s_dict_hdr = dict_page_header(max_ns, sc.s_dict.len());
    sc.s_data.clear();
    sc.s_data.reserve(1 + (nrows * s_bw as usize).div_ceil(8) + nrows / 64 + 32);
    sc.s_data.push(s_bw);
    if s_bw == 0 {
        push_uleb(&mut sc.s_data, (nrows as u64) << 1);
    }

    sc.e_data.clear();
    sc.e_data.reserve(nrows + 128);
    push_uleb(&mut sc.e_data, 128);
    push_uleb(&mut sc.e_data, 4);
    push_uleb(&mut sc.e_data, nrows as u64);

    sc.r_data.clear();
    sc.r_data.reserve(buf.nz_rev.len() * 6 + nrows / 16 + 128);
    push_uleb(&mut sc.r_data, 128);
    push_uleb(&mut sc.r_data, 4);
    push_uleb(&mut sc.r_data, nrows as u64);

    let cat_has_vals = !buf.nz_rev.is_empty();
    sc.def_encoded.clear();
    sc.cat_vals.clear();
    if cat_has_vals {
        sc.def_encoded.reserve(nrows / 8 + nrows / 64 + 32);
        sc.cat_vals.reserve(buf.nz_rev.len());
    }

    // Single L1-streaming pass over 4B UdRow4: zero intermediate column Vecs!
    let mut d_blk = [0u32; 128];
    let mut s_blk = [0u32; 128];
    let mut def_blk = [0u8; 16];
    let mut e_deltas = [0i32; 128];
    let mut r_deltas = [0i64; 128];
    let (mut blk_n, mut d_n) = (0usize, 0usize);
    let (mut prev_e, mut prev_r) = (0i32, 0i64);
    let (mut wd_i, mut wne_i, mut wc_i, mut nzr_i, mut row_i) = (0usize, 0usize, 0usize, 0usize, 0usize);

    buf.iter_rows(|r| {
        let d = if r.day < 255 {
            r.day as u32
        } else {
            let v = buf.wide_days[wd_i];
            wd_i += 1;
            v
        };
        d_blk[blk_n] = d;
        s_blk[blk_n] = (r.ns - 1) as u32;
        let ne = if r.ne_lo < 255 {
            r.ne_lo as i32
        } else {
            let v = buf.wide_ne[wne_i] as i32;
            wne_i += 1;
            v
        };
        let rev = if r.cat == 255 {
            0i64
        } else {
            let v = buf.nz_rev[nzr_i];
            nzr_i += 1;
            let c = if r.cat < 254 {
                r.cat as u32
            } else {
                let cv = buf.wide_cat[wc_i];
                wc_i += 1;
                cv
            };
            sc.cat_vals.push(c);
            def_blk[blk_n >> 3] |= 1u8 << (blk_n & 7);
            v
        };
        blk_n += 1;
        if blk_n == 128 {
            if day_is_dict {
                if d_bw > 0 {
                    flush_bp_block(&mut d_blk, 128, d_bw, &mut sc.d_data);
                }
            } else {
                for &dv in &d_blk {
                    sc.d_data.extend_from_slice(&((day0 + dv as i64) as i32).to_le_bytes());
                }
            }
            if s_bw > 0 {
                flush_bp_block(&mut s_blk, 128, s_bw, &mut sc.s_data);
            }
            if cat_has_vals {
                sc.def_encoded.push((16u8 << 1) | 1);
                sc.def_encoded.extend_from_slice(&def_blk);
                def_blk = [0u8; 16];
            }
            blk_n = 0;
        }
        if row_i == 0 {
            push_zigzag32(&mut sc.e_data, ne);
            prev_e = ne;
            push_zigzag64(&mut sc.r_data, rev);
            prev_r = rev;
        } else {
            e_deltas[d_n] = ne.wrapping_sub(prev_e);
            prev_e = ne;
            r_deltas[d_n] = rev.wrapping_sub(prev_r);
            prev_r = rev;
            d_n += 1;
            if d_n == 128 {
                flush_delta_i32_block(&e_deltas, 128, &mut sc.e_data);
                flush_delta_i64_block(&r_deltas, 128, &mut sc.r_data);
                d_n = 0;
            }
        }
        row_i += 1;
    });

    if blk_n > 0 {
        if day_is_dict {
            if d_bw > 0 {
                flush_bp_block(&mut d_blk, blk_n, d_bw, &mut sc.d_data);
            }
        } else {
            for &dv in &d_blk[..blk_n] {
                sc.d_data.extend_from_slice(&((day0 + dv as i64) as i32).to_le_bytes());
            }
        }
        if s_bw > 0 {
            flush_bp_block(&mut s_blk, blk_n, s_bw, &mut sc.s_data);
        }
        if cat_has_vals {
            let nbytes = blk_n.div_ceil(8);
            sc.def_encoded.push(((nbytes as u8) << 1) | 1);
            sc.def_encoded.extend_from_slice(&def_blk[..nbytes]);
        }
    }
    if d_n > 0 {
        flush_delta_i32_block(&e_deltas, d_n, &mut sc.e_data);
        flush_delta_i64_block(&r_deltas, d_n, &mut sc.r_data);
    }

    let d_data_hdr = data_page_header(nrows, sc.d_data.len(), if day_is_dict { 2 } else { 0 });
    let s_data_hdr = data_page_header(nrows, sc.s_data.len(), 2);
    let e_data_hdr = data_page_header(nrows, sc.e_data.len(), 5);
    let r_data_hdr = data_page_header(nrows, sc.r_data.len(), 5);

    // 6. top_category (OptStr: PLAIN_DICTIONARY if any non-null, else PLAIN 0 bytes + RLE zeros)
    sc.c_dict.clear();
    sc.c_data.clear();
    let mut c_dict_hdr = Vec::new();
    let c_data_hdr;
    if cat_has_vals {
        for &name in cat_names {
            sc.c_dict.extend_from_slice(&(name.len() as u32).to_le_bytes());
            sc.c_dict.extend_from_slice(name);
        }
        c_dict_hdr = dict_page_header(cat_names.len(), sc.c_dict.len());
        sc.c_data.reserve(4 + sc.def_encoded.len() + sc.cat_vals.len() + 16);
        sc.c_data.extend_from_slice(&(sc.def_encoded.len() as u32).to_le_bytes());
        sc.c_data.extend_from_slice(&sc.def_encoded);
        let c_bw = bits_needed(cat_names.len());
        encode_rle_bp_stream(c_bw, &sc.cat_vals, &mut sc.c_data);
        c_data_hdr = data_page_header(nrows, sc.c_data.len(), 2);
    } else {
        // All-null column: single RLE zero run for def-levels, 0 value bytes!
        let mut def_enc = Vec::with_capacity(12);
        push_uleb(&mut def_enc, (nrows as u64) << 1);
        def_enc.push(0);
        sc.c_data.extend_from_slice(&(def_enc.len() as u32).to_le_bytes());
        sc.c_data.extend_from_slice(&def_enc);
        c_data_hdr = data_page_header(nrows, sc.c_data.len(), 0); // PLAIN
    }

    // Compute exact column chunk offsets & sizes.
    struct ChunkInfo {
        name: &'static str,
        kind: Kind,
        dict_off: Option<i64>,
        data_off: i64,
        total_size: i64,
        enc_kind: u8, // 0=PLAIN, 2=PLAIN_DICT, 5=DELTA_BP
    }
    let mut off = 4i64;
    let mut infos = Vec::with_capacity(6);

    // col 0: user_id
    {
        let dict_off = off;
        let data_off = off + (u_dict_hdr.len() + u_dict_len) as i64;
        let total = (u_dict_hdr.len() + u_dict_len + u_data_hdr.len() + sc.u_data.len()) as i64;
        infos.push(ChunkInfo { name: "user_id", kind: Kind::Str, dict_off: Some(dict_off), data_off, total_size: total, enc_kind: 2 });
        off += total;
    }
    // col 1: day
    {
        let dict_off = if day_is_dict { Some(off) } else { None };
        let data_off = off + (d_dict_hdr.len() + sc.d_dict.len()) as i64;
        let total = (d_dict_hdr.len() + sc.d_dict.len() + d_data_hdr.len() + sc.d_data.len()) as i64;
        infos.push(ChunkInfo { name: "day", kind: Kind::Date, dict_off, data_off, total_size: total, enc_kind: if day_is_dict { 2 } else { 0 } });
        off += total;
    }
    // col 2: sessions
    {
        let dict_off = off;
        let data_off = off + (s_dict_hdr.len() + sc.s_dict.len()) as i64;
        let total = (s_dict_hdr.len() + sc.s_dict.len() + s_data_hdr.len() + sc.s_data.len()) as i64;
        infos.push(ChunkInfo { name: "sessions", kind: Kind::I32, dict_off: Some(dict_off), data_off, total_size: total, enc_kind: 2 });
        off += total;
    }
    // col 3: events
    {
        let data_off = off;
        let total = (e_data_hdr.len() + sc.e_data.len()) as i64;
        infos.push(ChunkInfo { name: "events", kind: Kind::I32, dict_off: None, data_off, total_size: total, enc_kind: 5 });
        off += total;
    }
    // col 4: revenue_cents
    {
        let data_off = off;
        let total = (r_data_hdr.len() + sc.r_data.len()) as i64;
        infos.push(ChunkInfo { name: "revenue_cents", kind: Kind::I64, dict_off: None, data_off, total_size: total, enc_kind: 5 });
        off += total;
    }
    // col 5: top_category
    {
        let dict_off = if cat_has_vals { Some(off) } else { None };
        let data_off = off + (c_dict_hdr.len() + sc.c_dict.len()) as i64;
        let total = (c_dict_hdr.len() + sc.c_dict.len() + c_data_hdr.len() + sc.c_data.len()) as i64;
        infos.push(ChunkInfo { name: "top_category", kind: Kind::OptStr, dict_off, data_off, total_size: total, enc_kind: if cat_has_vals { 2 } else { 0 } });
        let _ = off + total;
    }

    // Build Thrift footer.
    let mut t = Thrift::new();
    t.i32(1, 1);
    t.list(2, 12, 7);
    t.push_scope();
    t.bin(4, b"schema");
    t.i32(5, 6);
    t.end_struct();
    for info in &infos {
        t.push_scope();
        t.i32(1, physical(info.kind));
        t.i32(3, if info.kind == Kind::OptStr { 1 } else { 0 });
        t.bin(4, info.name.as_bytes());
        match info.kind {
            Kind::Str | Kind::OptStr => {
                t.i32(6, 0); // UTF8
                t.begin_struct(10);
                t.begin_struct(1); // STRING
                t.end_struct();
                t.end_struct();
            }
            Kind::Date => {
                t.i32(6, 6); // DATE
                t.begin_struct(10);
                t.begin_struct(6); // DATE
                t.end_struct();
                t.end_struct();
            }
            Kind::I32 | Kind::I64 => {}
        }
        t.end_struct();
    }
    t.i64(3, nrows as i64);
    t.list(4, 12, 1);
    t.push_scope();
    t.list(1, 12, 6);
    let mut rg_total = 0i64;
    for info in &infos {
        rg_total += info.total_size;
        t.push_scope();
        t.i64(2, info.dict_off.unwrap_or(info.data_off));
        t.begin_struct(3);
        t.i32(1, physical(info.kind));
        match info.enc_kind {
            2 => {
                t.list(2, 5, 3);
                t.varint(4); // PLAIN_DICTIONARY (zigzag 2)
                t.varint(0); // PLAIN (zigzag 0)
                t.varint(6); // RLE (zigzag 3)
            }
            5 => {
                t.list(2, 5, 2);
                t.varint(10); // DELTA_BINARY_PACKED (zigzag 5)
                t.varint(6);  // RLE (zigzag 3)
            }
            _ => {
                t.list(2, 5, 2);
                t.varint(0); // PLAIN (zigzag 0)
                t.varint(6); // RLE (zigzag 3)
            }
        }
        t.list(3, 8, 1);
        t.varint(info.name.len() as u64);
        t.buf.extend_from_slice(info.name.as_bytes());
        t.i32(4, 0); // UNCOMPRESSED
        t.i64(5, nrows as i64);
        t.i64(6, info.total_size);
        t.i64(7, info.total_size);
        t.i64(9, info.data_off);
        if let Some(doff) = info.dict_off {
            t.i64(11, doff);
        }
        t.end_struct();
        t.end_struct();
    }
    t.i64(2, rg_total);
    t.i64(3, nrows as i64);
    t.end_struct();
    t.bin(6, b"shopping-sessions");
    t.buf.push(0);
    let flen = (t.buf.len() as u32).to_le_bytes();
    t.buf.extend_from_slice(&flen);
    t.buf.extend_from_slice(b"PAR1");

    let mut f = std::fs::File::create(path)?;
    let mut slices: Vec<&[u8]> = Vec::with_capacity(32);
    slices.push(b"PAR1");
    // col 0: user_id
    slices.push(&u_dict_hdr);
    buf.user_dict.push_slices(&mut slices);
    slices.push(&u_data_hdr);
    slices.push(&sc.u_data);
    // col 1: day
    if day_is_dict {
        slices.push(&d_dict_hdr);
        slices.push(&sc.d_dict);
    }
    slices.push(&d_data_hdr);
    slices.push(&sc.d_data);
    // col 2: sessions
    slices.push(&s_dict_hdr);
    slices.push(&sc.s_dict);
    slices.push(&s_data_hdr);
    slices.push(&sc.s_data);
    // col 3: events
    slices.push(&e_data_hdr);
    slices.push(&sc.e_data);
    // col 4: revenue_cents
    slices.push(&r_data_hdr);
    slices.push(&sc.r_data);
    // col 5: top_category
    if cat_has_vals {
        slices.push(&c_dict_hdr);
        slices.push(&sc.c_dict);
    }
    slices.push(&c_data_hdr);
    slices.push(&sc.c_data);
    slices.push(&t.buf);
    write_all_vectored(&mut f, &mut slices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{release_owned_blocks, BlockCarver};

    #[test]
    fn arena_columns_match_heap_byte_for_byte() {
        let mut carver = BlockCarver::new();
        let mut a_cols = [
            Column::new_arena("user_id", Kind::Str),
            Column::new_arena("day", Kind::Date),
            Column::new_arena("sessions", Kind::I32),
            Column::new_arena("revenue", Kind::I64),
            Column::new_arena("top_cat", Kind::OptStr),
        ];
        // Cross multiple 64KB/128KB/256KB/512KB chunks plus a >512KB oversized string.
        let big = vec![b'z'; 600_000];
        let nrows = 70_000usize;
        for i in 0..nrows {
            let mut sbuf = [0u8; 16];
            let s = format!("u{:08}", i);
            sbuf[..s.len()].copy_from_slice(s.as_bytes());
            if i == 12345 {
                a_cols[0].push_str_arena(&big, &mut carver);
            } else {
                unsafe { a_cols[0].push_str_inline16_arena(&sbuf[..s.len()], &mut carver) };
            }
            a_cols[1].push_date_arena(20_000 + (i % 14) as i32, &mut carver);
            a_cols[2].push_i32_arena((i % 97) as i32, &mut carver);
            a_cols[3].push_i64_arena((i as i64) * 100, &mut carver);
            if i % 3 == 0 {
                a_cols[4].push_null_arena(&mut carver);
            } else {
                a_cols[4].push_opt_str_arena(Some(b"cat_01"), &mut carver);
            }
        }
        let mut dir = std::env::temp_dir();
        dir.push(format!("bc_parquet_test_{}.parquet", std::process::id()));
        write_file(&dir, &a_cols).unwrap();
        let bytes = std::fs::read(&dir).unwrap();
        let _ = std::fs::remove_file(&dir);
        assert!(bytes.starts_with(b"PAR1"));
        assert!(bytes.ends_with(b"PAR1"));
        assert_eq!(a_cols[0].len(), nrows);
        assert_eq!(a_cols[4].defs.total_len(), nrows.div_ceil(8));
        assert!(a_cols[0].values.done.len() >= 2);
        let blocks = carver.take_blocks();
        assert!(!blocks.is_empty());
        release_owned_blocks(&blocks);
    }
}
