//! Fast non-cryptographic 64-bit string hash (wyhash-style mixing).

const P0: u64 = 0xa076_1d64_78bd_642f;
const P1: u64 = 0xe703_7ed1_a0b4_28db;

#[inline(always)]
fn mum(a: u64, b: u64) -> u64 {
    let r = (a as u128).wrapping_mul(b as u128);
    (r as u64) ^ ((r >> 64) as u64)
}

#[inline(always)]
unsafe fn r8(p: *const u8) -> u64 {
    u64::from_le((p as *const u64).read_unaligned())
}

#[inline(always)]
unsafe fn r4(p: *const u8) -> u64 {
    u32::from_le((p as *const u32).read_unaligned()) as u64
}

#[inline(always)]
pub fn hash(s: &[u8], seed: u64) -> u64 {
    let len = s.len();
    let p = s.as_ptr();
    let mut seed = seed ^ P0;
    let a;
    let b;
    unsafe {
        if len <= 16 {
            if len >= 4 {
                let off = (len >> 3) << 2;
                a = (r4(p) << 32) | r4(p.add(off));
                b = (r4(p.add(len - 4)) << 32) | r4(p.add(len - 4 - off));
            } else if len > 0 {
                a = ((*p as u64) << 16) | ((*p.add(len >> 1) as u64) << 8) | (*p.add(len - 1) as u64);
                b = 0;
            } else {
                a = 0;
                b = 0;
            }
        } else {
            let mut i = len;
            let mut q = p;
            while i > 16 {
                seed = mum(r8(q) ^ P1, r8(q.add(8)) ^ seed);
                q = q.add(16);
                i -= 16;
            }
            a = r8(p.add(len - 16));
            b = r8(p.add(len - 8));
        }
    }
    mum(P1 ^ (len as u64), mum(a ^ P1, b ^ seed))
}
