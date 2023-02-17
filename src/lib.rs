#![feature(portable_simd)]

use std::simd::{simd_swizzle, u32x4};

#[derive(Clone)]
pub struct Gimli {
    pub bytes: [u8; 48],
}

impl Gimli {
    pub fn new() -> Self {
        Self { bytes: [0; 48] }
    }

    pub fn permute(&mut self) {
        let (mut a, mut b, mut c) = self.unpack();

        for round_constant in [
            0x9e377918, 0x9e377914, 0x9e377910, 0x9e37790c, 0x9e377908, 0x9e377904,
        ] {
            sp_box(&mut a, &mut b, &mut c);
            a = simd_swizzle!(a, [1, 0, 3, 2]);
            a ^= u32x4::from_array([round_constant, 0, 0, 0]);

            sp_box(&mut a, &mut b, &mut c);

            sp_box(&mut a, &mut b, &mut c);
            a = simd_swizzle!(a, [2, 3, 0, 1]);

            sp_box(&mut a, &mut b, &mut c);
        }

        self.pack(a, b, c);
    }

    #[inline]
    fn unpack(&self) -> (u32x4, u32x4, u32x4) {
        (
            u32x4::from_le_bytes(self.bytes[00..16].try_into().unwrap()),
            u32x4::from_le_bytes(self.bytes[16..32].try_into().unwrap()),
            u32x4::from_le_bytes(self.bytes[32..48].try_into().unwrap()),
        )
    }

    #[inline]
    fn pack(&mut self, a: u32x4, b: u32x4, c: u32x4) {
        self.bytes[00..16].copy_from_slice(&a.to_le_bytes());
        self.bytes[16..32].copy_from_slice(&b.to_le_bytes());
        self.bytes[32..48].copy_from_slice(&c.to_le_bytes());
    }
}

trait GimliInternal {
    fn from_le_bytes(bytes: [u8; 16]) -> Self;

    fn to_le_bytes(self) -> [u8; 16];

    fn rotate_left<const OFFSET: u32>(&self) -> Self;
}

impl GimliInternal for u32x4 {
    #[inline]
    fn from_le_bytes(bytes: [u8; 16]) -> Self {
        let mut words: u32x4 = unsafe { std::mem::transmute(bytes) };
        words
            .as_mut_array()
            .iter_mut()
            .for_each(|lane| *lane = lane.to_le());
        words
    }

    #[inline]
    fn to_le_bytes(mut self) -> [u8; 16] {
        self.as_mut_array()
            .iter_mut()
            .for_each(|lane| *lane = u32::from_le(*lane));
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    fn rotate_left<const OFFSET: u32>(&self) -> Self {
        (self << Self::splat(OFFSET)) | (self >> Self::splat(32 - OFFSET))
    }
}

#[inline]
fn sp_box(a: &mut u32x4, b: &mut u32x4, c: &mut u32x4) {
    let x = a.rotate_left::<24>();
    let y = b.rotate_left::<09>();
    let z = *c;

    *c = x ^ (z << u32x4::splat(1)) ^ ((y & z) << u32x4::splat(2));
    *b = y ^ x ^ ((x | z) << u32x4::splat(1));
    *a = z ^ y ^ ((x & y) << u32x4::splat(3));
}

impl Default for Gimli {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Gimli;

    #[test]
    fn it_works() {
        let mut gimli = Gimli::default();

        for _ in 0..384 {
            gimli.permute();
        }

        assert_eq!(
            gimli.bytes,
            [
                0xf7, 0xb2, 0xd5, 0x86, 0x5e, 0x79, 0x28, 0x27, 0xcb, 0xad, 0xe4, 0x14, 0x07, 0x5f,
                0x6e, 0x3e, 0x40, 0x8a, 0xcc, 0x2f, 0xdb, 0xb7, 0xbb, 0x56, 0x47, 0x08, 0x9c, 0xf4,
                0xef, 0xc6, 0xc1, 0x23, 0xf1, 0x21, 0x5b, 0x75, 0x22, 0x2c, 0x72, 0x85, 0xb8, 0xdb,
                0x63, 0x01, 0xe9, 0x0a, 0x73, 0x0c,
            ]
        );
    }
}
