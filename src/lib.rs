#![feature(portable_simd)]

use std::simd::{simd_swizzle, u32x4};

#[derive(Clone)]
pub struct Gimli {
    a: u32x4,
    b: u32x4,
    c: u32x4,
}

impl Gimli {
    pub fn new() -> Self {
        Self {
            a: u32x4::splat(0),
            b: u32x4::splat(0),
            c: u32x4::splat(0),
        }
    }

    pub fn permute(&mut self) {
        self.a.from_le();
        self.b.from_le();
        self.c.from_le();

        for round_constant in [
            0x9e377918, 0x9e377914, 0x9e377910, 0x9e37790c, 0x9e377908, 0x9e377904,
        ] {
            self.sp_box();
            self.a = simd_swizzle!(self.a, [1, 0, 3, 2]);
            self.a ^= u32x4::from_array([round_constant, 0, 0, 0]);

            self.sp_box();

            self.sp_box();
            self.a = simd_swizzle!(self.a, [2, 3, 0, 1]);

            self.sp_box();
        }

        self.a.to_le();
        self.b.to_le();
        self.c.to_le();
    }

    #[inline(always)]
    pub fn bytes_view(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self as *const _ as *const u8, 48) }
    }

    #[inline(always)]
    pub fn bytes_view_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut _ as *mut u8, 48) }
    }

    #[inline(always)]
    fn sp_box(&mut self) {
        let x = self.a.rotate_left::<24>();
        let y = self.b.rotate_left::<09>();
        let z = self.c;

        self.c = x ^ (z << u32x4::splat(1)) ^ ((y & z) << u32x4::splat(2));
        self.b = y ^ x ^ ((x | z) << u32x4::splat(1));
        self.a = z ^ y ^ ((x & y) << u32x4::splat(3));
    }
}

trait GimliInternal {
    fn from_le(&mut self);

    fn rotate_left<const OFFSET: u32>(&self) -> Self;

    fn to_le(&mut self);
}

impl GimliInternal for u32x4 {
    #[inline(always)]
    fn from_le(&mut self) {
        self.as_mut_array()
            .iter_mut()
            .for_each(|lane| *lane = u32::from_le(*lane));
    }

    #[inline(always)]
    fn rotate_left<const OFFSET: u32>(&self) -> Self {
        (self << Self::splat(OFFSET)) | (self >> Self::splat(32 - OFFSET))
    }

    #[inline(always)]
    fn to_le(&mut self) {
        self.as_mut_array()
            .iter_mut()
            .for_each(|lane| *lane = lane.to_le());
    }
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
        {
            let mut gimli = Gimli::default();

            for _ in 0..384 {
                gimli.permute();
            }

            assert_eq!(
                gimli.bytes_view(),
                [
                    0xf7, 0xb2, 0xd5, 0x86, 0x5e, 0x79, 0x28, 0x27, 0xcb, 0xad, 0xe4, 0x14, 0x07,
                    0x5f, 0x6e, 0x3e, 0x40, 0x8a, 0xcc, 0x2f, 0xdb, 0xb7, 0xbb, 0x56, 0x47, 0x08,
                    0x9c, 0xf4, 0xef, 0xc6, 0xc1, 0x23, 0xf1, 0x21, 0x5b, 0x75, 0x22, 0x2c, 0x72,
                    0x85, 0xb8, 0xdb, 0x63, 0x01, 0xe9, 0x0a, 0x73, 0x0c,
                ]
            );
        }
    }
}
