#![feature(portable_simd)]

use std::simd::{simd_swizzle, u32x4, u8x16, ToBytes};

#[derive(Clone)]
pub struct Gimli(pub [u8; 48]);

impl Gimli {
    pub const fn new() -> Self {
        Self([0; 48])
    }

    pub fn permute(&mut self) {
        let mut x = u32x4::from_le_bytes(u8x16::from_array(self.0[00..16].try_into().unwrap()));
        let mut y = u32x4::from_le_bytes(u8x16::from_array(self.0[16..32].try_into().unwrap()));
        let mut z = u32x4::from_le_bytes(u8x16::from_array(self.0[32..48].try_into().unwrap()));

        for round_constant in [
            0x9e377918, 0x9e377914, 0x9e377910, 0x9e37790c, 0x9e377908, 0x9e377904,
        ] {
            (x, y, z) = sp_box(x, y, z);
            x = simd_swizzle!(x, [1, 0, 3, 2]);
            x ^= u32x4::from_array([round_constant, 0, 0, 0]);

            (x, y, z) = sp_box(x, y, z);

            (x, y, z) = sp_box(x, y, z);
            x = simd_swizzle!(x, [2, 3, 0, 1]);

            (x, y, z) = sp_box(x, y, z);
        }

        self.0[00..16].copy_from_slice(x.to_le_bytes().as_array());
        self.0[16..32].copy_from_slice(y.to_le_bytes().as_array());
        self.0[32..48].copy_from_slice(z.to_le_bytes().as_array());
    }
}

#[inline(always)]
fn sp_box(x: u32x4, y: u32x4, z: u32x4) -> (u32x4, u32x4, u32x4) {
    let x = rotate_left::<24>(x);
    let y = rotate_left::<09>(y);
    (
        z ^ y << 0 ^ (x & y) << 3,
        y ^ x << 0 ^ (x | z) << 1,
        x ^ z << 1 ^ (y & z) << 2,
    )
}

#[inline(always)]
fn rotate_left<const OFFSET: u32>(x: u32x4) -> u32x4 {
    x << OFFSET | x >> (u32::BITS - OFFSET)
}

#[cfg(test)]
mod tests {
    use super::Gimli;

    #[test]
    fn it_works() {
        {
            let mut gimli = Gimli::new();

            for _ in 0..384 {
                gimli.permute();
            }

            assert_eq!(
                gimli.0,
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
