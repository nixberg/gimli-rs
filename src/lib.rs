use packed_simd_2::{shuffle, u32x4, u8x16, IntoBits};

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

        let round_constants = &[
            0x9e377918, 0x9e377914, 0x9e377910, 0x9e37790c, 0x9e377908, 0x9e377904,
        ];

        #[inline]
        fn apply_sp_box(a: &mut u32x4, b: &mut u32x4, c: &mut u32x4) {
            let x = rotate_lanes(*a, 24);
            let y = rotate_lanes(*b, 9);
            let z = *c;

            *c = x ^ (z << 1) ^ ((y & z) << 2);
            *b = y ^ x ^ ((x | z) << 1);
            *a = z ^ y ^ ((x & y) << 3);
        }

        for &round_constant in round_constants {
            apply_sp_box(&mut a, &mut b, &mut c);
            a = small_swap(a);
            a ^= u32x4::new(round_constant, 0, 0, 0);

            apply_sp_box(&mut a, &mut b, &mut c);

            apply_sp_box(&mut a, &mut b, &mut c);
            a = big_swap(a);

            apply_sp_box(&mut a, &mut b, &mut c);
        }

        self.pack(a, b, c);
    }

    #[inline]
    fn unpack(&self) -> (u32x4, u32x4, u32x4) {
        let a_le: u32x4 = u8x16::from_slice_unaligned(&self.bytes[00..16]).into_bits();
        let b_le: u32x4 = u8x16::from_slice_unaligned(&self.bytes[16..32]).into_bits();
        let c_le: u32x4 = u8x16::from_slice_unaligned(&self.bytes[32..48]).into_bits();
        (
            u32x4::from_le(a_le),
            u32x4::from_le(b_le),
            u32x4::from_le(c_le),
        )
    }

    #[inline]
    fn pack(&mut self, a: u32x4, b: u32x4, c: u32x4) {
        let a_bytes: u8x16 = u32x4::to_le(a).into_bits();
        let b_bytes: u8x16 = u32x4::to_le(b).into_bits();
        let c_bytes: u8x16 = u32x4::to_le(c).into_bits();
        a_bytes.write_to_slice_unaligned(&mut self.bytes[00..16]);
        b_bytes.write_to_slice_unaligned(&mut self.bytes[16..32]);
        c_bytes.write_to_slice_unaligned(&mut self.bytes[32..48]);
    }
}

#[inline]
fn rotate_lanes(x: u32x4, n: u32) -> u32x4 {
    x.rotate_left(u32x4::splat(n))
}

#[inline]
fn small_swap(x: u32x4) -> u32x4 {
    shuffle!(x, [1, 0, 3, 2])
}

#[inline]
fn big_swap(x: u32x4) -> u32x4 {
    shuffle!(x, [2, 3, 0, 1])
}

#[cfg(test)]
mod tests {
    use super::Gimli;

    #[test]
    fn it_works() {
        let mut gimli = Gimli::new();

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
