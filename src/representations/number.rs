use bytes::{Buf, BufMut};

const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const U8_DEN: u8 = 0b00010000;
const U16_DEN: u8 = 0b00100000;
const U32_DEN: u8 = 0b00110000;
const U64_DEN: u8 = 0b01000000;
const NUM_MASK: u8 = 0b00001111;
const DEN_MASK: u8 = 0b11110000;

/// A generalized rational number. The first byte indicates the size and type of the numerator and denominator.
/// The highest four bits give the byte size of the numerator and the lower bits of the denominator.
/// Any size beyond 4 will have a special meaning, such as signaling that the number is a rational polynomial instead
/// or that the number is in a finite field.
pub trait RationalNumberWriter {
    /// Write a single number.
    fn write_num(&self, dest: &mut Vec<u8>);
    /// Write a fraction.
    fn write_frac(&self, den: Self, dest: &mut Vec<u8>);
    /// Write a fraction to a fixed-size buffer.
    fn write_frac_fixed(&self, den: Self, dest: &mut [u8]);
}

/// A reader for generalized rational numbers. See [`RationalNumberWriter`].
pub trait RationalNumberReader {
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_one(&self) -> bool;
}

impl RationalNumberReader for [u8] {
    #[inline(always)]
    fn skip_rational(&self) -> &[u8] {
        let mut dest = self;
        let var_size = dest.get_u8();
        let size = (var_size & NUM_MASK) + ((var_size & DEN_MASK) >> 4);
        dest.advance(size as usize);
        dest
    }

    #[inline(always)]
    fn get_frac_u64(&self) -> (u64, u64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & NUM_MASK {
            1 => {
                let v = source.get_u8();
                (v as u64, source)
            }
            2 => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            3 => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            4 => {
                let v = source.get_u64_le();
                (v as u64, source)
            }
            x => {
                unreachable!("Unsupported numerator type {}", x)
            }
        };

        let den;
        (den, source) = match (disc & DEN_MASK) >> 4 {
            0 => (1u64, source),
            1 => {
                let v = source.get_u8();
                (v as u64, source)
            }
            2 => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            3 => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            4 => {
                let v = source.get_u64_le();
                (v as u64, source)
            }
            x => {
                unreachable!("Unsupported denominator type {}", x)
            }
        };

        (num, den, source)
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self[1] == 1 && self[2] == 1
    }
}

impl RationalNumberWriter for u64 {
    #[inline(always)]
    fn write_num(&self, dest: &mut Vec<u8>) {
        if *self < u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(*self as u8);
        } else if *self < u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(*self as u16);
        } else if *self < u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(*self as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(*self);
        }
    }

    #[inline(always)]
    fn write_frac(&self, den: u64, dest: &mut Vec<u8>) {
        let p = dest.len();

        if *self < u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(*self as u8);
        } else if *self < u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(*self as u16);
        } else if *self < u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(*self as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(*self);
        }

        if den == 1 {
        } else if den < u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(den as u8);
        } else if den < u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(den as u16);
        } else if den < u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u8(3);
            dest.put_u32_le(den as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(den);
        }
    }

    #[inline(always)]
    fn write_frac_fixed(&self, den: u64, mut dest: &mut [u8]) {
        let p = dest.len();

        if *self < u8::MAX as u64 {
            dest.put_u8(1);
            dest.put_u8(*self as u8);
        } else if *self < u16::MAX as u64 {
            dest.put_u8(2);
            dest.put_u16_le(*self as u16);
        } else if *self < u32::MAX as u64 {
            dest.put_u8(3);
            dest.put_u32_le(*self as u32);
        } else {
            dest.put_u8(4);
            dest.put_u64_le(*self);
        }

        if den == 1 {
        } else if den < u8::MAX as u64 {
            dest[p] |= 0b00010000;
            dest.put_u8(den as u8);
        } else if den < u16::MAX as u64 {
            dest[p] |= 0b00100000;
            dest.put_u16_le(den as u16);
        } else if den < u32::MAX as u64 {
            dest[p] |= 0b00110000;
            dest.put_u8(3);
            dest.put_u32_le(den as u32);
        } else {
            dest[p] |= 0b01000000;
            dest.put_u64_le(den);
        }
    }
}
