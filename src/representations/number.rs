use bytes::{Buf, BufMut};

/// A generalized rational number. The first byte indicates the size and type of the numerator and denominator.
/// The highest four bits give the byte size of the numerator and the lower bits of the denominator.
/// Any size beyond 4 will have a special meaning, such as signaling that the number is a rational polynomial instead
/// or that the number is in a finite field.
pub trait RationalNumber {
    fn write_frac(&self, den: u64, dest: &mut Vec<u8>);
    fn write_frac_fixed(&self, den: u64, dest: &mut [u8]);
    fn get_frac_u64(source: &[u8]) -> (u64, u64, &[u8]);

    fn skip_rational(mut dest: &[u8]) -> &[u8] {
        let var_size = dest.get_u8();
        let size = (var_size & 0b00001111) + ((var_size & 0b11110000) >> 4);
        dest.advance(size as usize);
        dest
    }
}

impl RationalNumber for u64 {
    #[inline(always)]
    fn write_frac(&self, den: u64, dest: &mut Vec<u8>) {
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

    #[inline(always)]
    fn get_frac_u64(mut source: &[u8]) -> (u64, u64, &[u8]) {
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & 0b00001111 {
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
            _ => {
                unreachable!("FAIL")
            }
        };

        let den;
        (den, source) = match (disc & 0b11110000) >> 4 {
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
            _ => {
                unreachable!("FAIL")
            }
        };

        (num, den, source)
    }
}
