use bytes::{Buf, BufMut};
use rug::{Integer, Rational};

use crate::utils;

const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const ARB_NUM: u8 = 0b00000111;
const U8_DEN: u8 = 0b00010000;
const U16_DEN: u8 = 0b00100000;
const U32_DEN: u8 = 0b00110000;
const U64_DEN: u8 = 0b01000000;
const ARB_DEN: u8 = 0b01110000;
const NUM_MASK: u8 = 0b00001111;
const DEN_MASK: u8 = 0b01110000;
const SIGN: u8 = 0b10000000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    Natural(i64, i64),
    Large(Rational),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowedNumber<'a> {
    Natural(i64, i64),
    Large(&'a Rational),
}

impl BorrowedNumber<'_> {
    pub fn to_owned(&self) -> Number {
        match self {
            BorrowedNumber::Natural(num, den) => Number::Natural(*num, *den),
            BorrowedNumber::Large(r) => Number::Large((*r).clone()),
        }
    }
}

impl BorrowedNumber<'_> {
    pub fn add(&self, other: &BorrowedNumber<'_>) -> Number {
        match (self, other) {
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Natural(n2, d2)) => {
                if let Some(lcm) = d2.checked_mul(d1 / utils::gcd_signed(*d1, *d2)) {
                    if let Some(num2) = n2.checked_mul(lcm / d2) {
                        if let Some(num1) = n1.checked_mul(lcm / d1) {
                            if let Some(num) = num1.checked_add(num2) {
                                if num % lcm == 0 {
                                    return Number::Natural(num / lcm, 1);
                                } else {
                                    return Number::Natural(num, lcm);
                                }
                            }
                        }
                    }
                }
                Number::Large(Rational::from((*n1, *d1)) + Rational::from((*n2, *d2)))
            }
            // TODO: check downcast
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Large(r2))
            | (BorrowedNumber::Large(r2), BorrowedNumber::Natural(n1, d1)) => {
                let r1 = Rational::from((*n1, *d1));
                Number::Large(r1 + *r2)
            }
            (BorrowedNumber::Large(r1), BorrowedNumber::Large(r2)) => {
                Number::Large((*r1 + *r2).into())
            }
        }
    }

    pub fn mul(&self, other: &BorrowedNumber<'_>) -> Number {
        match (self, other) {
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Natural(n2, d2)) => {
                let gcd1 = utils::gcd_signed(*n1 as i64, *d2 as i64);
                let gcd2 = utils::gcd_signed(*d1 as i64, *n2 as i64);

                match (n2 / gcd2).checked_mul(n1 / gcd1) {
                    Some(nn) => match (d1 / gcd2).checked_mul(d2 / gcd1) {
                        Some(nd) => Number::Natural(nn, nd),
                        None => Number::Large(Rational::from((
                            nn,
                            Integer::from(d1 / gcd2) * Integer::from(d2 / gcd1),
                        ))),
                    },
                    None => Number::Large(Rational::from((
                        Integer::from(n1 / gcd1) * Integer::from(n2 / gcd2),
                        Integer::from(d1 / gcd2) * Integer::from(d2 / gcd1),
                    ))),
                }
            }
            // TODO: check downcast
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Large(r2))
            | (BorrowedNumber::Large(r2), BorrowedNumber::Natural(n1, d1)) => {
                let r1 = Rational::from((*n1, *d1));
                Number::Large(r1 * *r2)
            }
            (BorrowedNumber::Large(r1), BorrowedNumber::Large(r2)) => {
                Number::Large((*r1 * *r2).into())
            }
        }
    }
}

impl PackedRationalNumberWriter for Number {
    fn write_packed(self, dest: &mut Vec<u8>) {
        match self {
            Number::Natural(num, den) => (num, den).write_packed(dest),
            Number::Large(r) => {
                dest.put_u8(ARB_NUM | ARB_DEN);

                // note that this is not a linear representation
                let v = std::mem::ManuallyDrop::new(r);
                let lin_buf = unsafe { utils::any_as_u8_slice(&v) };

                dest.extend(lin_buf);
            }
        }
    }

    fn write_packed_fixed(self, dest: &mut [u8]) {
        match self {
            Number::Natural(num, den) => (num, den).write_packed_fixed(dest),
            Number::Large(_) => {
                todo!("Writing large packed rational not implemented")
            }
        }
    }
}

/// A generalized rational number. The first byte indicates the sign, size and type of the numerator and denominator.
/// The highest four bits give the byte size of the numerator and the lower bits of the denominator.
pub trait PackedRationalNumberWriter {
    /// Write a single number.
    fn write_packed(self, dest: &mut Vec<u8>);
    /// Write a fraction to a fixed-size buffer.
    fn write_packed_fixed(self, dest: &mut [u8]);
}

/// A reader for generalized rational numbers. See [`RationalNumberWriter`].
pub trait PackedRationalNumberReader {
    fn get_number_view(&self) -> (BorrowedNumber, &[u8]);
    fn get_frac_i64(&self) -> (i64, i64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_zero_rat(&self) -> bool;
    fn is_one_rat(&self) -> bool;
}

impl PackedRationalNumberReader for [u8] {
    #[inline(always)]
    fn get_number_view(&self) -> (BorrowedNumber, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        if (disc & NUM_MASK) == ARB_NUM {
            let rat: &Rational = unsafe { std::mem::transmute(&source[0]) };
            (
                BorrowedNumber::Large(rat),
                &source[std::mem::size_of::<Rational>()..],
            )
        } else {
            let (num, den, source) = self.get_frac_i64();
            (BorrowedNumber::Natural(num, den), source)
        }
    }

    #[inline(always)]
    fn get_frac_i64(&self) -> (i64, i64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & NUM_MASK {
            U8_NUM => {
                let v = source.get_u8();
                (v as i64, source)
            }
            U16_NUM => {
                let v = source.get_u16_le();
                (v as i64, source)
            }
            U32_NUM => {
                let v = source.get_u32_le();
                (v as i64, source)
            }
            U64_NUM => {
                let v = source.get_u64_le();
                (v as i64, source)
            }
            ARB_NUM => {
                panic!("Overflow")
            }
            x => {
                unreachable!("Unsupported numerator type {}", x)
            }
        };

        let den;
        (den, source) = match disc & DEN_MASK {
            0 => (1i64, source),
            U8_DEN => {
                let v = source.get_u8();
                (v as i64, source)
            }
            U16_DEN => {
                let v = source.get_u16_le();
                (v as i64, source)
            }
            U32_DEN => {
                let v = source.get_u32_le();
                (v as i64, source)
            }
            U64_DEN => {
                let v = source.get_u64_le();
                (v as i64, source)
            }
            ARB_DEN => {
                panic!("Overflow")
            }
            x => {
                unreachable!("Unsupported denominator type {}", x)
            }
        };

        if disc & SIGN != 0 {
            (-num, den, source)
        } else {
            (num, den, source)
        }
    }

    #[inline(always)]
    fn skip_rational(&self) -> &[u8] {
        let mut dest = self;
        let var_size = dest.get_u8();

        if var_size & NUM_MASK == ARB_NUM {
            dest.advance(std::mem::size_of::<Rational>());
            dest
        } else {
            let size = (var_size & NUM_MASK) + ((var_size & DEN_MASK) >> 4);
            dest.advance(size as usize);
            dest
        }
    }

    #[inline(always)]
    fn is_zero_rat(&self) -> bool {
        // TODO: make a zero have no number at all (i.e., self[1] = 0)
        self[1] == 1 && self[2] == 0
    }

    #[inline(always)]
    fn is_one_rat(&self) -> bool {
        self[1] == 1 && self[2] == 1
    }
}

impl PackedRationalNumberWriter for (i64, i64) {
    #[inline(always)]
    fn write_packed(self, dest: &mut Vec<u8>) {
        let p = dest.len();

        let num_u64 = self.0.abs() as u64;
        let den_u64 = self.1.abs() as u64;
        (num_u64, den_u64).write_packed(dest);

        if self.0 >= 0 && self.1 < 0 || self.0 < 0 && self.1 >= 0 {
            dest[p] |= SIGN;
        }
    }

    #[inline(always)]
    fn write_packed_fixed(self, dest: &mut [u8]) {
        let p = dest.len();

        let num_u64 = self.0.abs() as u64;
        let den_u64 = self.1.abs() as u64;
        (num_u64, den_u64).write_packed_fixed(dest);

        if self.0 >= 0 && self.1 < 0 || self.0 < 0 && self.1 >= 0 {
            dest[p] |= SIGN;
        }
    }
}

impl PackedRationalNumberWriter for (u64, u64) {
    #[inline(always)]
    fn write_packed(self, dest: &mut Vec<u8>) {
        let p = dest.len();

        if self.0 < u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(self.0 as u8);
        } else if self.0 < u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(self.0 as u16);
        } else if self.0 < u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(self.0 as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 < u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 < u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 < u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u8(3);
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline(always)]
    fn write_packed_fixed(self, mut dest: &mut [u8]) {
        let p = dest.len();

        if self.0 < u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(self.0 as u8);
        } else if self.0 < u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(self.0 as u16);
        } else if self.0 < u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(self.0 as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 < u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 < u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 < u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u8(3);
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }
}
