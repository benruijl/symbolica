use bytes::{Buf, BufMut};
use rug::integer::Order;

use crate::{
    coefficient::{Coefficient, CoefficientView, SerializedRational},
    domains::{
        finite_field::FiniteFieldElement, integer::IntegerRing, rational::Rational,
        rational_polynomial::RationalPolynomial,
    },
    state::FiniteFieldIndex,
    utils,
};

const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const FIN_NUM: u8 = 0b00000101;
const ARB_NUM: u8 = 0b00000111;
const RAT_POLY: u8 = 0b00001000;
const U8_DEN: u8 = 0b00010000;
const U16_DEN: u8 = 0b00100000;
const U32_DEN: u8 = 0b00110000;
const U64_DEN: u8 = 0b01000000;
const ARB_DEN: u8 = 0b01110000;
const NUM_MASK: u8 = 0b00001111;
const DEN_MASK: u8 = 0b01110000;
const SIGN: u8 = 0b10000000;

#[inline(always)]
fn get_size_of_natural(num_type: u8) -> u8 {
    match num_type {
        0 => 0,
        U8_NUM => 1,
        U16_NUM => 2,
        U32_NUM => 4,
        U64_NUM => 8,
        _ => unreachable!(),
    }
}

/// A generalized rational number. The first byte indicates the sign, size and type of the numerator and denominator.
/// The highest four bits give the byte size of the numerator and the lower bits of the denominator.
pub trait PackedRationalNumberWriter {
    /// Write a single number.
    fn write_packed(&self, dest: &mut Vec<u8>);
    /// Write a fraction to a fixed-size buffer.
    fn write_packed_fixed(&self, dest: &mut [u8]);
    /// Get the number of bytes of the packed representation.
    fn get_packed_size(&self) -> u64;
}

impl PackedRationalNumberWriter for Coefficient {
    fn write_packed(&self, dest: &mut Vec<u8>) {
        match self {
            Coefficient::Rational(r) => match r {
                Rational::Natural(num, den) => (*num, *den).write_packed(dest),
                Rational::Large(r) => {
                    dest.put_u8(ARB_NUM | ARB_DEN);

                    let num_digits = r.numer().significant_digits::<u8>();
                    let den_digits = r.denom().significant_digits::<u8>();

                    if r.numer() < &0 {
                        (-(num_digits as i64), den_digits as i64).write_packed(dest);
                    } else {
                        (num_digits as i64, den_digits as i64).write_packed(dest);
                    }

                    let old_len = dest.len();
                    dest.resize(old_len + num_digits + den_digits, 0);
                    r.numer().write_digits(&mut dest[old_len..], Order::Lsf);
                    r.denom()
                        .write_digits(&mut dest[old_len + num_digits..], Order::Lsf);
                }
            },
            Coefficient::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (num.0, f.0 as u64).write_packed(dest); // this adds an extra tag
            }
            Coefficient::RationalPolynomial(p) => {
                dest.put_u8(RAT_POLY);
                // note that this is not a linear representation
                // FIXME: pointer alignment
                let p = p.clone();
                let v = std::mem::ManuallyDrop::new(p);
                let lin_buf = unsafe { utils::any_as_u8_slice(&v) };

                dest.extend(lin_buf);
            }
        }
    }

    fn write_packed_fixed(&self, mut dest: &mut [u8]) {
        match self {
            Coefficient::Rational(r) => match r {
                Rational::Natural(num, den) => (*num, *den).write_packed_fixed(dest),
                Rational::Large(_) => todo!("Writing large packed rational not implemented"),
            },
            Coefficient::RationalPolynomial(_) => {
                todo!("Writing packed rational polynomial not implemented")
            }
            Coefficient::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (num.0, f.0 as u64).write_packed_fixed(dest);
            }
        }
    }

    fn get_packed_size(&self) -> u64 {
        match self {
            Coefficient::Rational(r) => match r {
                Rational::Natural(num, den) => (*num, *den).get_packed_size(),
                Rational::Large(l) => {
                    let n = l.numer().significant_digits::<u8>() as i64;
                    let d = l.denom().significant_digits::<u8>() as i64;
                    1 + (n, d).get_packed_size() + n as u64 + d as u64
                }
            },
            Coefficient::FiniteField(m, i) => 2 + (m.0, i.0 as u64).get_packed_size(),
            Coefficient::RationalPolynomial(_) => {
                1 + std::mem::size_of::<RationalPolynomial<IntegerRing, u16>>() as u64
            }
        }
    }
}

/// A reader for generalized rational numbers. See [`MultiPrecisionRationalNumberWriter`].
pub trait PackedRationalNumberReader {
    fn get_coeff_view(&self) -> (CoefficientView, &[u8]);
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
    fn get_frac_i64(&self) -> (i64, i64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_zero_rat(&self) -> bool;
    fn is_one_rat(&self) -> bool;
}

impl PackedRationalNumberReader for [u8] {
    #[inline(always)]
    fn get_coeff_view(&self) -> (CoefficientView, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        if disc == RAT_POLY {
            let rat = unsafe { std::mem::transmute(&source[0]) };
            (
                CoefficientView::RationalPolynomial(rat),
                &source[std::mem::size_of::<RationalPolynomial<IntegerRing, u16>>()..],
            )
        } else if (disc & NUM_MASK) == ARB_NUM {
            let (num, den);
            (num, den, source) = source.get_frac_i64();
            let num_len = num.unsigned_abs() as usize;
            let den_len = den.unsigned_abs() as usize;
            let num_limbs = &source[..num_len];
            let den_limbs = &source[num_len..num_len + den_len];

            (
                CoefficientView::Large(SerializedRational {
                    is_negative: num < 0,
                    num_digits: num_limbs,
                    den_digits: den_limbs,
                }),
                &source[num_len + den_len..],
            )
        } else if (disc & NUM_MASK) == FIN_NUM {
            let (num, fi);
            (num, fi, source) = source.get_frac_u64();
            (
                CoefficientView::FiniteField(
                    FiniteFieldElement(num),
                    FiniteFieldIndex(fi as usize),
                ),
                source,
            )
        } else {
            let (num, den, source) = self.get_frac_i64();
            (CoefficientView::Natural(num, den), source)
        }
    }

    #[inline(always)]
    fn get_frac_u64(&self) -> (u64, u64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & NUM_MASK {
            U8_NUM => {
                let v = source.get_u8();
                (v as u64, source)
            }
            U16_NUM => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            U32_NUM => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            U64_NUM => {
                let v = source.get_u64_le();
                (v, source)
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
            0 => (1u64, source),
            U8_DEN => {
                let v = source.get_u8();
                (v as u64, source)
            }
            U16_DEN => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            U32_DEN => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            U64_DEN => {
                let v = source.get_u64_le();
                (v, source)
            }
            ARB_DEN => {
                panic!("Overflow")
            }
            x => {
                unreachable!("Unsupported denominator type {}", x)
            }
        };

        (num, den, source)
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

        let v_num = var_size & NUM_MASK;
        if v_num == ARB_NUM {
            let (num_size, den_size);
            (num_size, den_size, dest) = dest.get_frac_i64();
            let num_size = num_size.unsigned_abs() as usize;
            let den_size = den_size.unsigned_abs() as usize;
            dest.advance(num_size + den_size);
            dest
        } else if v_num == RAT_POLY {
            dest.advance(std::mem::size_of::<RationalPolynomial<IntegerRing, u16>>());
            dest
        } else if v_num == FIN_NUM {
            let var_size = dest.get_u8();
            let size = get_size_of_natural(var_size & NUM_MASK)
                + get_size_of_natural((var_size & DEN_MASK) >> 4);
            dest.advance(size as usize);
            dest
        } else {
            let size = get_size_of_natural(v_num) + get_size_of_natural((var_size & DEN_MASK) >> 4);
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
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();

        let num_u64 = self.0.unsigned_abs();
        let den_u64 = self.1.unsigned_abs();
        (num_u64, den_u64).write_packed(dest);

        if self.0 >= 0 && self.1 < 0 || self.0 < 0 && self.1 >= 0 {
            dest[p] |= SIGN;
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        let num_u64 = self.0.unsigned_abs();
        let den_u64 = self.1.unsigned_abs();
        (num_u64, den_u64).write_packed_fixed(dest);

        if self.0 >= 0 && self.1 < 0 || self.0 < 0 && self.1 >= 0 {
            dest[0] |= SIGN;
        }
    }

    fn get_packed_size(&self) -> u64 {
        (self.0 as u64, self.1 as u64).get_packed_size()
    }
}

impl PackedRationalNumberWriter for (u64, u64) {
    #[inline(always)]
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();

        if self.0 <= u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(self.0 as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        let (tag, mut dest) = dest.split_first_mut().unwrap();

        if self.0 <= u8::MAX as u64 {
            *tag = U8_NUM;
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            *tag = U16_NUM;
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            *tag = U32_NUM;
            dest.put_u32_le(self.0 as u32);
        } else {
            *tag = U64_NUM;
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            *tag |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            *tag |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            *tag |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            *tag |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    fn get_packed_size(&self) -> u64 {
        let mut size = 1;
        size += if self.0 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.0 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.0 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };

        size += if self.1 == 1 {
            0
        } else if self.1 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.1 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.1 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };
        size as u64
    }
}
