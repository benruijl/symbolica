use std::cmp::Ordering;

use bytes::{Buf, BufMut};
use rug::{
    ops::Pow as RP, Integer as ArbitraryPrecisionInteger, Rational as ArbitraryPrecisionRational,
};

use crate::{
    poly::polynomial::MultivariatePolynomial,
    rings::{
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, ToFiniteField,
        },
        integer::{Integer, IntegerRing},
        rational::{Rational, RationalField},
        rational_polynomial::RationalPolynomial,
        Field, Ring,
    },
    state::{FiniteFieldIndex, State},
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

pub trait ConvertToRing: Ring {
    /// Convert from a Symbolica `Number` to a Ring.
    fn from_number(&self, number: Number) -> Self::Element;

    /// Convert from a Symbolica `BorrowedNumber` to a Ring.
    fn from_borrowed_number(&self, number: BorrowedNumber<'_>) -> Self::Element;
}

// TODO: rename to Coefficient
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Number {
    Natural(i64, i64),
    Large(ArbitraryPrecisionRational),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(RationalPolynomial<IntegerRing, u16>),
}

impl Number {
    pub fn to_borrowed<'a>(&'a self) -> BorrowedNumber<'a> {
        match self {
            Number::Natural(num, den) => BorrowedNumber::Natural(*num, *den),
            Number::Large(r) => BorrowedNumber::Large(r),
            Number::FiniteField(num, field) => BorrowedNumber::FiniteField(*num, *field),
            Number::RationalPolynomial(r) => BorrowedNumber::RationalPolynomial(r),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Number::Natural(num, _den) => *num == 0,
            Number::Large(_r) => false,
            Number::FiniteField(num, _field) => num.0 == 0,
            Number::RationalPolynomial(r) => r.numerator.is_zero(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowedNumber<'a> {
    Natural(i64, i64),
    Large(&'a ArbitraryPrecisionRational),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(&'a RationalPolynomial<IntegerRing, u16>),
}

impl<'a> ConvertToRing for RationalField {
    #[inline]
    fn from_number(&self, number: Number) -> Self::Element {
        match number {
            Number::Natural(r, d) => Rational::Natural(r, d),
            Number::Large(r) => Rational::Large(r),
            Number::FiniteField(_, _) => panic!("Cannot convert finite field to rational"),
            Number::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn from_borrowed_number(&self, number: BorrowedNumber<'_>) -> Rational {
        match number {
            BorrowedNumber::Natural(r, d) => Rational::Natural(r, d),
            BorrowedNumber::Large(r) => Rational::Large(r.clone()),
            BorrowedNumber::FiniteField(_, _) => panic!("Cannot convert finite field to rational"),
            BorrowedNumber::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }
}

impl<'a> ConvertToRing for IntegerRing {
    #[inline]
    fn from_number(&self, number: Number) -> Integer {
        match number {
            Number::Natural(r, d) => {
                debug_assert!(d == 1);
                Integer::Natural(r)
            }
            Number::Large(r) => {
                let (n, d) = r.into_numer_denom();
                debug_assert!(d == 1);
                Integer::Large(n)
            }
            Number::FiniteField(_, _) => panic!("Cannot convert finite field to integer"),
            Number::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn from_borrowed_number(&self, number: BorrowedNumber<'_>) -> Integer {
        match number {
            BorrowedNumber::Natural(r, d) => {
                debug_assert!(d == 1);
                Integer::Natural(r)
            }
            BorrowedNumber::Large(r) => {
                debug_assert!(r.denom() == &1);
                Integer::Large(r.numer().clone())
            }
            BorrowedNumber::FiniteField(_, _) => panic!("Cannot convert finite field to integer"),
            BorrowedNumber::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to integer")
            }
        }
    }
}

impl<'a, UField: FiniteFieldWorkspace> ConvertToRing for FiniteField<UField>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    Integer: ToFiniteField<UField>,
{
    #[inline]
    fn from_number(&self, number: Number) -> <FiniteField<UField> as Ring>::Element {
        match number {
            Number::Natural(n, d) => self.div(
                &Integer::new(n).to_finite_field(self),
                &Integer::new(d).to_finite_field(self),
            ),
            Number::Large(r) => {
                let (n, d) = r.into_numer_denom();
                self.div(
                    &Integer::Large(n).to_finite_field(self),
                    &Integer::Large(d).to_finite_field(self),
                )
            }
            Number::FiniteField(_, _) => panic!("Cannot convert finite field to other one"),
            Number::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }

    #[inline]
    fn from_borrowed_number(
        &self,
        number: BorrowedNumber<'_>,
    ) -> <FiniteField<UField> as Ring>::Element {
        match number {
            BorrowedNumber::Natural(n, d) => self.div(
                &Integer::new(n).to_finite_field(self),
                &Integer::new(d).to_finite_field(self),
            ),
            BorrowedNumber::Large(r) => self.div(
                &Integer::Large(r.numer().clone()).to_finite_field(self),
                &Integer::Large(r.denom().clone()).to_finite_field(self),
            ),
            BorrowedNumber::FiniteField(_, _) => panic!("Cannot convert finite field to other one"),
            BorrowedNumber::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }
}

impl BorrowedNumber<'_> {
    pub fn normalize(&self) -> Number {
        match self {
            BorrowedNumber::Natural(num, den) => {
                let gcd = utils::gcd_signed(*num, *den);
                Number::Natural(*num / gcd, *den / gcd)
            }
            BorrowedNumber::Large(_)
            | BorrowedNumber::FiniteField(_, _)
            | BorrowedNumber::RationalPolynomial(_) => self.to_owned(),
        }
    }

    pub fn to_owned(&self) -> Number {
        match self {
            BorrowedNumber::Natural(num, den) => Number::Natural(*num, *den),
            BorrowedNumber::Large(r) => Number::Large((*r).clone()),
            BorrowedNumber::FiniteField(num, field) => Number::FiniteField(*num, *field),
            BorrowedNumber::RationalPolynomial(p) => Number::RationalPolynomial((*p).clone()),
        }
    }

    pub fn add(&self, other: &BorrowedNumber<'_>, state: &State) -> Number {
        match (self, other) {
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Natural(n2, d2)) => {
                let r = &Rational::Natural(*n1, *d1) + &Rational::Natural(*n2, *d2);
                match r {
                    Rational::Natural(n, d) => Number::Natural(n, d),
                    Rational::Large(r) => Number::Large(r),
                }
            }
            // TODO: check downcast
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Large(r2))
            | (BorrowedNumber::Large(r2), BorrowedNumber::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Number::Large(r1 + *r2)
            }
            (BorrowedNumber::Large(r1), BorrowedNumber::Large(r2)) => {
                Number::Large((*r1 + *r2).into())
            }
            (BorrowedNumber::FiniteField(n1, i1), BorrowedNumber::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(*i1).get_prime(),
                        state.get_finite_field(*i2).get_prime()
                    );
                }
                let f = state.get_finite_field(*i1);
                Number::FiniteField(f.add(n1, n2), *i1)
            }
            (BorrowedNumber::FiniteField(_, _), _) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (_, BorrowedNumber::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (BorrowedNumber::Natural(n, d), BorrowedNumber::RationalPolynomial(p))
            | (BorrowedNumber::RationalPolynomial(p), BorrowedNumber::Natural(n, d)) => {
                let r = (*p).clone();
                let r2 = RationalPolynomial {
                    numerator: MultivariatePolynomial::new_from_constant(
                        &p.numerator,
                        Integer::Natural(*n),
                    ),
                    denominator: MultivariatePolynomial::new_from_constant(
                        &p.denominator,
                        Integer::Natural(*d),
                    ),
                };
                Number::RationalPolynomial(&r + &r2)
            }
            (BorrowedNumber::Large(l), BorrowedNumber::RationalPolynomial(p))
            | (BorrowedNumber::RationalPolynomial(p), BorrowedNumber::Large(l)) => {
                let r = (*p).clone();
                let r2 = RationalPolynomial {
                    numerator: MultivariatePolynomial::new_from_constant(
                        &p.numerator,
                        Integer::Large(l.numer().clone()),
                    ),
                    denominator: MultivariatePolynomial::new_from_constant(
                        &p.denominator,
                        Integer::Large(l.denom().clone()),
                    ),
                };
                Number::RationalPolynomial(&r + &r2)
            }
            (BorrowedNumber::RationalPolynomial(p1), BorrowedNumber::RationalPolynomial(p2)) => {
                if p1.get_var_map() != p2.get_var_map() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_var_map(&mut p2);
                    Number::RationalPolynomial(&p1 + &p2)
                } else {
                    Number::RationalPolynomial(*p1 + *p2)
                }
            }
        }
    }

    pub fn mul(&self, other: &BorrowedNumber<'_>, state: &State) -> Number {
        match (self, other) {
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Natural(n2, d2)) => {
                let r = &Rational::Natural(*n1, *d1) * &Rational::Natural(*n2, *d2);
                match r {
                    Rational::Natural(n, d) => Number::Natural(n, d),
                    Rational::Large(r) => Number::Large(r),
                }
            }
            // TODO: check downcast
            (BorrowedNumber::Natural(n1, d1), BorrowedNumber::Large(r2))
            | (BorrowedNumber::Large(r2), BorrowedNumber::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Number::Large(r1 * *r2)
            }
            (BorrowedNumber::Large(r1), BorrowedNumber::Large(r2)) => {
                Number::Large((*r1 * *r2).into())
            }
            (BorrowedNumber::FiniteField(n1, i1), BorrowedNumber::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(*i1).get_prime(),
                        state.get_finite_field(*i2).get_prime()
                    );
                }
                let f = state.get_finite_field(*i1);
                Number::FiniteField(f.mul(n1, n2), *i1)
            }
            (BorrowedNumber::FiniteField(_, _), _) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (_, BorrowedNumber::FiniteField(_, _)) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (BorrowedNumber::Natural(n, d), BorrowedNumber::RationalPolynomial(p))
            | (BorrowedNumber::RationalPolynomial(p), BorrowedNumber::Natural(n, d)) => {
                let mut r = (*p).clone();
                r.numerator = r.numerator.mul_coeff(Integer::Natural(*n));
                r.denominator = r.denominator.mul_coeff(Integer::Natural(*d));
                Number::RationalPolynomial(r)
            }
            (BorrowedNumber::Large(l), BorrowedNumber::RationalPolynomial(p))
            | (BorrowedNumber::RationalPolynomial(p), BorrowedNumber::Large(l)) => {
                let mut r = (*p).clone();
                r.numerator = r.numerator.mul_coeff(Integer::Large(l.numer().clone()));
                r.denominator = r.denominator.mul_coeff(Integer::Large(l.denom().clone()));
                Number::RationalPolynomial(r)
            }
            (BorrowedNumber::RationalPolynomial(p1), BorrowedNumber::RationalPolynomial(p2)) => {
                if p1.get_var_map() != p2.get_var_map() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_var_map(&mut p2);
                    Number::RationalPolynomial(&p1 * &p2)
                } else {
                    Number::RationalPolynomial(*p1 * *p2)
                }
            }
        }
    }

    pub fn pow(&self, other: &BorrowedNumber<'_>, _state: &State) -> (Number, Number) {
        // TODO: normalize 4^1/3 to 2^(2/3)?
        match (self, other) {
            (&BorrowedNumber::Natural(mut n1, mut d1), &BorrowedNumber::Natural(mut n2, d2)) => {
                if n2 < 0 {
                    n2 = n2.saturating_abs();
                    (n1, d1) = (d1, n1);
                }

                if n2 < u32::MAX as i64 {
                    if let Some(pn) = n1.checked_pow(n2 as u32) {
                        if let Some(pd) = d1.checked_pow(n2 as u32) {
                            // TODO: simplify 4^(1/2)
                            return (Number::Natural(pn, pd), Number::Natural(1, d2));
                        }
                    }

                    (
                        Number::Large(ArbitraryPrecisionRational::from((n1, d1)).pow(n2 as u32)),
                        Number::Natural(1, d2),
                    )
                } else {
                    panic!("Power is too large: {}", n2);
                }
            }
            (&BorrowedNumber::RationalPolynomial(r), &BorrowedNumber::Natural(n2, d2)) => {
                if n2.saturating_abs() >= u32::MAX as i64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.clone().inv();
                    (
                        Number::RationalPolynomial(r.pow(n2.saturating_abs() as u64)),
                        Number::Natural(1, d2),
                    )
                } else {
                    (
                        Number::RationalPolynomial(r.pow(n2 as u64)),
                        Number::Natural(1, d2),
                    )
                }
            }
            _ => {
                unimplemented!(
                    "Power of configuration {:?}^{:?} is not implemented",
                    self,
                    other
                );
            }
        }
    }

    pub fn cmp(&self, other: &BorrowedNumber) -> Ordering {
        match (self, other) {
            (&BorrowedNumber::Natural(n1, d1), &BorrowedNumber::Natural(n2, d2)) => {
                // TODO: improve
                if n1 < 0 && n2 > 0 {
                    return Ordering::Less;
                }
                if n1 > 0 && n2 < 0 {
                    return Ordering::Greater;
                }

                match n1.checked_mul(d2) {
                    Some(a1) => match n2.checked_mul(d1) {
                        Some(a2) => a1.cmp(&a2),
                        None => ArbitraryPrecisionInteger::from(a1).cmp(
                            &(ArbitraryPrecisionInteger::from(n2)
                                * ArbitraryPrecisionInteger::from(d1)),
                        ),
                    },
                    None => (ArbitraryPrecisionInteger::from(n1)
                        * ArbitraryPrecisionInteger::from(d2))
                    .cmp(
                        &(ArbitraryPrecisionInteger::from(n2)
                            * ArbitraryPrecisionInteger::from(d1)),
                    ),
                }
            }
            (BorrowedNumber::Large(n1), BorrowedNumber::Large(n2)) => n1.cmp(n2),
            (BorrowedNumber::FiniteField(n1, _), BorrowedNumber::FiniteField(n2, _)) => {
                n1.0.cmp(&n2.0)
            }
            (&BorrowedNumber::Natural(n1, d1), BorrowedNumber::Large(n2)) => {
                ArbitraryPrecisionRational::from((n1, d1)).cmp(&n2)
            }
            (BorrowedNumber::Large(n1), &BorrowedNumber::Natural(n2, d2)) => {
                n1.cmp(&&ArbitraryPrecisionRational::from((n2, d2)))
            }
            _ => unreachable!(),
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
            Number::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (num.0, f.0 as u64).write_packed(dest); // this adds an extra tag
            }
            Number::RationalPolynomial(p) => {
                dest.put_u8(RAT_POLY);
                // note that this is not a linear representation
                let v = std::mem::ManuallyDrop::new(p);
                let lin_buf = unsafe { utils::any_as_u8_slice(&v) };

                dest.extend(lin_buf);
            }
        }
    }

    fn write_packed_fixed(self, mut dest: &mut [u8]) {
        match self {
            Number::Natural(num, den) => (num, den).write_packed_fixed(dest),
            Number::Large(_) | Number::RationalPolynomial(_) => {
                todo!("Writing large packed rational not implemented")
            }
            Number::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (num.0, f.0 as u64).write_packed_fixed(dest);
            }
        }
    }

    fn get_packed_size(&self) -> u64 {
        match self {
            Number::Natural(num, den) => (*num, *den).get_packed_size(),
            Number::Large(_) => 1 + std::mem::size_of::<ArbitraryPrecisionRational>() as u64,
            Number::FiniteField(m, i) => 2 + (m.0, i.0 as u64).get_packed_size(),
            Number::RationalPolynomial(_) => {
                1 + std::mem::size_of::<RationalPolynomial<IntegerRing, u16>>() as u64
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
    /// Get the number of bytes of the packed representation.
    fn get_packed_size(&self) -> u64;
}

/// A reader for generalized rational numbers. See [`ArbitraryPrecisionRationalNumberWriter`].
pub trait PackedRationalNumberReader {
    fn get_number_view(&self) -> (BorrowedNumber, &[u8]);
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
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
        if disc == RAT_POLY {
            let rat = unsafe { std::mem::transmute(&source[0]) };
            (
                BorrowedNumber::RationalPolynomial(rat),
                &source[std::mem::size_of::<RationalPolynomial<IntegerRing, u16>>()..],
            )
        } else if (disc & NUM_MASK) == ARB_NUM {
            let rat: &ArbitraryPrecisionRational = unsafe { std::mem::transmute(&source[0]) };
            (
                BorrowedNumber::Large(rat),
                &source[std::mem::size_of::<ArbitraryPrecisionRational>()..],
            )
        } else if (disc & NUM_MASK) == FIN_NUM {
            let (num, fi);
            (num, fi, source) = source.get_frac_u64();
            (
                BorrowedNumber::FiniteField(FiniteFieldElement(num), FiniteFieldIndex(fi as usize)),
                source,
            )
        } else {
            let (num, den, source) = self.get_frac_i64();
            (BorrowedNumber::Natural(num, den), source)
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
                (v as u64, source)
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
                (v as u64, source)
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
            dest.advance(std::mem::size_of::<ArbitraryPrecisionRational>());
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
    fn write_packed(self, dest: &mut Vec<u8>) {
        let p = dest.len();

        let num_u64 = self.0.abs() as u64; // FIXME: may overflow
        let den_u64 = self.1.abs() as u64;
        (num_u64, den_u64).write_packed(dest);

        if self.0 >= 0 && self.1 < 0 || self.0 < 0 && self.1 >= 0 {
            dest[p] |= SIGN;
        }
    }

    #[inline(always)]
    fn write_packed_fixed(self, dest: &mut [u8]) {
        let num_u64 = self.0.abs() as u64;
        let den_u64 = self.1.abs() as u64;
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
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline(always)]
    fn write_packed_fixed(self, dest: &mut [u8]) {
        let (tag, mut dest) = dest.split_first_mut().unwrap();

        if self.0 < u8::MAX as u64 {
            *tag = U8_NUM;
            dest.put_u8(self.0 as u8);
        } else if self.0 < u16::MAX as u64 {
            *tag = U16_NUM;
            dest.put_u16_le(self.0 as u16);
        } else if self.0 < u32::MAX as u64 {
            *tag = U32_NUM;
            dest.put_u32_le(self.0 as u32);
        } else {
            *tag = U64_NUM;
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 < u8::MAX as u64 {
            *tag |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 < u16::MAX as u64 {
            *tag |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 < u32::MAX as u64 {
            *tag |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            *tag |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    fn get_packed_size(&self) -> u64 {
        let mut size = 1;
        size += if self.0 < u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.0 < u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.0 < u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };

        size += if self.1 == 1 {
            0
        } else if self.1 < u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.1 < u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.1 < u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };
        size as u64
    }
}
