use std::{fmt::Display, ops::Neg};

use rand::Rng;
use rug::{
    ops::{Pow, RemRounding},
    Integer as ArbitraryPrecisionInteger,
};

use crate::utils;

use super::{
    finite_field::{FiniteField, ToFiniteField},
    rational::Rational,
    EuclideanDomain, Ring,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IntegerRing;

impl IntegerRing {
    pub fn new() -> IntegerRing {
        IntegerRing
    }
}

// FIXME: PartialEq can only work when Large simplifies to Natural whenever possible
#[derive(Clone, PartialEq, Debug)]
pub enum Integer {
    Natural(i64),
    Large(ArbitraryPrecisionInteger),
}

impl ToFiniteField<u32> for Integer {
    fn to_finite_field(&self, field: FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        match self {
            &Integer::Natural(n) => {
                let mut ff = if n < u32::MAX as i64 {
                    field.to_element(n.abs() as u32)
                } else {
                    field.to_element((n.abs() as u64 % field.get_prime() as u64) as u32)
                };

                if n < 0 {
                    ff = field.neg(&ff);
                }

                ff
            }
            Integer::Large(r) => field.to_element(r.mod_u(field.get_prime())),
        }
    }
}

impl Integer {
    pub fn new(num: i64) -> Integer {
        Integer::Natural(num)
    }

    pub fn from_finite_field_u32(
        field: FiniteField<u32>,
        element: &<FiniteField<u32> as Ring>::Element,
    ) -> Integer {
        Integer::Natural(field.from_element(*element) as i64)
    }

    pub fn to_rational(&self) -> Rational {
        match self {
            Integer::Natural(n) => Rational::Natural(*n, 1),
            Integer::Large(r) => Rational::Large(r.into()),
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Integer::Natural(n) => *n < 0,
            Integer::Large(r) => ArbitraryPrecisionInteger::from(r.signum_ref()) == -1,
        }
    }

    /// Use Garner's algorithm for the Chinese remainder theorem
    /// to reconstruct an x that satisfies n1 = x % p1 and n2 = x % p2.
    /// The x will be in the range [-p1*p2/2,p1*p2/2].
    pub fn chinese_remainder(n1: Integer, n2: Integer, p1: Integer, p2: Integer) -> Integer {
        // make sure n1 < n2
        if match (&n1, &n2) {
            (Integer::Natural(n1), Integer::Natural(n2)) => n1 > n2,
            (Integer::Natural(_), Integer::Large(_)) => false,
            (Integer::Large(_), Integer::Natural(_)) => true,
            (Integer::Large(r1), Integer::Large(r2)) => r1 > r2,
        } {
            return Self::chinese_remainder(n2, n1, p2, p1);
        }

        let p1 = match p1 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let p2 = match p2 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        let n1 = match n1 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let n2 = match n2 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        // convert to mixed-radix notation
        let gamma1 = (p1.clone() % p2.clone())
            .invert(&p2)
            .expect(&format!("Could not invert {} in {}", p1, p2));

        let v1 = ((n2.clone() - n1.clone()) * gamma1.clone()) % p2.clone();

        // convert to standard representation
        let r = v1 * p1.clone() + n1;

        let res = if r.clone() * 2 > p1.clone() * p2.clone() {
            r - p1 * p2
        } else {
            r
        };

        // potentially downgrade from bigint
        if let Some(r) = res.to_i64() {
            Integer::Natural(r)
        } else {
            Integer::Large(ArbitraryPrecisionInteger::from(res))
        }
    }
}

impl Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Integer::Natural(n) => n.fmt(f),
            Integer::Large(r) => r.fmt(f),
        }
    }
}

impl Display for IntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Ring for IntegerRing {
    type Element = Integer;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_add(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) + ArbitraryPrecisionInteger::from(*n2),
                    )
                }
            }
            // TODO: check downcast
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::Large(r1 + r2)
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::Large((r1 + r2).into()),
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(nn) = n1.checked_mul(*n2) {
                    Integer::Natural(nn)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) * ArbitraryPrecisionInteger::from(*n2),
                    )
                }
            }
            // FIXME: downcast
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::Large(r1 * r2)
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::Large((r1 * r2).into()),
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        // TODO: optimize
        *a = self.add(a, b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        match a {
            Integer::Natural(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::Natural(neg)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n).neg())
                }
            }
            Integer::Large(r) => Integer::Large(r.neg().into()),
        }
    }

    fn zero() -> Self::Element {
        Integer::Natural(0)
    }

    fn one(&self) -> Self::Element {
        Integer::Natural(1)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        match b {
            Integer::Natural(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Natural(pn)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Large(r) => Integer::Large(r.pow(e).into()),
        }
    }

    fn is_zero(a: &Self::Element) -> bool {
        match a {
            Integer::Natural(r) => *r == 0,
            // TODO: not needed anymore when automatically simplifying
            Integer::Large(r) => r.to_u8().map(|x| x == 0).unwrap_or(false),
        }
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        match a {
            Integer::Natural(r) => *r == 1,
            Integer::Large(r) => r.to_u8().map(|x| x == 1).unwrap_or(false),
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Integer::Natural(r)
    }
}

impl EuclideanDomain for IntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Integer::Natural(a), Integer::Natural(b)) => Integer::Natural(a.rem_euclid(*b)),
            // TODO: downcast
            (Integer::Natural(a), Integer::Large(b)) => {
                Integer::Large(ArbitraryPrecisionInteger::from(*a).rem_euc(b).into())
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                Integer::Large(a.rem_euc(ArbitraryPrecisionInteger::from(*b)).into())
            }
            (Integer::Large(a), Integer::Large(b)) => Integer::Large(a.rem_euc(b).into()),
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        match (a, b) {
            (Integer::Natural(a), Integer::Natural(b)) => (
                Integer::Natural(a.div_euclid(*b)),
                Integer::Natural(a.rem_euclid(*b)),
            ),
            // TODO: downcast
            (Integer::Natural(a), Integer::Large(b)) => {
                let r = ArbitraryPrecisionInteger::from(*a).div_rem_euc(b.clone());
                (Integer::Large(r.0), Integer::Large(r.1))
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                let r = a.clone().div_rem_euc(ArbitraryPrecisionInteger::from(*b));
                (Integer::Large(r.0), Integer::Large(r.1))
            }
            (Integer::Large(a), Integer::Large(b)) => {
                let r = a.clone().div_rem_euc(b.clone());
                (Integer::Large(r.0), Integer::Large(r.1))
            }
        }
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                Integer::Natural(utils::gcd_signed(*n1 as i64, *n2 as i64))
            }
            // FIXME: downcast
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::Large(ArbitraryPrecisionInteger::from(r1.clone().gcd(r2)))
            }
            (Integer::Large(r1), Integer::Large(r2)) => {
                Integer::Large(ArbitraryPrecisionInteger::from(r1.clone().gcd(r2)))
            }
        }
    }
}
