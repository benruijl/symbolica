use std::{
    cmp::Ordering,
    fmt::{Display, Error, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use rug::{
    integer::IntegerExt64,
    ops::{Pow, RemRounding},
    Integer as ArbitraryPrecisionInteger,
};

use crate::utils;

use super::{
    finite_field::{FiniteField, FiniteFieldCore, ToFiniteField},
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
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Integer {
    Natural(i64),
    Large(ArbitraryPrecisionInteger),
}

impl ToFiniteField<u32> for Integer {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        match self {
            &Integer::Natural(n) => field.to_element(n.rem_euclid(field.get_prime() as i64) as u32),
            Integer::Large(r) => field.to_element(r.mod_u(field.get_prime())),
        }
    }
}

impl ToFiniteField<u64> for Integer {
    fn to_finite_field(&self, field: &FiniteField<u64>) -> <FiniteField<u64> as Ring>::Element {
        match self {
            &Integer::Natural(n) => {
                if field.get_prime() >= i64::MAX as u64 {
                    field.to_element((n as i128).rem_euclid(field.get_prime() as i128) as u64)
                } else {
                    field.to_element(n.rem_euclid(field.get_prime() as i64) as u64)
                }
            }
            Integer::Large(r) => field.to_element(r.mod_u64(field.get_prime())),
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

    pub fn abs(&self) -> Integer {
        match self {
            Integer::Natural(n) => {
                if *n == i64::MIN {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n).abs())
                } else {
                    Integer::Natural(n.abs())
                }
            }
            Integer::Large(n) => Integer::Large(n.clone().abs()),
        }
    }

    pub fn abs_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if n1 == n2 {
                    Ordering::Equal
                } else if *n1 == i64::MIN {
                    Ordering::Greater
                } else {
                    n1.abs().cmp(&n2.abs())
                }
            }
            (Integer::Natural(n1), Integer::Large(n2)) => {
                if *n1 == i64::MIN {
                    ArbitraryPrecisionInteger::from(*n1).as_abs().cmp(n2)
                } else {
                    n2.as_abs()
                        .partial_cmp(&n1.abs())
                        .unwrap_or(Ordering::Equal)
                        .reverse()
                }
            }
            (Integer::Large(n1), Integer::Natural(n2)) => {
                if *n1 == i64::MIN {
                    n1.as_abs()
                        .cmp(&ArbitraryPrecisionInteger::from(*n2).as_abs())
                } else {
                    n1.as_abs()
                        .partial_cmp(&n2.abs())
                        .unwrap_or(Ordering::Equal)
                }
            }
            (Integer::Large(n1), Integer::Large(n2)) => n1.as_abs().cmp(&n2.as_abs()),
        }
    }

    /// Compute the binomial coefficient `(n k) = n!/(k!(n-k)!)`.
    ///
    /// The implementation does not to overflow.
    pub fn binom(n: i64, mut k: i64) -> Integer {
        if n < 0 || k < 0 || k > n {
            return Integer::Natural(0);
        }
        if k > n / 2 {
            k = n - k
        }
        let mut res = Integer::Natural(1);
        for i in 1..=k {
            res *= n - k + i;
            res /= i;
        }
        res
    }

    /// Compute the multinomial coefficient `(k_1+...+k_n)!/(k_1!*...*k_n!)`
    ///
    /// The implementation does not to overflow.
    pub fn multinom(k: &[u32]) -> Integer {
        let mut mcr = Integer::Natural(1);
        let mut accum = 0i64;
        for v in k {
            if let Some(res) = accum.checked_add(*v as i64) {
                accum = res;
            } else {
                panic!("Sum of occurrences exceeds i64: {:?}", k);
            }

            mcr *= &Self::binom(accum, *v as i64);
        }
        mcr
    }

    pub fn pow(&self, e: u64) -> Integer {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        match self {
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

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Integer::Natural(n1), Integer::Natural(n2)) => n1.partial_cmp(n2),
            (Integer::Natural(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Natural(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
        }
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Ring for IntegerRing {
    type Element = Integer;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    fn zero(&self) -> Self::Element {
        Integer::Natural(0)
    }

    fn one(&self) -> Self::Element {
        Integer::Natural(1)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e)
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

    fn fmt_display(&self, element: &Self::Element, f: &mut Formatter<'_>) -> Result<(), Error> {
        element.fmt(f)
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

impl<'a, 'b> Add<&'b Integer> for &'a Integer {
    type Output = Integer;

    fn add(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
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
}

impl<'a, 'b> Sub<&'b Integer> for &'a Integer {
    type Output = Integer;

    fn sub(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_sub(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) - ArbitraryPrecisionInteger::from(*n2),
                    )
                }
            }
            // TODO: check downcast
            (Integer::Natural(n1), Integer::Large(r2)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::Large(r1 - r2)
            }
            (Integer::Large(r1), Integer::Natural(n2)) => {
                let r2 = ArbitraryPrecisionInteger::from(*n2);
                Integer::Large(r1 - r2)
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::Large((r1 - r2).into()),
        }
    }
}

impl<'a, 'b> Mul<&'b Integer> for &'a Integer {
    type Output = Integer;

    fn mul(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
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
}

impl<'a, 'b> Div<&'b Integer> for &'a Integer {
    type Output = Integer;

    fn div(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(nn) = n1.checked_div(*n2) {
                    Integer::Natural(nn)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) / ArbitraryPrecisionInteger::from(*n2),
                    )
                }
            }
            // FIXME: downcast
            (Integer::Natural(n1), Integer::Large(r2)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::Large(r1 / r2)
            }
            (Integer::Large(r1), Integer::Natural(n2)) => {
                let r2 = ArbitraryPrecisionInteger::from(*n2);
                Integer::Large(r1 / r2)
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::Large((r1 / r2).into()),
        }
    }
}

impl<'a, 'b> Add<i64> for &'a Integer {
    type Output = Integer;

    fn add(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_add(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) + ArbitraryPrecisionInteger::from(rhs),
                    )
                }
            }
            Integer::Large(n1) => Integer::Large(n1 + ArbitraryPrecisionInteger::from(rhs)),
        }
    }
}

impl<'a, 'b> Sub<i64> for &'a Integer {
    type Output = Integer;

    fn sub(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_sub(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) - ArbitraryPrecisionInteger::from(rhs),
                    )
                }
            }
            Integer::Large(n1) => Integer::Large(n1 - ArbitraryPrecisionInteger::from(rhs)),
        }
    }
}

impl<'a, 'b> Mul<i64> for &'a Integer {
    type Output = Integer;

    fn mul(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_mul(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) * ArbitraryPrecisionInteger::from(rhs),
                    )
                }
            }
            Integer::Large(n1) => Integer::Large(n1 * ArbitraryPrecisionInteger::from(rhs)),
        }
    }
}

impl<'a, 'b> Div<i64> for &'a Integer {
    type Output = Integer;

    fn div(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_div(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Large(
                        ArbitraryPrecisionInteger::from(*n1) / ArbitraryPrecisionInteger::from(rhs),
                    )
                }
            }
            Integer::Large(n1) => Integer::Large(n1 / ArbitraryPrecisionInteger::from(rhs)),
        }
    }
}

impl<'a> AddAssign<&'a Integer> for Integer {
    fn add_assign(&mut self, rhs: &'a Integer) {
        match self {
            Integer::Natural(n1) => match rhs {
                Integer::Natural(n2) => {
                    if let Some(nn) = n1.checked_add(*n2) {
                        *n1 = nn;
                    } else {
                        let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                        r1.add_assign(ArbitraryPrecisionInteger::from(*n2));
                        *self = Integer::Large(r1)
                    }
                }
                Integer::Large(r2) => {
                    let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                    r1.add_assign(r2);
                    *self = Integer::Large(r1)
                }
            },
            Integer::Large(r1) => match rhs {
                Integer::Natural(n2) => r1.add_assign(ArbitraryPrecisionInteger::from(*n2)),
                Integer::Large(r2) => r1.add_assign(r2),
            },
        };
    }
}

impl<'a> SubAssign<&'a Integer> for Integer {
    fn sub_assign(&mut self, rhs: &'a Integer) {
        match self {
            Integer::Natural(n1) => match rhs {
                Integer::Natural(n2) => {
                    if let Some(nn) = n1.checked_sub(*n2) {
                        *n1 = nn;
                    } else {
                        let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                        r1.sub_assign(ArbitraryPrecisionInteger::from(*n2));
                        *self = Integer::Large(r1)
                    }
                }
                Integer::Large(r2) => {
                    let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                    r1.sub_assign(r2);
                    *self = Integer::Large(r1)
                }
            },
            Integer::Large(r1) => match rhs {
                Integer::Natural(n2) => r1.sub_assign(ArbitraryPrecisionInteger::from(*n2)),
                Integer::Large(r2) => r1.sub_assign(r2),
            },
        };
    }
}

impl<'a> MulAssign<&'a Integer> for Integer {
    fn mul_assign(&mut self, rhs: &'a Integer) {
        match self {
            Integer::Natural(n1) => match rhs {
                Integer::Natural(n2) => {
                    if let Some(nn) = n1.checked_mul(*n2) {
                        *n1 = nn;
                    } else {
                        let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                        r1.mul_assign(ArbitraryPrecisionInteger::from(*n2));
                        *self = Integer::Large(r1)
                    }
                }
                Integer::Large(r2) => {
                    let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                    r1.mul_assign(r2);
                    *self = Integer::Large(r1)
                }
            },
            Integer::Large(r1) => match rhs {
                Integer::Natural(n2) => r1.mul_assign(ArbitraryPrecisionInteger::from(*n2)),
                Integer::Large(r2) => r1.mul_assign(r2),
            },
        };
    }
}

impl<'a> DivAssign<&'a Integer> for Integer {
    fn div_assign(&mut self, rhs: &'a Integer) {
        match self {
            Integer::Natural(n1) => match rhs {
                Integer::Natural(n2) => {
                    if let Some(nn) = n1.checked_div(*n2) {
                        *n1 = nn;
                    } else {
                        let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                        r1.div_assign(ArbitraryPrecisionInteger::from(*n2));
                        *self = Integer::Large(r1)
                    }
                }
                Integer::Large(r2) => {
                    let mut r1 = ArbitraryPrecisionInteger::from(*n1);
                    r1.div_assign(r2);
                    *self = Integer::Large(r1)
                }
            },
            Integer::Large(r1) => match rhs {
                Integer::Natural(n2) => r1.div_assign(ArbitraryPrecisionInteger::from(*n2)),
                Integer::Large(r2) => r1.div_assign(r2),
            },
        };
    }
}

impl<'a> MulAssign<i64> for Integer {
    fn mul_assign(&mut self, rhs: i64) {
        *self = (&*self) * rhs;
    }
}

impl<'a> DivAssign<i64> for Integer {
    fn div_assign(&mut self, rhs: i64) {
        *self = (&*self) / rhs;
    }
}

impl<'a> Neg for &'a Integer {
    type Output = Integer;

    fn neg(self) -> Self::Output {
        match self {
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
}
