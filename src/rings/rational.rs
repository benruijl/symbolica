use std::{fmt::Display, ops::Neg};

use rug::{ops::Pow, Integer, Rational as ArbitraryPrecisionRational};

use crate::utils;

use super::{EuclideanDomain, Field, Ring};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RationalField;

impl RationalField {
    pub fn new() -> RationalField {
        RationalField
    }
}

// FIXME: PartialEq can only work when Large simplifies to Natural whenever possible
#[derive(Clone, PartialEq, Debug)]
pub enum Rational {
    Natural(i64, i64),
    Large(ArbitraryPrecisionRational),
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Rational {
        Rational::Natural(num, den)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rational::Natural(n, d) => {
                if *d == 1 {
                    f.write_fmt(format_args!("{}", n))
                } else {
                    f.write_fmt(format_args!("{}/{}", n, d))
                }
            }
            Rational::Large(r) => r.fmt(f),
        }
    }
}

impl Display for RationalField {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Ring for RationalField {
    type Element = Rational;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                if let Some(lcm) = d2.checked_mul(d1 / utils::gcd_signed(*d1, *d2)) {
                    if let Some(num2) = n2.checked_mul(lcm / d2) {
                        if let Some(num1) = n1.checked_mul(lcm / d1) {
                            if let Some(num) = num1.checked_add(num2) {
                                let g = utils::gcd_signed(num, lcm);
                                return Rational::Natural(num / g, lcm / g);
                            }
                        }
                    }
                }
                Rational::Large(
                    ArbitraryPrecisionRational::from((*n1, *d1))
                        + ArbitraryPrecisionRational::from((*n2, *d2)),
                )
            }
            // TODO: check downcast
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::Large(r1 + r2)
            }
            (Rational::Large(r1), Rational::Large(r2)) => Rational::Large((r1 + r2).into()),
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                let gcd1 = utils::gcd_signed(*n1 as i64, *d2 as i64);
                let gcd2 = utils::gcd_signed(*d1 as i64, *n2 as i64);

                match (n2 / gcd2).checked_mul(n1 / gcd1) {
                    Some(nn) => match (d1 / gcd2).checked_mul(d2 / gcd1) {
                        Some(nd) => Rational::Natural(nn, nd),
                        None => Rational::Large(ArbitraryPrecisionRational::from((
                            nn,
                            Integer::from(d1 / gcd2) * Integer::from(d2 / gcd1),
                        ))),
                    },
                    None => Rational::Large(ArbitraryPrecisionRational::from((
                        Integer::from(n1 / gcd1) * Integer::from(n2 / gcd2),
                        Integer::from(d1 / gcd2) * Integer::from(d2 / gcd1),
                    ))),
                }
            }
            // FIXME: downcast
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::Large(r1 * r2)
            }
            (Rational::Large(r1), Rational::Large(r2)) => Rational::Large((r1 * r2).into()),
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        // TODO: optimize
        *a = self.add(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        match a {
            Rational::Natural(n, d) => {
                if let Some(neg) = n.checked_neg() {
                    Rational::Natural(neg, *d)
                } else {
                    Rational::Large(ArbitraryPrecisionRational::from((*n, *d)).neg())
                }
            }
            Rational::Large(r) => Rational::Large(r.neg().into()),
        }
    }

    fn zero() -> Self::Element {
        Rational::Natural(0, 1)
    }

    fn one(&self) -> Self::Element {
        Rational::Natural(1, 1)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        match b {
            Rational::Natural(n1, d1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    if let Some(pd) = d1.checked_pow(e) {
                        return Rational::Natural(pn, pd);
                    }
                }

                Rational::Large(ArbitraryPrecisionRational::from((*n1, *d1)).pow(e))
            }
            Rational::Large(r) => Rational::Large(r.pow(e).into()),
        }
    }

    fn is_zero(a: &Self::Element) -> bool {
        match a {
            Rational::Natural(r, _) => *r == 0,
            // TODO: not needed anymore when automatically simplifying
            Rational::Large(r) => r.numer().to_u8().map(|x| x == 0).unwrap_or(false),
        }
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        match a {
            Rational::Natural(r, d) => *r == 1 && *d == 1,
            Rational::Large(r) => {
                r.numer().to_u8().map(|x| x == 1).unwrap_or(false)
                    && r.denom().to_u8().map(|x| x == 1).unwrap_or(false)
            }
        }
    }
}

impl EuclideanDomain for RationalField {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Rational::Natural(0, 0)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), Rational::Natural(0, 0))
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                let gcd1 = utils::gcd_signed(*n1 as i64, *d2 as i64);
                let gcd2 = utils::gcd_signed(*d1 as i64, *n2 as i64);

                if let Some(lcm) = d2.checked_mul(d1 / gcd2) {
                    Rational::Natural(gcd1, lcm)
                } else {
                    Rational::Large(ArbitraryPrecisionRational::from((
                        Integer::from(gcd1),
                        Integer::from(*d2) * Integer::from(d1 / gcd2),
                    )))
                }
            }
            // FIXME: downcast
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::Large(ArbitraryPrecisionRational::from((
                    r1.numer().clone().gcd(r2.numer()),
                    r1.denom().clone().lcm(r2.denom()),
                )))
            }
            (Rational::Large(r1), Rational::Large(r2)) => {
                Rational::Large(ArbitraryPrecisionRational::from((
                    r1.numer().clone().gcd(r2.numer()),
                    r1.denom().clone().lcm(r2.denom()),
                )))
            }
        }
    }
}

impl Field for RationalField {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        match a {
            Rational::Natural(n, d) => {
                if *n < 0 {
                    if let Some(neg) = d.checked_neg() {
                        Rational::Natural(neg, -n)
                    } else {
                        Rational::Large(ArbitraryPrecisionRational::from((*n, *d)).clone().recip())
                    }
                } else {
                    Rational::Natural(*d, *n)
                }
            }
            Rational::Large(r) => Rational::Large(r.clone().recip()),
        }
    }
}
