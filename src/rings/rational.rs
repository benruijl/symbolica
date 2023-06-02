use std::{
    fmt::{Display, Error, Formatter, Write},
    ops::{Add, Div, Mul, Neg, Sub},
};

use rand::Rng;
use rug::{ops::Pow, Integer as ArbitraryPrecisionInteger, Rational as ArbitraryPrecisionRational};

use crate::utils;

use super::{
    finite_field::{FiniteField, FiniteFieldCore, ToFiniteField},
    integer::Integer,
    EuclideanDomain, Field, Ring,
};

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

impl ToFiniteField<u32> for Rational {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        match self {
            &Rational::Natural(n, d) => {
                let mut ff = field.to_element(n.rem_euclid(field.get_prime() as i64) as u32);

                if d != 1 {
                    let df = field.to_element(d.rem_euclid(field.get_prime() as i64) as u32);
                    field.div_assign(&mut ff, &df);
                }

                ff
            }
            Rational::Large(r) => field.div(
                &field.to_element(r.numer().mod_u(field.get_prime())),
                &field.to_element(r.denom().mod_u(field.get_prime())),
            ),
        }
    }
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Rational {
        Rational::Natural(num, den)
    }

    pub fn from_finite_field_u32(
        field: FiniteField<u32>,
        element: &<FiniteField<u32> as Ring>::Element,
    ) -> Rational {
        Rational::Natural(field.from_element(*element) as i64, 1)
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Rational::Natural(n, _) => *n < 0,
            Rational::Large(r) => ArbitraryPrecisionInteger::from(r.numer().signum_ref()) == -1,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Rational::Natural(_, d) => *d == 1,
            Rational::Large(r) => r.is_integer(),
        }
    }

    pub fn numerator(&self) -> Integer {
        match self {
            Rational::Natural(n, _) => Integer::Natural(*n),
            Rational::Large(r) => Integer::Large(r.numer().clone()),
        }
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rational::Natural(n, d) => {
                if *d == 1 {
                    n.fmt(f)
                } else {
                    n.fmt(f)?;
                    f.write_char('/')?;
                    write!(f, "{}", d)
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
                            ArbitraryPrecisionInteger::from(d1 / gcd2)
                                * ArbitraryPrecisionInteger::from(d2 / gcd1),
                        ))),
                    },
                    None => Rational::Large(ArbitraryPrecisionRational::from((
                        ArbitraryPrecisionInteger::from(n1 / gcd1)
                            * ArbitraryPrecisionInteger::from(n2 / gcd2),
                        ArbitraryPrecisionInteger::from(d1 / gcd2)
                            * ArbitraryPrecisionInteger::from(d2 / gcd1),
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

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &(b * c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &(b * c));
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

    fn zero(&self) -> Self::Element {
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

    fn get_unit(&self, a: &Self::Element) -> Self::Element {
        a.clone()
    }

    fn get_inv_unit(&self, a: &Self::Element) -> Self::Element {
        self.inv(a)
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Rational::Natural(r, 1)
    }

    fn fmt_display(&self, element: &Self::Element, f: &mut Formatter<'_>) -> Result<(), Error> {
        element.fmt(f)
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
                let gcd_num = utils::gcd_signed(*n1 as i64, *n2 as i64);
                let gcd_den = utils::gcd_signed(*d1 as i64, *d2 as i64);

                if let Some(lcm) = d2.checked_mul(d1 / gcd_den) {
                    Rational::Natural(gcd_num, lcm)
                } else {
                    Rational::Large(ArbitraryPrecisionRational::from((
                        ArbitraryPrecisionInteger::from(gcd_num),
                        ArbitraryPrecisionInteger::from(*d2)
                            * ArbitraryPrecisionInteger::from(d1 / gcd_den),
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

impl<'a, 'b> Add<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        RationalField::new().add(self, other)
    }
}

impl Sub for Rational {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, 'b> Sub<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        (self.clone()).sub(other.clone())
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        RationalField::new().neg(&self)
    }
}

impl<'a, 'b> Mul<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        RationalField::new().mul(self, other)
    }
}

impl<'a, 'b> Div<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        RationalField::new().div(self, other)
    }
}
