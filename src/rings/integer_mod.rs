use crate::{rings::EuclideanDomain, state::State};
use std::{
    fmt::{Display, Error, Formatter},
    ops::Neg,
};

use super::{
    integer::{Integer, IntegerRing},
    Field, Ring,
};

/// The modular ring `Z / mZ`, where `m` can be any integer.
/// Use this class only if `m` is not a prime number or
/// if it is larger than 2^64. In all other cases, use
/// [`FiniteField`](crate::rings::finite_field::FiniteField<UField>) instead.
///
/// This ring also implements `Field`. The user *must* make sure
/// to only use field features such as inverses when the input is coprime.
#[derive(Clone, PartialEq, Debug)]
pub struct IntegerMod(Integer);

impl IntegerMod {
    pub fn new(m: Integer) -> IntegerMod {
        IntegerMod(m)
    }

    pub fn to_element(&self, a: &Integer) -> Integer {
        a.symmetric_mod(&self.0)
    }
}

impl Display for IntegerMod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " % {}", self.0)
    }
}

impl Ring for IntegerMod {
    type Element = Integer;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a + b).symmetric_mod(&self.0)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a - b).symmetric_mod(&self.0)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a * b).symmetric_mod(&self.0)
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
        a.neg().symmetric_mod(&self.0)
    }

    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        IntegerRing::new().nth(n).symmetric_mod(&self.0)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        // FIXME: intermediate mods
        b.pow(e).symmetric_mod(&self.0)
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        IntegerRing::new().sample(rng, range).symmetric_mod(&self.0)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        _state: Option<&State>,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl EuclideanDomain for IntegerMod {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), Integer::zero())
    }

    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::one()
    }
}

impl Field for IntegerMod {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Compute the inverse when `a` and the modulus are coprime,
    /// otherwise panic.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(!a.is_zero(), "0 is not invertible");

        let mut u1 = Integer::one();
        let mut u3 = a.clone();
        let mut v1 = Integer::zero();
        let mut v3 = self.0.clone();
        let mut even_iter: bool = true;

        while !v3.is_zero() {
            let (q, t3) = IntegerRing::new().quot_rem(&u3, &v3);
            let t1 = &u1 + &(&q * &v1);
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        assert!(u3.is_one(), "{} is not invertible mod {}", a, self.0);
        if even_iter {
            u1
        } else {
            &self.0 - &u1
        }
    }
}
