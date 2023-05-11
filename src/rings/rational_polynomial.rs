use std::{
    fmt::{Display, Error, Formatter},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::{
    poly::{gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent, INLINED_EXPONENTS},
    representations::Identifier,
};

use super::{integer::IntegerRing, rational::RationalField, EuclideanDomain, Field, Ring, finite_field::FiniteField};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RationalPolynomialField<R: Ring, E: Exponent> {
    ring: R,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring, E: Exponent> RationalPolynomialField<R, E> {
    pub fn new(coeff_ring: R) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: coeff_ring,
            _phantom_exp: PhantomData,
        }
    }
}

pub trait FromNumeratorAndDenominator<R: Ring, OR: Ring, E: Exponent> {
    fn from_num_den(
        num: MultivariatePolynomial<R, E>,
        den: MultivariatePolynomial<R, E>,
        field: OR,
    ) -> RationalPolynomial<OR, E>;
}

#[derive(Clone, PartialEq, Debug)]
pub struct RationalPolynomial<R: Ring, E: Exponent> {
    pub numerator: MultivariatePolynomial<R, E>,
    pub denominator: MultivariatePolynomial<R, E>,
}

impl<R: Ring, E: Exponent> RationalPolynomial<R, E> {
    pub fn new(field: R, var_map: Option<&[Identifier]>) -> RationalPolynomial<R, E> {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new(
                var_map.map(|x| x.len()).unwrap_or(0),
                field,
                None,
                var_map.map(|x| x.iter().cloned().collect()),
            ),
            denominator: {
                let d = MultivariatePolynomial::new(
                    var_map.map(|x| x.len()).unwrap_or(0),
                    field,
                    None,
                    var_map.map(|x| x.iter().cloned().collect()),
                );
                d.add_monomial(field.one())
            },
        }
    }

    pub fn get_var_map(
        &self,
    ) -> &Option<smallvec::SmallVec<[crate::representations::Identifier; INLINED_EXPONENTS]>> {
        &self.numerator.var_map
    }

    pub fn unify_var_map(&mut self, other: &mut Self) {
        assert!(
            self.numerator.var_map == self.denominator.var_map
                && other.numerator.var_map == other.denominator.var_map
        );

        self.numerator.unify_var_map(&mut other.numerator);
        self.denominator.unify_var_map(&mut other.denominator);
    }
}

impl<E: Exponent> FromNumeratorAndDenominator<RationalField, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        num: MultivariatePolynomial<RationalField, E>,
        den: MultivariatePolynomial<RationalField, E>,
        field: IntegerRing,
    ) -> RationalPolynomial<IntegerRing, E> {
        let content = num.field.gcd(&num.content(), &den.content());

        let mut num_int = MultivariatePolynomial::new(
            num.nvars,
            IntegerRing::new(),
            Some(num.nterms),
            num.var_map.clone(),
        );

        for t in num.into_iter() {
            let coeff = num.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            num_int.append_monomial(coeff.numerator(), t.exponents);
        }

        let mut den_int = MultivariatePolynomial::new(
            den.nvars,
            IntegerRing::new(),
            Some(den.nterms),
            den.var_map.clone(),
        );

        for t in den.into_iter() {
            let coeff = den.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            den_int.append_monomial(coeff.numerator(), t.exponents);
        }

        <RationalPolynomial<IntegerRing, E> as FromNumeratorAndDenominator<
            IntegerRing,
            IntegerRing,
            E,
        >>::from_num_den(num_int, den_int, field)
    }
}

impl<E: Exponent> FromNumeratorAndDenominator<IntegerRing, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut den: MultivariatePolynomial<IntegerRing, E>,
        _field: IntegerRing,
    ) -> Self {
        num.unify_var_map(&mut den);

        let gcd = MultivariatePolynomial::gcd(&num, &den);

        RationalPolynomial {
            numerator: num / &gcd,
            denominator: den / &gcd,
        }
    }
}

impl<E: Exponent> FromNumeratorAndDenominator<FiniteField<u32>, FiniteField<u32>, E>
    for RationalPolynomial<FiniteField<u32>, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FiniteField<u32>, E>,
        mut den: MultivariatePolynomial<FiniteField<u32>, E>,
        _field: FiniteField<u32>,
    ) -> Self {
        num.unify_var_map(&mut den);

        let gcd = MultivariatePolynomial::gcd(&num, &den);

        RationalPolynomial {
            numerator: num / &gcd,
            denominator: den / &gcd,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> RationalPolynomial<R, E> {
    pub fn inv(&self) -> Self {
        // TODO: normalize the leading monomial
        RationalPolynomial {
            numerator: self.denominator.clone(),
            denominator: self.numerator.clone(),
        }
    }

    pub fn pow(&self, e: u64) -> Self {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        // TODO: do binary exponentation
        let mut poly = RationalPolynomial {
            numerator: MultivariatePolynomial::new_from(&self.numerator, None),
            denominator: MultivariatePolynomial::new_from(&self.denominator, None),
        };
        poly.numerator = poly.numerator.add_monomial(self.numerator.field.one());
        poly.denominator = poly.denominator.add_monomial(self.numerator.field.one());

        for _ in 0..e {
            poly = &poly * self;
        }
        poly
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let gcd_num = MultivariatePolynomial::gcd(&self.numerator, &other.numerator);
        let gcd_den = MultivariatePolynomial::gcd(&self.denominator, &other.denominator);

        RationalPolynomial {
            numerator: gcd_num,
            denominator: (&other.denominator / &gcd_den) * &self.denominator,
        }
    }
}

impl<R: Ring, E: Exponent> Display for RationalPolynomial<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominator.is_one() {
            self.numerator.fmt(f)
        } else {
            f.write_fmt(format_args!("({})/({})", self.numerator, self.denominator))
        }
    }
}

impl<R: Ring, E: Exponent> Display for RationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(()) // FIXME
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Ring for RationalPolynomialField<R, E> {
    type Element = RationalPolynomial<R, E>;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
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
        a.clone().neg()
    }

    fn zero(&self) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new(0, self.ring, None, None),
            denominator: MultivariatePolynomial::from_constant(self.ring.one(), 0, self.ring),
        }
    }

    fn one(&self) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::from_constant(self.ring.one(), 0, self.ring),
            denominator: MultivariatePolynomial::from_constant(self.ring.one(), 0, self.ring),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        // TODO: do binary exponentation
        let mut poly = RationalPolynomial {
            numerator: MultivariatePolynomial::new_from(&b.numerator, None),
            denominator: MultivariatePolynomial::new_from(&b.denominator, None),
        };
        poly.numerator = poly.numerator.add_monomial(self.ring.one());
        poly.denominator = poly.denominator.add_monomial(self.ring.one());

        for _ in 0..e {
            poly = self.mul(&poly, &b);
        }
        poly
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.numerator.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.numerator.is_one() && a.denominator.is_one()
    }

    fn sample(&self, _rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        todo!("Sampling a polynomial is not possible yet")
    }

    fn fmt_display(&self, element: &Self::Element, f: &mut Formatter<'_>) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> EuclideanDomain
    for RationalPolynomialField<R, E>
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new_from(&a.numerator, None),
            denominator: MultivariatePolynomial::from_constant(
                a.numerator.field.one(),
                a.numerator.nvars,
                a.numerator.field,
            ),
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Field for RationalPolynomialField<R, E> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a / b
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        a.inv()
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E> + PolynomialGCD<E>, E: Exponent>
    Add<&'a RationalPolynomial<R, E>> for &'b RationalPolynomial<R, E>
{
    type Output = RationalPolynomial<R, E>;

    fn add(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        let denom_gcd = MultivariatePolynomial::gcd(&self.denominator, &other.denominator);
        let a_denom_red = &self.denominator / &denom_gcd;
        let lcm = &a_denom_red * &other.denominator;
        let num1 = (&other.denominator / &denom_gcd) * &self.numerator;
        let num2 = a_denom_red * &other.numerator;
        let num = num1 + num2;
        let g = MultivariatePolynomial::gcd(&num, &lcm);
        RationalPolynomial {
            numerator: num / &g,
            denominator: lcm / &g,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Sub for RationalPolynomial<R, E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Sub<&'a RationalPolynomial<R, E>>
    for &'b RationalPolynomial<R, E>
{
    type Output = RationalPolynomial<R, E>;

    fn sub(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        (self.clone()).sub(other.clone())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Neg for RationalPolynomial<R, E> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        RationalPolynomial {
            numerator: self.numerator.neg(),
            denominator: self.denominator,
        }
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Mul<&'a RationalPolynomial<R, E>>
    for &'b RationalPolynomial<R, E>
{
    type Output = RationalPolynomial<R, E>;

    fn mul(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        let gcd1 = MultivariatePolynomial::gcd(&self.numerator, &other.denominator);
        let gcd2 = MultivariatePolynomial::gcd(&self.denominator, &other.numerator);

        RationalPolynomial {
            numerator: (&self.numerator / &gcd1) * &(&other.numerator / &gcd2),
            denominator: (&self.denominator / &gcd2) * &(&other.denominator / &gcd1),
        }
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Div<&'a RationalPolynomial<R, E>>
    for &'b RationalPolynomial<R, E>
{
    type Output = RationalPolynomial<R, E>;

    fn div(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        // TODO: optimize
        self * &other.inv()
    }
}
