use std::{
    fmt::{Display, Error, Formatter},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use rand::Rng;

use crate::{
    poly::{polynomial::MultivariatePolynomial, Exponent, INLINED_EXPONENTS},
    representations::Identifier,
};

use super::{
    integer::{Integer, IntegerRing},
    rational::RationalField,
    EuclideanDomain, Field, Ring,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RationalPolynomialField<E: Exponent> {
    _phantom_data: PhantomData<E>,
}

impl<E: Exponent> RationalPolynomialField<E> {
    pub fn new() -> RationalPolynomialField<E> {
        RationalPolynomialField {
            _phantom_data: PhantomData,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct RationalPolynomial<E: Exponent> {
    pub numerator: MultivariatePolynomial<IntegerRing, E>,
    pub denominator: MultivariatePolynomial<IntegerRing, E>,
}

impl<E: Exponent> RationalPolynomial<E> {
    pub fn new(var_map: Option<&[Identifier]>) -> RationalPolynomial<E> {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new(
                var_map.map(|x| x.len()).unwrap_or(0),
                IntegerRing::new(),
                None,
                var_map.map(|x| x.iter().cloned().collect()),
            ),
            denominator: {
                let d = MultivariatePolynomial::new(
                    var_map.map(|x| x.len()).unwrap_or(0),
                    IntegerRing::new(),
                    None,
                    var_map.map(|x| x.iter().cloned().collect()),
                );
                d.add_monomial(Integer::Natural(1))
            },
        }
    }

    pub fn from_rat_num_and_den(
        num: &MultivariatePolynomial<RationalField, E>,
        den: &MultivariatePolynomial<RationalField, E>,
    ) -> RationalPolynomial<E> {
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

        Self::from_num_and_den(num_int, den_int)
    }

    pub fn from_num_and_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut den: MultivariatePolynomial<IntegerRing, E>,
    ) -> RationalPolynomial<E> {
        num.unify_var_map(&mut den);

        let gcd = MultivariatePolynomial::gcd(&num, &den);

        RationalPolynomial {
            numerator: num / &gcd,
            denominator: den / &gcd,
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

impl<E: Exponent> Display for RationalPolynomial<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominator.is_one() {
            self.numerator.fmt(f)
        } else {
            f.write_fmt(format_args!("({})/({})", self.numerator, self.denominator))
        }
    }
}

impl<E: Exponent> Display for RationalPolynomialField<E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<E: Exponent> Ring for RationalPolynomialField<E> {
    type Element = RationalPolynomial<E>;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let denom_gcd = MultivariatePolynomial::gcd(&a.denominator, &b.denominator);
        let a_denom_red = &a.denominator / &denom_gcd;
        let lcm = &a_denom_red * &b.denominator;
        let num1 = (&b.denominator / &denom_gcd) * &a.numerator;
        let num2 = a_denom_red * &b.numerator;
        let num = num1 + num2;
        let g = MultivariatePolynomial::gcd(&num, &lcm);
        RationalPolynomial {
            numerator: num / &g,
            denominator: lcm / &g,
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let gcd1 = MultivariatePolynomial::gcd(&a.numerator, &b.denominator);
        let gcd2 = MultivariatePolynomial::gcd(&a.denominator, &b.numerator);

        RationalPolynomial {
            numerator: (&a.numerator / &gcd1) * &(&b.numerator / &gcd2),
            denominator: (&a.denominator / &gcd2) * &(&b.denominator / &gcd1),
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
        RationalPolynomial {
            numerator: a.numerator.clone().neg(),
            denominator: a.denominator.clone(),
        }
    }

    fn zero() -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new(0, IntegerRing::new(), None, None),
            denominator: MultivariatePolynomial::from_constant(
                Integer::Natural(1),
                0,
                IntegerRing::new(),
            ),
        }
    }

    fn one(&self) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::from_constant(
                Integer::Natural(1),
                0,
                IntegerRing::new(),
            ),
            denominator: MultivariatePolynomial::from_constant(
                Integer::Natural(1),
                0,
                IntegerRing::new(),
            ),
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
        poly.numerator = poly.numerator.add_monomial(Integer::Natural(1));
        poly.denominator = poly.denominator.add_monomial(Integer::Natural(1));

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

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = Integer::Natural(rng.gen_range(range.0..range.1));
        RationalPolynomial {
            numerator: MultivariatePolynomial::from_constant(r, 0, IntegerRing::new()),
            denominator: MultivariatePolynomial::from_constant(
                Integer::Natural(1),
                0,
                IntegerRing::new(),
            ),
        }
    }

    fn fmt_display(&self, element: &Self::Element, f: &mut Formatter<'_>) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl<E: Exponent> EuclideanDomain for RationalPolynomialField<E> {
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new_from(&a.numerator, None),
            denominator: MultivariatePolynomial::from_constant(
                Integer::Natural(1),
                a.numerator.nvars,
                IntegerRing::new(),
            ),
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), RationalPolynomialField::zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let gcd_num = MultivariatePolynomial::gcd(&a.numerator, &b.numerator);
        let gcd_den = MultivariatePolynomial::gcd(&a.denominator, &b.denominator);

        RationalPolynomial {
            numerator: gcd_num,
            denominator: (&b.denominator / &gcd_den) * &a.denominator,
        }
    }
}

impl<E: Exponent> Field for RationalPolynomialField<E> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        // FIXME: what about a sign? -1/x may become x/-1
        // how to quickly check this?
        RationalPolynomial {
            numerator: a.denominator.clone(),
            denominator: a.numerator.clone(),
        }
    }
}

impl<'a, 'b, E: Exponent> Add<&'a RationalPolynomial<E>> for &'b RationalPolynomial<E> {
    type Output = RationalPolynomial<E>;

    fn add(self, other: &'a RationalPolynomial<E>) -> Self::Output {
        RationalPolynomialField::new().add(self, other)
    }
}

impl<E: Exponent> Sub for RationalPolynomial<E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, 'b, E: Exponent> Sub<&'a RationalPolynomial<E>> for &'b RationalPolynomial<E> {
    type Output = RationalPolynomial<E>;

    fn sub(self, other: &'a RationalPolynomial<E>) -> Self::Output {
        (self.clone()).sub(other.clone())
    }
}

impl<E: Exponent> Neg for RationalPolynomial<E> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        RationalPolynomialField::new().neg(&self)
    }
}

impl<'a, 'b, E: Exponent> Mul<&'a RationalPolynomial<E>> for &'b RationalPolynomial<E> {
    type Output = RationalPolynomial<E>;

    fn mul(self, other: &'a RationalPolynomial<E>) -> Self::Output {
        RationalPolynomialField::new().mul(self, other)
    }
}

impl<'a, 'b, E: Exponent> Div<&'a RationalPolynomial<E>> for &'b RationalPolynomial<E> {
    type Output = RationalPolynomial<E>;

    fn div(self, other: &'a RationalPolynomial<E>) -> Self::Output {
        RationalPolynomialField::new().div(self, other)
    }
}
