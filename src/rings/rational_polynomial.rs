use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{Display, Error, Formatter, Write},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

use ahash::HashMap;

use crate::{
    poly::{gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent, Variable},
    printer::{PrintOptions, RationalPolynomialPrinter},
    state::State,
};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::IntegerRing,
    rational::RationalField,
    EuclideanDomain, Field, Ring,
};

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
        field: &OR,
        do_gcd: bool,
    ) -> RationalPolynomial<OR, E>;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RationalPolynomial<R: Ring, E: Exponent> {
    pub numerator: MultivariatePolynomial<R, E>,
    pub denominator: MultivariatePolynomial<R, E>,
}

impl<R: Ring, E: Exponent> PartialOrd for RationalPolynomial<R, E> {
    /// An ordering of rational polynomials that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.numerator
                .exponents
                .cmp(&other.numerator.exponents)
                .then_with(|| self.denominator.exponents.cmp(&other.denominator.exponents))
                .then_with(|| {
                    self.numerator
                        .coefficients
                        .partial_cmp(&other.numerator.coefficients)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| {
                    self.denominator
                        .coefficients
                        .partial_cmp(&other.denominator.coefficients)
                        .unwrap_or(Ordering::Equal)
                }),
        )
    }
}

impl<R: Ring, E: Exponent> RationalPolynomial<R, E> {
    pub fn new(field: &R, var_map: Option<&[Variable]>) -> RationalPolynomial<R, E> {
        let num = MultivariatePolynomial::new(
            var_map.map(|x| x.len()).unwrap_or(0),
            field,
            None,
            var_map,
        );
        let den = num.new_from_constant(field.one());

        RationalPolynomial {
            numerator: num,
            denominator: den,
        }
    }

    pub fn get_var_map(&self) -> Option<&[Variable]> {
        self.numerator.var_map.as_ref().map(|x| x.as_slice())
    }

    pub fn unify_var_map(&mut self, other: &mut Self) {
        assert_eq!(self.numerator.var_map, self.denominator.var_map);
        assert_eq!(other.numerator.var_map, other.denominator.var_map);

        self.numerator.unify_var_map(&mut other.numerator);
        self.denominator.unify_var_map(&mut other.denominator);
    }

    /// Constuct a pretty-printer for the rational polynomial.
    pub fn printer<'a, 'b>(&'a self, state: &'b State) -> RationalPolynomialPrinter<'a, 'b, R, E> {
        RationalPolynomialPrinter::new(self, state)
    }

    /// Convert the coefficient from the current field to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> RationalPolynomial<FiniteField<UField>, E>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Ring>::Element: Copy,
    {
        // check the gcd, since the rational polynomial may simplify
        RationalPolynomial::from_num_den(
            self.numerator.to_finite_field(field),
            self.denominator.to_finite_field(field),
            field,
            true,
        )
    }
}

impl<E: Exponent> FromNumeratorAndDenominator<RationalField, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        num: MultivariatePolynomial<RationalField, E>,
        den: MultivariatePolynomial<RationalField, E>,
        field: &IntegerRing,
        do_gcd: bool,
    ) -> RationalPolynomial<IntegerRing, E> {
        let content = num.field.gcd(&num.content(), &den.content());

        let mut num_int = MultivariatePolynomial::new(
            num.nvars,
            &IntegerRing::new(),
            None,
            num.var_map.as_ref().map(|x| x.as_slice()),
        );
        num_int.nterms = num.nterms;
        num_int.exponents = num.exponents;

        let mut den_int = MultivariatePolynomial::new(
            den.nvars,
            &IntegerRing::new(),
            Some(den.nterms),
            den.var_map.as_ref().map(|x| x.as_slice()),
        );
        den_int.nterms = den.nterms;
        den_int.exponents = den.exponents;

        if num.field.is_one(&content) {
            num_int.coefficients = num
                .coefficients
                .into_iter()
                .map(|c| c.numerator())
                .collect();
            den_int.coefficients = den
                .coefficients
                .into_iter()
                .map(|c| c.numerator())
                .collect();
        } else {
            num_int.coefficients = num
                .coefficients
                .into_iter()
                .map(|c| num.field.div(&c, &content).numerator())
                .collect();
            den_int.coefficients = den
                .coefficients
                .into_iter()
                .map(|c| den.field.div(&c, &content).numerator())
                .collect();
        }

        <RationalPolynomial<IntegerRing, E> as FromNumeratorAndDenominator<
            IntegerRing,
            IntegerRing,
            E,
        >>::from_num_den(num_int, den_int, field, do_gcd)
    }
}

impl<E: Exponent> FromNumeratorAndDenominator<IntegerRing, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut den: MultivariatePolynomial<IntegerRing, E>,
        _field: &IntegerRing,
        do_gcd: bool,
    ) -> Self {
        num.unify_var_map(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = MultivariatePolynomial::gcd(&num, &den);

                if !gcd.is_one() {
                    num = num / &gcd;
                    den = den / &gcd;
                }
            }

            // normalize denominator to have positive leading coefficient
            if den.lcoeff().is_negative() {
                num = -num;
                den = -den;
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<UField: FiniteFieldWorkspace, E: Exponent>
    FromNumeratorAndDenominator<FiniteField<UField>, FiniteField<UField>, E>
    for RationalPolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FiniteField<UField>, E>,
        mut den: MultivariatePolynomial<FiniteField<UField>, E>,
        field: &FiniteField<UField>,
        do_gcd: bool,
    ) -> Self {
        num.unify_var_map(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = MultivariatePolynomial::gcd(&num, &den);

                if !gcd.is_one() {
                    num = num / &gcd;
                    den = den / &gcd;
                }
            }

            // normalize denominator to have leading coefficient of one
            if !field.is_one(&den.lcoeff()) {
                let c = den.lcoeff();
                num = num.div_coeff(&c);
                den = den.div_coeff(&c);
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> RationalPolynomial<R, E>
where
    Self: FromNumeratorAndDenominator<R, R, E>,
{
    #[inline]
    pub fn inv(self) -> Self {
        if self.numerator.is_zero() {
            panic!("Cannot invert 0");
        }

        let field = self.numerator.field.clone();
        Self::from_num_den(self.denominator, self.numerator, &field, false)
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

    /// Convert the rational polynomial to a polynomial in the specified
    /// variables, with rational polynomial coefficients.
    /// If the specified variables appear in the denominator, an `Err` is returned.
    ///
    /// If `ignore_denominator` is `True`, the denominator is considered to be 1,
    /// after the variable check.
    pub fn to_polynomial(
        &self,
        variables: &[Variable],
        ignore_denominator: bool,
    ) -> Result<MultivariatePolynomial<RationalPolynomialField<R, E>, E>, &'static str> {
        let var_map = self.get_var_map().ok_or("Variable map missing")?;
        let index_mask: Vec<_> = var_map
            .iter()
            .map(|v| variables.iter().position(|vv| vv == v))
            .collect();

        for e in self.denominator.exponents.chunks(self.denominator.nvars) {
            for (c, p) in index_mask.iter().zip(e) {
                if c.is_some() && *p > E::zero() {
                    return Err("Not a polynomial");
                }
            }
        }

        let mut hm: HashMap<Vec<E>, RationalPolynomial<R, E>> = HashMap::default();

        let mut e_list = vec![E::zero(); variables.len()];

        let mut e_list_coeff = vec![E::zero(); self.numerator.nvars];
        for e in self.numerator.into_iter() {
            for ee in &mut e_list {
                *ee = E::zero();
            }

            for ((elc, ee), m) in e_list_coeff.iter_mut().zip(e.exponents).zip(&index_mask) {
                if let Some(p) = m {
                    e_list[*p] = *ee;
                    *elc = E::zero();
                } else {
                    *elc = *ee;
                }
            }

            // TODO: remove variables from the var_map of the coefficient
            if let Some(r) = hm.get_mut(e_list.as_slice()) {
                r.numerator
                    .append_monomial(e.coefficient.clone(), &e_list_coeff);
            } else {
                let mut r = RationalPolynomial::new(&self.numerator.field.clone(), Some(var_map));
                r.numerator
                    .append_monomial(e.coefficient.clone(), &e_list_coeff);
                hm.insert(e_list.clone(), r);
            }
        }

        let mut poly = MultivariatePolynomial::new(
            variables.len(),
            &RationalPolynomialField::new(self.numerator.field.clone()),
            Some(hm.len()),
            Some(variables),
        );

        if !ignore_denominator {
            let denom = RationalPolynomial::from_num_den(
                MultivariatePolynomial::new_from_constant(
                    &self.denominator,
                    self.denominator.field.one(),
                ),
                self.denominator.clone(),
                &self.denominator.field,
                false,
            );

            for coeff in hm.values_mut() {
                // divide by the denominator
                *coeff = coeff.mul(&denom);
            }
        }

        for (exp, coeff) in hm {
            poly.append_monomial(coeff, &exp);
        }

        Ok(poly)
    }
}

impl<R: Ring, E: Exponent> Display for RationalPolynomial<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominator.is_one() {
            self.numerator.fmt(f)
        } else {
            if f.sign_plus() {
                f.write_char('+')?;
            }

            f.write_fmt(format_args!("({})/({})", self.numerator, self.denominator))
        }
    }
}

impl<R: Ring, E: Exponent> Display for RationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(()) // FIXME
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Ring for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
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

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &(b * c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &(b * c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }

    fn zero(&self) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new(0, &self.ring, None, None),
            denominator: MultivariatePolynomial::one(&self.ring),
        }
    }

    fn one(&self) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::one(&self.ring),
            denominator: MultivariatePolynomial::one(&self.ring),
        }
    }

    fn nth(&self, n: u64) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::one(&self.ring).mul_coeff(self.ring.nth(n)),
            denominator: MultivariatePolynomial::one(&self.ring),
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
            poly = self.mul(&poly, b);
        }
        poly
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.numerator.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.numerator.is_one() && a.denominator.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn sample(&self, _rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        todo!("Sampling a polynomial is not possible yet")
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        state: Option<&State>,
        in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        if let Some(state) = state {
            f.write_fmt(format_args!(
                "{}",
                RationalPolynomialPrinter {
                    poly: element,
                    state: state,
                    opts: PrintOptions::default(),
                    add_parentheses: in_product
                },
            ))
        } else if in_product {
            f.write_fmt(format_args!("({})", element))
        } else {
            f.write_fmt(format_args!("{}", element))
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> EuclideanDomain
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        RationalPolynomial {
            numerator: MultivariatePolynomial::new_from(&a.numerator, None),
            denominator: MultivariatePolynomial::new_from_constant(
                &a.numerator,
                a.numerator.field.one(),
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

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Field for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a / b
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        a.clone().inv()
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E> + PolynomialGCD<E>, E: Exponent>
    Add<&'a RationalPolynomial<R, E>> for &'b RationalPolynomial<R, E>
{
    type Output = RationalPolynomial<R, E>;

    fn add(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        let denom_gcd = MultivariatePolynomial::gcd(&self.denominator, &other.denominator);

        let mut a_denom_red = Cow::Borrowed(&self.denominator);
        let mut b_denom_red = Cow::Borrowed(&other.denominator);

        if !denom_gcd.is_one() {
            a_denom_red = Cow::Owned(&self.denominator / &denom_gcd);
            b_denom_red = Cow::Owned(&other.denominator / &denom_gcd);
        }

        let num1 = &self.numerator * &b_denom_red;
        let num2 = &other.numerator * &a_denom_red;
        let mut num = num1 + num2;

        // prefer small * large over medium * medium sized polynomials
        let mut den = if self.denominator.nterms > other.denominator.nterms
            && self.denominator.nterms > a_denom_red.nterms
        {
            b_denom_red.as_ref() * &self.denominator
        } else {
            a_denom_red.as_ref() * &other.denominator
        };

        let g = MultivariatePolynomial::gcd(&num, &denom_gcd);

        if !g.is_one() {
            num = num / &g;
            den = den / &g;
        }

        RationalPolynomial {
            numerator: num,
            denominator: den,
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

        if gcd1.is_one() {
            if gcd2.is_one() {
                RationalPolynomial {
                    numerator: &self.numerator * &other.numerator,
                    denominator: &self.denominator * &other.denominator,
                }
            } else {
                RationalPolynomial {
                    numerator: &self.numerator * &(&other.numerator / &gcd2),
                    denominator: (&self.denominator / &gcd2) * &other.denominator,
                }
            }
        } else if gcd2.is_one() {
            RationalPolynomial {
                numerator: (&self.numerator / &gcd1) * &other.numerator,
                denominator: &self.denominator * &(&other.denominator / &gcd1),
            }
        } else {
            RationalPolynomial {
                numerator: (&self.numerator / &gcd1) * &(&other.numerator / &gcd2),
                denominator: (&self.denominator / &gcd2) * &(&other.denominator / &gcd1),
            }
        }
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Div<&'a RationalPolynomial<R, E>>
    for &'b RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = RationalPolynomial<R, E>;

    fn div(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        // TODO: optimize
        self * &other.clone().inv()
    }
}
