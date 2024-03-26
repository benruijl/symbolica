use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{Display, Error, Formatter, Write},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use ahash::HashMap;

use crate::{
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent,
        Variable,
    },
    printer::{PrintOptions, RationalPolynomialPrinter},
};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::{IntegerRing, Z},
    rational::RationalField,
    EuclideanDomain, Field, Ring,
};

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RationalPolynomialField<R: Ring, E: Exponent> {
    ring: R,
    var_map: Arc<Vec<Variable>>,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring, E: Exponent> RationalPolynomialField<R, E> {
    pub fn new(coeff_ring: R, var_map: Arc<Vec<Variable>>) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: coeff_ring,
            var_map,
            _phantom_exp: PhantomData,
        }
    }

    pub fn new_from_poly(poly: &MultivariatePolynomial<R, E>) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: poly.field.clone(),
            var_map: poly.variables.clone(),
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

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
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
    pub fn new(field: &R, var_map: Arc<Vec<Variable>>) -> RationalPolynomial<R, E> {
        let num = MultivariatePolynomial::new(field, None, var_map);
        let den = num.one();

        RationalPolynomial {
            numerator: num,
            denominator: den,
        }
    }

    pub fn get_variables(&self) -> &Arc<Vec<Variable>> {
        &self.numerator.variables
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        assert_eq!(self.numerator.variables, self.denominator.variables);
        assert_eq!(other.numerator.variables, other.denominator.variables);

        self.numerator.unify_variables(&mut other.numerator);
        self.denominator.unify_variables(&mut other.denominator);
    }

    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominator.is_constant()
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
            self.numerator
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
            self.denominator
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
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

        let mut num_int = MultivariatePolynomial::new(&Z, None, num.variables.into());
        num_int.exponents = num.exponents;

        let mut den_int = MultivariatePolynomial::new(&Z, Some(den.nterms()), den.variables.into());
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
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = num.gcd(&den);

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
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = num.gcd(&den);

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
            numerator: self.numerator.one(),
            denominator: self.denominator.one(),
        };

        for _ in 0..e {
            poly = &poly * self;
        }
        poly
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let gcd_num = self.numerator.gcd(&other.numerator);
        let gcd_den = self.denominator.gcd(&other.denominator);

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
        let index_mask: Vec<_> = self
            .numerator
            .variables
            .iter()
            .map(|v| variables.iter().position(|vv| vv == v))
            .collect();

        for e in self.denominator.exponents_iter() {
            for (c, p) in index_mask.iter().zip(e) {
                if c.is_some() && *p > E::zero() {
                    return Err("Not a polynomial");
                }
            }
        }

        let mut hm: HashMap<Vec<E>, RationalPolynomial<R, E>> = HashMap::default();

        let mut e_list = vec![E::zero(); variables.len()];

        let mut e_list_coeff = vec![E::zero(); self.numerator.nvars()];
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
                let mut r = RationalPolynomial::new(
                    &self.numerator.field.clone(),
                    self.numerator.variables.clone(),
                );
                r.numerator
                    .append_monomial(e.coefficient.clone(), &e_list_coeff);
                hm.insert(e_list.clone(), r);
            }
        }

        let v = Arc::new(variables.to_vec());
        let field = RationalPolynomialField::new(self.numerator.field.clone(), v.clone());
        let mut poly = MultivariatePolynomial::new(&field, Some(hm.len()), v.into());

        if !ignore_denominator {
            let denom = RationalPolynomial::from_num_den(
                self.denominator.one(),
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
        RationalPolynomialPrinter::new(self).fmt(f)
    }
}

impl<R: Ring, E: Exponent> Display for RationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
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
        let num = MultivariatePolynomial::new(&self.ring, None, self.var_map.clone().into());
        RationalPolynomial {
            denominator: num.one(),
            numerator: num,
        }
    }

    fn one(&self) -> Self::Element {
        let num = MultivariatePolynomial::new(&self.ring, None, self.var_map.clone().into()).one();
        RationalPolynomial {
            numerator: num.clone(),
            denominator: num,
        }
    }

    fn nth(&self, n: u64) -> Self::Element {
        let mut r = self.one();
        r.numerator = r.numerator.mul_coeff(self.ring.nth(n));
        r
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        // TODO: do binary exponentation
        let mut poly = RationalPolynomial {
            numerator: b.numerator.zero(),
            denominator: b.denominator.zero(),
        };
        poly.numerator = poly.numerator.add_constant(self.ring.one());
        poly.denominator = poly.denominator.add_constant(self.ring.one());

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

    fn is_characteristic_zero(&self) -> bool {
        self.ring.is_characteristic_zero()
    }

    fn sample(&self, _rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        todo!("Sampling a polynomial is not possible yet")
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        f.write_fmt(format_args!(
            "{}",
            RationalPolynomialPrinter {
                poly: element,
                opts: *opts,
                add_parentheses: in_product
            },
        ))
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> EuclideanDomain
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        RationalPolynomial {
            numerator: a.numerator.zero(),
            denominator: a.numerator.one(),
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
        let denom_gcd = self.denominator.gcd(&other.denominator);

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
        let mut den = if self.denominator.nterms() > other.denominator.nterms()
            && self.denominator.nterms() > a_denom_red.nterms()
        {
            b_denom_red.as_ref() * &self.denominator
        } else {
            a_denom_red.as_ref() * &other.denominator
        };

        let g = num.gcd(&denom_gcd);

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
        let gcd1 = self.numerator.gcd(&other.denominator);
        let gcd2 = self.denominator.gcd(&other.numerator);

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

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Compute the partial fraction decomposition of the rational polynomial in `var`.
    pub fn apart(&self, var: usize) -> Vec<Self> {
        let fs = self.denominator.factor();

        if fs.len() == 1 {
            return vec![self.clone()];
        }

        let rat_field = RationalPolynomialField::new_from_poly(&self.numerator);

        let mut poly_univ = vec![];
        let mut exp = vec![E::zero(); self.numerator.nvars()];
        for (f, p) in &fs {
            let f = f.clone().pow(*p);

            let l = f.to_univariate_polynomial_list(var);
            let mut res: MultivariatePolynomial<_, E> = MultivariatePolynomial::new(
                &rat_field,
                Some(l.len()),
                self.numerator.variables.clone().into(),
            );

            for (p, e) in l {
                exp[var] = e;
                let one = p.one();
                res.append_monomial(
                    RationalPolynomial::from_num_den(p, one, &self.numerator.field, false),
                    &exp,
                );
            }
            poly_univ.push(res);
        }

        let rhs = poly_univ[0].one();
        let deltas = MultivariatePolynomial::diophantine_univariate(&mut poly_univ, &rhs);

        let mut factors = Vec::with_capacity(deltas.len());
        for (d, (p, pe)) in deltas.into_iter().zip(fs.into_iter()) {
            let mut unfold = rat_field.zero();
            for (c, e) in d
                .coefficients
                .into_iter()
                .zip(d.exponents.chunks(self.numerator.nvars()))
            {
                unfold = &unfold
                    + &(&c
                        * &RationalPolynomial::from_num_den(
                            unfold
                                .numerator
                                .monomial(self.numerator.field.one(), e.to_vec()),
                            unfold.numerator.one(),
                            &self.numerator.field,
                            true,
                        ));
            }

            unfold = &unfold
                * &RationalPolynomial::from_num_den(
                    self.numerator.clone(),
                    p.pow(pe),
                    &self.numerator.field,
                    false,
                );

            factors.push(unfold.clone());
        }

        factors
    }
}
