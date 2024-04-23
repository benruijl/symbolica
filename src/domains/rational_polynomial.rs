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
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial,
        univariate::UnivariatePolynomial, Exponent, Variable,
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

impl<R: Ring, E: Exponent> From<MultivariatePolynomial<R, E>> for RationalPolynomial<R, E>
where
    Self: FromNumeratorAndDenominator<R, R, E>,
{
    fn from(poly: MultivariatePolynomial<R, E>) -> Self {
        let d = poly.one();
        let field = poly.field.clone();
        Self::from_num_den(poly, d, &field, false)
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

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
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

    // Convert from a univariate polynomial with rational polynomial coefficients to a rational polynomial.
    pub fn from_univariate(
        f: UnivariatePolynomial<RationalPolynomialField<R, E>>,
    ) -> RationalPolynomial<R, E> {
        let Some(pos) = f
            .field
            .var_map
            .iter()
            .position(|x| x == f.variable.as_ref())
        else {
            panic!("Variable not found in the field");
        };

        let mut res = RationalPolynomial::new(&f.field.ring, f.field.var_map.clone());

        let mut exp = vec![E::zero(); f.field.var_map.len()];
        exp[pos] = E::one();
        let v: RationalPolynomial<R, E> = res
            .numerator
            .monomial(res.numerator.field.one(), exp)
            .into();

        for (p, c) in f.coefficients.into_iter().enumerate() {
            res = &res + &(&v.pow(p as u64) * &c);
        }

        res
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
        if self.denominator.degree(var) == E::zero() {
            return vec![self.clone()];
        }

        let rat_field = RationalPolynomialField::new_from_poly(&self.numerator);
        let n = self
            .numerator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        let mut hs = vec![];

        let rem = if self.numerator.degree(var) >= self.denominator.degree(var) {
            let d = self
                .denominator
                .to_univariate(var)
                .map_coeff(|c| c.clone().into(), rat_field.clone());
            let (q, r) = n.quot_rem(&d);
            if !q.is_zero() {
                hs.push(Self::from_univariate(q));
            }
            r
        } else {
            n
        };

        // partial fraction the denominator
        let mut fs = self
            .denominator
            .factor()
            .into_iter()
            .map(|(x, p)| (x.to_univariate(var), p))
            .collect::<Vec<_>>();

        let mut constant = fs[0].0.one();
        fs.retain(|x| {
            if x.0.is_constant() {
                constant = &constant * &x.0.pow(x.1);
                false
            } else {
                true
            }
        });

        let constant =
            Self::from_univariate(constant.map_coeff(|c| c.clone().into(), rat_field.clone()))
                .inv();

        assert!(!fs.is_empty());

        let mut expanded = fs
            .iter()
            .map(|(f, p)| f.pow(*p).map_coeff(|c| c.clone().into(), rat_field.clone()))
            .collect::<Vec<_>>();

        // perform partial fractioning
        let deltas = if expanded.len() > 1 {
            UnivariatePolynomial::diophantine(&mut expanded, &rem)
        } else {
            vec![rem]
        };

        let fs = fs
            .into_iter()
            .map(|(x, e)| (x.map_coeff(|c| c.clone().into(), rat_field.clone()), e))
            .collect::<Vec<_>>();

        for (d, (p, p_pow)) in deltas.into_iter().zip(fs) {
            let exp = d.p_adic_expansion(&p);
            let p_rat = Self::from_univariate(p);
            for (pow, d_exp) in exp.into_iter().enumerate() {
                hs.push(
                    &(&Self::from_univariate(d_exp) / &p_rat.pow(p_pow as u64 - pow as u64))
                        * &constant,
                );
            }
        }

        hs
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Integrate the rational function in `var`. It returns a tuple
    /// `(ps, ls)` where `ps` should be interpreted as the sum of the rational parts
    /// and `ls` as a sum of logarithmic parts. Each logarithmic part is a tuple `(r, a)`
    /// that represents `sum_(r(z) = 0) z*log(a(var, z))` if `r` is dependent on `z`,
    /// else it is `r*log(a)`.
    pub fn integrate(&self, var: usize) -> (Vec<Self>, Vec<(Self, Self)>) {
        let rat_field = RationalPolynomialField::new_from_poly(&self.numerator);
        let n = self
            .numerator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        let d = self
            .denominator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        if d.is_constant() {
            let r = &Self::from_univariate(n.integrate()) / &Self::from_univariate(d);
            return (vec![r], vec![]);
        }

        let (q, r) = n.quot_rem(&d);

        let mut v = if q.is_zero() {
            vec![]
        } else {
            let n_conv = q.map_coeff(|c| c.clone().into(), rat_field.clone());
            vec![Self::from_univariate(n_conv.integrate())]
        };

        // partial fraction the denominator
        let mut fs = self
            .denominator
            .square_free_factorization()
            .into_iter()
            .map(|(x, p)| (x.to_univariate(var), p))
            .collect::<Vec<_>>();

        let mut constant = fs[0].0.one();
        fs.retain(|x| {
            if x.0.is_constant() {
                constant = &constant * &x.0.pow(x.1);
                false
            } else {
                true
            }
        });

        let mut constant =
            Self::from_univariate(constant.map_coeff(|c| c.clone().into(), rat_field.clone()))
                .inv();

        assert!(!fs.is_empty());

        let mut expanded = fs
            .iter()
            .map(|(f, p)| f.pow(*p).map_coeff(|c| c.clone().into(), rat_field.clone()))
            .collect::<Vec<_>>();

        let rem = r.map_coeff(|c| c.clone().into(), rat_field.clone());

        // perform partial fractioning
        let deltas = if expanded.len() > 1 {
            UnivariatePolynomial::diophantine(&mut expanded, &rem)
        } else {
            vec![rem]
        };

        let fs = fs
            .into_iter()
            .map(|(x, e)| (x.map_coeff(|c| c.clone().into(), rat_field.clone()), e))
            .collect::<Vec<_>>();

        let mut hs = vec![];
        for (d, (p, p_pow)) in deltas.into_iter().zip(&fs) {
            let p_diff = p.derivative();
            let (_, s, t) = p.eea(&p_diff);

            let p_full = Self::from_univariate(p.clone());

            let mut d_exp = d.p_adic_expansion(&p);

            // grow to the same level as the pow
            if d_exp.len() < *p_pow {
                d_exp.resize(*p_pow, d.zero());
            }

            // highest degree in 1/p last
            d_exp.reverse();

            // perform Hermite reduction
            for i in (1..d_exp.len()).rev() {
                let dd = d_exp[i].clone();
                let s_cor = s.clone() * &dd;
                let t_cor = t.clone() * &dd;

                d_exp[i - 1] = d_exp[i - 1].clone()
                    + s_cor
                    + t_cor.derivative().div_coeff(&(t_cor.field.nth(i as u64)));

                let t_full = Self::from_univariate(t_cor);

                let r = -(&t_full / &(&rat_field.nth(i as u64) * &p_full.pow(i as u64)));
                if !r.is_zero() {
                    v.push(r);
                }
            }

            if !d_exp[0].is_zero() {
                hs.push(d_exp.swap_remove(0));
            }
        }

        // create new temporary variable
        let mut t = MultivariatePolynomial::new(
            &self.numerator.field,
            None,
            Arc::new(vec![Variable::Temporary(0)]),
        )
        .monomial(self.numerator.field.one(), vec![E::one()])
        .into();

        let mut w = vec![];

        for (mut h, (mut p, _)) in hs.into_iter().zip(fs.into_iter()) {
            for c in &mut p.coefficients {
                c.unify_variables(&mut t);
            }
            for c in &mut h.coefficients {
                c.unify_variables(&mut t);
            }

            constant.unify_variables(&mut t);

            // adding a variable changes the field
            p.field = RationalPolynomialField::new_from_poly(&p.coefficients[0].numerator);
            h.field = p.field.clone();
            let new_var = p.coefficients[0].numerator.nvars() - 1;

            let b = h.clone() - p.derivative().mul_coeff(&t);

            // TODO: use resultant_prs instead?
            let r = p.resultant(&b);

            // drop the denominator as it is constant in x
            let mut sqf = r.numerator.square_free_factorization();
            sqf.retain(|(x, _)| !x.is_constant());

            let factors: Vec<(Vec<_>, _, _)> = sqf
                .into_iter()
                .map(|(s, p)| (s.factor().into_iter().map(|x| x.0).collect(), s, p))
                .collect();

            // TODO: if there is only one factor, we know there will be no merging and we can give the result
            // in terms of a root sum of the original denominator instead of the more complicated one
            // this requires returning 3 terms: the root to solve for, the residue and the log content

            // perform monic euclidean algorithm and collect the remainders of the powers that appear in the factorized resultant
            // TODO: this may be wrong, see
            // A Note on Subresultants and the Lazard/Rioboo/Trager Formula in Rational Function Integration
            let mut prs = vec![];
            prs.push(p.clone().make_monic());
            prs.push(b.clone().make_monic());
            while !prs.last().unwrap().is_zero() {
                let r = prs[prs.len() - 2].rem(&prs[prs.len() - 1]);
                if RationalPolynomialField::is_zero(&r.lcoeff()) {
                    break;
                }
                prs.push(r.make_monic());
            }

            for r in &prs {
                let Some((xs, sqf, _)) = factors.iter().find(|(_, _, pp)| r.degree() == *pp) else {
                    continue;
                };

                // write the polynomial as a polynomial in x and t only with rational polynomial
                // coefficients in all other variables
                // since we will make the polynomial monic, the denominator drops out
                let ll = RationalPolynomial::from_univariate(r.clone())
                    .numerator
                    .to_multivariate_polynomial_list(&[var, new_var], true);
                let mut bivar_poly =
                    MultivariatePolynomial::new(&p.field, Some(ll.len()), p.field.var_map.clone());
                for (e, p) in ll {
                    bivar_poly.append_monomial(p.into(), &e);
                }

                // convert defining polynomial to a univariate polynomial in t with rational polynomial coefficients
                let def_uni = sqf
                    .to_univariate(new_var)
                    .map_coeff(|c| c.clone().into(), p.field.clone());

                // write the polynomial in x and t as a polynomial in x with rational polynomial coefficients in t and
                // all other variables and solve a diophantine equation
                let lcoeff = bivar_poly.to_univariate(var).lcoeff();
                let aa = lcoeff.to_univariate_from_univariate(new_var);
                let (_, s, _) = aa.eea(&def_uni);

                // convert the s to a multivariate polynomial and multiply with the bivariate polynomial
                // and mod with the defining polynomial
                // this is equivalent to making the bivariate polynomial monic in Q[t][x]
                // TODO: write conversion routine
                let mut ss = bivar_poly.zero();
                let mut exp = vec![E::zero(); bivar_poly.nvars()];
                for (e, p) in s.coefficients.into_iter().enumerate() {
                    exp[new_var] = E::from_u32(e as u32);
                    ss.append_monomial(p, &exp);
                }

                let bivar_poly_scaled = bivar_poly * &ss;

                let mut def_biv = lcoeff.zero();
                for (e, p) in def_uni.coefficients.into_iter().enumerate() {
                    exp[new_var] = E::from_u32(e as u32);
                    def_biv.append_monomial(p, &exp);
                }

                let monic = bivar_poly_scaled.rem(&def_biv);

                // convert the result to a multivariate rational polynomial
                let mut res = p.field.zero();
                for t in &monic {
                    let mut exp = vec![E::zero(); p.lcoeff().numerator.nvars()];
                    exp.copy_from_slice(t.exponents);
                    let mm = p.coefficients[0]
                        .numerator
                        .monomial(self.numerator.field.one(), exp);
                    res = &res + &(t.coefficient * &mm.into());
                }

                for ff in xs {
                    // solve linear equation
                    if ff.degree(new_var) == E::one() {
                        let a = ff.to_univariate(new_var);
                        let sol = RationalPolynomial::from_num_den(
                            -a.coefficients[0].clone(),
                            a.coefficients[1].clone(),
                            &a.coefficients[0].field,
                            true,
                        );

                        let eval = monic.replace(new_var, &sol);

                        let mut res = p.field.zero();
                        for t in &eval {
                            let mut exp = vec![E::zero(); p.lcoeff().numerator.nvars()];
                            exp.copy_from_slice(t.exponents);
                            let mm = p.coefficients[0]
                                .numerator
                                .monomial(self.numerator.field.one(), exp);
                            res = &res + &(t.coefficient * &mm.into());
                        }

                        w.push((&sol * &constant, res));
                    } else {
                        w.push((&ff.clone().into() * &constant, res.clone()));
                    }
                }
            }
        }

        (v, w)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        domains::{integer::Z, rational::Q, rational_polynomial::RationalPolynomial, Ring},
        state::State,
    };

    use super::RationalPolynomialField;

    #[test]
    fn field() {
        let field = RationalPolynomialField::<_, u8>::new(Z, Arc::new(vec![]));
        let one = field.one();
        let t = format!("{}", field.printer(&one));
        assert_eq!(t, "1");
    }

    #[test]
    fn hermite_reduction() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse("1/(v1 + 1)^5")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, l) = p.integrate(0);

        assert_eq!(
            r,
            vec![Atom::parse("-1/(4+16*v1+24*v1^2+16*v1^3+4*v1^4)")
                .unwrap()
                .to_rational_polynomial::<_, _, u8>(&Q, &Z, r[0].get_variables().clone().into())]
        );
        assert_eq!(l, vec![]);
    }

    #[test]
    fn constant() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse("(v1^4+v2+v1*v2+2*v1)/(v2 + 1)")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(
                &Q,
                &Z,
                Some(Arc::new(vec![
                    State::get_symbol("v1").into(),
                    State::get_symbol("v2").into(),
                ])),
            );

        let (r, l) = p.integrate(0);
        assert_eq!(
            r,
            vec![
                Atom::parse("(10*v1*v2+10*v1^2+5*v1^2*v2+2*v1^5)/(10+10*v2)")
                    .unwrap()
                    .to_rational_polynomial::<_, _, u8>(
                        &Q,
                        &Z,
                        r[0].get_variables().clone().into()
                    )
            ]
        );
        assert_eq!(l, vec![]);
    }

    #[test]
    fn mixed_denominator() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse("(v1^4+v2+v1*v2+2*v1)/(v1)/(v2 + 1)")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(
                &Q,
                &Z,
                Some(Arc::new(vec![
                    State::get_symbol("v1").into(),
                    State::get_symbol("v2").into(),
                ])),
            );

        let (r, l) = p.integrate(0);

        let v = l[0].0.get_variables().clone();

        assert_eq!(
            r,
            vec![Atom::parse("(8*v1+4*v1*v2+v1^4)/(4+4*v2)")
                .unwrap()
                .to_rational_polynomial::<_, _, u8>(&Q, &Z, r[0].get_variables().clone().into())]
        );
        assert_eq!(
            l,
            vec![(
                Atom::parse("v2/(1+v2)")
                    .unwrap()
                    .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                Atom::parse("v1")
                    .unwrap()
                    .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
            ),]
        );
    }

    #[test]
    fn three_factors() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> =
            Atom::parse("(36v1^2+1167v1+3549/2)/(v1^3+23/30v1^2-2/15v1-2/15)")
                .unwrap()
                .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, l) = p.integrate(0);

        let v = l[0].0.get_variables().clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![
                (
                    Atom::parse("-8000")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("(1+2*v1)/2")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                ),
                (
                    Atom::parse("91125/16")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("(2+3*v1)/3")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                ),
                (
                    Atom::parse("37451/16")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("(-2+5*v1)/5")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                )
            ]
        );
    }

    #[test]
    fn multiple_residues() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse(
            "(7v1^13+10v1^8+4v1^7-7v1^6-4v1^3-4v1^2+3v1+3)/(v1^14-2v1^8-2v1^7-2v1^4-4v1^3-v1^2+2v1+1)",
        )
        .unwrap()
        .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, mut l) = p.integrate(0);
        let new_var = State::get_symbol("v2");

        // root sum in the answer, rename the temporary variable
        // TODO: add rename function
        let mut v = l[0].0.get_variables().as_ref().clone();
        *v.last_mut().unwrap() = new_var.clone().into();
        let new_map = Arc::new(v);

        l[0].0.numerator.variables = new_map.clone();
        l[0].0.denominator.variables = new_map.clone();
        l[0].1.numerator.variables = new_map.clone();
        l[0].1.denominator.variables = new_map.clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![(
                Atom::parse("1+4*v2-4*v2^2")
                    .unwrap()
                    .to_rational_polynomial::<_, _, u8>(&Q, &Z, new_map.clone().into()),
                Atom::parse("-1-2*v1*v2+v1^2-2*v1^2*v2+v1^7")
                    .unwrap()
                    .to_rational_polynomial::<_, _, u8>(&Q, &Z, new_map.clone().into()),
            )]
        );
    }

    #[test]
    fn multi_factor() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse("1/(v1^3+v1)")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, l) = p.integrate(0);

        let v = l[0].0.get_variables().clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![
                (
                    Atom::parse("-1/2")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("1+v1^2")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                ),
                (
                    Atom::parse("1")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("v1")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                )
            ]
        );
    }

    #[test]
    fn multiple_variables() {
        use crate::atom::Atom;
        let p: RationalPolynomial<_, _> = Atom::parse("(v1^4+v2+v1*v2+2*v1)/((v1-v2)(v1-2)(v1-4))")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, l) = p.integrate(0);

        let v = l[0].0.get_variables().clone();

        assert_eq!(
            r,
            vec![Atom::parse("(12*v1+2*v1*v2+v1^2)/2")
                .unwrap()
                .to_rational_polynomial::<_, _, u8>(&Q, &Z, r[0].get_variables().clone().into())]
        );
        assert_eq!(
            l,
            vec![
                (
                    Atom::parse("(20+3*v2)/(-4+2*v2)")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("-2+v1")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                ),
                (
                    Atom::parse("(-264-5*v2)/(-8+2*v2)")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("-4+v1")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                ),
                (
                    Atom::parse("(3*v2+v2^2+v2^4)/(8-6*v2+v2^2)")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                    Atom::parse("-v2+v1")
                        .unwrap()
                        .to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone().into()),
                )
            ]
        );
    }
}
