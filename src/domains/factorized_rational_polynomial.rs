use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{Display, Error, Formatter, Write},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    poly::{
        factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent,
        Variable,
    },
    printer::{FactorizedRationalPolynomialPrinter, PrintOptions},
    state::State,
};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::IntegerRing,
    rational::RationalField,
    EuclideanDomain, Field, Ring, RingPrinter,
};

#[derive(Clone, PartialEq, Debug)]
pub struct FactorizedRationalPolynomialField<R: Ring, E: Exponent> {
    ring: R,
    nvars: usize,
    var_map: Option<Arc<Vec<Variable>>>,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring, E: Exponent> FactorizedRationalPolynomialField<R, E> {
    pub fn new(
        coeff_ring: R,
        nvars: usize,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> FactorizedRationalPolynomialField<R, E> {
        FactorizedRationalPolynomialField {
            ring: coeff_ring,
            nvars,
            var_map,
            _phantom_exp: PhantomData,
        }
    }

    pub fn new_from_poly(
        poly: &MultivariatePolynomial<R, E>,
    ) -> FactorizedRationalPolynomialField<R, E> {
        FactorizedRationalPolynomialField {
            ring: poly.field.clone(),
            nvars: poly.nvars,
            var_map: poly.var_map.clone(),
            _phantom_exp: PhantomData,
        }
    }
}

pub trait FromNumeratorAndFactorizedDenominator<R: Ring, OR: Ring, E: Exponent> {
    /// Construct a rational polynomial from a numerator and a factorized denominator.
    /// An empty denominator means a denominator of 1.
    fn from_num_den(
        num: MultivariatePolynomial<R, E>,
        dens: Vec<(MultivariatePolynomial<R, E>, usize)>,
        field: &OR,
        do_factor: bool,
    ) -> FactorizedRationalPolynomial<OR, E>;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FactorizedRationalPolynomial<R: Ring, E: Exponent> {
    pub numerator: MultivariatePolynomial<R, E>,
    pub numer_coeff: R::Element,
    pub denom_coeff: R::Element,
    pub denominators: Vec<(MultivariatePolynomial<R, E>, usize)>, // TODO: sort factors?
}

impl<R: Ring, E: Exponent> PartialOrd for FactorizedRationalPolynomial<R, E> {
    /// An ordering of rational polynomials that has no intuitive meaning.
    fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
        todo!()
    }
}

impl<R: Ring, E: Exponent> FactorizedRationalPolynomial<R, E> {
    pub fn new(
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> FactorizedRationalPolynomial<R, E> {
        let num = MultivariatePolynomial::new(
            var_map.as_ref().map(|x| x.len()).unwrap_or(0),
            field,
            None,
            var_map,
        );

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: field.one(),
            denom_coeff: field.one(),
            denominators: vec![],
        }
    }

    pub fn get_var_map(&self) -> Option<&[Variable]> {
        self.numerator.var_map.as_ref().map(|x| x.as_slice())
    }

    pub fn unify_var_map(&mut self, other: &mut Self) {
        self.numerator.unify_var_map(&mut other.numerator);

        for d in &mut self.denominators {
            d.0.unify_var_map(&mut other.numerator);
        }
    }

    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominators.is_empty()
    }

    /// Constuct a pretty-printer for the rational polynomial.
    pub fn printer<'a, 'b>(
        &'a self,
        state: &'b State,
    ) -> FactorizedRationalPolynomialPrinter<'a, 'b, R, E> {
        FactorizedRationalPolynomialPrinter::new(self, state)
    }

    /// Convert the coefficient from the current field to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> FactorizedRationalPolynomial<FiniteField<UField>, E>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Ring>::Element: Copy,
    {
        let constant = field.div(
            &self.numer_coeff.to_finite_field(field),
            &self.denom_coeff.to_finite_field(field),
        );

        // check the gcd, since the rational polynomial may simplify
        FactorizedRationalPolynomial::from_num_den(
            self.numerator
                .map_coeff(|c| c.to_finite_field(field), field.clone())
                .mul_coeff(constant),
            self.denominators
                .iter()
                .map(|(f, p)| (f.map_coeff(|c| c.to_finite_field(field), field.clone()), *p))
                .collect(),
            field,
            true,
        )
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    fn is_one(&self) -> bool {
        self.numerator.is_one()
            && self.denominators.is_empty()
            && self.numerator.field.is_one(&self.numer_coeff)
            && self.numerator.field.is_one(&self.denom_coeff)
    }
}

impl<E: Exponent> FromNumeratorAndFactorizedDenominator<RationalField, IntegerRing, E>
    for FactorizedRationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        num: MultivariatePolynomial<RationalField, E>,
        dens: Vec<(MultivariatePolynomial<RationalField, E>, usize)>,
        field: &IntegerRing,
        do_factor: bool,
    ) -> FactorizedRationalPolynomial<IntegerRing, E> {
        let mut content = num.content();
        for (d, _) in &dens {
            content = d.field.gcd(&content, &d.content());
        }

        let (num_int, dens_int) = if num.field.is_one(&content) {
            (
                num.map_coeff(|c| c.numerator(), IntegerRing::new()),
                dens.iter()
                    .map(|(d, p)| (d.map_coeff(|c| c.numerator(), IntegerRing::new()), *p))
                    .collect(),
            )
        } else {
            (
                num.map_coeff(
                    |c| num.field.div(c, &content).numerator(),
                    IntegerRing::new(),
                ),
                dens.iter()
                    .map(|(d, p)| {
                        (
                            d.map_coeff(
                                |c| num.field.div(c, &content).numerator(),
                                IntegerRing::new(),
                            ),
                            *p,
                        )
                    })
                    .collect(),
            )
        };

        <FactorizedRationalPolynomial<IntegerRing, E> as FromNumeratorAndFactorizedDenominator<
            IntegerRing,
            IntegerRing,
            E,
        >>::from_num_den(num_int, dens_int, field, do_factor)
    }
}

impl<E: Exponent> FromNumeratorAndFactorizedDenominator<IntegerRing, IntegerRing, E>
    for FactorizedRationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut dens: Vec<(MultivariatePolynomial<IntegerRing, E>, usize)>,
        _field: &IntegerRing,
        do_factor: bool,
    ) -> Self {
        for _ in 0..2 {
            for (d, _) in &mut dens {
                num.unify_var_map(d);
            }
        }

        let mut num_const = num.field.one();
        let mut den_const = num.field.one();

        if dens.is_empty() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: num_const,
                denom_coeff: den_const,
                denominators: dens,
            };
        }

        if do_factor {
            for (d, _) in &mut dens {
                let gcd = MultivariatePolynomial::gcd(&num, d);

                if !gcd.is_one() {
                    num = num / &gcd;
                    *d = &*d / &gcd;
                }
            }

            // factor all denominators, as they may be unfactored
            // TODO: add extra flag for this?
            let mut factored = vec![];
            for (d, p) in dens {
                for (f, p2) in d.factor() {
                    factored.push((f, p * p2));
                }
            }

            // TODO: fuse factors that are the same

            dens = factored;
        }

        dens.retain(|f| {
            if f.0.is_constant() {
                den_const = &den_const * &f.0.lcoeff().pow(f.1 as u64);
                false
            } else {
                true
            }
        });

        if den_const.is_negative() {
            num_const = num_const.neg();
            den_const = den_const.neg();
        }

        // normalize denominator to have positive leading coefficient
        for (d, _) in &mut dens {
            if d.lcoeff().is_negative() {
                num_const = num_const.neg();
                *d = -d.clone(); // TODO: prevent clone
            }
        }

        if do_factor {
            let g = num.content();
            if !g.is_one() {
                num = num.div_coeff(&g);
            }
            num_const = &num_const * &g;
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: num_const,
            denom_coeff: den_const,
            denominators: dens,
        }
    }
}

impl<UField: FiniteFieldWorkspace, E: Exponent>
    FromNumeratorAndFactorizedDenominator<FiniteField<UField>, FiniteField<UField>, E>
    for FactorizedRationalPolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FiniteField<UField>, E>,
        mut dens: Vec<(MultivariatePolynomial<FiniteField<UField>, E>, usize)>,
        field: &FiniteField<UField>,
        do_factor: bool,
    ) -> Self {
        for _ in 0..2 {
            for (d, _) in &mut dens {
                num.unify_var_map(d);
            }
        }

        let mut constant = num.field.one();

        if dens.is_empty() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: constant,
                denom_coeff: constant,
                denominators: dens,
            };
        }

        if do_factor {
            for (d, _) in &mut dens {
                let gcd = MultivariatePolynomial::gcd(&num, d);

                if !gcd.is_one() {
                    num = num / &gcd;
                    *d = &*d / &gcd;
                }
            }

            // factor all denominators, as they may be unfactored
            // TODO: add extra flag for this?
            let mut factored = vec![];
            for (d, p) in dens {
                for (f, p2) in d.factor() {
                    factored.push((f, p * p2));
                }
            }

            // TODO: fuse factors that are the same

            dens = factored;
        }

        dens.retain(|f| {
            if f.0.is_constant() {
                field.mul_assign(&mut constant, &field.pow(&f.0.coefficients[0], f.1 as u64));
                false
            } else {
                true
            }
        });

        num = num.mul_coeff(field.inv(&constant));
        constant = field.one();

        // normalize denominator to have leading coefficient of one
        for (d, _) in &mut dens {
            if !field.is_one(&d.lcoeff()) {
                let c = field.inv(&d.lcoeff());
                num = num.mul_coeff(c);
                *d = d.clone().mul_coeff(c); // TODO: prevent clone
            }
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: constant,
            denom_coeff: constant,
            denominators: dens,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> FactorizedRationalPolynomial<R, E>
where
    Self: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Invert a factored rational polynomial. This is an expensive operation, as it requires
    /// factoring the numerator.
    #[inline]
    pub fn inv(self) -> Self {
        if self.numerator.is_zero() {
            panic!("Cannot invert 0");
        }

        let mut num = self.numerator.constant(self.denom_coeff);
        for (d, p) in self.denominators {
            num = num * &d.pow(p);
        }

        let dens = self.numerator.factor();

        let field = self.numerator.field.clone();
        Self::from_num_den(num, dens, &field, false)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> FactorizedRationalPolynomial<R, E>
where
    Self: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    pub fn pow(&self, e: u64) -> Self {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        // TODO: do binary exponentation
        let mut poly = FactorizedRationalPolynomial {
            numerator: self.numerator.constant(self.numerator.field.one()),
            numer_coeff: self.numerator.field.one(),
            denom_coeff: self.numerator.field.one(),
            denominators: vec![],
        };

        for _ in 0..e {
            poly = &poly * self;
        }
        poly
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let gcd_num = MultivariatePolynomial::gcd(&self.numerator, &other.numerator);

        let mut disjoint_factors = vec![];

        for (d, p) in &self.denominators {
            let mut found = false;
            for (d2, p2) in &other.denominators {
                if d == d2 {
                    disjoint_factors.push((d.clone(), *p.min(p2)));
                    found = true;
                    break;
                }
            }

            if !found {
                disjoint_factors.push((d.clone(), *p));
            }
        }

        for (d, p) in &other.denominators {
            let mut found = false;
            for (d2, _) in &self.denominators {
                if d == d2 {
                    found = true;
                    break;
                }
            }

            if !found {
                disjoint_factors.push((d.clone(), *p));
            }
        }

        let field = &self.numerator.field;
        FactorizedRationalPolynomial {
            numerator: gcd_num,
            numer_coeff: field.gcd(&self.numer_coeff, &other.numer_coeff),
            denom_coeff: field.mul(
                &field
                    .quot_rem(
                        &self.denom_coeff,
                        &field.gcd(&self.denom_coeff, &other.denom_coeff),
                    )
                    .0,
                &other.denom_coeff,
            ),
            denominators: disjoint_factors,
        }
    }
}

impl<R: Ring, E: Exponent> Display for FactorizedRationalPolynomial<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denominators.is_empty() && self.numerator.field.is_one(&self.denom_coeff) {
            self.numerator.fmt(f)
        } else {
            if f.sign_plus() {
                f.write_char('+')?;
            }

            if !self.numerator.field.is_one(&self.numer_coeff) {
                f.write_fmt(format_args!(
                    "({})*",
                    RingPrinter {
                        ring: &self.numerator.field,
                        element: &self.numer_coeff,
                        state: None,
                        opts: &PrintOptions::default(),
                        in_product: false,
                    }
                ))?;
            }

            f.write_fmt(format_args!("({})/(", self.numerator))?;

            if !self.numerator.field.is_one(&self.denom_coeff) {
                f.write_fmt(format_args!(
                    "({})",
                    RingPrinter {
                        ring: &self.numerator.field,
                        element: &self.denom_coeff,
                        state: None,
                        opts: &PrintOptions::default(),
                        in_product: false,
                    }
                ))?;
            }

            for (d, p) in &self.denominators {
                if *p == 1 {
                    f.write_fmt(format_args!("({})", d))?;
                } else {
                    f.write_fmt(format_args!("({})^{}", d, p))?;
                }
            }

            f.write_char(')')
        }
    }
}

impl<R: Ring, E: Exponent> Display for FactorizedRationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(()) // FIXME
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Ring
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    type Element = FactorizedRationalPolynomial<R, E>;

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
        FactorizedRationalPolynomial {
            numerator: MultivariatePolynomial::new(
                self.nvars,
                &self.ring,
                None,
                self.var_map.clone(),
            ),
            numer_coeff: self.ring.zero(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        }
    }

    fn one(&self) -> Self::Element {
        FactorizedRationalPolynomial {
            numerator: MultivariatePolynomial::new(
                self.nvars,
                &self.ring,
                None,
                self.var_map.clone(),
            )
            .one(),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
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
        let mut poly = FactorizedRationalPolynomial {
            numerator: b.numerator.constant(self.ring.one()),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        };

        for _ in 0..e {
            poly = self.mul(&poly, b);
        }
        poly
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.numerator.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.numerator.is_one()
            && a.denominators.is_empty()
            && a.numerator.field.is_one(&a.denom_coeff)
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
        state: Option<&State>,
        opts: &PrintOptions,
        in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        if let Some(state) = state {
            f.write_fmt(format_args!(
                "{}",
                FactorizedRationalPolynomialPrinter {
                    poly: element,
                    state,
                    opts: *opts,
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
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        FactorizedRationalPolynomial {
            numerator: a.numerator.zero(),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Field
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
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
    Add<&'a FactorizedRationalPolynomial<R, E>> for &'b FactorizedRationalPolynomial<R, E>
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn add(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        if self.is_zero() {
            return other.clone();
        } else if other.is_zero() {
            return self.clone();
        }

        let mut den = Vec::with_capacity(self.denominators.len() + other.denominators.len());
        let mut num_1 = self.numerator.clone();
        let mut num_2 = other.numerator.clone();

        for (d, p) in &self.denominators {
            if let Some((_, p2)) = other.denominators.iter().find(|(d2, _)| d == d2) {
                if p > p2 {
                    num_2 = num_2 * &d.pow(*p - *p2);
                } else if p < p2 {
                    num_1 = num_1 * &d.pow(*p2 - *p);
                }
                den.push((d.clone(), *p.max(p2)));
                continue;
            }
            num_2 = num_2 * &d.pow(*p);
            den.push((d.clone(), *p));
        }
        for (d, p) in &other.denominators {
            if self.denominators.iter().any(|(d2, _)| d == d2) {
                continue;
            }
            num_1 = num_1 * &d.pow(*p);
            den.push((d.clone(), *p));
        }

        if self.denom_coeff != other.denom_coeff {
            num_1 = num_1.mul_coeff(other.denom_coeff.clone());
            num_2 = num_2.mul_coeff(self.denom_coeff.clone());
        }

        let mut num =
            num_1.mul_coeff(self.numer_coeff.clone()) + num_2.mul_coeff(other.numer_coeff.clone());

        if num.is_zero() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: self.numerator.field.zero(),
                denom_coeff: self.numerator.field.one(),
                denominators: vec![],
            };
        }

        // TODO: are there some factors we can skip the division check for?
        for (d, p) in &mut den {
            while *p > 0 {
                if let Some(q) = num.divides(d) {
                    num = q;
                    *p -= 1;
                } else {
                    break;
                }
            }
        }

        den.retain(|(_, p)| *p > 0);

        let mut constant = if self.denom_coeff != other.denom_coeff {
            num.field.mul(&self.denom_coeff, &other.denom_coeff)
        } else {
            self.denom_coeff.clone()
        };

        let mut c = num.content();
        if !num.field.is_one(&c) {
            num = num.div_coeff(&c);
        }

        let g = num.field.gcd(&c, &constant);
        if !num.field.is_one(&g) {
            c = num.field.quot_rem(&c, &g).0;
            constant = num.field.quot_rem(&constant, &g).0;
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: c,
            denom_coeff: constant,
            denominators: den,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Sub
    for FactorizedRationalPolynomial<R, E>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent>
    Sub<&'a FactorizedRationalPolynomial<R, E>> for &'b FactorizedRationalPolynomial<R, E>
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn sub(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        (self.clone()).sub(other.clone())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> Neg
    for FactorizedRationalPolynomial<R, E>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        FactorizedRationalPolynomial {
            numer_coeff: self.numerator.field.neg(&self.numer_coeff),
            numerator: self.numerator,
            denom_coeff: self.denom_coeff,
            denominators: self.denominators,
        }
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent>
    Mul<&'a FactorizedRationalPolynomial<R, E>> for &'b FactorizedRationalPolynomial<R, E>
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn mul(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        if self.is_one() {
            return other.clone();
        } else if other.is_one() {
            return self.clone();
        }

        let mut reduced_numerator_1 = Cow::Borrowed(&self.numerator);
        let mut reduced_numerator_2 = Cow::Borrowed(&other.numerator);

        let mut den = Vec::with_capacity(self.denominators.len() + other.denominators.len());

        for (d, p) in &other.denominators {
            if let Some((_, p2)) = self.denominators.iter().find(|(d2, _)| d == d2) {
                den.push((d.clone(), p + p2));
                continue;
            }

            let mut p = *p;
            while p > 0 {
                if let Some(q) = reduced_numerator_1.divides(d) {
                    reduced_numerator_1 = Cow::Owned(q);
                    p -= 1;
                } else {
                    break;
                }
            }

            if p > 0 {
                den.push((d.clone(), p));
            }
        }

        for (d, p) in &self.denominators {
            if other.denominators.iter().any(|(d2, _)| d == d2) {
                continue;
            }

            let mut p = *p;
            while p > 0 {
                if let Some(q) = reduced_numerator_2.divides(d) {
                    reduced_numerator_2 = Cow::Owned(q);
                    p -= 1;
                } else {
                    break;
                }
            }

            if p > 0 {
                den.push((d.clone(), p));
            }
        }

        let field = &self.numerator.field;
        let mut constant = field.one();

        let mut numer_coeff;
        if !field.is_one(&self.denom_coeff) {
            let g = field.gcd(&other.numer_coeff, &self.denom_coeff);
            numer_coeff = field.quot_rem(&other.numer_coeff, &g).0;
            constant = field.quot_rem(&self.denom_coeff, &g).0;
        } else {
            numer_coeff = other.numer_coeff.clone();
        };
        if !field.is_one(&other.denom_coeff) {
            let g = field.gcd(&self.numer_coeff, &other.denom_coeff);
            field.mul_assign(&mut numer_coeff, &field.quot_rem(&self.numer_coeff, &g).0);
            field.mul_assign(&mut constant, &field.quot_rem(&other.denom_coeff, &g).0);
        } else {
            field.mul_assign(&mut numer_coeff, &self.numer_coeff);
        }

        FactorizedRationalPolynomial {
            numerator: reduced_numerator_1.as_ref() * reduced_numerator_2.as_ref(),
            numer_coeff,
            denom_coeff: constant,
            denominators: den,
        }
    }
}

impl<'a, 'b, R: EuclideanDomain + PolynomialGCD<E>, E: Exponent>
    Div<&'a FactorizedRationalPolynomial<R, E>> for &'b FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn div(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        // TODO: optimize
        // the factored form can be kept intact and cancellations can be achieved before writing out the new numerator
        self * &other.clone().inv()
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Compute the partial fraction decomposition of the rational polynomial in `var`.
    pub fn apart(&self, var: usize) -> Vec<Self> {
        if self.denominators.len() == 1 {
            return vec![self.clone()];
        }

        let rat_field = FactorizedRationalPolynomialField::new_from_poly(&self.numerator);

        let mut poly_univ = vec![];
        for (f, p) in &self.denominators {
            let f = f.clone().pow(*p);

            let l = f.to_univariate_polynomial_list(var);
            let mut res: MultivariatePolynomial<_, E> = MultivariatePolynomial::new(
                self.numerator.nvars,
                &rat_field,
                Some(l.len()),
                self.numerator.var_map.clone(),
            );

            let mut exp = vec![E::zero(); self.numerator.nvars];
            for (p, e) in l {
                exp[var] = e;
                res.append_monomial(
                    FactorizedRationalPolynomial::from_num_den(
                        p,
                        vec![],
                        &self.numerator.field,
                        false,
                    ),
                    &exp,
                );
            }
            poly_univ.push(res);
        }

        let rhs = poly_univ[0].one();
        let deltas = MultivariatePolynomial::diophantine_univariate(&mut poly_univ, &rhs);

        let mut factors = Vec::with_capacity(deltas.len());
        for (d, (p, pe)) in deltas.into_iter().zip(&self.denominators) {
            let mut unfold = rat_field.zero();
            for (c, e) in d
                .coefficients
                .into_iter()
                .zip(d.exponents.chunks(self.numerator.nvars))
            {
                unfold = &unfold
                    + &(&c
                        * &FactorizedRationalPolynomial::from_num_den(
                            unfold
                                .numerator
                                .monomial(self.numerator.field.one(), e.to_vec()),
                            vec![],
                            &self.numerator.field,
                            true,
                        ));
            }

            unfold = &unfold
                * &(&FactorizedRationalPolynomial::from_num_den(
                    self.numerator.clone(),
                    vec![
                        (p.clone(), *pe),
                        (self.numerator.constant(self.denom_coeff.clone()), 1),
                    ],
                    &self.numerator.field,
                    false,
                ) * &FactorizedRationalPolynomial {
                    numerator: self.numerator.one(),
                    numer_coeff: self.numer_coeff.clone(),
                    denom_coeff: self.numerator.field.one(),
                    denominators: vec![],
                });

            factors.push(unfold.clone());
        }

        factors
    }
}
