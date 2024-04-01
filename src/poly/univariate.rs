use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    domains::{EuclideanDomain, Field, Ring, RingPrinter},
    printer::PrintOptions,
};

use super::Variable;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct UnivariatePolynomialRing<R: Ring> {
    ring: R,
    variable: Arc<Variable>,
}

impl<R: Ring> UnivariatePolynomialRing<R> {
    pub fn new(coeff_ring: R, var_map: Arc<Variable>) -> UnivariatePolynomialRing<R> {
        UnivariatePolynomialRing {
            ring: coeff_ring,
            variable: var_map,
        }
    }

    pub fn new_from_poly(poly: &UnivariatePolynomial<R>) -> UnivariatePolynomialRing<R> {
        UnivariatePolynomialRing {
            ring: poly.field.clone(),
            variable: poly.variable.clone(),
        }
    }
}

impl<R: Ring> std::fmt::Display for UnivariatePolynomialRing<R> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: Ring> Ring for UnivariatePolynomialRing<R> {
    type Element = UnivariatePolynomial<R>;

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
        *a = std::mem::replace(a, b.zero()) + b.clone();
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b.clone();
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) * b;
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * c
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * c
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }

    fn zero(&self) -> Self::Element {
        UnivariatePolynomial::new(&self.ring, None, self.variable.clone())
    }

    fn one(&self) -> Self::Element {
        self.zero().one()
    }

    fn nth(&self, n: u64) -> Self::Element {
        self.zero().constant(self.ring.nth(n))
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e as usize)
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
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
        _opts: &PrintOptions,
        in_product: bool,
        f: &mut std::fmt::Formatter<'_>,
    ) -> Result<(), std::fmt::Error> {
        if f.sign_plus() {
            f.write_str("+")?;
        }

        if in_product {
            f.write_str("(")?;
        }

        <Self::Element as std::fmt::Display>::fmt(element, f)?;

        if in_product {
            f.write_str(")")?;
        }

        Ok(())
    }
}

impl<R: EuclideanDomain> EuclideanDomain for UnivariatePolynomialRing<R> {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.rem(b)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b)
    }

    fn gcd(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        todo!("Implement univariate GCD for non-fields")
    }
}

/// A dense univariate polynomial.
#[derive(Clone)]
pub struct UnivariatePolynomial<F: Ring> {
    pub coefficients: Vec<F::Element>,
    pub variable: Arc<Variable>,
    pub field: F,
}

impl<F: Ring + std::fmt::Debug> std::fmt::Debug for UnivariatePolynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for c in self.coefficients.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{{ {:?} }}", c)?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + std::fmt::Display> std::fmt::Display for UnivariatePolynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let v = self.variable.to_string();

        let mut first = true;
        for (e, c) in self.coefficients.iter().enumerate() {
            if F::is_zero(c) {
                continue;
            }

            if first {
                first = false;
            } else {
                write!(f, "+")?;
            }

            write!(
                f,
                "{}*{}^{}",
                RingPrinter {
                    element: c,
                    ring: &self.field,
                    opts: PrintOptions::default(),
                    in_product: true
                },
                v,
                e
            )?;
        }
        Ok(())
    }
}

impl<F: Ring> UnivariatePolynomial<F> {
    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(field: &F, cap: Option<usize>, variable: Arc<Variable>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            field: field.clone(),
            variable,
        }
    }

    /// Constructs a zero polynomial, inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a zero polynomial with the given number of variables and capacity,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero_with_capacity(&self, cap: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap),
            field: self.field.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a constant polynomial,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        if F::is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            field: self.field.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a polynomial that is one, inheriting the field and variable map from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponent: usize) -> Self {
        if F::is_zero(&coeff) {
            return self.zero();
        }

        let mut coefficients = vec![self.field.zero(); exponent + 1];
        coefficients[exponent] = coeff;

        Self {
            coefficients,
            field: self.field.clone(),
            variable: self.variable.clone(),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.degree() == 1 && self.field.is_one(&self.coefficients[0])
    }

    /// Returns true if the polynomial is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1
    }

    /// Get the constant term of the polynomial.
    #[inline]
    pub fn get_constant(&self) -> F::Element {
        if self.is_zero() {
            return self.field.zero();
        }

        self.coefficients[0].clone()
    }

    /// Get a copy of the variable/
    pub fn get_vars(&self) -> Arc<Variable> {
        self.variable.clone()
    }

    /// Get a reference to the variables
    pub fn get_vars_ref(&self) -> &Variable {
        self.variable.as_ref()
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        self.coefficients
            .last()
            .unwrap_or(&self.field.zero())
            .clone()
    }

    pub fn degree(&self) -> usize {
        if self.is_zero() {
            return 0; // TODO: return None?
        }

        self.coefficients.len() - 1
    }

    /// Compute `self^pow`.
    pub fn pow(&self, mut pow: usize) -> Self {
        if pow == 0 {
            return self.one();
        }

        let mut x = self.clone();
        let mut y = self.one();
        while pow != 1 {
            if pow % 2 == 1 {
                y = &y * &x;
                pow -= 1;
            }

            x = &x * &x;
            pow /= 2;
        }

        x * &y
    }

    pub fn mul_exp(&self, exp: usize) -> Self {
        if exp == 0 {
            return self.clone();
        }

        let mut a = self.zero();
        a.coefficients = vec![self.field.zero(); self.degree() + exp + 1];

        for (cn, c) in a.coefficients.iter_mut().skip(exp).zip(&self.coefficients) {
            *cn = c.clone();
        }

        a
    }

    pub fn mul_coeff(mut self, coeff: &F::Element) -> Self {
        for c in &mut self.coefficients {
            if !F::is_zero(c) {
                self.field.mul_assign(c, coeff);
            }
        }

        self
    }

    fn truncate(&mut self) {
        let d = self
            .coefficients
            .iter_mut()
            .rev()
            .position(|c| !F::is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.coefficients.truncate(self.coefficients.len() - d);
    }

    pub fn evaluate(&self, x: &F::Element) -> F::Element {
        let mut res = self.field.zero();

        let mut last_non_zero = 0;
        for c in self.coefficients.iter().rev() {
            if !F::is_zero(c) {
                if last_non_zero == 1 {
                    self.field.mul_assign(&mut res, x);
                } else {
                    let p = self.field.pow(x, last_non_zero + 1); // TODO: cache powers?
                    self.field.mul_assign(&mut res, &p);
                }

                self.field.add_assign(&mut res, c);
                last_non_zero = 0;
            } else {
                last_non_zero += 1;
            }
        }

        if last_non_zero == 1 {
            self.field.mul_assign(&mut res, x);
        } else if last_non_zero > 1 {
            let p = self.field.pow(x, last_non_zero + 1);
            self.field.mul_assign(&mut res, &p);
        }

        res
    }
}

impl<F: Ring> PartialEq for UnivariatePolynomial<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable != other.variable {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.is_zero() != other.is_zero() {
                return false;
            }

            if self.is_zero() {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            // TODO: what is expected here?
            unimplemented!(
                "Cannot compare non-constant polynomials with different variable maps yet"
            );
        }

        if self.degree() != other.degree() {
            return false;
        }
        self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring> std::hash::Hash for UnivariatePolynomial<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.variable.hash(state);
    }
}

impl<F: Ring> Eq for UnivariatePolynomial<F> {}

impl<R: Ring> PartialOrd for UnivariatePolynomial<R> {
    /// An ordering of polynomials that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coefficients.partial_cmp(&other.coefficients)
    }
}

impl<F: Ring> Add for UnivariatePolynomial<F> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.field, other.field);
        assert_eq!(self.variable, other.variable);

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        if self.degree() < other.degree() {
            std::mem::swap(&mut self, &mut other);
        }

        for (i, c) in other.coefficients.iter().enumerate() {
            self.field.add_assign(&mut self.coefficients[i], c);
        }

        self.truncate();

        self
    }
}

impl<'a, 'b, F: Ring> Add<&'a UnivariatePolynomial<F>> for &'b UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring> Sub for UnivariatePolynomial<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, 'b, F: Ring> Sub<&'a UnivariatePolynomial<F>> for &'b UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring> Neg for UnivariatePolynomial<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(c);
        }
        self
    }
}

impl<'a, 'b, F: Ring> Mul<&'a UnivariatePolynomial<F>> for &'b UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    #[inline]
    fn mul(self, rhs: &'a UnivariatePolynomial<F>) -> Self::Output {
        let n = self.degree();
        let m = rhs.degree();

        if n == 0 || m == 0 {
            return self.zero();
        }

        if n == 1 {
            let mut r = rhs.clone();
            for c in &mut r.coefficients {
                self.field.mul_assign(c, &self.coefficients[0]);
            }
            return r;
        }

        if m == 1 {
            let mut r = self.clone();
            for c in &mut r.coefficients {
                self.field.mul_assign(c, &rhs.coefficients[0]);
            }
            return r;
        }

        let mut res = self.zero();
        res.coefficients = vec![self.field.zero(); n + m + 1];

        for (e1, c1) in self.coefficients.iter().enumerate() {
            if F::is_zero(c1) {
                continue;
            }

            for (e2, c2) in rhs.coefficients.iter().enumerate() {
                if !F::is_zero(c2) {
                    self.field
                        .add_mul_assign(&mut res.coefficients[e1 + e2], c1, c2);
                }
            }
        }

        res.truncate();
        res
    }
}

impl<'a, F: Ring> Mul<&'a UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    #[inline]
    fn mul(self, rhs: &'a UnivariatePolynomial<F>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a, 'b, F: EuclideanDomain> Div<&'a UnivariatePolynomial<F>> for &'b UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn div(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        self.divides(other)
            .unwrap_or_else(|| panic!("No clean division of {} by {}", self, other))
    }
}

impl<'a, F: EuclideanDomain> Div<&'a UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn div(self: UnivariatePolynomial<F>, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: EuclideanDomain> UnivariatePolynomial<F> {
    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.field.zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.field.is_one(&c) {
                break;
            }

            c = self.field.gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        for c in &mut self.coefficients {
            let (quot, rem) = self.field.quot_rem(c, other);
            debug_assert!(F::is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the polynomial primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }

    pub fn divides(&self, div: &UnivariatePolynomial<F>) -> Option<UnivariatePolynomial<F>> {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return Some(self.clone());
        }

        // check if the leading coefficients divide
        if !F::is_zero(&self.field.rem(&self.lcoeff(), &div.lcoeff())) {
            return None;
        }

        if self.degree() < div.degree() {
            return None;
        }

        if self.field.is_characteristic_zero() {
            // test division of constant term (evaluation at x_i = 0)
            let c = div.get_constant();
            if !F::is_zero(&c)
                && !self.field.is_one(&c)
                && !F::is_zero(&self.field.rem(&self.get_constant(), &c))
            {
                return None;
            }

            // test division at x_i = 1
            let mut num = self.field.zero();
            for c in &self.coefficients {
                if !F::is_zero(c) {
                    self.field.add_assign(&mut num, c);
                }
            }
            let mut den = self.field.zero();
            for c in &div.coefficients {
                if !F::is_zero(c) {
                    self.field.add_assign(&mut den, c);
                }
            }

            if !F::is_zero(&den)
                && !self.field.is_one(&den)
                && !F::is_zero(&self.field.rem(&num, &den))
            {
                return None;
            }
        }

        let (a, b) = self.quot_rem(div);
        if b.is_zero() {
            Some(a)
        } else {
            None
        }
    }

    /// Compute the remainder `self % div`.
    pub fn rem(&self, div: &UnivariatePolynomial<F>) -> Self {
        self.quot_rem(div).1
    }

    pub fn quot_rem(&self, div: &Self) -> (Self, Self) {
        if div.is_zero() {
            panic!("Cannot divide by 0");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let mut n = self.degree();
        let m = div.degree();

        let mut q = self.zero();
        q.coefficients = vec![self.field.zero(); n - m + 1];

        let mut r = self.clone();

        while n >= m {
            let (qq, rr) = self
                .field
                .quot_rem(&r.coefficients[n], &div.coefficients[m]);
            if !F::is_zero(&rr) {
                return (self.zero(), r);
            }

            r = r - div.mul_exp(n - m).mul_coeff(&qq);
            q.coefficients[n - m] = qq;

            if r.is_zero() {
                break;
            }

            n = r.degree();
        }

        q.truncate();

        (q, r)
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    pub fn gcd(&self, b: &Self) -> Self {
        if self.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return self.clone();
        }

        let mut c = self.clone();
        let mut d = b.clone();
        if self.degree() < b.degree() {
            std::mem::swap(&mut c, &mut d);
        }

        // TODO: there exists an efficient algorithm for univariate poly
        // division in a finite field using FFT
        let mut r = c.quot_rem(&mut d).1;
        while !r.is_zero() {
            c = d;
            d = r;
            r = c.quot_rem(&mut d).1;
        }

        // normalize the gcd
        let l = d.coefficients.last().unwrap().clone();
        for x in &mut d.coefficients {
            self.field.div_assign(x, &l);
        }

        d
    }

    /// Optimized division routine for univariate polynomials over a field, which
    /// makes the divisor monic first.
    pub fn quot_rem_field(
        &self,
        div: &mut UnivariatePolynomial<F>,
    ) -> (UnivariatePolynomial<F>, UnivariatePolynomial<F>) {
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let mut n = self.degree();
        let m = div.degree();

        let u = self.field.inv(&div.coefficients[m]);

        let mut q = self.zero();
        q.coefficients = vec![self.field.zero(); n - m + 1];

        let mut r = self.clone();

        while n >= m {
            let qq = self.field.mul(&r.coefficients[n], &u);
            r = r - div.mul_exp(n - m).mul_coeff(&qq);
            q.coefficients[n - m] = qq;
            n = r.degree();
        }

        q.truncate();

        (q, r)
    }
}
