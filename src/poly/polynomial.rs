use ahash::{HashMap, HashMapExt};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt::Display;
use std::fmt::{self, Write};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::printer::PolynomialPrinter;
use crate::rings::finite_field::{
    FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField,
};
use crate::rings::integer::{Integer, IntegerRing};
use crate::rings::rational::RationalField;
use crate::rings::{EuclideanDomain, Field, Ring, RingPrinter};
use crate::state::State;

use super::gcd::PolynomialGCD;
use super::{Exponent, LexOrder, MonomialOrder, Variable, INLINED_EXPONENTS};
use smallvec::{smallvec, SmallVec};

/// Multivariate polynomial with a sparse degree and variable dense representation.
// TODO: implement EuclideanDomain for MultivariatePolynomial
#[derive(Clone)]
pub struct MultivariatePolynomial<F: Ring, E: Exponent, O: MonomialOrder = LexOrder> {
    // Data format: the i-th monomial is stored as coefficients[i] and
    // exponents[i * nvars .. (i + 1) * nvars]. Keep coefficients.len() == nterms and
    // exponents.len() == nterms * nvars. Terms are always expanded and sorted by the exponents via
    // cmp_exponents().
    pub coefficients: Vec<F::Element>,
    pub exponents: Vec<E>,
    pub nterms: usize,
    pub nvars: usize,
    pub field: F,
    pub var_map: Option<SmallVec<[Variable; INLINED_EXPONENTS]>>,
    pub(crate) _phantom: PhantomData<O>,
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Constructs a zero polynomial.
    #[inline]
    pub fn new(nvars: usize, field: &F, cap: Option<usize>, var_map: Option<&[Variable]>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            exponents: Vec::with_capacity(cap.unwrap_or(0) * nvars),
            nterms: 0,
            nvars,
            field: field.clone(),
            var_map: var_map.map(|x| x.into()),
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial with the given number of variables and capacity,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn new_from(&self, cap: Option<usize>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            exponents: Vec::with_capacity(cap.unwrap_or(0) * self.nvars),
            nterms: 0,
            nvars: self.nvars,
            field: self.field.clone(),
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a constant polynomial,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn new_from_constant(&self, coeff: F::Element) -> Self {
        if F::is_zero(&coeff) {
            return self.new_from(None);
        }

        Self {
            coefficients: vec![coeff],
            exponents: vec![E::zero(); self.nvars],
            nterms: 1,
            nvars: self.nvars,
            field: self.field.clone(),
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn new_from_monomial(&self, coeff: F::Element, exponents: Vec<E>) -> Self {
        debug_assert!(self.nvars == exponents.len());

        if F::is_zero(&coeff) {
            return self.new_from(None);
        }

        Self {
            coefficients: vec![coeff],
            nvars: exponents.len(),
            exponents,
            nterms: 1,
            field: self.field.clone(),
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constuct a pretty-printer for the polynomial.
    pub fn printer<'a, 'b>(&'a self, state: &'b State) -> PolynomialPrinter<'a, 'b, F, E, O> {
        PolynomialPrinter::new(self, state)
    }

    /// Get the ith monomial
    pub fn to_monomial_view(&self, i: usize) -> MonomialView<F, E> {
        assert!(i < self.nterms);

        MonomialView {
            coefficient: &self.coefficients[i],
            exponents: self.exponents(i),
        }
    }

    #[inline]
    pub fn reserve(&mut self, cap: usize) -> &mut Self {
        self.coefficients.reserve(cap);
        self.exponents.reserve(cap * self.nvars);
        self
    }

    #[inline]
    pub fn zero(field: &F) -> Self {
        Self::new(0, field, None, None)
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.nterms == 0
    }

    #[inline]
    pub fn one(field: &F) -> Self {
        // TODO: inherit var_map from somewhere
        Self {
            coefficients: vec![field.one()],
            exponents: vec![],
            nterms: 1,
            nvars: 0,
            field: field.clone(),
            var_map: None,
            _phantom: PhantomData,
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.nterms == 1
            && self.field.is_one(&self.coefficients[0])
            && self.exponents.iter().all(|x| x.is_zero())
    }

    /// Returns the number of terms in the polynomial.
    #[inline]
    pub fn nterms(&self) -> usize {
        self.nterms
    }

    /// Returns the number of variables in the polynomial.
    #[inline]
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Returns true if the polynomial is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        if self.nterms >= 2 {
            return false;
        }
        debug_assert!(!F::is_zero(self.coefficients.first().unwrap()));
        return self.exponents.iter().all(|e| e.is_zero());
    }

    /// Returns the `index`th monomial, starting from the back.
    #[inline]
    pub fn coefficient_back(&self, index: usize) -> &F::Element {
        &self.coefficients[self.nterms - index - 1]
    }

    /// Returns the slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents(&self, index: usize) -> &[E] {
        //&self.exponents[index * self.nvars..(index + 1) * self.nvars]
        unsafe {
            self.exponents
                .get_unchecked(index * self.nvars..(index + 1) * self.nvars)
        }
    }

    /// Returns the slice for the exponents of the specified monomial
    /// starting from the back.
    #[inline]
    pub fn exponents_back(&self, index: usize) -> &[E] {
        let index = self.nterms - index - 1;
        &self.exponents[index * self.nvars..(index + 1) * self.nvars]
    }

    pub fn last_exponents(&self) -> &[E] {
        assert!(self.nterms > 0);
        &self.exponents[(self.nterms - 1) * self.nvars..self.nterms * self.nvars]
    }

    /// Returns the mutable slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents_mut(&mut self, index: usize) -> &mut [E] {
        &mut self.exponents[index * self.nvars..(index + 1) * self.nvars]
    }

    /// Returns the number of variables in the polynomial.
    #[inline]
    pub fn clear(&mut self) {
        self.nterms = 0;
        self.coefficients.clear();
        self.exponents.clear();
    }

    /// Get the variable map.
    pub fn get_var_map(&self) -> &Option<smallvec::SmallVec<[Variable; INLINED_EXPONENTS]>> {
        &self.var_map
    }

    /// Unify the variable maps of two polynomials, i.e.
    /// rewrite a polynomial in `x` and one in `y` to a
    /// two polynomial in `x` and `y`.
    ///
    /// The variable map will be inherited from
    /// `self` and will be extended by variables occurring
    /// in `other`.
    pub fn unify_var_map(&mut self, other: &mut Self) {
        assert!(
            (self.var_map.is_some() || self.nvars == 0)
                && (other.var_map.is_some() || other.nvars == 0)
        );

        if self.var_map == other.var_map {
            return;
        }

        let mut new_var_map = self.var_map.clone().unwrap_or(SmallVec::new());
        let mut new_var_pos_other = vec![0; other.nvars];
        for (pos, v) in new_var_pos_other
            .iter_mut()
            .zip(other.var_map.as_ref().unwrap_or(&SmallVec::new()))
        {
            if let Some(p) = new_var_map.iter().position(|x| x == v) {
                *pos = p;
            } else {
                *pos = new_var_map.len();
                new_var_map.push(*v);
            }
        }

        let mut newexp = vec![E::zero(); new_var_map.len() * self.nterms];

        for t in 0..self.nterms {
            newexp[t * new_var_map.len()..t * new_var_map.len() + self.nvars]
                .copy_from_slice(self.exponents(t));
        }

        self.var_map = Some(new_var_map.clone());
        self.exponents = newexp;
        self.nvars = new_var_map.len();

        // reconstruct 'other' with correct monomial ordering
        let mut newother = Self::new(
            new_var_map.len(),
            &other.field,
            Some(other.nterms),
            Some(&new_var_map),
        );
        let mut newexp: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); new_var_map.len()];
        for t in other.into_iter() {
            for c in &mut newexp {
                *c = E::zero();
            }

            for (var, e) in t.exponents.iter().enumerate() {
                newexp[new_var_pos_other[var]] = *e;
            }
            newother.append_monomial(t.coefficient.clone(), &newexp);
        }
        *other = newother;
    }

    /// Reverse the monomial ordering in-place.
    fn reverse(&mut self) {
        if self.nterms < 2 {
            return;
        }

        self.coefficients.reverse();

        let midu = if self.nterms % 2 == 0 {
            self.nvars * (self.nterms / 2)
        } else {
            self.nvars * (self.nterms / 2 + 1)
        };

        let (l, r) = self.exponents.split_at_mut(midu);

        let rend = r.len();
        for i in 0..self.nterms / 2 {
            l[i * self.nvars..(i + 1) * self.nvars]
                .swap_with_slice(&mut r[rend - (i + 1) * self.nvars..rend - i * self.nvars]);
        }
    }

    /// Grow the exponent list so the variable index fits in.
    pub fn grow_to(&mut self, var: usize) {
        if self.nterms() < var {
            // move all the exponents
            self.exponents.resize(var, E::zero());
            unimplemented!()
        }
    }

    /// Check if the polynomial is sorted and has only non-zero coefficients
    pub fn check_consistency(&self) {
        assert_eq!(self.coefficients.len(), self.nterms);
        assert_eq!(self.exponents.len(), self.nterms * self.nvars);

        for c in &self.coefficients {
            if F::is_zero(c) {
                panic!("Inconsistent polynomial (0 coefficient): {}", self);
            }
        }

        for t in 1..self.nterms {
            match O::cmp(self.exponents(t), self.exponents(t - 1)) {
                Ordering::Equal => panic!("Inconsistent polynomial (equal monomials): {}", self),
                Ordering::Less => panic!(
                    "Inconsistent polynomial (wrong monomial ordering): {}",
                    self
                ),
                Ordering::Greater => {}
            }
        }
    }

    /// Append a monomial to the back. It merges with the last monomial if the
    /// exponents are equal.
    #[inline]
    pub fn append_monomial_back(&mut self, coefficient: F::Element, exponents: &[E]) {
        if F::is_zero(&coefficient) {
            return;
        }

        if self.nterms > 0 && exponents == self.last_exponents() {
            self.field
                .add_assign(&mut self.coefficients[self.nterms - 1], &coefficient);

            if F::is_zero(&self.coefficients[self.nterms - 1]) {
                self.coefficients.pop();
                self.exponents.truncate((self.nterms - 1) * self.nvars);
                self.nterms -= 1;
            }
        } else {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
            self.nterms += 1;
        }
    }

    /// Appends a monomial to the polynomial.
    pub fn append_monomial(&mut self, coefficient: F::Element, exponents: &[E]) {
        if F::is_zero(&coefficient) {
            return;
        }
        if self.nvars != exponents.len() {
            panic!(
                "nvars mismatched: got {}, expected {}",
                exponents.len(),
                self.nvars
            );
        }

        // should we append to the back?
        if self.nterms == 0 || O::cmp(self.last_exponents(), exponents).is_lt() {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
            self.nterms += 1;
            return;
        }

        if O::cmp(self.exponents(0), exponents).is_gt() {
            self.coefficients.insert(0, coefficient);
            self.exponents.splice(0..0, exponents.iter().cloned());
            self.nterms += 1;
            return;
        }

        // Binary search to find the insert-point.
        let mut l = 0;
        let mut r = self.nterms;

        while l <= r {
            let m = (l + r) / 2;
            let c = O::cmp(exponents, self.exponents(m)); // note the reversal

            match c {
                Ordering::Equal => {
                    // Add the two coefficients.
                    self.field
                        .add_assign(&mut self.coefficients[m], &coefficient);
                    if F::is_zero(&self.coefficients[m]) {
                        // The coefficient becomes zero. Remove this monomial.
                        self.coefficients.remove(m);
                        let i = m * self.nvars;
                        self.exponents.splice(i..i + self.nvars, Vec::new());
                        self.nterms -= 1;
                    }
                    return;
                }
                Ordering::Greater => {
                    l = m + 1;

                    if l == self.nterms {
                        self.coefficients.push(coefficient);
                        self.exponents.extend_from_slice(exponents);
                        self.nterms += 1;
                        return;
                    }
                }
                Ordering::Less => {
                    if m == 0 {
                        self.coefficients.insert(0, coefficient);
                        self.exponents.splice(0..0, exponents.iter().cloned());
                        self.nterms += 1;
                        return;
                    }

                    r = m - 1;
                }
            }
        }

        self.coefficients.insert(l, coefficient);
        let i = l * self.nvars;
        self.exponents.splice(i..i, exponents.iter().cloned());
        self.nterms += 1;
    }

    /// Take the derivative of the polynomial w.r.t the variable `var`.
    pub fn derivative(&self, var: usize) -> Self {
        debug_assert!(var < self.nvars);

        let mut res = MultivariatePolynomial::new_from(self, Some(self.nterms));

        let mut exp = vec![E::zero(); self.nvars];
        for x in self {
            if x.exponents[var] > E::zero() {
                exp.copy_from_slice(x.exponents);
                let pow = exp[var].to_u32() as u64;
                exp[var] = exp[var] - E::one();
                res.append_monomial(self.field.mul(x.coefficient, &self.field.nth(pow)), &exp);
            }
        }

        res
    }
}

impl<F: Ring + fmt::Debug, E: Exponent + fmt::Debug, O: MonomialOrder> fmt::Debug
    for MultivariatePolynomial<F, E, O>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for monomial in self {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{{ {:?}, {:?} }}",
                monomial.coefficient, monomial.exponents
            )?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + Display, E: Exponent, O: MonomialOrder> Display for MultivariatePolynomial<F, E, O> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        let mut is_first_term = true;
        for monomial in self {
            let mut is_first_factor = true;
            if self.field.is_one(monomial.coefficient) {
                if !is_first_term {
                    write!(f, "+")?;
                }
            } else if monomial.coefficient.eq(&self.field.neg(&self.field.one())) {
                write!(f, "-")?;
            } else {
                if is_first_term {
                    self.field
                        .fmt_display(monomial.coefficient, None, true, f)?;
                } else {
                    write!(
                        f,
                        "{:+}",
                        RingPrinter {
                            ring: &self.field,
                            element: monomial.coefficient,
                            state: None,
                            in_product: true
                        }
                    )?;
                }
                is_first_factor = false;
            }
            is_first_term = false;
            for (i, e) in monomial.exponents.iter().enumerate() {
                if e.is_zero() {
                    continue;
                }
                if is_first_factor {
                    is_first_factor = false;
                } else {
                    write!(f, "*")?;
                }
                write!(f, "x{}", i)?;
                if e.to_u32() != 1 {
                    write!(f, "^{}", e)?;
                }
            }
            if is_first_factor {
                write!(f, "1")?;
            }
        }
        if is_first_term {
            write!(f, "0")?;
        }

        Display::fmt(&self.field, f)
    }
}

impl<F: Ring + PartialEq, E: Exponent, O: MonomialOrder> PartialEq
    for MultivariatePolynomial<F, E, O>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.nvars != other.nvars {
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
        if self.nterms != other.nterms {
            return false;
        }
        self.exponents.eq(&other.exponents) && self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring + Eq, E: Exponent, O: MonomialOrder> Eq for MultivariatePolynomial<F, E, O> {}

impl<F: Ring, E: Exponent, O: MonomialOrder> Add for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        debug_assert_eq!(self.field, other.field);
        debug_assert!(other.var_map.is_none() || self.var_map == other.var_map); // TODO: remove?

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        if self.nvars != other.nvars {
            panic!("nvars mismatched");
        }

        // Merge the two polynomials, which are assumed to be already sorted.

        let mut new_coefficients = vec![self.field.zero(); self.nterms + other.nterms];
        let mut new_exponents: Vec<E> = vec![E::zero(); self.nvars * (self.nterms + other.nterms)];
        let mut new_nterms = 0;
        let mut i = 0;
        let mut j = 0;

        macro_rules! insert_monomial {
            ($source:expr, $index:expr) => {
                mem::swap(
                    &mut new_coefficients[new_nterms],
                    &mut $source.coefficients[$index],
                );

                new_exponents[new_nterms * $source.nvars..(new_nterms + 1) * $source.nvars]
                    .clone_from_slice($source.exponents($index));
                new_nterms += 1;
            };
        }

        while i < self.nterms && j < other.nterms {
            let c = O::cmp(self.exponents(i), other.exponents(j));
            match c {
                Ordering::Less => {
                    insert_monomial!(self, i);
                    i += 1;
                }
                Ordering::Greater => {
                    insert_monomial!(other, j);
                    j += 1;
                }
                Ordering::Equal => {
                    self.field
                        .add_assign(&mut self.coefficients[i], &other.coefficients[j]);
                    if !F::is_zero(&self.coefficients[i]) {
                        insert_monomial!(self, i);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.nterms {
            insert_monomial!(self, i);
            i += 1;
        }

        while j < other.nterms {
            insert_monomial!(other, j);
            j += 1;
        }

        new_coefficients.truncate(new_nterms);
        new_exponents.truncate(self.nvars * new_nterms);

        Self {
            coefficients: new_coefficients,
            exponents: new_exponents,
            nterms: new_nterms,
            nvars: self.nvars,
            field: self.field,
            var_map: self.var_map,
            _phantom: PhantomData,
        }
    }
}

impl<'a, 'b, F: Ring, E: Exponent, O: MonomialOrder> Add<&'a MultivariatePolynomial<F, E, O>>
    for &'b MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn add(self, other: &'a MultivariatePolynomial<F, E, O>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Sub for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, 'b, F: Ring, E: Exponent, O: MonomialOrder> Sub<&'a MultivariatePolynomial<F, E, O>>
    for &'b MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn sub(self, other: &'a MultivariatePolynomial<F, E, O>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Neg for MultivariatePolynomial<F, E, O> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(c);
        }
        self
    }
}

impl<'a, 'b, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &'b MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    #[inline]
    fn mul(self, other: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        self.heap_mul(other)
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    #[inline]
    fn mul(self, other: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        self.heap_mul(other)
    }
}

impl<'a, 'b, F: EuclideanDomain, E: Exponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &'b MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(self, other: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        self.divides(other)
            .unwrap_or_else(|| panic!("No clean division of {} by {}", self, other))
    }
}

impl<'a, F: EuclideanDomain, E: Exponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(
        self: MultivariatePolynomial<F, E, LexOrder>,
        other: &'a MultivariatePolynomial<F, E, LexOrder>,
    ) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Change the monomial order of the polynomial from `O` to `ON`.
    pub fn reorder<ON: MonomialOrder>(&self) -> MultivariatePolynomial<F, E, ON> {
        let mut sorted_index: Vec<_> = (0..self.nterms).collect();
        sorted_index.sort_by(|a, b| ON::cmp(self.exponents(*a), self.exponents(*b)));

        let coefficients: Vec<_> = sorted_index
            .iter()
            .map(|i| self.coefficients[*i].clone())
            .collect();
        let exponents: Vec<_> = sorted_index
            .iter()
            .map(|i| self.exponents(*i))
            .flatten()
            .cloned()
            .collect();

        MultivariatePolynomial {
            coefficients,
            exponents,
            nterms: self.nterms,
            nvars: self.nvars,
            field: self.field.clone(),
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Multiply every coefficient with `other`.
    pub fn mul_coeff(mut self, other: F::Element) -> Self {
        for c in &mut self.coefficients {
            self.field.mul_assign(c, &other);
        }

        for i in (0..self.nterms).rev() {
            if F::is_zero(&self.coefficients[i]) {
                self.coefficients.remove(i);
                self.exponents.drain(i * self.nvars..(i + 1) * self.nvars);
                self.nterms -= 1;
            }
        }

        self
    }

    /// Map a coefficient using the function `f`.
    pub fn map_coeff<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        f: T,
        field: U,
    ) -> MultivariatePolynomial<U, E, O> {
        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        let mut exponents = Vec::with_capacity(self.exponents.len());

        for m in self.into_iter() {
            let nc = f(m.coefficient);
            if !U::is_zero(&nc) {
                coefficients.push(nc);
                exponents.extend(m.exponents);
            }
        }

        MultivariatePolynomial {
            nterms: coefficients.len(),
            coefficients,
            exponents,
            nvars: self.nvars,
            field,
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Add `exponents` to every exponent.
    pub fn mul_exp(mut self, exponents: &[E]) -> Self {
        debug_assert_eq!(self.nvars, exponents.len());
        for e in self.exponents.chunks_mut(self.nvars) {
            for (e1, e2) in e.iter_mut().zip(exponents) {
                *e1 = e1.checked_add(e2).expect("overflow in adding exponents");
            }
        }

        self
    }

    #[inline]
    pub fn max_coeff(&self) -> &F::Element {
        self.coefficients.last().unwrap()
    }

    #[inline]
    pub fn max_exp(&self) -> &[E] {
        if self.coefficients.is_empty() {
            panic!("Cannot get max exponent of empty polynomial");
        }

        &self.exponents[(self.nterms - 1) * self.nvars..self.nterms * self.nvars]
    }

    /// Add a new monomial with coefficient `other` and exponent one.
    pub fn add_monomial(mut self, other: F::Element) -> Self {
        let nvars = self.nvars;
        self.append_monomial(other, &vec![E::zero(); nvars]);
        self
    }

    #[inline]
    fn mul_monomial(self, coefficient: &F::Element, exponents: &[E]) -> Self {
        self.mul_coeff(coefficient.clone()).mul_exp(exponents)
    }

    /// Get the degree of the variable `x`.
    /// This operation is O(n).
    pub fn degree(&self, x: usize) -> E {
        let mut max = E::zero();
        for e in self.exponents.iter().skip(x).step_by(self.nvars) {
            if max < *e {
                max = *e;
            }
        }
        max
    }

    // Get the highest degree of a variable in the leading monomial.
    pub fn ldegree(&self, v: usize) -> E {
        if self.is_zero() {
            return E::zero();
        }
        self.last_exponents()[v]
    }

    /// Get the highest degree of the leading monomial.
    pub fn ldegree_max(&self) -> E {
        if self.is_zero() {
            return E::zero();
        }
        *self.last_exponents().iter().max().unwrap_or(&E::zero())
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        if self.is_zero() {
            return self.field.one();
        }
        self.coefficients.last().unwrap().clone()
    }

    /// Convert the coefficient from the current field to a finite field.
    /// TODO: deprecate, use map_coeff
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> MultivariatePolynomial<FiniteField<UField>, E>
    where
        F::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        let mut exponents = Vec::with_capacity(self.exponents.len());

        for m in self.into_iter() {
            let nc = m.coefficient.to_finite_field(&field);
            if !FiniteField::<UField>::is_zero(&nc) {
                coefficients.push(nc);
                exponents.extend(m.exponents);
            }
        }

        MultivariatePolynomial {
            nterms: coefficients.len(),
            coefficients,
            exponents,
            nvars: self.nvars,
            field: field.clone(),
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }

    /// Perform self % var^pow.
    pub fn mod_var(&self, var: usize, pow: E) -> Self {
        let mut m = self.new_from(None);
        for t in self.into_iter() {
            if t.exponents[var] < pow {
                m.append_monomial(t.coefficient.clone(), t.exponents);
            }
        }
        m
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Get the leading coefficient under a given variable ordering.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_varorder(&self, vars: &[usize]) -> F::Element {
        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff();
        }

        let mut highest = vec![E::zero(); self.nvars];
        let mut highestc = &self.field.zero();

        'nextmon: for m in self.into_iter() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            highestc = m.coefficient;
        }
        debug_assert!(!F::is_zero(highestc));
        highestc.clone()
    }

    /// Get the leading coefficient of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_lcoeff(&self, x: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        let d = self.degree(x);
        let mut lcoeff = self.new_from(None);

        if self.coefficients.is_empty() {
            return lcoeff;
        }

        if d == E::zero() {
            return self.clone();
        }

        let mut e = vec![E::zero(); self.nvars];
        for t in self {
            if t.exponents[x] == d {
                e.copy_from_slice(t.exponents);
                e[x] = E::zero();
                lcoeff.append_monomial(t.coefficient.clone(), &e);
            }
        }

        lcoeff
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables except the last variable `n`.
    pub fn lcoeff_last(&self, n: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return MultivariatePolynomial::zero(&self.field);
        }
        // the last variable should have the least sorting priority,
        // so the last term should still be the lcoeff
        let last = self.last_exponents();

        let mut res = self.new_from(None);
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars];

        for t in (0..self.nterms()).rev() {
            if (0..self.nvars - 1).all(|i| self.exponents(t)[i] == last[i] || i == n) {
                e[n] = self.exponents(t)[n];
                res.append_monomial(self.coefficients[t].clone(), &e);
                e[n] = E::zero();
            } else {
                break;
            }
        }

        res
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables with order as described in `vars` except the last variable in `vars`.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_last_varorder(&self, vars: &[usize]) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return MultivariatePolynomial::zero(&self.field);
        }

        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff_last(*vars.last().unwrap());
        }

        let (vars, lastvar) = vars.split_at(vars.len() - 1);

        let mut highest = vec![E::zero(); self.nvars];
        let mut indices = Vec::with_capacity(10);

        'nextmon: for (i, m) in self.into_iter().enumerate() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            indices.clear();
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            indices.push(i);
        }

        let mut res = self.new_from(None);
        let mut e = vec![E::zero(); self.nvars];
        for i in indices {
            e[lastvar[0]] = self.exponents(i)[lastvar[0]];
            res.append_monomial(self.coefficients[i].clone(), &e);
            e[lastvar[0]] = E::zero();
        }
        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    ///
    /// Note that the polynomial `var_map` is not updated.
    pub fn rearrange(
        &self,
        order: &[usize],
        inverse: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut new_exp = vec![E::zero(); self.nterms * self.nvars];
        for (e, er) in new_exp
            .chunks_mut(self.nvars)
            .zip(self.exponents.chunks(self.nvars))
        {
            for x in 0..order.len() {
                if !inverse {
                    e[x] = er[order[x]];
                } else {
                    e[order[x]] = er[x];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * self.nvars..(i + 1) * self.nvars]);

        let mut res = self.new_from(Some(self.nterms));

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * self.nvars..(i + 1) * self.nvars],
            );
        }

        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The order may contain `None`, to signal unmapped indices. This operation
    /// allows the polynomial to grow in size.
    ///
    /// Note that the polynomial `var_map` is not updated.
    pub fn rearrange_with_growth(
        &self,
        order: &[Option<usize>],
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut new_exp = vec![E::zero(); self.nterms * order.len()];
        for (e, er) in new_exp
            .chunks_mut(order.len())
            .zip(self.exponents.chunks(self.nvars))
        {
            for x in 0..order.len() {
                if let Some(v) = order[x] {
                    e[x] = er[v];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * order.len()..(i + 1) * order.len()]);

        let mut res = MultivariatePolynomial::new(
            order.len(),
            &self.field,
            Some(self.nterms),
            self.var_map.as_ref().map(|x| x.as_slice()),
        );

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * order.len()..(i + 1) * order.len()],
            );
        }

        res
    }

    /// Replace a variable `n` in the polynomial by an element from
    /// the ring `v`.
    pub fn replace(&self, n: usize, v: &F::Element) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut res = self.new_from(Some(self.nterms));
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars];

        for t in self {
            if t.exponents[n] == E::zero() {
                res.append_monomial(t.coefficient.clone(), t.exponents);
                continue;
            }

            let c = self.field.mul(
                t.coefficient,
                &self.field.pow(v, t.exponents[n].to_u32() as u64),
            );

            e.copy_from_slice(t.exponents);
            e[n] = E::zero();
            res.append_monomial(c, &e);
        }

        res
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// the ring.
    pub fn replace_all_except(
        &self,
        v: usize,
        r: &[(usize, F::Element)],
        cache: &mut [Vec<F::Element>],
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut tm: HashMap<E, F::Element> = HashMap::new();

        for t in self {
            let mut c = t.coefficient.clone();
            for (n, vv) in r {
                let p = t.exponents[*n].to_u32() as usize;
                if p > 0 {
                    if p < cache[*n].len() {
                        if F::is_zero(&cache[*n][p]) {
                            cache[*n][p] = self.field.pow(vv, p as u64);
                        }

                        self.field.mul_assign(&mut c, &cache[*n][p]);
                    } else {
                        self.field.mul_assign(&mut c, &self.field.pow(vv, p as u64));
                    }
                }
            }

            tm.entry(t.exponents[v])
                .and_modify(|e| self.field.add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = self.new_from(None);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm {
            e[v] = k;
            res.append_monomial(c, &e);
            e[v] = E::zero();
        }

        res
    }

    /// Create a univariate polynomial coefficient list out of a multivariate polynomial.
    /// The output is sorted in the degree.
    // TODO: allow a MultivariatePolynomial as a coefficient
    pub fn to_univariate_polynomial_list(
        &self,
        x: usize,
    ) -> Vec<(MultivariatePolynomial<F, E, LexOrder>, E)> {
        if self.coefficients.is_empty() {
            return vec![];
        }

        // get maximum degree for variable x
        let mut maxdeg = E::zero();
        for t in 0..self.nterms {
            let d = self.exponents(t)[x];
            if d > maxdeg {
                maxdeg = d;
            }
        }

        // construct the coefficient per power of x
        let mut result = vec![];
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars];
        for d in 0..maxdeg.to_u32() + 1 {
            // TODO: add bounds estimate
            let mut a = self.new_from(None);
            for t in 0..self.nterms {
                if self.exponents(t)[x].to_u32() == d {
                    for (i, ee) in self.exponents(t).iter().enumerate() {
                        e[i] = *ee;
                    }
                    e[x] = E::zero();
                    a.append_monomial(self.coefficients[t].clone(), &e);
                }
            }

            if !a.is_zero() {
                result.push((a, E::from_u32(d)));
            }
        }

        result
    }

    /// Split the polynomial as a polynomial in `xs` if include is true,
    /// else excluding `xs`.
    pub fn to_multivariate_polynomial_list(
        &self,
        xs: &[usize],
        include: bool,
    ) -> HashMap<SmallVec<[E; INLINED_EXPONENTS]>, MultivariatePolynomial<F, E, LexOrder>> {
        if self.coefficients.is_empty() {
            return HashMap::new();
        }

        let mut tm: HashMap<
            SmallVec<[E; INLINED_EXPONENTS]>,
            MultivariatePolynomial<F, E, LexOrder>,
        > = HashMap::new();
        let mut e_not_in_xs = smallvec![E::zero(); self.nvars];
        let mut e_in_xs = smallvec![E::zero(); self.nvars];
        for t in self {
            for (i, ee) in t.exponents.iter().enumerate() {
                e_not_in_xs[i] = *ee;
                e_in_xs[i] = E::zero();
            }

            for x in xs {
                e_in_xs[*x] = e_not_in_xs[*x];
                e_not_in_xs[*x] = E::zero();
            }

            if include {
                tm.entry(e_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_not_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::new_from_monomial(
                            self,
                            t.coefficient.clone(),
                            e_not_in_xs.to_vec(),
                        )
                    });
            } else {
                tm.entry(e_not_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::new_from_monomial(
                            self,
                            t.coefficient.clone(),
                            e_in_xs.to_vec(),
                        )
                    });
            }
        }

        tm
    }

    /// Multiplication for multivariate polynomials using a custom variation of the heap method
    /// described in "Sparse polynomial division using a heap" by Monagan, Pearce (2011) and using
    /// the sorting described in "Sparse Polynomial Powering Using Heaps".
    /// It uses a heap to obtain the next monomial of the result in an ordered fashion.
    /// Additionally, this method uses a hashmap with the monomial exponent as a key and a vector of all pairs
    /// of indices in `self` and `other` that have that monomial exponent when multiplied together.
    /// When a multiplication of two monomials is considered, its indices are added to the hashmap,
    /// but they are only added to the heap if the monomial exponent is new. As a result, the heap
    /// only has new monomials, and by taking (and removing) the corresponding entry from the hashmap, all
    /// monomials that have that exponent can be summed. Then, new monomials combinations are added that
    /// should be considered next as they are smaller than the current monomial.
    pub fn heap_mul(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.nterms == 0 || other.nterms == 0 {
            return MultivariatePolynomial::new_from(self, None);
        }

        if self.nterms == 1 {
            return other
                .clone()
                .mul_monomial(&self.coefficients[0], &self.exponents);
        }

        if other.nterms == 1 {
            return self
                .clone()
                .mul_monomial(&other.coefficients[0], &other.exponents);
        }

        // place the smallest polynomial first, as this is faster
        // in the heap algorithm
        if self.nterms > other.nterms {
            return other.heap_mul(self);
        }

        // use a special routine if the exponents can be packed into a u64
        let mut pack_u8 = true;
        if self.nvars <= 8
            && (0..self.nvars).all(|i| {
                let deg = self
                    .degree(i)
                    .to_u32()
                    .saturating_add(other.degree(i).to_u32());
                if deg > 255 {
                    pack_u8 = false;
                }

                deg <= 255 || self.nvars <= 4 && deg <= 65535
            })
        {
            return self.heap_mul_packed_exp(other, pack_u8);
        }

        let mut res = self.new_from(Some(self.nterms));

        let mut cache: BTreeMap<Vec<E>, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<Vec<E>>> = BinaryHeap::with_capacity(self.nterms);

        let monom: Vec<E> = self
            .exponents(0)
            .iter()
            .zip(other.exponents(0))
            .map(|(e1, e2)| *e1 + *e2)
            .collect();
        cache.insert(monom.clone(), vec![(0, 0)]);
        h.push(Reverse(monom));

        let mut m_cache: Vec<E> = vec![E::zero(); self.nvars];

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; other.nterms];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; other.nterms];
        in_heap[0] = true;

        while !h.is_empty() {
            let cur_mon = h.pop().unwrap();

            let mut coefficient = self.field.zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.field.add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &other.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms && (j == 0 || merged_index[j - 1] > i + 1) {
                    for ((m, e1), e2) in m_cache
                        .iter_mut()
                        .zip(self.exponents(i + 1))
                        .zip(other.exponents(j))
                    {
                        *m = *e1 + *e2;
                    }

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((i + 1, j));
                    } else {
                        h.push(Reverse(m_cache.clone())); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < other.nterms && !in_heap[j + 1] {
                    for ((m, e1), e2) in m_cache
                        .iter_mut()
                        .zip(self.exponents(i))
                        .zip(other.exponents(j + 1))
                    {
                        *m = *e1 + *e2;
                    }

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((i, j + 1));
                    } else {
                        h.push(Reverse(m_cache.clone())); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !F::is_zero(&coefficient) {
                res.coefficients.push(coefficient);
                res.exponents.extend_from_slice(&cur_mon.0);
                res.nterms += 1;
            }
        }
        res
    }

    /// Heap multiplication, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 65535 if there are four or fewer variables,
    /// or 255 if there are 8 or fewer variables.
    pub fn heap_mul_packed_exp(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
        pack_u8: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut res = self.new_from(Some(self.nterms));

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents
                .chunks(self.nvars)
                .map(|c| E::pack(c))
                .collect()
        } else {
            self.exponents
                .chunks(self.nvars)
                .map(|c| E::pack_u16(c))
                .collect()
        };
        let pack_b: Vec<_> = if pack_u8 {
            other
                .exponents
                .chunks(self.nvars)
                .map(|c| E::pack(c))
                .collect()
        } else {
            other
                .exponents
                .chunks(self.nvars)
                .map(|c| E::pack_u16(c))
                .collect()
        };

        let mut cache: BTreeMap<u64, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(self.nterms);

        let monom: u64 = pack_a[0] + pack_b[0];
        cache.insert(monom, vec![(0, 0)]);
        h.push(Reverse(monom));

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; other.nterms];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; other.nterms];
        in_heap[0] = true;

        while !h.is_empty() {
            let cur_mon = h.pop().unwrap();

            let mut coefficient = self.field.zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.field.add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &other.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = pack_a[i + 1] + pack_b[j];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(Reverse(m)); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < other.nterms && !in_heap[j + 1] {
                    let m = pack_a[i] + pack_b[j + 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(Reverse(m)); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !F::is_zero(&coefficient) {
                res.coefficients.push(coefficient);
                let len = res.exponents.len();

                res.exponents.resize(len + self.nvars, E::zero());

                if pack_u8 {
                    E::unpack(cur_mon.0, &mut res.exponents[len..len + self.nvars]);
                } else {
                    E::unpack_u16(cur_mon.0, &mut res.exponents[len..len + self.nvars]);
                }
                res.nterms += 1;
            }
        }
        res
    }

    /// Synthetic division for univariate polynomials, where `div` is monic.
    // TODO: create UnivariatePolynomial?
    pub fn quot_rem_univariate_monic(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        debug_assert_eq!(div.lcoeff(), self.field.one());
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let mut dividendpos = self.nterms - 1; // work from the back

        let mut q = self.new_from(Some(self.nterms));
        let mut r = self.new_from(None);

        // determine the variable
        let mut var = 0;
        for (i, x) in self.last_exponents().iter().enumerate() {
            if !x.is_zero() {
                var = i;
                break;
            }
        }

        let m = div.ldegree_max();
        let mut pow = self.ldegree_max();

        loop {
            // find the power in the dividend if it exists
            let mut coeff = loop {
                if self.exponents(dividendpos)[var] == pow {
                    break self.coefficients[dividendpos].clone();
                }
                if dividendpos == 0 || self.exponents(dividendpos)[var] < pow {
                    break self.field.zero();
                }
                dividendpos -= 1;
            };

            let mut qindex = 0; // starting from highest
            let mut bindex = 0; // starting from lowest
            while bindex < div.nterms && qindex < q.nterms {
                while bindex + 1 < div.nterms
                    && div.exponents(bindex)[var] + q.exponents(qindex)[var] < pow
                {
                    bindex += 1;
                }

                if div.exponents(bindex)[var] + q.exponents(qindex)[var] == pow {
                    self.field.sub_mul_assign(
                        &mut coeff,
                        &div.coefficients[bindex],
                        &q.coefficients[qindex],
                    );
                }

                qindex += 1;
            }

            if !F::is_zero(&coeff) {
                // can the division be performed? if not, add to rest
                // TODO: refactor
                let (quot, div) = if pow >= m {
                    (coeff, true)
                } else {
                    (coeff, false)
                };

                if div {
                    q.coefficients.push(quot);
                    q.exponents.resize((q.nterms + 1) * q.nvars, E::zero());
                    q.exponents[q.nterms * q.nvars + var] = pow - m;
                    q.nterms += 1;
                } else {
                    r.coefficients.push(quot);
                    r.exponents.resize((r.nterms + 1) * r.nvars, E::zero());
                    r.exponents[r.nterms * r.nvars + var] = pow;
                    r.nterms += 1;
                }
            }

            if pow.is_zero() {
                break;
            }

            pow = pow - E::one();
        }

        q.reverse();
        r.reverse();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({})/({}): q={}, r={}", self, div, q, r);
            }
        }

        (q, r)
    }

    /// Shift a variable `var` to `var+shift`.
    pub fn shift_var(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_u32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);

        let mut v = vec![self.new_from(None); d + 1];
        for (x_poly, p) in y_poly {
            v[p.to_u32() as usize] = x_poly;
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1].clone().mul_coeff(shift.clone());
            }
        }

        let mut poly = self.new_from(None);
        for (i, mut v) in v.into_iter().enumerate() {
            for x in v.exponents.chunks_mut(self.nvars) {
                x[var] = E::from_u32(i as u32);
            }

            for m in &v {
                poly.append_monomial(m.coefficient.clone(), m.exponents);
            }
        }

        poly
    }
}

impl<F: EuclideanDomain, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
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

    pub fn divides(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if self.is_zero() {
            return Some(self.clone());
        }

        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if (0..self.nvars).any(|v| self.degree(v) < div.degree(v)) {
            return None;
        }

        let (a, b) = self.quot_rem(div, true);
        if b.nterms == 0 {
            Some(a)
        } else {
            None
        }
    }

    /// Compute the remainder `self % div`.
    pub fn rem(&self, div: &MultivariatePolynomial<F, E, LexOrder>) -> Self {
        self.quot_rem(div, false).1
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    pub fn quot_rem(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.is_one() {
            return (self.clone(), self.new_from(None));
        }

        if self.nterms == div.nterms {
            if self == div {
                return (
                    self.new_from_constant(self.field.one()),
                    self.new_from(None),
                );
            }

            // check if one is a multiple of the other
            let (q, r) = self.field.quot_rem(&self.lcoeff(), &div.lcoeff());

            if F::is_zero(&r)
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| t1.exponents == t2.exponents)
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| &self.field.mul(t2.coefficient, &q) == t1.coefficient)
            {
                return (self.new_from_constant(q), self.new_from(None));
            }
        }

        if div.nterms == 1 {
            let mut q = self.clone();
            let dive = div.to_monomial_view(0);

            for ee in q.exponents.chunks_mut(q.nvars) {
                for (e1, e2) in ee.iter_mut().zip(dive.exponents) {
                    if *e1 >= *e2 {
                        *e1 = *e1 - *e2;
                    } else {
                        return (MultivariatePolynomial::new_from(self, None), self.clone());
                    }
                }
            }

            for c in &mut q.coefficients {
                let (quot, rem) = q.field.quot_rem(c, dive.coefficient);
                *c = quot;
                if !F::is_zero(&rem) {
                    // TODO: support upgrade to a RationalField
                    return (MultivariatePolynomial::new_from(self, None), self.clone());
                }
            }

            return (q, self.new_from(None));
        }

        let mut pack_u8 = true;
        if self.nvars <= 8
            && (0..self.nvars).all(|i| {
                let deg = self.degree(i).to_u32();
                if deg > 127 {
                    pack_u8 = false;
                }

                deg <= 127 || self.nvars <= 4 && deg <= 32767
            })
        {
            self.heap_division_packed_exp(div, abort_on_remainder, pack_u8)
        } else {
            self.heap_division(div, abort_on_remainder)
        }
    }

    /// Heap division for multivariate polynomials, using a cache so that only unique
    /// monomial exponents appear in the heap.
    /// Reference: "Sparse polynomial division using a heap" by Monagan, Pearce (2011)
    pub fn heap_division(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let mut q = self.new_from(Some(self.nterms));
        let mut r = self.new_from(None);

        let mut div_monomial_in_heap = vec![false; div.nterms];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms];

        let mut cache: BTreeMap<Vec<E>, Vec<(usize, usize, bool)>> = BTreeMap::new();

        let mut h: BinaryHeap<Vec<E>> = BinaryHeap::with_capacity(self.nterms);
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = vec![];

        let mut m = vec![E::zero(); div.nvars];
        let mut m_cache = vec![E::zero(); div.nvars];
        let mut c;

        let mut k = 0;
        while !h.is_empty() || k < self.nterms {
            if k < self.nterms && (h.is_empty() || self.exponents_back(k) >= h.peek().unwrap()) {
                for (s, e) in m.iter_mut().zip(self.exponents_back(k)) {
                    *s = *e;
                }

                c = self.coefficient_back(k).clone();
                k += 1;
            } else {
                for (s, e) in m.iter_mut().zip(h.peek().unwrap().as_slice()) {
                    *s = *e;
                }
                c = self.field.zero();
            }

            if let Some(monomial) = h.peek() {
                if &m == monomial {
                    h.pop().unwrap();

                    for (i, j, next_in_divisor) in cache.remove(&m).unwrap() {
                        // TODO: use fraction-free routines
                        self.field.sub_mul_assign(
                            &mut c,
                            &q.coefficients[i],
                            div.coefficient_back(j),
                        );

                        if next_in_divisor && j + 1 < div.nterms {
                            // quotient heap product
                            for ((m, e1), e2) in m_cache
                                .iter_mut()
                                .zip(q.exponents(i))
                                .zip(div.exponents_back(j + 1))
                            {
                                *m = *e1 + *e2;
                            }

                            // TODO: make macro
                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, true));
                            } else {
                                h.push(m_cache.clone()); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, true));
                                    cache.insert(m_cache.clone(), qq);
                                } else {
                                    cache.insert(m_cache.clone(), vec![(i, j + 1, true)]);
                                }
                            }
                        } else if !next_in_divisor {
                            merged_index_of_div_monomial_in_quotient[j] = i + 1;

                            if i + 1 < q.nterms
                                && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                            {
                                for ((m, e1), e2) in m_cache
                                    .iter_mut()
                                    .zip(q.exponents(i + 1))
                                    .zip(div.exponents_back(j))
                                {
                                    *m = *e1 + *e2;
                                }

                                if let Some(e) = cache.get_mut(&m_cache) {
                                    e.push((i + 1, j, false));
                                } else {
                                    h.push(m_cache.clone()); // only add when new
                                    if let Some(mut qq) = q_cache.pop() {
                                        qq.push((i + 1, j, false));
                                        cache.insert(m_cache.clone(), qq);
                                    } else {
                                        cache.insert(m_cache.clone(), vec![(i + 1, j, false)]);
                                    }
                                }
                            } else {
                                div_monomial_in_heap[j] = false;
                            }

                            if j + 1 < div.nterms && !div_monomial_in_heap[j + 1] {
                                for ((m, e1), e2) in m_cache
                                    .iter_mut()
                                    .zip(q.exponents(i))
                                    .zip(div.exponents_back(j + 1))
                                {
                                    *m = *e1 + *e2;
                                }

                                if let Some(e) = cache.get_mut(&m_cache) {
                                    e.push((i, j + 1, false));
                                } else {
                                    h.push(m_cache.clone()); // only add when new

                                    if let Some(mut qq) = q_cache.pop() {
                                        qq.push((i, j + 1, false));
                                        cache.insert(m_cache.clone(), qq);
                                    } else {
                                        cache.insert(m_cache.clone(), vec![(i, j + 1, false)]);
                                    }
                                }

                                div_monomial_in_heap[j + 1] = true;
                            }
                        }
                    }
                }
            }

            if !F::is_zero(&c) && div.last_exponents().iter().zip(&m).all(|(ge, me)| me >= ge) {
                let (quot, rem) = self.field.quot_rem(&c, &div.lcoeff());
                if !F::is_zero(&rem) {
                    // TODO: support upgrade to a RationalField
                    if abort_on_remainder {
                        r.nterms += 1;
                        return (q, r);
                    } else {
                        return (MultivariatePolynomial::new_from(self, None), self.clone());
                    }
                }

                q.coefficients.push(quot);
                q.exponents.extend(
                    div.last_exponents()
                        .iter()
                        .zip(&m)
                        .map(|(ge, me)| *me - *ge),
                );
                q.nterms += 1;

                if div.nterms == 1 {
                    continue;
                }

                for ((m, e1), e2) in m_cache
                    .iter_mut()
                    .zip(q.last_exponents())
                    .zip(div.exponents_back(1))
                {
                    *m = *e1 + *e2;
                }

                if q.nterms < div.nterms {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms - 1, 1, true));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms - 1, 1, true));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms - 1, 1, true)]);
                        }
                    }
                } else if q.nterms >= div.nterms {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms - 1, 1, false));
                        } else {
                            h.push(m_cache.clone()); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms - 1, 1, false));
                                cache.insert(m_cache.clone(), qq);
                            } else {
                                cache.insert(m_cache.clone(), vec![(q.nterms - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms - 1, 1, false));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms - 1, 1, false));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms - 1, 1, false)]);
                        }
                    }
                }
            } else if !F::is_zero(&c) {
                r.nterms += 1;

                if abort_on_remainder {
                    // return something, the only guarantee is that r.nterms > 0
                    return (q, r);
                } else {
                    r.coefficients.push(c);
                    r.exponents.extend(&m);
                }
            }
        }

        // q and r have the highest monomials first
        q.reverse();
        r.reverse();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({})/({}): q={}, r={}", self, div, q, r);
            }
        }

        (q, r)
    }

    /// Heap division, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 32767 if there are 5 or fewer variables,
    /// or 127 if there are 8 or fewer variables, such that the last bit per byte can
    /// be used to check for subtraction overflow, serving as a division test.
    pub fn heap_division_packed_exp(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
        pack_u8: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let mut q = self.new_from(Some(self.nterms));
        let mut r = self.new_from(None);

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents
                .chunks(self.nvars)
                .map(|c| E::pack(c))
                .collect()
        } else {
            self.exponents
                .chunks(self.nvars)
                .map(|c| E::pack_u16(c))
                .collect()
        };
        let pack_div: Vec<_> = if pack_u8 {
            div.exponents
                .chunks(div.nvars)
                .map(|c| E::pack(c))
                .collect()
        } else {
            div.exponents
                .chunks(div.nvars)
                .map(|c| E::pack_u16(c))
                .collect()
        };

        let mut div_monomial_in_heap = vec![false; div.nterms];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms];

        let mut cache: BTreeMap<u64, Vec<(usize, usize, bool)>> = BTreeMap::new();

        #[inline(always)]
        fn divides(a: u64, b: u64, pack_u8: bool) -> Option<u64> {
            let d = a.overflowing_sub(b).0;
            if pack_u8 && (d & 9259542123273814144u64 == 0)
                || !pack_u8 && (d & 9223512776490647552u64 == 0)
            {
                Some(d)
            } else {
                None
            }
        }

        let mut h: BinaryHeap<u64> = BinaryHeap::with_capacity(self.nterms);
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = Vec::with_capacity(self.nterms);

        let mut m;
        let mut m_cache;
        let mut c;

        let mut q_exp = Vec::with_capacity(self.nterms);

        let mut k = 0;
        while !h.is_empty() || k < self.nterms {
            if k < self.nterms
                && (h.is_empty() || pack_a[self.nterms - k - 1] >= *h.peek().unwrap())
            {
                m = pack_a[self.nterms - k - 1];

                c = self.coefficient_back(k).clone();

                k += 1;
            } else {
                m = *h.peek().unwrap();
                c = self.field.zero();
            }

            if let Some(monomial) = h.peek() {
                if &m == monomial {
                    h.pop().unwrap();

                    for (i, j, next_in_divisor) in cache.remove(&m).unwrap() {
                        // TODO: use fraction-free routines
                        self.field.sub_mul_assign(
                            &mut c,
                            &q.coefficients[i],
                            div.coefficient_back(j),
                        );

                        if next_in_divisor && j + 1 < div.nterms {
                            // quotient heap product
                            m_cache = q_exp[i] + pack_div[div.nterms - (j + 1) - 1];

                            // TODO: make macro
                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, true));
                            } else {
                                h.push(m_cache); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, true));
                                    cache.insert(m_cache, qq);
                                } else {
                                    cache.insert(m_cache, vec![(i, j + 1, true)]);
                                }
                            }
                        } else if !next_in_divisor {
                            merged_index_of_div_monomial_in_quotient[j] = i + 1;

                            if i + 1 < q.nterms
                                && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                            {
                                m_cache = q_exp[i + 1] + pack_div[div.nterms - j - 1];

                                if let Some(e) = cache.get_mut(&m_cache) {
                                    e.push((i + 1, j, false));
                                } else {
                                    h.push(m_cache); // only add when new
                                    if let Some(mut qq) = q_cache.pop() {
                                        qq.push((i + 1, j, false));
                                        cache.insert(m_cache, qq);
                                    } else {
                                        cache.insert(m_cache, vec![(i + 1, j, false)]);
                                    }
                                }
                            } else {
                                div_monomial_in_heap[j] = false;
                            }

                            if j + 1 < div.nterms && !div_monomial_in_heap[j + 1] {
                                m_cache = q_exp[i] + pack_div[div.nterms - (j + 1) - 1];

                                if let Some(e) = cache.get_mut(&m_cache) {
                                    e.push((i, j + 1, false));
                                } else {
                                    h.push(m_cache); // only add when new

                                    if let Some(mut qq) = q_cache.pop() {
                                        qq.push((i, j + 1, false));
                                        cache.insert(m_cache, qq);
                                    } else {
                                        cache.insert(m_cache, vec![(i, j + 1, false)]);
                                    }
                                }

                                div_monomial_in_heap[j + 1] = true;
                            }
                        }
                    }
                }
            }

            let q_e = divides(m, pack_div[pack_div.len() - 1], pack_u8);
            if !F::is_zero(&c) && q_e.is_some() {
                let (quot, rem) = self.field.quot_rem(&c, &div.lcoeff());
                if !F::is_zero(&rem) {
                    // TODO: support upgrade to a RationalField
                    if abort_on_remainder {
                        r.nterms += 1;
                        return (q, r);
                    } else {
                        return (MultivariatePolynomial::new_from(self, None), self.clone());
                    }
                }

                let q_e = q_e.unwrap();
                q.coefficients.push(quot);
                let len = q.exponents.len();
                q.exponents.resize(len + self.nvars, E::zero());

                if pack_u8 {
                    E::unpack(q_e, &mut q.exponents[len..len + self.nvars]);
                } else {
                    E::unpack_u16(q_e, &mut q.exponents[len..len + self.nvars]);
                }
                q.nterms += 1;
                q_exp.push(q_e);

                if div.nterms == 1 {
                    continue;
                }

                m_cache = q_exp.last().unwrap() + pack_div[pack_div.len() - 2];

                if q.nterms < div.nterms {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms - 1, 1, true));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms - 1, 1, true));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms - 1, 1, true)]);
                        }
                    }
                } else if q.nterms >= div.nterms {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms - 1, 1, false));
                        } else {
                            h.push(m_cache); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms - 1, 1, false));
                                cache.insert(m_cache, qq);
                            } else {
                                cache.insert(m_cache, vec![(q.nterms - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms - 1, 1, false));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms - 1, 1, false));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms - 1, 1, false)]);
                        }
                    }
                }
            } else if !F::is_zero(&c) {
                r.nterms += 1;

                if abort_on_remainder {
                    // return something, the only guarantee is that r.nterms > 0
                    return (q, r);
                } else {
                    r.coefficients.push(c);
                    let len = r.exponents.len();
                    r.exponents.resize(len + self.nvars, E::zero());

                    if pack_u8 {
                        E::unpack(m, &mut r.exponents[len..len + self.nvars]);
                    } else {
                        E::unpack_u16(m, &mut r.exponents[len..len + self.nvars]);
                    }
                }
            }
        }

        // q and r have the highest monomials first
        q.reverse();
        r.reverse();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({})/({}): q={}, r={}", self, div, q, r);
            }
        }

        (q, r)
    }
}

impl<F: Field, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Make the polynomial monic, i.e., make the leading coefficient `1` by
    /// multiplying all monomials with `1/lcoeff`.
    pub fn make_monic(self) -> Self {
        if self.lcoeff() != self.field.one() {
            let ci = self.field.inv(&self.lcoeff());
            self.mul_coeff(ci)
        } else {
            self
        }
    }
}

impl<F: Field, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Optimized division routine for univariate polynomials over a field, which
    /// makes the divisor monic first.
    pub fn quot_rem_univariate(
        &self,
        div: &mut MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.nterms == 1 {
            // calculate inverse once
            let inv = self.field.inv(&div.coefficients[0]);

            if div.is_constant() {
                let mut q = self.clone();
                for c in &mut q.coefficients {
                    self.field.mul_assign(c, &inv);
                }

                return (q, self.new_from(None));
            }

            let mut q = self.new_from(Some(self.nterms));
            let mut r = self.new_from(None);
            let dive = div.exponents(0);

            for m in self.into_iter() {
                if m.exponents.iter().zip(dive).all(|(a, b)| a >= b) {
                    q.coefficients.push(self.field.mul(m.coefficient, &inv));

                    for (ee, ed) in m.exponents.iter().zip(dive) {
                        q.exponents.push(*ee - *ed);
                    }
                    q.nterms += 1;
                } else {
                    r.coefficients.push(m.coefficient.clone());
                    r.exponents.extend(m.exponents);
                    r.nterms += 1;
                }
            }
            return (q, r);
        }

        // normalize the lcoeff to 1 to prevent a costly inversion
        if !self.field.is_one(&div.lcoeff()) {
            let o = div.lcoeff();
            let inv = self.field.inv(&div.lcoeff());

            for c in &mut div.coefficients {
                self.field.mul_assign(c, &inv);
            }

            let mut res = self.quot_rem_univariate_monic(div);

            for c in &mut res.0.coefficients {
                self.field.mul_assign(c, &inv);
            }

            for c in &mut div.coefficients {
                self.field.mul_assign(c, &o);
            }

            return res;
        }

        self.quot_rem_univariate_monic(div)
    }

    /// Compute self^n % m where m is a polynomial
    pub fn exp_mod_univariate(&self, mut n: Integer, m: &mut Self) -> Self {
        if n.is_zero() {
            return Self::new_from_constant(self, self.field.one());
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.quot_rem_univariate(m).1;
        let mut y = Self::new_from_constant(self, self.field.one());
        while !n.is_one() {
            if (&n % &Integer::Natural(2)).is_one() {
                y = (&y * &x).quot_rem_univariate(m).1;
                n -= &Integer::one();
            }

            x = (&x * &x).quot_rem_univariate(m).1;
            n /= 2;
        }

        (x * &y).quot_rem_univariate(m).1
    }

    /// Compute `(g, s, t)` where `self * s + other * t = g`
    /// by means of the extended Euclidean algorithm.
    pub fn eea_univariate(&self, other: &Self) -> (Self, Self, Self) {
        let mut r0 = self.clone().make_monic();
        let mut r1 = other.clone().make_monic();
        let mut s0 = self.new_from_constant(self.field.inv(&self.lcoeff()));
        let mut s1 = self.new_from(None);
        let mut t0 = self.new_from(None);
        let mut t1 = self.new_from_constant(self.field.inv(&other.lcoeff()));

        while !r1.is_zero() {
            let (q, r) = r0.quot_rem_univariate(&mut r1);
            let a = self.field.inv(&r.lcoeff());
            (r1, r0) = (r.mul_coeff(a.clone()), r1);
            (s1, s0) = ((s0 - &q * &s1).mul_coeff(a.clone()), s1);
            (t1, t0) = ((t0 - q * &t1).mul_coeff(a), t1);
        }

        (r0, s0, t0)
    }

    /// Compute `(s1,...,n2)` where `A0 * s0 + ... + An * sn = g`
    /// where `Ai = prod(polys[j], j != i)`
    /// by means of the extended Euclidean algorithm.
    ///
    /// The `polys` must be pairwise co-prime.
    pub fn diophantine_univariate(polys: &mut [Self], b: &Self) -> Vec<Self> {
        let mut cur = polys.last().unwrap().clone();
        let mut a = vec![cur.clone()];
        for x in polys[1..].iter().rev().skip(1) {
            cur = cur * x;
            a.push(cur.clone());
        }
        a.reverse();

        let mut ss = vec![];
        let mut cur_s = b.clone();
        for (p, aa) in polys.iter_mut().zip(&mut a) {
            let (g, s, t) = p.eea_univariate(aa);
            debug_assert!(g.is_one());
            let new_s = (t * &cur_s).quot_rem_univariate(p).1;
            ss.push(new_s);
            cur_s = (s * &cur_s).quot_rem_univariate(aa).1;
        }

        ss.push(cur_s);
        ss
    }

    /// Find a rational fraction `n(x)/d(x)`, the Pade approximant,
    ///  such that `d(x)*self-n(x)=0 mod x^(deg_n+deg_d+1)` and
    /// `deg(d(x)) <= deg_d` and `deg(n(x) <= deg_n` using the extended Euclidean algorithm.
    pub fn rational_approximant_univariate(&self, deg_n: u32, deg_d: u32) -> Option<(Self, Self)>
    where
        F: PolynomialGCD<E>,
    {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return Some((self.clone(), self.new_from_constant(self.field.one())));
        };

        let mut exp = self.last_exponents().to_vec();
        exp[var] = E::from_u32(deg_n) + E::from_u32(deg_d) + E::one();
        let mut v0 = self.new_from_monomial(self.field.one(), exp);
        let mut v1 = self.new_from(None);

        let mut w0 = self.clone();
        let mut w1 = self.new_from_constant(self.field.one());

        while w0.degree(var).to_u32() > deg_n {
            let (q, r) = v0.quot_rem_univariate(&mut w0);
            (w1, v1) = (v1 - q * &w1, w1);
            (v0, w0) = (w0, r);
        }

        // TODO: normalize denominator?
        let r = MultivariatePolynomial::gcd(&w0, &w1);

        Some((w0 / &r, w1 / &r))
    }

    /// Shift a variable `var` to `var+shift`, using an optimized routine that
    /// uses a power cache. If working in a finite field, the characteristic
    /// should be larger than the degree of the polynomial.
    pub fn shift_var_cached(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_u32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);
        let mut sample_powers = Vec::with_capacity(d + 1);
        let mut accum = self.field.one();

        sample_powers.push(self.field.one());
        let d = d as usize;
        for _ in 0..d {
            self.field.mul_assign(&mut accum, shift);
            sample_powers.push(accum.clone());
        }
        let mut v = vec![self.new_from(None); d + 1];
        for (x_poly, p) in y_poly {
            let i = p.to_u32() as usize;
            v[i] = x_poly.mul_coeff(sample_powers[i].clone());
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1];
            }
        }

        let mut poly = self.new_from(None);
        let mut accum_inv = self.field.one();
        let sample_point_inv = self.field.inv(&shift);
        for (i, mut v) in v.into_iter().enumerate() {
            v = v.mul_coeff(accum_inv.clone());

            for x in v.exponents.chunks_mut(self.nvars) {
                x[var] = E::from_u32(i as u32);
            }

            for m in &v {
                poly.append_monomial(m.coefficient.clone(), m.exponents);
            }

            self.field.mul_assign(&mut accum_inv, &sample_point_inv);
        }

        poly
    }
}

impl<E: Exponent> Into<MultivariatePolynomial<RationalField, E>>
    for &MultivariatePolynomial<IntegerRing, E>
{
    fn into(self) -> MultivariatePolynomial<RationalField, E> {
        MultivariatePolynomial {
            coefficients: self.coefficients.iter().map(|x| x.into()).collect(),
            exponents: self.exponents.clone(),
            nterms: self.nterms,
            nvars: self.nvars,
            field: RationalField,
            var_map: self.var_map.clone(),
            _phantom: PhantomData,
        }
    }
}

/// View object for a term in a multivariate polynomial.
#[derive(Copy, Clone, Debug)]
pub struct MonomialView<'a, F: 'a + Ring, E: 'a + Exponent> {
    pub coefficient: &'a F::Element,
    pub exponents: &'a [E],
}

/// Iterator over terms in a multivariate polynomial.
pub struct MonomialViewIterator<'a, F: Ring, E: Exponent, O: MonomialOrder> {
    poly: &'a MultivariatePolynomial<F, E, O>,
    index: usize,
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> Iterator for MonomialViewIterator<'a, F, E, O> {
    type Item = MonomialView<'a, F, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.poly.nterms {
            None
        } else {
            let view = MonomialView {
                coefficient: &self.poly.coefficients[self.index],
                exponents: self.poly.exponents(self.index),
            };
            self.index += 1;
            Some(view)
        }
    }
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> IntoIterator
    for &'a MultivariatePolynomial<F, E, O>
{
    type Item = MonomialView<'a, F, E>;
    type IntoIter = MonomialViewIterator<'a, F, E, O>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            poly: self,
            index: 0,
        }
    }
}
