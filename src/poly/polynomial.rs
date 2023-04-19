use ahash::{HashMap, HashMapExt};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;
use std::fmt::Display;
use std::mem;
use std::ops::{Add, Mul, Neg, Sub};

use crate::rings::{EuclideanDomain, Ring};

use super::monomial::{Monomial, MonomialView};
use super::Exponent;
use smallvec::smallvec;

/// Multivariate polynomial with a sparse degree and variable dense representation.
#[derive(Clone, Hash)]
pub struct MultivariatePolynomial<F: Ring, E: Exponent> {
    // Data format: the i-th monomial is stored as coefficients[i] and
    // exponents[i * nvars .. (i + 1) * nvars]. Keep coefficients.len() == nterms and
    // exponents.len() == nterms * nvars. Terms are always expanded and sorted by the exponents via
    // cmp_exponents().
    pub coefficients: Vec<F::Element>,
    pub exponents: Vec<E>,
    pub nterms: usize,
    pub nvars: usize,
    pub field: F,
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E> {
    /// Constructs a zero polynomial.
    #[inline]
    pub fn new(field: F) -> Self {
        Self {
            coefficients: Vec::new(),
            exponents: Vec::new(),
            nterms: 0,
            nvars: 0,
            field,
        }
    }

    /// Constructs a zero polynomial with the given number of variables.
    #[inline]
    pub fn with_nvars(nvars: usize, field: F) -> Self {
        Self {
            coefficients: Vec::new(),
            exponents: Vec::new(),
            nterms: 0,
            nvars,
            field,
        }
    }

    /// Constructs a zero polynomial with the given number of variables and capacity.
    #[inline]
    pub fn with_nvars_and_capacity(nvars: usize, cap: usize, field: F) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap),
            exponents: Vec::with_capacity(cap * nvars),
            nterms: 0,
            nvars,
            field,
        }
    }

    /// Constructs a constant polynomial with the given number of variables.
    #[inline]
    pub fn from_constant_with_nvars(constant: F::Element, nvars: usize, field: F) -> Self {
        if F::is_zero(&constant) {
            return Self::with_nvars(nvars, field);
        }
        Self {
            coefficients: vec![constant],
            exponents: vec![E::zero(); nvars],
            nterms: 1,
            nvars,
            field,
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn from_monomial(coefficient: F::Element, exponents: Vec<E>, field: F) -> Self {
        if F::is_zero(&coefficient) {
            return Self::with_nvars(exponents.len(), field);
        }
        Self {
            coefficients: vec![coefficient],
            nvars: exponents.len(),
            exponents,
            nterms: 1,
            field,
        }
    }

    /// Get the ith monomial
    pub fn to_monomial(&self, i: usize) -> Monomial<F, E> {
        assert!(i < self.nterms);

        Monomial::new(
            self.coefficients[i].clone(),
            self.exponents(i).iter().cloned().collect(),
            self.field,
        )
    }

    /// Get the ith monomial
    pub fn to_monomial_view(&self, i: usize) -> MonomialView<F, E> {
        assert!(i < self.nterms);

        MonomialView {
            coefficient: &self.coefficients[i],
            exponents: &self.exponents(i),
        }
    }

    #[inline]
    pub fn zero(field: F) -> Self {
        Self::new(field)
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.nterms == 0
    }

    #[inline]
    pub fn one(field: F) -> Self {
        MultivariatePolynomial::from_constant_with_nvars(field.one(), 0, field)
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.nterms == 1
            && self.field.is_one(&self.coefficients[0])
            && self.exponents.iter().all(|x| x.is_zero())
    }

    /// F::Elementeturns the number of terms in the polynomial.
    #[inline]
    pub fn nterms(&self) -> usize {
        return self.nterms;
    }

    /// F::Elementeturns the number of variables in the polynomial.
    #[inline]
    pub fn nvars(&self) -> usize {
        return self.nvars;
    }

    /// F::Elementeturns true if the polynomial is constant.
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

    /// F::Elementeturns the slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents(&self, index: usize) -> &[E] {
        &self.exponents[index * self.nvars..(index + 1) * self.nvars]
    }

    pub fn last_exponents(&self) -> &[E] {
        assert!(self.nterms > 0);
        &self.exponents[(self.nterms - 1) * self.nvars..self.nterms * self.nvars]
    }

    /// F::Elementeturns the mutable slice for the exponents of the specified monomial.
    #[inline]
    fn exponents_mut(&mut self, index: usize) -> &mut [E] {
        &mut self.exponents[index * self.nvars..(index + 1) * self.nvars]
    }

    /// F::Elementeturns the number of variables in the polynomial.
    #[inline]
    pub fn clear(&mut self) {
        self.nterms = 0;
        self.coefficients.clear();
        self.exponents.clear();
    }

    /// F::Elementeverse the monomial ordering in-place.
    fn reverse(&mut self) {
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

    /// Compares exponent vectors of two monomials.
    #[inline]
    fn cmp_exponents(a: &[E], b: &[E]) -> Ordering {
        debug_assert!(a.len() == b.len());
        // TODO: Introduce other term orders.
        a.cmp(b)
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
            match MultivariatePolynomial::<F, E>::cmp_exponents(
                self.exponents(t),
                &self.exponents(t - 1),
            ) {
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
        if self.nterms == 0 || self.last_exponents() < exponents {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
            self.nterms += 1;
            return;
        }

        // Binary search to find the insert-point.
        let mut l = 0;
        let mut r = self.nterms;

        while l <= r {
            let m = (l + r) / 2;
            let c = Self::cmp_exponents(exponents, self.exponents(m)); // note the reversal

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
}

impl<F: Ring + fmt::Debug, E: Exponent + fmt::Debug> fmt::Debug for MultivariatePolynomial<F, E> {
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

impl<F: Ring + Display, E: Exponent> Display for MultivariatePolynomial<F, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut is_first_term = true;
        for monomial in self {
            let mut is_first_factor = true;
            if self.field.is_one(&monomial.coefficient) {
                if !is_first_term {
                    write!(f, "+")?;
                }
            } else if monomial.coefficient.eq(&self.field.neg(&self.field.one())) {
                write!(f, "-")?;
            } else {
                if is_first_term {
                    write!(f, "{}", monomial.coefficient)?;
                } else {
                    write!(f, "+{}", monomial.coefficient)?;
                }
                is_first_factor = false;
            }
            is_first_term = false;
            for (i, e) in monomial.exponents.into_iter().enumerate() {
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

impl<F: Ring + PartialEq, E: Exponent> PartialEq for MultivariatePolynomial<F, E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.nvars != other.nvars {
            if self.is_zero() && other.is_zero() {
                // Both are 0.
                return true;
            }
            if self.is_zero() || other.is_zero() {
                // One of them is 0.
                return false;
            }
            panic!("nvars mismatched");
        }
        if self.nterms != other.nterms {
            return false;
        }
        self.exponents.eq(&other.exponents) && self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring + Eq, E: Exponent> Eq for MultivariatePolynomial<F, E> {}

impl<F: Ring, E: Exponent> Add for MultivariatePolynomial<F, E> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        debug_assert_eq!(self.field, other.field);

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

        let mut new_coefficients = vec![F::zero(); self.nterms + other.nterms];
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
            let c = Self::cmp_exponents(self.exponents(i), other.exponents(j));
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
        }
    }
}

impl<F: Ring, E: Exponent> Sub for MultivariatePolynomial<F, E> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<F: Ring, E: Exponent> Neg for MultivariatePolynomial<F, E> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(c);
        }
        self
    }
}

impl<F: Ring, E: Exponent> Mul for MultivariatePolynomial<F, E> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        self * &other
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E>>
    for MultivariatePolynomial<F, E>
{
    type Output = Self;

    fn mul(self, other: &'a MultivariatePolynomial<F, E>) -> Self::Output {
        debug_assert_eq!(self.field, other.field);
        if self.is_zero() {
            return Self::with_nvars(other.nvars, self.field);
        }
        if other.is_zero() {
            return Self::with_nvars(self.nvars, self.field);
        }
        if self.nvars != other.nvars {
            panic!("nvars mismatched");
        }
        // TODO: this is a quick implementation. To be improved.
        let mut new_poly = Self::with_nvars(self.nvars, self.field);
        for m in other {
            let p = self.clone().mul_monomial(m.coefficient, m.exponents);
            new_poly = new_poly.add(p);
        }
        new_poly
    }
}

// FIXME: cannot implement Add<F::Element> because F::Element could be MultivariatePolynomial<F, E>
impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E> {
    /// Multiply every coefficient with `other`.
    pub fn mul_coeff(mut self, other: F::Element) -> Self {
        for c in &mut self.coefficients {
            self.field.mul_assign(c, &other);
        }
        self
    }

    /// Add a new monomial with coefficient `other` and exponent one.
    pub fn add_monomial(mut self, other: F::Element) -> Self {
        let nvars = self.nvars;
        self.append_monomial(other, &vec![E::zero(); nvars]);
        self
    }

    #[inline]
    fn mul_monomial(mut self, coefficient: &F::Element, exponents: &[E]) -> Self {
        debug_assert_eq!(self.nvars, exponents.len());
        debug_assert!(self.nterms > 0);
        debug_assert!(!F::is_zero(coefficient));
        for c in &mut self.coefficients {
            self.field.mul_assign(c, coefficient);
        }
        for i in 0..self.nterms {
            let ee = self.exponents_mut(i);
            for (e1, e2) in ee.iter_mut().zip(exponents) {
                *e1 = e1.checked_add(e2).expect("overflow in adding exponents");
            }
        }
        self
    }

    /// Get the degree of the variable `x`.
    /// This operation is O(n).
    pub fn degree(&self, x: usize) -> E {
        let mut max = E::zero();
        for t in 0..self.nterms {
            if max < self.exponents(t)[x] {
                max = self.exponents(t)[x];
            }
        }
        max
    }

    // Get the highest degree of a variable in the leading monomial.
    pub fn ldegree(&self, v: usize) -> E {
        if self.is_zero() {
            return E::zero();
        }
        self.last_exponents()[v].clone()
    }

    /// Get the highest degree of the leading monomial.
    pub fn ldegree_max(&self) -> E {
        if self.is_zero() {
            return E::zero();
        }
        self.last_exponents()
            .iter()
            .max()
            .unwrap_or(&E::zero())
            .clone()
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        if self.is_zero() {
            return F::zero();
        }
        self.coefficients.last().unwrap().clone()
    }

    /// Get the leading coefficient under a given variable ordering.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_varorder(&self, vars: &[usize]) -> F::Element {
        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff();
        }

        let mut highest = vec![E::zero(); self.nvars];
        let mut highestc = &F::zero();

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
            highestc = &m.coefficient;
        }
        debug_assert!(!F::is_zero(highestc));
        highestc.clone()
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables except the last variable `n`.
    pub fn lcoeff_last(&self, n: usize) -> MultivariatePolynomial<F, E> {
        if self.is_zero() {
            return MultivariatePolynomial::zero(self.field);
        }
        // the last variable should have the least sorting priority,
        // so the last term should still be the lcoeff
        let last = self.last_exponents();

        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut e = vec![E::zero(); self.nvars];
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
    pub fn lcoeff_last_varorder(&self, vars: &[usize]) -> MultivariatePolynomial<F, E> {
        if self.is_zero() {
            return MultivariatePolynomial::zero(self.field);
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

        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut e = vec![E::zero(); self.nvars];
        for i in indices {
            e[lastvar[0]] = self.exponents(i)[lastvar[0]];
            res.append_monomial(self.coefficients[i].clone(), &e);
            e[lastvar[0]] = E::zero();
        }
        res
    }

    /// Change the order of the variables in the polynomial, using `varmap`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    pub fn rearrange(&self, varmap: &[usize], inverse: bool) -> MultivariatePolynomial<F, E> {
        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut newe = vec![E::zero(); self.nvars];
        for m in self.into_iter() {
            for x in 0..varmap.len() {
                if !inverse {
                    newe[x] = m.exponents[varmap[x]];
                } else {
                    newe[varmap[x]] = m.exponents[x];
                }
            }

            res.append_monomial(m.coefficient.clone(), &newe);
        }
        res
    }

    /// Replace a variable `n' in the polynomial by an element from
    /// the ring `v'.
    pub fn replace(&self, n: usize, v: F::Element) -> MultivariatePolynomial<F, E> {
        let mut res =
            MultivariatePolynomial::with_nvars_and_capacity(self.nvars, self.nterms, self.field);
        let mut e = vec![E::zero(); self.nvars];
        for t in 0..self.nterms {
            let c = self.field.mul(
                &self.coefficients[t],
                &self.field.pow(&v, self.exponents(t)[n].to_u32() as u64),
            );

            for (i, ee) in self.exponents(t).iter().enumerate() {
                e[i] = *ee;
            }

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
    ) -> MultivariatePolynomial<F, E> {
        let mut tm: HashMap<E, F::Element> = HashMap::new();

        for t in 0..self.nterms {
            let mut c = self.coefficients[t].clone();
            for (n, vv) in r {
                let p = self.exponents(t)[*n].to_u32() as usize;
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

            tm.entry(self.exponents(t)[v])
                .and_modify(|e| self.field.add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm {
            e[v] = k;
            res.append_monomial(c, &e);
            e[v] = E::zero();
        }

        res
    }

    /// Create a univariate polynomial out of a multivariate one.
    /// TODO: allow a MultivariatePolynomial as a coefficient
    pub fn to_univariate_polynomial_list(
        &self,
        x: usize,
    ) -> Vec<(MultivariatePolynomial<F, E>, u32)> {
        if self.coefficients.is_empty() {
            return vec![];
        }

        // get maximum degree for variable x
        let mut maxdeg = 0;
        for t in 0..self.nterms {
            let d = self.exponents(t)[x].to_u32();
            if d > maxdeg {
                maxdeg = d.clone();
            }
        }

        // construct the coefficient per power of x
        let mut result = vec![];
        let mut e = vec![E::zero(); self.nvars];
        for d in 0..maxdeg + 1 {
            // TODO: add bounds estimate
            let mut a = MultivariatePolynomial::with_nvars(self.nvars, self.field);
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
                result.push((a, d));
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
    ) -> HashMap<Vec<E>, MultivariatePolynomial<F, E>> {
        if self.coefficients.is_empty() {
            return HashMap::new();
        }

        let mut tm: HashMap<Vec<E>, MultivariatePolynomial<F, E>> = HashMap::new();
        let mut e = vec![E::zero(); self.nvars];
        let mut me = vec![E::zero(); self.nvars];
        for t in 0..self.nterms {
            for (i, ee) in self.exponents(t).iter().enumerate() {
                e[i] = *ee;
                me[i] = E::zero();
            }

            for x in xs {
                me[*x] = e[*x].clone();
                e[*x] = E::zero();
            }

            if include {
                let add = match tm.get_mut(&me) {
                    Some(x) => {
                        x.append_monomial(self.coefficients[t].clone(), &e);
                        false
                    }
                    None => true,
                };

                if add {
                    tm.insert(
                        me.clone(),
                        // TODO: add nterms estimate
                        MultivariatePolynomial::from_monomial(
                            self.coefficients[t].clone(),
                            e.clone(),
                            self.field,
                        ),
                    );
                }
            } else {
                let add = match tm.get_mut(&e) {
                    Some(x) => {
                        x.append_monomial(self.coefficients[t].clone(), &me);
                        false
                    }
                    None => true,
                };

                if add {
                    tm.insert(
                        e.clone(),
                        MultivariatePolynomial::from_monomial(
                            self.coefficients[t].clone(),
                            me.clone(),
                            self.field,
                        ),
                    );
                }
            }
        }

        tm
    }
}

impl<F: EuclideanDomain, E: Exponent> MultivariatePolynomial<F, E> {
    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return F::zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            c = self.field.gcd(&c, cc);
        }
        c
    }

    /// Synthetic division for univariate polynomials
    // TODO: create UnivariatePolynomial?
    pub fn synthetic_division(
        &self,
        div: &MultivariatePolynomial<F, E>,
    ) -> (MultivariatePolynomial<F, E>, MultivariatePolynomial<F, E>) {
        let mut dividendpos = self.nterms - 1; // work from the back
        let norm = div.coefficients.last().unwrap();

        let mut q = MultivariatePolynomial::<F, E>::with_nvars_and_capacity(
            self.nvars,
            self.nterms,
            self.field,
        );
        let mut r = MultivariatePolynomial::<F, E>::with_nvars(self.nvars, self.field);

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
                    break F::zero();
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
                    self.field.add_assign(
                        &mut coeff,
                        &self.field.neg(
                            &self
                                .field
                                .mul(&div.coefficients[bindex], &q.coefficients[qindex]),
                        ),
                    );
                }

                qindex += 1;
            }

            if !F::is_zero(&coeff) {
                // can the division be performed? if not, add to rest

                let (quot, div) = if pow >= m {
                    let (quot, rem) = self.field.quot_rem(&coeff, &norm);
                    if F::is_zero(&rem) {
                        (quot, true)
                    } else {
                        (coeff, false)
                    }
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

            pow = pow - E::from_u32(1);
        }

        q.reverse();
        r.reverse();

        #[cfg(debug_assertions)]
        {
            if !(q.clone() * div.clone() + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({})/({}): q={}, r={}", self, div, q, r);
            }
        }

        (q, r)
    }

    /// Long division for multivarariate polynomial.
    /// If the ring `F` is not a field, and the coefficient does not cleanly divide,
    /// the division is stopped and the current quotient and rest term are returned.
    #[allow(dead_code)]
    fn long_division(
        &self,
        div: &MultivariatePolynomial<F, E>,
    ) -> (MultivariatePolynomial<F, E>, MultivariatePolynomial<F, E>) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        let mut q = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut r = self.clone();
        let divdeg = div.last_exponents();

        while !r.is_zero()
            && r.last_exponents()
                .iter()
                .zip(divdeg.iter())
                .all(|(re, de)| re >= de)
        {
            let (tc, rem) = self.field.quot_rem(
                &r.coefficients.last().unwrap(),
                &div.coefficients.last().unwrap(),
            );

            if !F::is_zero(&rem) {
                // long division failed, return the term as the rest
                return (q, r);
            }

            let tp: Vec<E> = r
                .last_exponents()
                .iter()
                .zip(divdeg.iter())
                .map(|(e1, e2)| e1.clone() - e2.clone())
                .collect();

            q.append_monomial(tc.clone(), &tp);
            r = r - div.clone().mul_monomial(&tc, &tp);
        }

        (q, r)
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    pub fn divmod(
        &self,
        div: &MultivariatePolynomial<F, E>,
    ) -> (MultivariatePolynomial<F, E>, MultivariatePolynomial<F, E>) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.is_one() {
            return (
                self.clone(),
                MultivariatePolynomial::with_nvars(self.nvars, self.field),
            );
        }

        if div.nterms == 1 {
            let mut q = MultivariatePolynomial::with_nvars_and_capacity(
                self.nvars,
                self.nterms,
                self.field,
            );
            let mut r = MultivariatePolynomial::with_nvars(self.nvars, self.field);
            let dive = div.to_monomial(0);

            for i in 0..self.nterms {
                let mut m = self.to_monomial(i);
                if m.try_div_assign(&dive) {
                    q.coefficients.push(m.coefficient);
                    q.exponents.extend_from_slice(&m.exponents);
                    q.nterms += 1;
                } else {
                    r.coefficients.push(m.coefficient);
                    r.exponents.extend_from_slice(&m.exponents);
                    r.nterms += 1;
                }
            }

            return (q, r);
        }

        // TODO: use other algorithm for univariate div
        self.heap_division(div)
    }

    /// Heap division for multivariate polynomials.
    /// Reference: "Polynomial Division Using Dynamic Arrays, Heaps, and Packed Exponent Vectors" by
    /// Monagan, Pearce (2007)
    /// TODO: implement "Sparse polynomial division using a heap" by Monagan, Pearce (2011)
    fn heap_division(
        &self,
        div: &MultivariatePolynomial<F, E>,
    ) -> (MultivariatePolynomial<F, E>, MultivariatePolynomial<F, E>) {
        let mut q =
            MultivariatePolynomial::with_nvars_and_capacity(self.nvars, self.nterms, self.field);
        let mut r = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut s = div.nterms - 1; // index viewed from the back
        let mut h = BinaryHeap::with_capacity(div.nterms);
        let mut t = Monomial {
            coefficient: F::zero(),
            exponents: smallvec![E::zero(); self.nvars],
            field: self.field,
        };

        let lm = Monomial {
            coefficient: div.lcoeff(),
            exponents: div.last_exponents().iter().cloned().collect(),
            field: self.field,
        };

        h.push((
            Monomial {
                coefficient: self.field.neg(&self.lcoeff()),
                exponents: div.last_exponents().iter().cloned().collect(),
                field: self.field,
            },
            0,                  // index in self/div viewed from the back (due to our poly ordering)
            usize::max_value(), // index in q, we set it out of bounds to signal we need new terms from f
        ));

        while h.len() > 0 {
            t.coefficient = F::zero();
            for e in t.exponents.iter_mut() {
                *e = E::zero();
            }

            loop {
                let (x, i, j) = h.pop().unwrap();

                if F::is_zero(&t.coefficient) {
                    t = -x;
                } else {
                    t = t - x;
                }

                // TODO: recycle memory from x for new element in h?
                if j == usize::max_value() {
                    if i + 1 < self.nterms {
                        // we need a new term from self
                        h.push((
                            Monomial {
                                coefficient: self
                                    .field
                                    .neg(&self.coefficients[self.nterms - i - 2]),
                                exponents: self
                                    .exponents(self.nterms - i - 2)
                                    .iter()
                                    .cloned()
                                    .collect(),
                                field: self.field,
                            },
                            i + 1,
                            j,
                        ));
                    }
                } else if j + 1 < q.nterms {
                    h.push((
                        q.to_monomial(j + 1) * div.to_monomial_view(div.nterms - i - 1),
                        i,
                        j + 1,
                    ));
                } else {
                    s += 1;
                }

                if h.len() == 0 || t != h.peek().unwrap().0 {
                    break;
                }
            }
            if !F::is_zero(&t.coefficient) && t.try_div_assign(&lm) {
                // add t to q
                q.coefficients.push(t.coefficient.clone());
                q.exponents.extend_from_slice(&t.exponents);
                q.nterms += 1;

                for i in div.nterms - s - 1..div.nterms - 1 {
                    h.push((div.to_monomial(i) * &t, div.nterms - i - 1, q.nterms - 1));
                }

                s = 0;
            } else {
                // add t to r
                if !F::is_zero(&t.coefficient) {
                    r.exponents.extend_from_slice(&t.exponents);
                    r.coefficients.push(t.coefficient);
                    r.nterms += 1;
                }
            }
        }

        // q and r have the highest monomials first
        q.reverse();
        r.reverse();

        #[cfg(debug_assertions)]
        {
            if !(q.clone() * div.clone() + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({})/({}): q={}, r={}", self, div, q, r);
            }
        }

        (q, r)
    }
}
