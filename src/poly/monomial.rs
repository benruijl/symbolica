use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use smallvec::SmallVec;

use crate::rings::{EuclideanDomain, Field, Ring};

use super::polynomial::MultivariatePolynomial;
use super::{Exponent, INLINED_EXPONENTS};

/// A monomial class. Equality and ordering is only considering the exponent.
#[derive(Debug, Clone)]
pub struct Monomial<F: Ring, E: Exponent> {
    pub coefficient: F::Element,
    pub exponents: SmallVec<[E; INLINED_EXPONENTS]>,
    pub field: F,
}

impl<F: Ring, E: Exponent> Monomial<F, E> {
    #[inline]
    pub fn new(
        coefficient: F::Element,
        exponents: SmallVec<[E; INLINED_EXPONENTS]>,
        field: F,
    ) -> Monomial<F, E> {
        Monomial {
            coefficient,
            exponents,
            field,
        }
    }
}

impl<F: EuclideanDomain, E: Exponent> Monomial<F, E> {
    #[inline]
    pub fn divides(&self, other: &Monomial<F, E>) -> bool {
        self.exponents
            .iter()
            .zip(&other.exponents)
            .all(|(a, b)| a >= b)
            && (self.field.is_one(&other.coefficient)
                || F::is_zero(&self.field.rem(&self.coefficient, &other.coefficient)))
    }

    pub fn try_div_assign(&mut self, other: &Monomial<F, E>) -> bool {
        if self
            .exponents
            .iter()
            .zip(&other.exponents)
            .any(|(a, b)| a < b)
        {
            return false;
        }

        let (q, r) = self.field.quot_rem(&self.coefficient, &other.coefficient);
        if !F::is_zero(&r) {
            return false;
        }

        for (ee, er) in self.exponents.iter_mut().zip(&other.exponents) {
            *ee = *ee - *er;
        }

        self.coefficient = q;
        true
    }
}

impl<'a, F: Field, E: Exponent> Div<&'a Monomial<F, E>> for Monomial<F, E> {
    type Output = Self;

    fn div(mut self, other: &Self) -> Self::Output {
        for (ee, er) in self.exponents.iter_mut().zip(&other.exponents) {
            if *ee < *er {
                panic!("Cannot divide monomials with exponents smaller than the divisor");
            }
            *ee = *ee - *er;
        }

        self.field
            .div_assign(&mut self.coefficient, &other.coefficient);
        self
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a Monomial<F, E>> for Monomial<F, E> {
    type Output = Self;

    fn mul(mut self, other: &Self) -> Self::Output {
        for (ee, er) in self.exponents.iter_mut().zip(&other.exponents) {
            *ee = *ee + *er;
        }

        self.field
            .mul_assign(&mut self.coefficient, &other.coefficient);
        self
    }
}

impl<'a, F: Ring, E: Exponent> Mul<MonomialView<'a, F, E>> for Monomial<F, E> {
    type Output = Self;

    fn mul(mut self, other: MonomialView<F, E>) -> Self::Output {
        for (ee, er) in self.exponents.iter_mut().zip(other.exponents) {
            *ee = *ee + *er;
        }

        self.field
            .mul_assign(&mut self.coefficient, &other.coefficient);
        self
    }
}

impl<F: Ring, E: Exponent> Add for Monomial<F, E> {
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        debug_assert_eq!(self.exponents, other.exponents);

        self.field
            .add_assign(&mut self.coefficient, &other.coefficient);
        self
    }
}

impl<F: Ring, E: Exponent> Sub for Monomial<F, E> {
    type Output = Self;

    fn sub(mut self, other: Self) -> Self::Output {
        debug_assert_eq!(self.exponents, other.exponents);

        self.field
            .add_assign(&mut self.coefficient, &self.field.neg(&other.coefficient));
        self
    }
}

impl<F: Ring, E: Exponent> Neg for Monomial<F, E> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.coefficient = self.field.neg(&self.coefficient);
        self
    }
}

impl<F: Ring, E: Exponent> PartialOrd for Monomial<F, E> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.exponents.partial_cmp(&other.exponents)
    }
}

impl<F: Ring, E: Exponent> Ord for Monomial<F, E> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<F: Ring, E: Exponent> PartialEq for Monomial<F, E> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.exponents.eq(&other.exponents)
    }
}

impl<F: Ring, E: Exponent> Eq for Monomial<F, E> {}

/// View object for a term in a multivariate polynomial.
#[derive(Copy, Clone, Debug)]
pub struct MonomialView<'a, F: 'a + Ring, E: 'a + Exponent> {
    pub coefficient: &'a F::Element,
    pub exponents: &'a [E],
}

/// Iterator over terms in a multivariate polynomial.
pub struct MonomialViewIterator<'a, F: Ring, E: Exponent> {
    poly: &'a MultivariatePolynomial<F, E>,
    index: usize,
}

impl<'a, F: Ring, E: Exponent> Iterator for MonomialViewIterator<'a, F, E> {
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

impl<'a, F: Ring, E: Exponent> IntoIterator for &'a MultivariatePolynomial<F, E> {
    type Item = MonomialView<'a, F, E>;
    type IntoIter = MonomialViewIterator<'a, F, E>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            poly: self,
            index: 0,
        }
    }
}
