use crate::domains::{EuclideanDomain, Field};

use super::univariate::UnivariatePolynomial;

impl<F: EuclideanDomain> UnivariatePolynomial<F> {
    /// Compute the resultant using Brown's polynomial remainder sequence algorithm.
    pub fn resultant_prs(&self, other: &Self) -> F::Element {
        if self.degree() < other.degree() {
            return other.resultant_prs(self);
        }

        let mut a = self.clone();
        let mut a_new = other.clone();

        let mut deg = a.degree() as u64 - a_new.degree() as u64;
        let mut neg_lc = self.field.one(); //unused
        let mut init = false;
        let mut beta = self.field.pow(&self.field.neg(&self.field.one()), deg + 1);
        let mut psi = self.field.neg(&self.field.one());
        while !a_new.is_constant() {
            if init {
                psi = if deg == 0 {
                    psi
                } else if deg == 1 {
                    self.field.pow(&neg_lc, deg)
                } else {
                    let (q, r) = self.field.quot_rem(&neg_lc, &psi);
                    debug_assert!(F::is_zero(&r));
                    self.field.mul(&q, &neg_lc)
                };

                deg = a.degree() as u64 - a_new.degree() as u64;
                beta = self.field.mul(&neg_lc, &self.field.pow(&psi, deg));
            } else {
                init = true;
            }

            neg_lc = self.field.neg(&a_new.coefficients.last().unwrap());

            let (_, mut r) = a
                .mul_coeff(&self.field.pow(&neg_lc, deg + 1))
                .quot_rem(&a_new);
            if (deg + 1) % 2 == 1 {
                r = -r;
            }

            (a, a_new) = (a_new, r.div_coeff(&beta));
        }

        a_new.coefficients.pop().unwrap()
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    /// Compute the resultant of the two polynomials.
    pub fn resultant(&self, other: &Self) -> F::Element {
        let mut a = self.clone();
        let mut a_new = other.clone();

        let mut v = vec![a.degree()];
        let mut c = vec![a.lcoeff()];

        while !a_new.is_constant() {
            let (_, r) = a.quot_rem(&mut a_new);
            (a, a_new) = (a_new, r);

            v.push(a.degree());
            c.push(a.lcoeff());
        }

        let r = a_new.lcoeff();
        if F::is_zero(&r) {
            return r;
        }

        let mut sign = 0;
        for w in v.windows(2) {
            sign += w[0] * w[1];
        }

        let mut res = self.field.pow(&r, *v.last().unwrap() as u64);
        if sign % 2 == 1 {
            res = self.field.neg(&res);
        };

        v.push(0);
        for i in 1..c.len() {
            self.field.mul_assign(
                &mut res,
                &self.field.pow(&c[i], v[i - 1] as u64 - v[i + 1] as u64),
            );
        }

        res
    }
}
