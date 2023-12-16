use ahash::HashMap;
use rand::{thread_rng, Rng};

use tracing::debug;

use crate::{
    combinatorics::CombinationIterator,
    poly::gcd::LARGE_U32_PRIMES,
    rings::{
        finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
        integer::{Integer, IntegerRing},
        integer_mod::IntegerMod,
        rational::RationalField,
        EuclideanDomain, Field, Ring,
    },
};

use super::{gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent, LexOrder};

pub trait Factorize: Sized {
    /// Perform a square-free factorization.
    /// The output is `a_1^e1*...*a_n^e_n`
    /// where each `a_i` is relative prime.
    fn square_free_factorization(&self) -> Vec<(Self, usize)>;
    /// Factor a polynomial over its coefficient ring.
    fn factor(&self) -> Result<Vec<(Self, usize)>, &'static str>;
}

impl<F: EuclideanDomain + PolynomialGCD<E>, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Find factors that do not contain all variables.
    pub fn factor_separable(&self) -> Vec<Self> {
        let mut stripped = self.clone();

        let mut factors = vec![];
        for x in 0..self.nvars {
            if self.degree(x) == E::zero() {
                continue;
            }

            let c = stripped.to_univariate_polynomial_list(x);
            let cs = c.into_iter().map(|x| x.0).collect();

            let gcd = PolynomialGCD::gcd_multiple(cs);

            if !gcd.is_constant() {
                stripped = stripped / &gcd;
                let mut fs = gcd.factor_separable();
                factors.extend(fs.drain(..));
            }
        }

        factors.push(stripped);
        factors
    }

    /// Perform a square free factorization using Yun's algorithm.
    ///
    /// The characteristic of the ring must be 0 and all variables
    /// must occur in every factor.
    fn square_free_factorization_0_char(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        // any variable can be selected
        // select the one with the lowest degree
        let lowest_rank_var = (0..self.nvars)
            .filter_map(|x| {
                let d = self.degree(x);
                if d > E::zero() {
                    Some((x, d))
                } else {
                    None
                }
            })
            .min_by_key(|a| a.1)
            .unwrap()
            .0;

        let b = self.derivative(lowest_rank_var);
        let c = MultivariatePolynomial::gcd(self, &b);

        if c.is_one() {
            return vec![(self.clone(), 1)];
        }

        let mut factors = vec![];

        let mut w = self / &c;
        let mut y = &b / &c;

        let mut i = 1;
        while !w.is_constant() {
            let z = y - w.derivative(lowest_rank_var);
            let g = MultivariatePolynomial::gcd(&w, &z);
            w = w / &g;
            y = z / &g;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        factors
    }
}

impl<E: Exponent> Factorize for MultivariatePolynomial<IntegerRing, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];

        if !c.is_one() {
            factors.push((Self::new_from_constant(self, c), 1));
        }

        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_0_char();
            factors.extend(nf.drain(..));
        }

        if factors.is_empty() {
            factors.push((Self::one(&self.field), 1))
        }

        factors
    }

    fn factor(&self) -> Result<Vec<(Self, usize)>, &'static str> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for v in 0..self.nvars {
                degrees[v] = f.degree(v).to_u32() as usize;
                if degrees[v] > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 | 1 => factors.extend(f.factor_reconstruct().into_iter().map(|ff| (ff, p))),
                2 => {
                    let mut d_it = degrees.iter().enumerate().filter(|(_, d)| **d > 0);
                    let main_var = d_it.next().unwrap().0;
                    let interpolation_var = d_it.next().unwrap().0;

                    factors.extend(
                        f.bivariate_factor_reconstruct(main_var, interpolation_var)
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
                _ => return Err("Cannot factor multivariate polynomial yet"),
            }
        }

        Ok(factors)
    }
}

impl<E: Exponent> Factorize for MultivariatePolynomial<RationalField, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.field.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            IntegerRing::new(),
        );

        let fs = stripped.square_free_factorization();

        let mut factors: Vec<_> = fs
            .into_iter()
            .map(|(f, e)| (f.map_coeff(|coeff| coeff.into(), RationalField::new()), e))
            .collect();

        if !c.is_one() {
            factors.push((Self::new_from_constant(&self, c), 1));
        }

        factors
    }

    fn factor(&self) -> Result<Vec<(Self, usize)>, &'static str> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.field.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            IntegerRing::new(),
        );

        let fs = stripped.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars];
        for (f, p) in fs {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for v in 0..self.nvars {
                degrees[v] = f.degree(v).to_u32() as usize;
                if degrees[v] > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 | 1 => factors.extend(
                    f.factor_reconstruct()
                        .into_iter()
                        .map(|ff| (ff.map_coeff(|coeff| coeff.into(), RationalField::new()), p)),
                ),
                2 => {
                    let mut d_it = degrees.iter().enumerate().filter(|(_, d)| **d > 0);
                    let main_var = d_it.next().unwrap().0;
                    let interpolation_var = d_it.next().unwrap().0;

                    factors.extend(
                        f.bivariate_factor_reconstruct(main_var, interpolation_var)
                            .into_iter()
                            .map(|ff| {
                                (ff.map_coeff(|coeff| coeff.into(), RationalField::new()), p)
                            }),
                    )
                }
                _ => return Err("Cannot factor multivariate polynomial yet"),
            }
        }
        if !c.is_one() {
            factors.push((Self::new_from_constant(&self, c), 1));
        }

        Ok(factors)
    }
}

impl<UField: FiniteFieldWorkspace, E: Exponent> Factorize
    for MultivariatePolynomial<FiniteField<UField>, E, LexOrder>
where
    FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
{
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];
        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_bernardin();
            factors.extend(nf.drain(..));
        }

        if factors.is_empty() || !self.field.is_one(&c) {
            factors.push((Self::new_from_constant(self, c), 1))
        }

        factors
    }

    fn factor(&self) -> Result<Vec<(Self, usize)>, &'static str> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for v in 0..self.nvars {
                degrees[v] = f.degree(v).to_u32() as usize;
                if degrees[v] > 0 {
                    var_count += 1;
                }
            }

            if var_count == 0 || var_count == 1 {
                for (d2, f2) in f.distinct_degree_factorization() {
                    debug!("DDF {} {}", f2, d2);
                    for f3 in f2.equal_degree_factorization(d2) {
                        factors.push((f3, p));
                    }
                }
            } else if var_count == 2 {
                let mut d_it = degrees.iter().enumerate().filter(|(_, d)| **d > 0);
                let main_var = d_it.next().unwrap().0;
                let interpolation_var = d_it.next().unwrap().0;
                for f in f.bivariate_factorization(main_var, interpolation_var) {
                    factors.push((f, p));
                }
            } else {
                return Err("Cannot factor multivariate polynomial yet");
            }
        }

        Ok(factors)
    }
}

impl<UField: FiniteFieldWorkspace, E: Exponent>
    MultivariatePolynomial<FiniteField<UField>, E, LexOrder>
where
    FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
{
    /// Bernardin's algorithm for square free factorization.
    fn square_free_factorization_bernardin(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        let mut f = self.clone();

        let mut h = HashMap::default();
        let mut hr;
        for var in 0..self.nvars {
            if f.degree(var) > E::zero() {
                (f, hr) = f.square_free_factorization_ff_yun(var);

                for (part, pow) in hr {
                    h.entry(pow)
                        .and_modify(|f| {
                            *f = &*f * &part;
                        })
                        .or_insert(part);
                }
            }
        }

        // take the pth root
        // the coefficients remain unchanged, since x^1/p = x
        // since the derivative in every var is 0, all powers are divisible by p
        let p = self.field.get_prime().to_u64() as usize;
        let mut b = f.clone();
        for es in b.exponents.chunks_mut(self.nvars) {
            for e in es {
                if e.is_zero() {
                    continue;
                }

                if p < u32::MAX as usize {
                    debug_assert_eq!(e.to_u32() as usize % p, 0);
                    *e = *e / E::from_u32(p as u32);
                } else {
                    // at the moment exponents are limited to 32-bits
                    // so only the case where e = 0 is supported
                    assert!(*e == E::zero());
                }
            }
        }

        let mut factors = vec![];
        let sub_factors = b.square_free_factorization_bernardin();

        for (mut k, n) in sub_factors {
            for (powh, hi) in &mut h {
                if *powh < p as usize {
                    let g = MultivariatePolynomial::gcd(&k, hi);
                    if !g.is_constant() {
                        k = k / &g;
                        *hi = &*hi / &g;
                        factors.push((g, n * p + *powh));
                    }
                }
            }

            if !k.is_constant() {
                factors.push((k, n * p));
            }
        }

        for (powh, hi) in h {
            if !hi.is_constant() {
                factors.push((hi, powh));
            }
        }

        factors
    }

    /// A modified version of Yun's square free factorization algorithm.
    fn square_free_factorization_ff_yun(&self, var: usize) -> (Self, Vec<(Self, usize)>) {
        let b = self.derivative(var);
        let mut c = MultivariatePolynomial::gcd(self, &b);
        let mut w = self / &c;
        let mut v = &b / &c;

        let mut factors = vec![];

        let mut i = 1;
        while !w.is_constant() && i < self.field.get_prime().to_u64() as usize {
            let z = v - w.derivative(var);
            let g = MultivariatePolynomial::gcd(&w, &z);
            w = w / &g;
            v = z / &g;
            c = c / &w;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        (c, factors)
    }

    /// Perform distinct degree factorization on a monic, univariate and square-free polynomial.
    pub fn distinct_degree_factorization(&self) -> Vec<(usize, Self)> {
        assert!(self.field.get_prime().to_u64() != 2);
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return vec![(0, self.clone())]; // constant polynomial
        };

        let mut e = self.last_exponents().to_vec();
        e[var] = E::one();
        let x = Self::new_from_monomial(&self, self.field.one(), e);

        let mut factors = vec![];
        let mut h = x.clone();
        let mut f = self.clone();
        let mut i: usize = 0;
        while !f.is_one() {
            i += 1;

            h = h.exp_mod_univariate(self.field.get_prime().to_u64().into(), &mut f);

            let mut g = MultivariatePolynomial::gcd(&(&h - &x), &f);

            if !g.is_one() {
                f = f.quot_rem_univariate(&mut g).0;
                factors.push((i, g));
            }

            if f.last_exponents()[var] < E::from_u32(2 * (i as u32 + 1)) {
                // f cannot be split more
                if !f.is_constant() {
                    factors.push((f.last_exponents()[var].to_u32() as usize, f));
                }
                break;
            }
        }

        factors
    }

    /// Perform Cantor-Zassenhaus's probabilistic algorithm for
    /// finding irreducible factors of degree `d`.
    pub fn equal_degree_factorization(&self, d: usize) -> Vec<Self> {
        assert!(self.field.get_prime().to_u64() != 2);
        let mut s = self.clone().make_monic();

        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            if d == 1 {
                return vec![s];
            } else {
                panic!("Degree mismatch");
            }
        };

        let n = self.degree(var).to_u32() as usize;

        if n == d {
            return vec![s];
        }

        let mut rng = thread_rng();
        let mut random_poly = Self::new_from(self, Some(d));
        let mut exp = vec![E::zero(); self.nvars];

        let factor = loop {
            // generate a random non-constant polynomial
            random_poly.clear();
            for i in 0..n {
                let r = self
                    .field
                    .nth(rng.gen_range(0..self.field.get_prime().to_u64()));
                if !FiniteField::<UField>::is_zero(&r) {
                    exp[var] = E::from_u32(i as u32);
                    random_poly.append_monomial(r, &exp);
                }
            }

            if random_poly.degree(var) == E::zero() {
                continue;
            }

            let g = MultivariatePolynomial::gcd(&random_poly, &s);

            if !g.is_one() {
                break g;
            }

            // TODO: use Frobenius map and modular composition to prevent computing large exponent poly^(p^d)
            let p: Integer = self.field.get_prime().to_u64().into();
            let b = random_poly
                .exp_mod_univariate(&(&p.pow(d as u64) - &1i64.into()) / &2i64.into(), &mut s)
                - Self::new_from_constant(&self, self.field.one());

            let g = MultivariatePolynomial::gcd(&b, &s);

            if !g.is_one() && g != s {
                break g;
            }
        };

        let mut factors = factor.equal_degree_factorization(d);
        factors.extend((self / &factor).equal_degree_factorization(d));
        factors
    }

    /// Perform distinct and equal degree factorization on a square-free univariate polynomial.
    fn factor_distinct_equal_degree(&self) -> Vec<Self> {
        let mut factors = vec![];
        for (d2, f2) in self.distinct_degree_factorization() {
            debug!("DDF {} {}", f2, d2);
            for f3 in f2.equal_degree_factorization(d2) {
                factors.push(f3);
            }
        }
        factors
    }

    /// Bernardin's algorithm based on
    /// "A new bivariate Hensel lifting algorithm for n factors"
    /// by Garrett Paluck. The formulation of the algorithm in other sources contain serious errors.
    // TODO: merge with an almost similar method for the integer case. A modification that needs
    // to be made here is to make the lcoeff_y=0 monic
    fn bivariate_hensel_lift_bernardin(
        &self,
        interpolation_var: usize,
        lcoeff: &Self,
        univariate_factors: &[Self],
        iterations: usize,
    ) -> Vec<Self> {
        let y_poly = self.to_univariate_polynomial_list(interpolation_var);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &self.field.zero())];
        factors.extend_from_slice(univariate_factors);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![self.new_from(None); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        let delta =
            Self::diophantine_univariate(&mut factors, &self.new_from_constant(self.field.one()));

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter_mut()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate(f).1;
            }

            for fs in u.iter() {
                let mut poly = fs[0].clone();
                for (p, ff) in fs[1..].iter().enumerate() {
                    let mut a = ff.clone();
                    for e in a.exponents.chunks_mut(self.nvars) {
                        e[interpolation_var] = E::from_u32((p + 1) as u32);
                    }
                    poly = &poly + &a;
                }
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = self.new_from(None);
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = self.new_from(Some(ts.len()));
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents.chunks_mut(f.nvars) {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }
                new_poly
            })
            .collect()
    }

    /// Compute the bivariate factorization of a square-free polynomial.
    fn bivariate_factorization(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        assert!(main_var != interpolation_var);

        // check for problems arising from canceling terms in the derivative
        let der = self.derivative(main_var);
        if self.derivative(main_var).is_zero() {
            return self.bivariate_factorization(interpolation_var, main_var);
        }

        let g = MultivariatePolynomial::gcd(&self, &der);
        if !g.is_constant() {
            let mut factors = g.bivariate_factorization(main_var, interpolation_var);
            factors.extend((self / &g).bivariate_factorization(main_var, interpolation_var));
            return factors;
        }

        let mut sample_point = self.field.zero();
        let mut uni_f = self.replace(interpolation_var, &sample_point);

        let mut i = 0;
        loop {
            if self.degree(main_var) == uni_f.degree(main_var)
                && MultivariatePolynomial::gcd(&uni_f, &uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            sample_point = self.field.nth(i);
            uni_f = self.replace(interpolation_var, &sample_point);
            i += 1;
        }

        let mut d = self.degree(interpolation_var).to_u32();

        let shifted_poly = if !FiniteField::<UField>::is_zero(&sample_point) {
            self.shift_var_cached(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        let fs = uni_f.factor_distinct_equal_degree();

        let mut lcoeff = shifted_poly.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut lc_d = lcoeff.degree(interpolation_var).to_u32();

        let iter = (d + lc_d + 1) as usize;
        let mut factors =
            shifted_poly.bivariate_hensel_lift_bernardin(interpolation_var, &lcoeff, &fs, iter);

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // TODO: multiply in the leading coefficient here,
                // then we can skip the Pade approximation and reduce the
                // number of iterations in the Hensel lifting to d + 1, like in the integer case?
                let mut g = MultivariatePolynomial::new_from_constant(&rest, rest.lcoeff());
                for i in 0..factors.len() {
                    if cs.contains(&i) {
                        g = (&g * &factors[i])
                            .mod_var(interpolation_var, E::from_u32(iter as u32 + 1));
                    }
                }

                let y_polys: Vec<_> = g
                    .to_univariate_polynomial_list(main_var)
                    .into_iter()
                    .map(|(x, _)| x)
                    .collect();

                let mut g_lcoeff = Self::lcoeff_reconstruct(&y_polys, d as u32, lc_d as u32);
                g = (&g * &g_lcoeff)
                    .mod_var(interpolation_var, E::from_u32(d as u32 + 1))
                    .make_monic();

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    lcoeff = lcoeff.quot_rem_univariate(&mut g_lcoeff).0;
                    lc_d = lcoeff.degree(interpolation_var).to_u32();
                    d = rest.degree(interpolation_var).to_u32();

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !FiniteField::<UField>::is_zero(&sample_point) {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var_cached(interpolation_var, &self.field.neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Reconstruct the leading coefficient using a Pade approximation with numerator degree `deg_n` and
    /// denominator degree `deg_d`. The resulting denominator should be a factor of the leading coefficient.
    fn lcoeff_reconstruct(coeffs: &[Self], deg_n: u32, deg_d: u32) -> Self {
        let mut lcoeff = coeffs[0].new_from_constant(coeffs[0].field.one());
        for x in coeffs {
            let d = x.rational_approximant_univariate(deg_n, deg_d).unwrap().1;
            if !d.is_one() {
                let g = MultivariatePolynomial::gcd(&d, &lcoeff);
                lcoeff = lcoeff * &(d / &g);
            }
        }
        lcoeff
    }
}

impl<E: Exponent> MultivariatePolynomial<IntegerRing, E, LexOrder> {
    /// Hensel lift a solution of `self = u * w mod p` to `self = u * w mod max_p`
    /// where `max_p` is a power of `p`.
    ///
    /// If the lifting is successful, i.e. the error is 0 at some stage,
    /// it will return `Ok((u,w))` where `u` and `w` are the true factors over
    /// the integers. If a true factorization is not possible, it returns
    /// `Err((u,w))` where `u` and `w` are monic.
    pub fn hensel_lift<UField: FiniteFieldWorkspace>(
        &self,
        mut u: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        mut w: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        gamma: Option<Integer>,
        max_p: &Integer,
    ) -> Result<(Self, Self), (Self, Self)>
    where
        FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
        Integer: ToFiniteField<UField>,
    {
        let lcoeff = self.lcoeff(); // lcoeff % p != 0
        let mut gamma = gamma.unwrap_or(lcoeff.clone());
        let lcoeff_p = lcoeff.to_finite_field(&u.field);
        let gamma_p = gamma.to_finite_field(&u.field);
        let field = u.field;
        let p = Integer::from(field.get_prime().to_u64());

        let a = self.clone().mul_coeff(gamma.clone());

        u = u.make_monic().mul_coeff(gamma_p.clone());
        w = w.make_monic().mul_coeff(lcoeff_p.clone());

        let (_, s, t) = u.eea_univariate(&w);

        debug_assert!((&s * &u + &t * &w).is_one());

        let sym_map = |e: &<FiniteField<UField> as Ring>::Element| {
            let i = field.from_element(e.clone()).to_u64().into();

            if &i * &2u64.into() > p {
                &i - &p
            } else {
                i
            }
        };

        let mut u_i = u.map_coeff(sym_map, IntegerRing::new());
        let mut w_i = w.map_coeff(sym_map, IntegerRing::new());

        // only replace the leading coefficient
        *u_i.coefficients.last_mut().unwrap() = gamma.clone();
        *w_i.coefficients.last_mut().unwrap() = lcoeff;

        let mut e = &a - &(&u_i * &w_i);

        let mut m = p.clone();

        while !e.is_zero() && &m <= max_p {
            let e_p = e.map_coeff(|c| (c / &m).to_finite_field(&field), field);
            let (q, r) = (&e_p * &s).quot_rem_univariate(&mut w);
            let tau = &e_p * &t + q * &u;

            u_i = u_i
                + tau
                    .map_coeff(sym_map, IntegerRing::new())
                    .mul_coeff(m.clone());
            w_i = w_i
                + r.map_coeff(sym_map, IntegerRing::new())
                    .mul_coeff(m.clone());
            e = &a - &(&u_i * &w_i);

            m = &m * &p;
        }

        if e.is_zero() {
            let content = u_i.content();
            if !content.is_one() {
                u_i = u_i.div_coeff(&content);
                gamma = &gamma / &content;
            }

            if !gamma.is_one() {
                w_i = w_i.div_coeff(&gamma); // true division is possible in this case
            }

            Ok((u_i, w_i))
        } else {
            if !u_i.lcoeff().is_one() {
                let inv = u_i.lcoeff().mod_inverse(&m);
                u_i = u_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), IntegerRing::new());
            }

            if !w_i.lcoeff().is_one() {
                let inv = w_i.lcoeff().mod_inverse(&m);
                w_i = w_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), IntegerRing::new());
            }

            Err((u_i, w_i))
        }
    }

    /// Lift multiple factors by creating a binary tree and lifting each product.
    fn multi_factor_hensel_lift(
        &self,
        hs: &[MultivariatePolynomial<FiniteField<u32>, E, LexOrder>],
        bound: &Integer,
        max_p: &Integer,
    ) -> Vec<Self> {
        let field = hs[0].field;

        if hs.len() == 1 {
            if self.lcoeff().is_one() {
                return vec![self.clone()];
            } else {
                let inv = self.lcoeff().mod_inverse(&max_p);
                let r = self.map_coeff(|c| (c * &inv).symmetric_mod(&max_p), IntegerRing::new());
                return vec![r];
            }
        }

        let (gs, hs) = hs.split_at(hs.len() / 2);

        let mut g = MultivariatePolynomial::new_from_constant(&gs[0], field.one());
        for x in gs {
            g = g * x;
        }

        let mut h = MultivariatePolynomial::new_from_constant(&hs[0], field.one());
        for x in hs {
            h = h * x;
        }

        let (g_i, h_i) = self.hensel_lift(g, h, None, max_p).unwrap_or_else(|e| e);
        debug!("g_i={}", g_i);
        debug!("h_i={}", h_i);

        let mut factors = g_i.multi_factor_hensel_lift(gs, bound, max_p);
        factors.extend(h_i.multi_factor_hensel_lift(hs, bound, max_p));
        factors
    }

    /// Factor a square-free univariate polynomial over the integers by Hensel lifting factors computed over
    /// a finite field image of the polynomial.
    fn factor_reconstruct(&self) -> Vec<Self> {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return vec![self.clone()]; // constant polynomial
        };
        let d = self.degree(var).to_u32();

        if d == 1 {
            return vec![self.clone()];
        }

        let max_norm = self.coefficients.iter().map(|x| x.abs()).max().unwrap();
        let bound: Integer =
            &Integer::from(((d + 1) as f64 * 2f64.powi(d as i32 + 1).sqrt()) as u64)
                * &(&Integer::from(2u64).pow(d as u64) * &(&max_norm * &self.lcoeff())); // NOTE: precision may be off

        // select a suitable prime
        let mut field;
        let mut f_p;
        let mut i = 0;
        loop {
            i += 1;
            if i == LARGE_U32_PRIMES.len() {
                panic!("Ran out of primes during factorization");
            }

            let p = LARGE_U32_PRIMES[i];

            if (&self.lcoeff() % &Integer::Natural(p as i64)).is_zero() {
                continue;
            }

            field = FiniteField::<u32>::new(p);
            f_p = self.map_coeff(|f| f.to_finite_field(&field), field);
            let df_p = f_p.derivative(var);

            // check is f_p remains square-free
            if MultivariatePolynomial::gcd(&f_p, &df_p).is_one() {
                break;
            }
        }

        let hs: Vec<_> = f_p.factor_distinct_equal_degree();

        if hs.len() == 1 {
            // the polynomial is irreducible
            return vec![self.clone()];
        }

        let p: Integer = (field.get_prime().to_u32() as i64).into();
        let mut max_p = p.clone();
        while max_p < bound {
            max_p = &max_p * &p;
        }

        let mut factors = self.multi_factor_hensel_lift(&hs, &bound, &max_p);

        #[cfg(debug_assertions)]
        for (h, h_p) in factors.iter().zip(&hs) {
            let hh_p = h.to_finite_field(&field).make_monic();
            if &hh_p != h_p {
                panic!("Mismatch of lifted factor: {} vs {} in {}", hh_p, h_p, self);
            }
        }

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = self.clone();
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // check if the constant term matches
                if rest.exponents[..rest.nvars].iter().all(|e| *e == E::zero()) {
                    let mut g1 = rest.lcoeff();
                    let mut h1 = rest.lcoeff();
                    for i in 0..factors.len() {
                        if factors[i].exponents[..rest.nvars]
                            .iter()
                            .all(|x| *x == E::zero())
                        {
                            if cs.contains(&i) {
                                g1 = (&g1 * &factors[i].coefficients[0]).symmetric_mod(&max_p);
                            } else {
                                h1 = (&h1 * &factors[i].coefficients[0]).symmetric_mod(&max_p);
                            }
                        }
                    }

                    if &g1 * &h1 != &rest.lcoeff() * &rest.coefficients[0] {
                        continue;
                    }
                }

                let mut g = MultivariatePolynomial::new_from_constant(&rest, rest.lcoeff());
                for i in 0..factors.len() {
                    if cs.contains(&i) {
                        g = &g * &factors[i];
                        g = g.map_coeff(|i| i.symmetric_mod(&max_p), IntegerRing::new());
                    }
                }

                let c = g.content();
                g = g.div_coeff(&c);

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    // should always happen happen when |g1|_1 * |h1|_1 <= bound
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    let c = h.content();
                    rest = h.div_coeff(&c);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);
        rec_factors
    }

    /// Lift a solution of `poly ≡ lcoeff * univariate_factors mod y mod p^k`
    /// to `mod y^iterations mod p^k`.
    ///
    /// Univariate factors must be monic and `lcoeff_y=0` should be as well.
    fn bivariate_hensel_lift_bernardin(
        poly: &MultivariatePolynomial<IntegerMod, E, LexOrder>,
        interpolation_var: usize,
        lcoeff: &MultivariatePolynomial<IntegerMod, E, LexOrder>,
        univariate_factors: &[MultivariatePolynomial<IntegerMod, E, LexOrder>],
        iterations: usize,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<IntegerMod, E, LexOrder>> {
        let finite_field = FiniteField::<u32>::new(p);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &poly.field.zero())];

        for f in univariate_factors {
            factors.push(f.clone());
        }

        let delta = Self::lift_diophantine_univariate(
            &mut factors,
            &poly.new_from_constant(poly.field.one()),
            finite_field.get_prime(),
            k,
        );

        let y_poly = poly.to_univariate_polynomial_list(interpolation_var);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![poly.new_from(None); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate_monic(f).1;
            }

            for fs in u.iter() {
                let mut poly = fs[0].clone();
                for (p, ff) in fs[1..].iter().enumerate() {
                    let mut a = ff.clone();
                    for e in a.exponents.chunks_mut(poly.nvars) {
                        e[interpolation_var] = E::from_u32((p + 1) as u32);
                    }
                    poly = &poly + &a;
                }
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = poly.new_from(None);
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = poly.new_from(Some(ts.len()));
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents.chunks_mut(f.nvars) {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }

                new_poly
            })
            .collect()
    }

    /// Factor a square-free bivariate polynomial over the integers.
    fn bivariate_factor_reconstruct(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        let d1 = self.degree(main_var).to_u32();
        let d2 = self.degree(interpolation_var).to_u32();

        // select a suitable evaluation point, as small as possible as to not change the coefficient bound
        let mut sample_point;
        let mut uni_f;
        let mut i = 0u64;
        loop {
            sample_point = i.into();
            uni_f = self.replace(interpolation_var, &sample_point);

            if self.degree(main_var) == uni_f.degree(main_var)
                && MultivariatePolynomial::gcd(&uni_f, &uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            i += 1;
        }

        // factor the univariate polynomial
        let mut uni_fs: Vec<_> = uni_f
            .factor()
            .unwrap()
            .into_iter()
            .map(|(f, p)| {
                debug_assert_eq!(p, 1);
                f
            })
            .collect();

        // strip potential content
        uni_fs.retain_mut(|f| !f.is_constant());

        // select a suitable prime
        let mut field;
        let mut i = 0;
        'new_prime: loop {
            i += 1;
            if i == LARGE_U32_PRIMES.len() {
                panic!("Ran out of primes during factorization");
            }

            let p = LARGE_U32_PRIMES[i];

            if (&uni_f.lcoeff() % &Integer::Natural(p as i64)).is_zero() {
                continue;
            }

            field = FiniteField::<u32>::new(p);

            // make sure the factors stay coprime
            let fs_p: Vec<_> = uni_fs.iter().map(|f| f.to_finite_field(&field)).collect();

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !MultivariatePolynomial::gcd(f, g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }

        let shifted_poly = if !sample_point.is_zero() {
            self.shift_var(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        let max_norm = shifted_poly
            .coefficients
            .iter()
            .map(|x| x.abs())
            .max()
            .unwrap();

        let bound: Integer = &Integer::from_f64(
            ((d1 + 1) as f64 * (d2 + 1) as f64 * 2f64.powi(2 * (d1 as i32 + d2 as i32 - 1))).sqrt(),
        ) * &(&max_norm * &self.lcoeff()); // NOTE: precision may be off

        let p = Integer::from(field.get_prime().to_u64());
        let mut max_p = p.clone();
        let mut k = 1;
        while &max_p * 2 < bound {
            max_p = &max_p * &p;
            k += 1;
        }

        let mod_field = IntegerMod::new(max_p.clone());

        // make all factors monic, this is possible since the lcoeff is invertible mod p^k
        let uni_fs_mod: Vec<_> = uni_fs
            .iter()
            .map(|f| {
                let f1 = f.map_coeff(|c| mod_field.to_element(c), mod_field.clone());
                let inv_coeff = mod_field.inv(f1.lcoeff().clone());
                f1.mul_coeff(inv_coeff)
            })
            .collect();

        let mut f_mod = shifted_poly.map_coeff(|c| c.symmetric_mod(&max_p), mod_field.clone());

        // make sure the lcoeff is monic at y=0
        let inv_coeff = mod_field.inv(uni_f.lcoeff().clone());
        let f_mod_monic = f_mod.clone().mul_coeff(inv_coeff);
        let lcoeff_monic = f_mod_monic.lcoeff_last_varorder(&[main_var, interpolation_var]);

        let mut factors = Self::bivariate_hensel_lift_bernardin(
            &f_mod_monic,
            interpolation_var,
            &lcoeff_monic,
            &uni_fs_mod,
            (d2 + 1) as usize,
            field.get_prime(),
            k,
        );

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                let mut g = lcoeff.clone();
                for i in 0..factors.len() {
                    if cs.contains(&i) {
                        g = (&g * &factors[i]).mod_var(interpolation_var, E::from_u32(d2 + 1));
                    }
                }

                // convert to integer
                let mut g_int = g.map_coeff(|c| c.clone(), IntegerRing::new());

                let content = g_int.univariate_content(main_var);
                g_int = &g_int / &content;

                let (h, r) = rest.quot_rem(&g_int, true);

                if r.is_zero() {
                    rec_factors.push(g_int);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    f_mod = rest.map_coeff(|c| c.symmetric_mod(&max_p), mod_field.clone());
                    lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !sample_point.is_zero() {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var(interpolation_var, &self.field.neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Solve a Diophantine equation over the ring `Z_p^k` using Newton iteration.
    /// All factors must be monic.
    fn lift_diophantine_univariate(
        factors: &mut [MultivariatePolynomial<IntegerMod, E, LexOrder>],
        rhs: &MultivariatePolynomial<IntegerMod, E, LexOrder>,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<IntegerMod, E, LexOrder>> {
        let field = FiniteField::<u32>::new(p);
        let prime: Integer = (p as u64).into();

        let mut f_p: Vec<_> = factors
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field))
            .collect();
        let rhs_p = rhs.map_coeff(|c| c.to_finite_field(&field), field);

        // TODO: recycle from finite field computation that must have happened earlier
        let mut delta =
            MultivariatePolynomial::<FiniteField<u32>, E, LexOrder>::diophantine_univariate(
                &mut f_p, &rhs_p,
            );

        let sym_map_p = |e: &<FiniteField<u32> as Ring>::Element| {
            let i = field.from_element(e.clone()).to_u64().into();

            if &i * &2u64.into() > prime {
                &i - &prime
            } else {
                i
            }
        };

        let mut deltas: Vec<_> = delta
            .iter()
            .map(|s| {
                s.map_coeff(sym_map_p, IntegerRing::new())
                    .map_coeff(|c| rhs.field.to_element(c), rhs.field.clone())
            })
            .collect();

        if k == 1 {
            return deltas;
        }

        let mut tot = rhs.new_from_constant(rhs.field.one());
        for f in factors.iter() {
            tot = &tot * f;
        }

        let pi = factors
            .iter()
            .map(|f| tot.quot_rem_univariate_monic(f).0)
            .collect::<Vec<_>>();

        let mut m = prime.clone();

        for _ in 1..k {
            // TODO: is breaking on e=0 safe?
            let mut e = rhs.clone();
            for (dd, pp) in deltas.iter_mut().zip(&pi) {
                e = &e - &(&*dd * pp);
            }

            let e_m = e.map_coeff(|c| (c / &m).to_finite_field(&field), field);

            for ((p, d_m), d) in f_p.iter_mut().zip(&mut delta).zip(deltas.iter_mut()) {
                let new_delta = (&e_m * &*d_m).quot_rem_univariate(p).1;

                *d = &*d
                    + &new_delta
                        .map_coeff(sym_map_p, IntegerRing::new())
                        .mul_coeff(m.clone())
                        .map_coeff(|c| rhs.field.to_element(c), rhs.field.clone());
            }

            m = &m * &prime;
        }

        deltas
    }
}
