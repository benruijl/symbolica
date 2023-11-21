use ahash::HashMap;
use rand::{thread_rng, Rng};
use tracing::debug;

use crate::{
    combinatorics::CombinationIterator,
    poly::gcd::LARGE_U32_PRIMES,
    rings::{
        finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
        integer::{Integer, IntegerRing},
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
    /// Factor a univariate polynomial over its coefficient ring.
    fn factor_univariate(&self) -> Vec<(Self, usize)>;
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
        // TODO: assert characteristic

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
            factors.push((Self::one(self.field), 1))
        }

        factors
    }

    fn factor_univariate(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);
            factors.extend(f.factor_reconstruct().into_iter().map(|ff| (ff, p)));
        }

        factors
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

    fn factor_univariate(&self) -> Vec<(Self, usize)> {
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
        for (f, p) in fs {
            factors.extend(
                f.factor_reconstruct()
                    .into_iter()
                    .map(|ff| (ff.map_coeff(|coeff| coeff.into(), RationalField::new()), p)),
            );
        }
        if !c.is_one() {
            factors.push((Self::new_from_constant(&self, c), 1));
        }

        factors
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

    fn factor_univariate(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);
            for (d2, f2) in f.distinct_degree_factorization() {
                debug!("DDF {} {}", f2, d2);
                for f3 in f2.equal_degree_factorization(d2) {
                    factors.push((f3, p));
                }
            }
        }

        factors
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
        // TODO: take p^l th root with max l
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
                // TODO: generate sparse polynomial?
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

        // TODO: simplify the polynomial when a true factor is found
        let (g_i, h_i) = self.hensel_lift(g, h, None, max_p).unwrap_or_else(|e| e);
        debug!("g_i={}", g_i);
        debug!("h_i={}", h_i);

        let mut factors = g_i.multi_factor_hensel_lift(gs, bound, max_p);
        factors.extend(h_i.multi_factor_hensel_lift(hs, bound, max_p));
        factors
    }

    /// Factor a square-free univariate polynomial over the integers by Hensel lifting factors computed over the
    /// a finite field image of the polynomial .
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
            let hh_p = h.to_finite_field(field);
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
}
