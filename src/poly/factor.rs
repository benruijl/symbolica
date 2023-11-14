use ahash::HashMap;
use rand::{thread_rng, Rng};
use tracing::debug;

use crate::rings::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace},
    integer::{Integer, IntegerRing},
    rational::RationalField,
    EuclideanDomain, Field, Ring,
};

use super::{gcd::PolynomialGCD, polynomial::MultivariatePolynomial, Exponent, LexOrder};

pub trait Factorize: Sized {
    /// Perform a square-free factorization.
    /// The output is `a_1^e1*...*a_n^e_n`
    /// where each `a_i` is relative prime.
    fn square_free_factorization(&self) -> Vec<(Self, usize)>;
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

    /// Perform a a square free factorization using Yun's algorithm.
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
}

impl<E: Exponent> Factorize for MultivariatePolynomial<RationalField, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.content();

        let mut stripped = MultivariatePolynomial::<_, E>::new(
            self.nvars,
            IntegerRing::new(),
            Some(self.nterms),
            self.var_map.as_ref().map(|x| x.as_slice()),
        );

        for t in self {
            let coeff = self.field.div(t.coefficient, &c);
            debug_assert!(coeff.is_integer());
            stripped.append_monomial(coeff.numerator(), t.exponents);
        }

        let mut factors = vec![];

        if !c.is_one() {
            factors.push((Self::new_from_constant(&self, c), 1));
        }

        let fs = stripped.factor_separable();

        for f in fs {
            let nf = f.square_free_factorization_0_char();

            for (p, pow) in nf {
                let mut p_rat = Self::new_from(&self, Some(p.nterms));
                for t in p.into_iter() {
                    p_rat.append_monomial(t.coefficient.into(), t.exponents);
                }
                factors.push((p_rat, pow));
            }
        }

        if factors.is_empty() {
            factors.push((Self::one(self.field), 1))
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
                if p < u32::MAX as usize {
                    debug_assert_eq!(*e % E::from_u32(p as u32), E::zero());
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

    pub fn factorize(&self) -> Vec<(Self, usize)> {
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
