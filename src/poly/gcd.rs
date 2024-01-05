use ahash::{HashMap, HashSet, HashSetExt};
use rand;
use smallvec::{smallvec, SmallVec};
use std::borrow::Cow;
use std::cmp::{max, min, Ordering};
use std::mem;
use std::ops::Add;
use tracing::{debug, instrument};

use crate::poly::INLINED_EXPONENTS;
use crate::rings::finite_field::{
    FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField,
};
use crate::rings::integer::{FromFiniteField, Integer, IntegerRing, SMALL_PRIMES};
use crate::rings::linear_system::{LinearSolverError, Matrix};
use crate::rings::rational::RationalField;
use crate::rings::{EuclideanDomain, Field, Ring};

use super::polynomial::MultivariatePolynomial;
use super::Exponent;

// 100 large u32 primes starting from the 203213901st prime number
pub const LARGE_U32_PRIMES: [u32; 100] = [
    4293490987, 4293491603, 4293492277, 4293492857, 4293491017, 4293491621, 4293492283, 4293492881,
    4293491023, 4293491639, 4293492293, 4293492893, 4293491051, 4293491659, 4293492331, 4293492941,
    4293491149, 4293491701, 4293492349, 4293492977, 4293491171, 4293491711, 4293492383, 4293493037,
    4293491221, 4293491747, 4293492403, 4293493049, 4293491261, 4293491779, 4293492421, 4293493069,
    4293491269, 4293491791, 4293492431, 4293493081, 4293491273, 4293491819, 4293492487, 4293493091,
    4293491281, 4293491849, 4293492499, 4293493117, 4293491299, 4293491863, 4293492523, 4293493121,
    4293491303, 4293491887, 4293492583, 4293493159, 4293491311, 4293491897, 4293492587, 4293493163,
    4293491327, 4293491911, 4293492649, 4293493207, 4293491329, 4293491953, 4293492661, 4293493229,
    4293491399, 4293491957, 4293492673, 4293493241, 4293491431, 4293492017, 4293492701, 4293493261,
    4293491467, 4293492023, 4293492739, 4293493319, 4293491509, 4293492097, 4293492751, 4293493363,
    4293491539, 4293492101, 4293492769, 4293493367, 4293491551, 4293492107, 4293492779, 4293493409,
    4293491561, 4293492113, 4293492781, 4293493423, 4293491567, 4293492139, 4293492811, 4293493433,
    4293491591, 4293492169, 4293492821, 4293493487,
];

pub const LARGE_U64_PRIMES: [u64; 50] = [
    18446744073709551557,
    18446744073709551533,
    18446744073709551521,
    18446744073709551437,
    18446744073709551427,
    18446744073709551359,
    18446744073709551337,
    18446744073709551293,
    18446744073709551263,
    18446744073709551253,
    18446744073709551191,
    18446744073709551163,
    18446744073709551113,
    18446744073709550873,
    18446744073709550791,
    18446744073709550773,
    18446744073709550771,
    18446744073709550719,
    18446744073709550717,
    18446744073709550681,
    18446744073709550671,
    18446744073709550593,
    18446744073709550591,
    18446744073709550539,
    18446744073709550537,
    18446744073709550381,
    18446744073709550341,
    18446744073709550293,
    18446744073709550237,
    18446744073709550147,
    18446744073709550141,
    18446744073709550129,
    18446744073709550111,
    18446744073709550099,
    18446744073709550047,
    18446744073709550033,
    18446744073709550009,
    18446744073709549951,
    18446744073709549861,
    18446744073709549817,
    18446744073709549811,
    18446744073709549777,
    18446744073709549757,
    18446744073709549733,
    18446744073709549667,
    18446744073709549621,
    18446744073709549613,
    18446744073709549583,
    18446744073709549571,
    18446744073709549519,
];

pub trait LargePrimes: Sized {
    /// Get a list of large primes that fit in this type.
    fn get_primes() -> &'static [Self];
}

impl LargePrimes for u32 {
    #[inline(always)]
    fn get_primes() -> &'static [Self] {
        &LARGE_U32_PRIMES
    }
}

impl LargePrimes for u64 {
    #[inline(always)]
    fn get_primes() -> &'static [Self] {
        &LARGE_U64_PRIMES
    }
}

/// The maximum power of a variable that is cached
pub const POW_CACHE_SIZE: usize = 1000;
pub const INITIAL_POW_MAP_SIZE: usize = 1000;

/// The upper bound of the range to be sampled during the computation of multiple gcds
pub const MAX_RNG_PREFACTOR: u32 = 50000;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum GCDError {
    BadOriginalImage,
    BadCurrentImage,
}

fn newton_interpolation<UField: FiniteFieldWorkspace, E: Exponent>(
    a: &[<FiniteField<UField> as Ring>::Element],
    u: &[MultivariatePolynomial<FiniteField<UField>, E>],
    x: usize, // the variable index to extend the polynomial by
) -> MultivariatePolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    let field = u[0].field;

    // compute inverses
    let mut gammas = Vec::with_capacity(a.len());
    for k in 1..a.len() {
        let mut pr = field.sub(&a[k], &a[0]);
        for i in 1..k {
            u[0].field.mul_assign(&mut pr, &field.sub(&a[k], &a[i]));
        }
        gammas.push(u[0].field.inv(&pr));
    }

    // compute Newton coefficients
    let mut v = vec![u[0].clone()];
    for k in 1..a.len() {
        let mut tmp = v[k - 1].clone();
        for j in (0..k - 1).rev() {
            tmp = tmp.mul_coeff(field.sub(&a[k], &a[j])).add(v[j].clone());
        }

        let mut r = u[k].clone() - tmp;
        r = r.mul_coeff(gammas[k - 1].clone());
        v.push(r);
    }

    // convert to standard form
    let mut e = vec![E::zero(); u[0].nvars];
    e[x] = E::one();
    let xp = MultivariatePolynomial::new_from_monomial(&u[0], field.one(), e);
    let mut u = v[v.len() - 1].clone();
    for k in (0..v.len() - 1).rev() {
        // TODO: prevent cloning
        u = u * &(xp.clone() - MultivariatePolynomial::new_from_constant(&v[0], a[k].clone()))
            + v[k].clone();
    }
    u
}

/// Evaluation of the exponents by filling in the variables
#[inline(always)]
fn evaluate_exponents<UField: FiniteFieldWorkspace, E: Exponent>(
    poly: &MultivariatePolynomial<FiniteField<UField>, E>,
    r: &[(usize, <FiniteField<UField> as Ring>::Element)],
    cache: &mut [Vec<<FiniteField<UField> as Ring>::Element>],
) -> Vec<<FiniteField<UField> as Ring>::Element>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    let mut eval = vec![poly.field.one(); poly.nterms];
    for (c, t) in eval.iter_mut().zip(poly) {
        // evaluate each exponent
        for &(n, v) in r {
            let exp = t.exponents[n].to_u32() as usize;
            if exp > 0 {
                if exp < cache[n].len() {
                    if FiniteField::<UField>::is_zero(&cache[n][exp]) {
                        cache[n][exp] = poly.field.pow(&v, exp as u64);
                    }

                    poly.field.mul_assign(c, &cache[n][exp]);
                } else {
                    poly.field.mul_assign(c, &poly.field.pow(&v, exp as u64));
                }
            }
        }
    }
    eval
}

/// Evaluate a polynomial using the evaluation of the exponent of every monomial.
#[inline(always)]
fn evaluate_using_exponents<UField: FiniteFieldWorkspace, E: Exponent>(
    poly: &MultivariatePolynomial<FiniteField<UField>, E>,
    exp_evals: &[<FiniteField<UField> as Ring>::Element],
    main_var: usize,
    out: &mut MultivariatePolynomial<FiniteField<UField>, E>,
) where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    out.clear();
    let mut c = poly.field.zero();
    let mut new_exp = vec![E::zero(); poly.nvars];
    for (aa, e) in poly.into_iter().zip(exp_evals) {
        if aa.exponents[main_var] != new_exp[main_var] {
            if !FiniteField::is_zero(&c) {
                out.coefficients.push(c);
                out.exponents.extend_from_slice(&new_exp);
                out.nterms += 1;

                c = poly.field.zero();
            }

            new_exp[main_var] = aa.exponents[main_var];
        }

        poly.field.add_mul_assign(&mut c, aa.coefficient, e);
    }

    if !FiniteField::is_zero(&c) {
        out.coefficients.push(c);
        out.exponents.extend_from_slice(&new_exp);
        out.nterms += 1;
    }
}

fn solve_vandermonde<UField: FiniteFieldWorkspace, E: Exponent>(
    a: &MultivariatePolynomial<FiniteField<UField>, E>,
    main_var: usize,
    shape: &[(MultivariatePolynomial<FiniteField<UField>, E>, E)],
    row_sample_values: Vec<Vec<<FiniteField<UField> as Ring>::Element>>,
    samples: Vec<Vec<<FiniteField<UField> as Ring>::Element>>,
) -> MultivariatePolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    let mut gp = MultivariatePolynomial::new(a.nvars, &a.field, None, None);

    // solve the transposed Vandermonde system
    for (((c, ex), sample), rhs) in shape.iter().zip(&row_sample_values).zip(&samples) {
        if c.nterms == 1 {
            let coeff = a.field.div(&rhs[0], &sample[0]);
            let mut ee: SmallVec<[E; INLINED_EXPONENTS]> = c.exponents(0).into();
            ee[main_var] = *ex;
            gp.append_monomial(coeff, &ee);
            continue;
        }

        // construct the master polynomial (1-s1)*(1-s2)*... efficiently
        let mut master = vec![a.field.zero(); sample.len() + 1];
        master[0] = a.field.one();

        for (i, x) in sample.iter().take(c.nterms).enumerate() {
            let first = &mut master[0];
            let mut old_last = *first;
            a.field.mul_assign(first, &a.field.neg(x));
            for m in &mut master[1..=i] {
                let ov = *m;
                a.field.mul_assign(m, &a.field.neg(x));
                a.field.add_assign(m, &old_last);
                old_last = ov;
            }
            master[i + 1] = a.field.one();
        }

        for (i, s) in sample.iter().take(c.nterms).enumerate() {
            let mut norm = a.field.one();

            // sample master/(1-s_i) by using the factorized form
            for (j, l) in sample.iter().enumerate() {
                if j != i {
                    a.field.mul_assign(&mut norm, &a.field.sub(s, l))
                }
            }

            // divide out 1-s_i
            let mut coeff = a.field.zero();
            let mut last_q = a.field.zero();
            for (m, rhs) in master.iter().skip(1).zip(rhs).rev() {
                last_q = a.field.add(m, &a.field.mul(s, &last_q));
                a.field.add_mul_assign(&mut coeff, &last_q, rhs);
            }
            a.field.div_assign(&mut coeff, &norm);

            // divide by the Vandermonde row since the Vandermonde matrices should start with a 1
            a.field.div_assign(&mut coeff, s);

            let mut ee: SmallVec<[E; INLINED_EXPONENTS]> = c.exponents(i).into();
            ee[main_var] = *ex;

            gp.append_monomial(coeff, &ee);
        }
    }

    gp
}

#[instrument(level = "trace", fields(%a, %b))]
fn construct_new_image_single_scale<UField: FiniteFieldWorkspace, E: Exponent>(
    a: &MultivariatePolynomial<FiniteField<UField>, E>,
    b: &MultivariatePolynomial<FiniteField<UField>, E>,
    a_ldegree: E,
    b_ldegree: E,
    bounds: &mut [E],
    single_scale: usize,
    vars: &[usize],
    main_var: usize,
    shape: &[(MultivariatePolynomial<FiniteField<UField>, E>, E)],
) -> Result<MultivariatePolynomial<FiniteField<UField>, E>, GCDError>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    if vars.is_empty() {
        // return gcd divided by the single scale factor
        let g = MultivariatePolynomial::univariate_gcd(a, b);

        if g.ldegree(main_var) < bounds[main_var] {
            // original image and form and degree bounds are unlucky
            // change the bound and try a new prime
            debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
            bounds[main_var] = g.ldegree(main_var);
            return Err(GCDError::BadOriginalImage);
        }

        if g.ldegree(main_var) > bounds[main_var] {
            return Err(GCDError::BadCurrentImage);
        }

        // check if all the monomials of the image appear in the shape
        // if not, the original shape is bad
        for m in g.into_iter() {
            if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                debug!("Bad shape: terms missing");
                return Err(GCDError::BadOriginalImage);
            }
        }

        // construct the scaling coefficient
        let (_, d) = &shape[single_scale];
        for t in &g {
            if t.exponents[main_var] == *d {
                let scale_factor = a.field.neg(&a.field.inv(t.coefficient)); // TODO: why -1?
                return Ok(g.mul_coeff(scale_factor));
            }
        }

        // the scaling term is missing, so the assumed form is wrong
        debug!("Bad original image");
        return Err(GCDError::BadOriginalImage);
    }

    let mut rng = rand::thread_rng();

    let mut failure_count = 0;

    // store a table for variables raised to a certain power
    let mut cache = (0..a.nvars)
        .map(|i| {
            vec![
                a.field.zero();
                min(
                    max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                    POW_CACHE_SIZE
                )
            ]
        })
        .collect::<Vec<_>>();

    // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
    let (row_sample_values, samples) = 'find_root_sample: loop {
        for v in &mut cache {
            for vi in v {
                *vi = a.field.zero();
            }
        }

        let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
            .iter()
            .map(|i| {
                (
                    *i,
                    a.field.sample(
                        &mut rng,
                        (
                            1,
                            a.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64) as i64,
                        ),
                    ),
                )
            })
            .collect();

        let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system
        let mut samples_needed = 0;
        for (c, _) in shape.iter() {
            samples_needed = samples_needed.max(c.nterms);
            let mut row = Vec::with_capacity(c.nterms);
            let mut seen = HashSet::new();

            for t in c {
                // evaluate each exponent
                let mut c = a.field.one();
                for &(n, v) in &r_orig {
                    let exp = t.exponents[n].to_u32() as usize;
                    if exp > 0 {
                        if exp < cache[n].len() {
                            if FiniteField::<UField>::is_zero(&cache[n][exp]) {
                                cache[n][exp] = a.field.pow(&v, exp as u64);
                            }

                            a.field.mul_assign(&mut c, &cache[n][exp]);
                        } else {
                            a.field.mul_assign(&mut c, &a.field.pow(&v, exp as u64));
                        }
                    }
                }
                row.push(c);

                // check if each element is unique
                if !seen.insert(a.field.from_element(c)) {
                    debug!("Duplicate element: restarting");
                    continue 'find_root_sample;
                }
            }

            row_sample_values.push(row);
        }

        let mut samples = vec![Vec::with_capacity(samples_needed); shape.len()];
        let mut r = r_orig.clone();

        let a_eval = evaluate_exponents(a, &r_orig, &mut cache);
        let b_eval = evaluate_exponents(b, &r_orig, &mut cache);

        let mut a_current = Cow::Borrowed(&a_eval);
        let mut b_current = Cow::Borrowed(&b_eval);

        let mut a_poly = MultivariatePolynomial::new(
            a.nvars,
            &a.field,
            Some(a.degree(main_var).to_u32() as usize + 1),
            None,
        );
        let mut b_poly = MultivariatePolynomial::new(
            b.nvars,
            &b.field,
            Some(b.degree(main_var).to_u32() as usize + 1),
            None,
        );

        for sample_index in 0..samples_needed {
            // sample at r^i
            if sample_index > 0 {
                for (c, rr) in r.iter_mut().zip(&r_orig) {
                    *c = (c.0, a.field.mul(&c.1, &rr.1));
                }

                for (c, e) in a_current.to_mut().iter_mut().zip(&a_eval) {
                    a.field.mul_assign(c, e);
                }
                for (c, e) in b_current.to_mut().iter_mut().zip(&b_eval) {
                    b.field.mul_assign(c, e);
                }
            }

            // now construct the univariate polynomials from the current evaluated monomials
            evaluate_using_exponents(a, &a_current, main_var, &mut a_poly);
            evaluate_using_exponents(b, &b_current, main_var, &mut b_poly);

            if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                continue 'find_root_sample;
            }

            let g = MultivariatePolynomial::univariate_gcd(&a_poly, &b_poly);
            debug!(
                "GCD of sample at point {:?} in main var {}: {}",
                r, main_var, g
            );

            if g.ldegree(main_var) < bounds[main_var] {
                // original image and form and degree bounds are unlucky
                // change the bound and try a new prime

                debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                bounds[main_var] = g.ldegree(main_var);
                return Err(GCDError::BadOriginalImage);
            }

            if g.ldegree(main_var) > bounds[main_var] {
                failure_count += 1;
                if failure_count > 2 {
                    // p is likely unlucky
                    debug!(
                        "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                        a,
                        b,
                        a.field.get_prime(),
                        r,
                        g
                    );
                    return Err(GCDError::BadCurrentImage);
                }
                debug!("Degree too high");
                continue 'find_root_sample;
            }

            // construct the scaling coefficient
            let mut scale_factor = a.field.one();
            let mut coeff = a.field.one();
            let (c, d) = &shape[single_scale];
            for (n, v) in r.iter() {
                // TODO: can be taken from row?
                a.field.mul_assign(
                    &mut coeff,
                    &a.field.pow(v, c.exponents(0)[*n].to_u32() as u64),
                );
            }

            let mut found = false;
            for t in &g {
                if t.exponents[main_var] == *d {
                    scale_factor = g.field.div(&coeff, t.coefficient);
                    found = true;
                    break;
                }
            }

            if !found {
                // the scaling term is missing, so the assumed form is wrong
                debug!("Bad original image");
                return Err(GCDError::BadOriginalImage);
            }

            // check if all the monomials of the image appear in the shape
            // if not, the original shape is bad
            for m in g.into_iter() {
                if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                    debug!("Bad shape: terms missing");
                    return Err(GCDError::BadOriginalImage);
                }
            }

            // construct the right-hand side
            'rhs: for (i, (rhs, (shape_part, exp))) in samples.iter_mut().zip(shape).enumerate() {
                // we may not need all terms
                if rhs.len() == shape_part.nterms {
                    continue;
                }

                // find the associated term in the sample, trying the usual place first
                if i < g.nterms && g.exponents(i)[main_var] == *exp {
                    rhs.push(a.field.neg(&a.field.mul(&g.coefficients[i], &scale_factor)));
                } else {
                    // find the matching term if it exists
                    for m in g.into_iter() {
                        if m.exponents[main_var] == *exp {
                            rhs.push(a.field.neg(&a.field.mul(m.coefficient, &scale_factor)));
                            continue 'rhs;
                        }
                    }

                    rhs.push(a.field.zero());
                }
            }
        }

        break (row_sample_values, samples);
    };

    Ok(solve_vandermonde(
        a,
        main_var,
        shape,
        row_sample_values,
        samples,
    ))
}

/// Construct an image in the case where no monomial in the main variable is a single term.
/// Using Javadi's method to solve the normalization problem, we first determine the coefficients of a single monomial using
/// Gaussian elimination. Then, we are back in the single term case and we use a Vandermonde
/// matrix to solve for every coefficient.
#[instrument(level = "trace", fields(%a, %b))]
fn construct_new_image_multiple_scales<UField: FiniteFieldWorkspace, E: Exponent>(
    a: &MultivariatePolynomial<FiniteField<UField>, E>,
    b: &MultivariatePolynomial<FiniteField<UField>, E>,
    a_ldegree: E,
    b_ldegree: E,
    bounds: &mut [E],
    vars: &[usize],
    main_var: usize,
    shape: &[(MultivariatePolynomial<FiniteField<UField>, E>, E)],
) -> Result<MultivariatePolynomial<FiniteField<UField>, E>, GCDError>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    let mut rng = rand::thread_rng();

    let mut failure_count = 0;

    // store a table for variables raised to a certain power
    let mut cache = (0..a.nvars)
        .map(|i| {
            vec![
                a.field.zero();
                min(
                    max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                    POW_CACHE_SIZE
                )
            ]
        })
        .collect::<Vec<_>>();

    // sort the shape based on the number of terms in the coefficient
    let mut shape_map: Vec<_> = (0..shape.len()).collect();
    shape_map.sort_unstable_by_key(|i| shape[*i].0.nterms);

    let mut scaling_var_relations: Vec<Vec<<FiniteField<UField> as Ring>::Element>> = vec![];

    let max_terms = shape[*shape_map.last().unwrap()].0.nterms;

    // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
    let (row_sample_values, samples) = 'find_root_sample: loop {
        for v in &mut cache {
            for vi in v {
                *vi = a.field.zero();
            }
        }

        let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
            .iter()
            .map(|i| {
                (
                    *i,
                    a.field.sample(
                        &mut rng,
                        (
                            1,
                            a.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64) as i64,
                        ),
                    ),
                )
            })
            .collect();

        let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system

        let max_samples_needed = 2 * max_terms - 1;
        for (c, _) in shape.iter() {
            let mut row = Vec::with_capacity(c.nterms);
            let mut seen = HashSet::new();

            for t in c {
                // evaluate each exponent
                let mut c = a.field.one();
                for &(n, v) in &r_orig {
                    let exp = t.exponents[n].to_u32() as usize;
                    if exp > 0 {
                        if exp < cache[n].len() {
                            if FiniteField::<UField>::is_zero(&cache[n][exp]) {
                                cache[n][exp] = a.field.pow(&v, exp as u64);
                            }

                            a.field.mul_assign(&mut c, &cache[n][exp]);
                        } else {
                            a.field.mul_assign(&mut c, &a.field.pow(&v, exp as u64));
                        }
                    }
                }
                row.push(c);

                // check if each element is unique
                if !seen.insert(a.field.from_element(c)) {
                    debug!("Duplicate element: restarting");
                    continue 'find_root_sample;
                }
            }

            row_sample_values.push(row);
        }

        let mut samples = vec![Vec::with_capacity(max_samples_needed); shape.len()];
        let mut r = r_orig.clone();

        let a_eval = evaluate_exponents(a, &r_orig, &mut cache);
        let b_eval = evaluate_exponents(b, &r_orig, &mut cache);

        let mut a_current = Cow::Borrowed(&a_eval);
        let mut b_current = Cow::Borrowed(&b_eval);

        let mut a_poly = MultivariatePolynomial::new(
            a.nvars,
            &a.field,
            Some(a.degree(main_var).to_u32() as usize + 1),
            None,
        );
        let mut b_poly = MultivariatePolynomial::new(
            b.nvars,
            &b.field,
            Some(b.degree(main_var).to_u32() as usize + 1),
            None,
        );

        let mut second_index = 1;
        let mut solved_coeff = None;
        for sample_index in 0..max_samples_needed {
            if solved_coeff.is_some() && sample_index >= max_terms {
                // we have enough samples
                break;
            }

            // sample at r^i
            if sample_index > 0 {
                for (c, rr) in r.iter_mut().zip(&r_orig) {
                    *c = (c.0, a.field.mul(&c.1, &rr.1));
                }

                for (c, e) in a_current.to_mut().iter_mut().zip(&a_eval) {
                    a.field.mul_assign(c, e);
                }
                for (c, e) in b_current.to_mut().iter_mut().zip(&b_eval) {
                    b.field.mul_assign(c, e);
                }
            }

            // now construct the univariate polynomials from the current evaluated monomials
            evaluate_using_exponents(a, &a_current, main_var, &mut a_poly);
            evaluate_using_exponents(b, &b_current, main_var, &mut b_poly);

            if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                continue 'find_root_sample;
            }

            let mut g = MultivariatePolynomial::univariate_gcd(&a_poly, &b_poly);
            debug!(
                "GCD of sample at point {:?} in main var {}: {}",
                r, main_var, g
            );

            if g.ldegree(main_var) < bounds[main_var] {
                // original image and form and degree bounds are unlucky
                // change the bound and try a new prime

                debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                bounds[main_var] = g.ldegree(main_var);
                return Err(GCDError::BadOriginalImage);
            }

            if g.ldegree(main_var) > bounds[main_var] {
                failure_count += 1;
                if failure_count > 2 {
                    // p is likely unlucky
                    debug!(
                        "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                        a,
                        b,
                        a.field.get_prime(),
                        r,
                        g
                    );
                    return Err(GCDError::BadCurrentImage);
                }
                debug!("Degree too high");
                continue 'find_root_sample;
            }

            // check if all the monomials of the image appear in the shape
            // if not, the original shape is bad
            for m in g.into_iter() {
                if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                    debug!("Bad shape: terms missing");
                    return Err(GCDError::BadOriginalImage);
                }
            }

            // set the coefficient of the scaling term in the gcd to 1
            let (_, d) = &shape[shape_map[0]];
            let mut found = false;
            for t in &g {
                if t.exponents[main_var] == *d {
                    let scale_factor = g.field.inv(t.coefficient);
                    g = g.mul_coeff(scale_factor);
                    found = true;
                    break;
                }
            }

            if !found {
                // the scaling term is missing, so the sample point is bad
                debug!("Bad sample point: scaling term missing");
                // TODO: check if this happen a number of times in a row
                // as the prime may be too small to generate n samples that
                // all contain the scaling term
                continue 'find_root_sample;
            }

            // construct the right-hand side
            'rhs: for (i, (rhs, (shape_part, exp))) in samples.iter_mut().zip(shape).enumerate() {
                // we may not need all terms
                if solved_coeff.is_some() && rhs.len() == shape_part.nterms {
                    continue;
                }

                // find the associated term in the sample, trying the usual place first
                if i < g.nterms && g.exponents(i)[main_var] == *exp {
                    rhs.push(g.coefficients[i]);
                } else {
                    // find the matching term if it exists
                    for m in g.into_iter() {
                        if m.exponents[main_var] == *exp {
                            rhs.push(*m.coefficient);
                            continue 'rhs;
                        }
                    }

                    rhs.push(a.field.zero());
                }
            }

            // see if we have collected enough samples to solve for the scaling factor
            while solved_coeff.is_none() {
                // try to solve the system!
                let vars_scale = shape[shape_map[0]].0.nterms - 1;
                let vars_second = shape[shape_map[second_index]].0.nterms;
                let samples_needed = vars_scale + vars_second;
                let rows = samples_needed + scaling_var_relations.len();

                if sample_index + 1 < samples_needed {
                    break; // obtain more samples
                }

                let mut gfm = SmallVec::with_capacity(rows * samples_needed);
                let mut new_rhs = SmallVec::with_capacity(rows);

                for sample_index in 0..samples_needed {
                    let rhs_sec = &samples[shape_map[second_index]][sample_index];
                    let row_eval_sec = &row_sample_values[shape_map[second_index]];
                    let row_eval_first = &row_sample_values[shape_map[0]];

                    // assume first constant is 1, which will form the rhs of our equation
                    let actual_rhs = a.field.mul(
                        rhs_sec,
                        &a.field.pow(&row_eval_first[0], sample_index as u64 + 1),
                    );

                    for aa in row_eval_sec {
                        gfm.push(a.field.pow(aa, sample_index as u64 + 1));
                    }

                    // place the scaling term variables at the end
                    for aa in &row_eval_first[1..] {
                        gfm.push(
                            a.field.neg(
                                &a.field
                                    .mul(rhs_sec, &a.field.pow(aa, sample_index as u64 + 1)),
                            ),
                        );
                    }

                    new_rhs.push(actual_rhs);
                }

                // add extra relations between the scaling term variables coming from previous tries
                // that yielded underdetermined systems
                for extra_relations in &scaling_var_relations {
                    for _ in 0..vars_second {
                        gfm.push(a.field.zero());
                    }

                    for v in &extra_relations[..vars_scale] {
                        gfm.push(*v);
                    }
                    new_rhs.push(*extra_relations.last().unwrap());
                }

                let m = Matrix {
                    shape: (rows as u32, samples_needed as u32),
                    data: gfm,
                    field: a.field,
                };
                let rhs = Matrix {
                    shape: (rows as u32, 1),
                    data: new_rhs,
                    field: a.field,
                };

                match m.solve(&rhs) {
                    Ok(r) => {
                        debug!(
                            "Solved with {} and {} term",
                            shape[shape_map[0]].0, shape[shape_map[second_index]].0
                        );

                        let mut r = r.data;
                        r.drain(0..vars_second);
                        solved_coeff = Some(r);
                    }
                    Err(LinearSolverError::Underdetermined {
                        row_reduced_matrix, ..
                    }) => {
                        // extract relations between the variables in the scaling term from the row reduced augmented matrix
                        let mat = row_reduced_matrix.expect("Row reduced matrix missing");

                        debug!(
                            "Underdetermined system {} and {} term; row reduction={}, rhs={}",
                            shape[shape_map[0]].0, shape[shape_map[second_index]].0, mat, rhs
                        );

                        for x in mat.row_iter() {
                            if x[..vars_second].iter().all(FiniteField::<UField>::is_zero)
                                && x.iter().any(|y| !FiniteField::<UField>::is_zero(y))
                            {
                                scaling_var_relations.push(x[vars_second..].to_vec());
                            }
                        }

                        second_index += 1;
                        if second_index == shape.len() {
                            panic!(
                                "Could not solve for the scaling coefficients: a={}, b={}, mat={}, rhs={}",
                                a,
                                b,
                                mat,
                                rhs,
                            );
                        }
                    }
                    Err(LinearSolverError::Inconsistent) => {
                        debug!("Inconsistent system");
                        return Err(GCDError::BadOriginalImage);
                    }
                    Err(LinearSolverError::NotSquare) => {
                        unreachable!()
                    }
                }
            }
        }

        if let Some(r) = solved_coeff {
            // evaluate the scaling term for every sample
            let mut lcoeff_cache = Vec::with_capacity(max_terms);
            for sample_index in 0..max_terms {
                let row_eval_first = &row_sample_values[shape_map[0]];
                let mut scaling_factor = a.field.pow(&row_eval_first[0], sample_index as u64 + 1); // coeff eval is 1
                for (exp_eval, coeff_eval) in row_sample_values[shape_map[0]][1..].iter().zip(&r) {
                    a.field.add_mul_assign(
                        &mut scaling_factor,
                        coeff_eval,
                        &a.field.pow(exp_eval, sample_index as u64 + 1),
                    );
                }
                lcoeff_cache.push(scaling_factor);
                debug!(
                    "Scaling fac {}: {}",
                    sample_index,
                    a.field.from_element(scaling_factor)
                );
            }

            for ((c, _), rhs) in shape.iter().zip(&mut samples) {
                rhs.truncate(c.nterms); // drop unneeded samples
                for (r, scale) in rhs.iter_mut().zip(&lcoeff_cache) {
                    a.field.mul_assign(r, scale);
                }
            }
        } else {
            debug!(
                "Could not solve the system with just 2 terms: a={}, b={}",
                a, b
            );
        }

        break (row_sample_values, samples);
    };

    Ok(solve_vandermonde(
        a,
        main_var,
        shape,
        row_sample_values,
        samples,
    ))
}

impl<UField: FiniteFieldWorkspace, E: Exponent> MultivariatePolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    fn univariate_gcd(a: &Self, b: &Self) -> Self {
        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }

        let mut c = a.clone();
        let mut d = b.clone();
        if a.ldegree_max() < b.ldegree_max() {
            mem::swap(&mut c, &mut d);
        }

        // TODO: there exists an efficient algorithm for univariate poly
        // division in a finite field using FFT
        let mut r = c.quot_rem_univariate(&mut d).1;
        while !r.is_zero() {
            c = d;
            d = r;
            r = c.quot_rem_univariate(&mut d).1;
        }

        // normalize the gcd
        let l = *d.coefficients.last().unwrap();
        for x in &mut d.coefficients {
            a.field.div_assign(x, &l);
        }

        d
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// a finite field of size `p`.
    pub fn sample_polynomial(
        &self,
        v: usize,
        r: &[(usize, <FiniteField<UField> as Ring>::Element)],
        cache: &mut [Vec<<FiniteField<UField> as Ring>::Element>],
        tm: &mut HashMap<E, <FiniteField<UField> as Ring>::Element>,
    ) -> Self {
        for mv in self.into_iter() {
            let mut c = *mv.coefficient;
            for &(n, vv) in r {
                let exp = mv.exponents[n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[n].len() {
                        if FiniteField::<UField>::is_zero(&cache[n][exp]) {
                            cache[n][exp] = self.field.pow(&vv, exp as u64);
                        }

                        self.field.mul_assign(&mut c, &cache[n][exp]);
                    } else {
                        self.field
                            .mul_assign(&mut c, &self.field.pow(&vv, exp as u64));
                    }
                }
            }

            tm.entry(mv.exponents[v])
                .and_modify(|e| self.field.add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = MultivariatePolynomial::new(self.nvars, &self.field, None, None);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm.drain() {
            if !FiniteField::<UField>::is_zero(&c) {
                e[v] = k;
                res.append_monomial(c, &e);
                e[v] = E::zero();
            }
        }

        res
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// a finite field of size `p`. The exponent of `v` should be small.
    pub fn sample_polynomial_small_exponent(
        &self,
        v: usize,
        r: &[(usize, <FiniteField<UField> as Ring>::Element)],
        cache: &mut [Vec<<FiniteField<UField> as Ring>::Element>],
        tm: &mut [<FiniteField<UField> as Ring>::Element],
    ) -> MultivariatePolynomial<FiniteField<UField>, E> {
        for mv in self.into_iter() {
            let mut c = *mv.coefficient;
            for &(n, vv) in r {
                let exp = mv.exponents[n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[n].len() {
                        if FiniteField::<UField>::is_zero(&cache[n][exp]) {
                            cache[n][exp] = self.field.pow(&vv, exp as u64);
                        }

                        self.field.mul_assign(&mut c, &cache[n][exp]);
                    } else {
                        self.field
                            .mul_assign(&mut c, &self.field.pow(&vv, exp as u64));
                    }
                }
            }

            let expv = mv.exponents[v].to_u32() as usize;
            self.field.add_assign(&mut tm[expv], &c);
        }

        // TODO: add bounds estimate
        let mut res = MultivariatePolynomial::new(self.nvars, &self.field, None, None);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm.iter_mut().enumerate() {
            if !FiniteField::<UField>::is_zero(c) {
                e[v] = E::from_u32(k as u32);
                res.append_monomial_back(mem::replace(c, self.field.zero()), &e);
                e[v] = E::zero();
            }
        }

        res
    }

    /// Find the upper bound of a variable `var` in the gcd.
    /// This is done by computing the univariate gcd by
    /// substituting all variables except `var`. This
    /// upper bound could be too tight due to an unfortunate
    /// sample point, but this is rare.
    fn get_gcd_var_bound(ap: &Self, bp: &Self, vars: &[usize], var: usize) -> E {
        let mut rng = rand::thread_rng();

        // store a table for variables raised to a certain power
        let mut cache = (0..ap.nvars)
            .map(|i| {
                vec![
                    ap.field.zero();
                    min(
                        max(ap.degree(i), bp.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // store a power map for the univariate polynomials that will be sampled
        // the sampling_polynomial routine will set the power to 0 after use
        let mut tm = HashMap::with_capacity_and_hasher(INITIAL_POW_MAP_SIZE, Default::default());

        // generate random numbers for all non-leading variables
        // TODO: apply a Horner scheme to speed up the substitution?
        let (_, a1, b1) = loop {
            for v in &mut cache {
                for vi in v {
                    *vi = ap.field.zero();
                }
            }

            let r: Vec<_> = vars
                .iter()
                .map(|i| {
                    (
                        *i,
                        ap.field.sample(
                            &mut rng,
                            (
                                1,
                                ap.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64) as i64,
                            ),
                        ),
                    )
                })
                .collect();

            let a1 = ap.sample_polynomial(var, &r, &mut cache, &mut tm);
            let b1 = bp.sample_polynomial(var, &r, &mut cache, &mut tm);

            if a1.ldegree(var) == ap.degree(var) && b1.ldegree(var) == bp.degree(var) {
                break (r, a1, b1);
            }

            debug!(
                "Degree error during sampling: trying again: a={}, a1=={}, bp={}, b1={}",
                ap, a1, bp, b1
            );
        };

        let g1 = MultivariatePolynomial::univariate_gcd(&a1, &b1);
        g1.ldegree_max()
    }

    /// Compute the gcd shape of two polynomials in a finite field by filling in random
    /// numbers.
    #[instrument(level = "debug", skip_all)]
    fn gcd_shape_modular(
        a: &Self,
        b: &Self,
        vars: &[usize],         // variables
        bounds: &mut [E],       // degree bounds
        tight_bounds: &mut [E], // tighter degree bounds
    ) -> Option<Self> {
        let lastvar = *vars.last().unwrap();

        // if we are in the univariate case, return the univariate gcd
        // TODO: this is a modification of the algorithm!
        if vars.len() == 1 {
            let gg = MultivariatePolynomial::univariate_gcd(a, b);
            if gg.degree(vars[0]) > bounds[vars[0]] {
                return None;
            }
            bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
            return Some(gg);
        }

        // the gcd of the content in the last variable should be 1
        let c = MultivariatePolynomial::multivariate_content_gcd(a, b, lastvar);
        if !c.is_one() {
            debug!("Content in last variable is not 1, but {}", c);
            // TODO: we assume that a content of -1 is also allowed
            // like in the special case gcd_(-x0*x1,-x0-x0*x1)
            if c.nterms != 1 || c.coefficients[0] != a.field.neg(&a.field.one()) {
                return None;
            }
        }

        let gamma = MultivariatePolynomial::univariate_gcd(
            &a.lcoeff_last_varorder(vars),
            &b.lcoeff_last_varorder(vars),
        );

        let mut rng = rand::thread_rng();

        let mut failure_count = 0;

        'newfirstnum: loop {
            // if we had two failures, it may be that the tight degree bound
            // was too tight due to an unfortunate prime/evaluation, so we relax it
            if failure_count == 2 {
                debug!(
                    "Changing tight bound for x{} from {} to {}",
                    lastvar, tight_bounds[lastvar], bounds[lastvar]
                );
                tight_bounds[lastvar] = bounds[lastvar];
            }
            failure_count += 1;

            let v = loop {
                let a = a.field.sample(
                    &mut rng,
                    (
                        1,
                        a.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64) as i64,
                    ),
                );
                if !gamma.replace(lastvar, &a).is_zero() {
                    break a;
                }
            };

            debug!("Chosen variable: {}", a.field.from_element(v));
            let av = a.replace(lastvar, &v);
            let bv = b.replace(lastvar, &v);

            // performance dense reconstruction
            let mut gv = if vars.len() > 2 {
                match MultivariatePolynomial::gcd_shape_modular(
                    &av,
                    &bv,
                    &vars[..vars.len() - 1],
                    bounds,
                    tight_bounds,
                ) {
                    Some(x) => x,
                    None => return None,
                }
            } else {
                let gg = MultivariatePolynomial::univariate_gcd(&av, &bv);
                if gg.degree(vars[0]) > bounds[vars[0]] {
                    return None;
                }
                bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
                gg
            };

            debug!(
                "GCD shape suggestion for sample point {} and gamma {}: {}",
                a.field.from_element(v),
                gamma,
                gv
            );

            // construct a new assumed form
            let gfu = gv.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gg that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms > nx {
                    nx = c.nterms;
                }
                if c.nterms == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gv.nterms() - 1) / (gfu.len() - 1);
                if (gv.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let mut lc = gv.lcoeff_varorder(vars);

            let mut gseq = vec![gv.clone().mul_coeff(
                gamma
                    .field
                    .div(&gamma.replace(lastvar, &v).coefficients[0], &lc),
            )];
            let mut vseq = vec![v];

            // sparse reconstruction
            'newnum: loop {
                if gseq.len()
                    == (tight_bounds[lastvar].to_u32() + gamma.ldegree_max().to_u32() + 1) as usize
                {
                    break;
                }

                let v = loop {
                    let v = a.field.sample(
                        &mut rng,
                        (
                            1,
                            a.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64) as i64,
                        ),
                    );
                    if !gamma.replace(lastvar, &v).is_zero() {
                        // we need unique sampling points
                        if !vseq.contains(&v) {
                            break v;
                        }
                    }
                };

                let av = a.replace(lastvar, &v);
                let bv = b.replace(lastvar, &v);

                let rec = if let Some(single_scale) = single_scale {
                    construct_new_image_single_scale(
                        &av,
                        &bv,
                        av.degree(vars[0]),
                        bv.degree(vars[0]),
                        bounds,
                        single_scale,
                        &vars[1..vars.len() - 1],
                        vars[0],
                        &gfu,
                    )
                } else {
                    construct_new_image_multiple_scales(
                        &av,
                        &bv,
                        // NOTE: different from paper where they use a.degree(..)
                        // it could be that the degree in av is lower than that of a
                        // which means the sampling will never terminate
                        av.degree(vars[0]),
                        bv.degree(vars[0]),
                        bounds,
                        &vars[1..vars.len() - 1],
                        vars[0],
                        &gfu,
                    )
                };

                match rec {
                    Ok(r) => {
                        gv = r;
                    }
                    Err(GCDError::BadOriginalImage) => {
                        debug!("Bad original image");
                        continue 'newfirstnum;
                    }
                    Err(GCDError::BadCurrentImage) => {
                        debug!("Bad current image");
                        continue 'newnum;
                    }
                }

                lc = gv.lcoeff_varorder(vars);

                gseq.push(
                    gv.clone().mul_coeff(
                        gamma
                            .field
                            .div(&gamma.replace(lastvar, &v).coefficients[0], &lc),
                    ),
                );
                vseq.push(v);
            }

            // use interpolation to construct x_n dependence
            let mut gc = newton_interpolation(&vseq, &gseq, lastvar);
            debug!("Interpolated: {}", gc);

            // remove content in x_n (wrt all other variables)
            let cont = gc.multivariate_content(lastvar);
            if !cont.is_one() {
                debug!("Removing content in x{}: {}", lastvar, cont);
                gc = gc.divides(&cont).unwrap();
            }

            // do a probabilistic division test
            let (g1, a1, b1) = loop {
                // store a table for variables raised to a certain power
                let mut cache = (0..a.nvars)
                    .map(|i| {
                        vec![
                            a.field.zero();
                            min(
                                max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                                POW_CACHE_SIZE
                            )
                        ]
                    })
                    .collect::<Vec<_>>();

                let r: Vec<_> = vars
                    .iter()
                    .skip(1)
                    .map(|i| {
                        (
                            *i,
                            a.field.sample(
                                &mut rng,
                                (
                                    1,
                                    a.field.get_prime().to_u64().min(MAX_RNG_PREFACTOR as u64)
                                        as i64,
                                ),
                            ),
                        )
                    })
                    .collect();

                let g1 = gc.replace_all_except(vars[0], &r, &mut cache);

                if g1.ldegree(vars[0]) == gc.degree(vars[0]) {
                    let a1 = a.replace_all_except(vars[0], &r, &mut cache);
                    let b1 = b.replace_all_except(vars[0], &r, &mut cache);
                    break (g1, a1, b1);
                }
            };

            if g1.is_one() || (a1.divides(&g1).is_some() && b1.divides(&g1).is_some()) {
                return Some(gc);
            }

            // if the gcd is bad, we had a bad number
            debug!(
                "Division test failed: gcd may be bad or probabilistic division test is unlucky: a1 {} b1 {} g1 {}", a1, b1, g1
            );
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent> MultivariatePolynomial<R, E> {
    /// Get the content of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let a = self.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(a.len());
        for (c, _) in a {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// multivariate polynomial in all variables except `x`.
    pub fn multivariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        PolynomialGCD::gcd_multiple(af.into_values().collect())
    }

    /// Compute the gcd of the univariate content in `x`.
    pub fn univariate_content_gcd(
        &self,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = self.to_univariate_polynomial_list(x);
        let bf = b.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(af.len() + bf.len());
        for (c, _) in af.into_iter().chain(bf.into_iter()) {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// multivariate polynomial in all variables except `x`.
    pub fn multivariate_content_gcd(
        a: &MultivariatePolynomial<R, E>,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = a.to_multivariate_polynomial_list(&[x], false);
        let bf = b.to_multivariate_polynomial_list(&[x], false);

        let f = af.into_values().chain(bf.into_values()).collect();

        PolynomialGCD::gcd_multiple(f)
    }

    /// Apply a GCD repeatedly to a list of polynomials.
    #[inline(always)]
    pub fn repeated_gcd(mut f: Vec<MultivariatePolynomial<R, E>>) -> MultivariatePolynomial<R, E> {
        if f.len() == 1 {
            return f.swap_remove(0);
        }

        if f.len() == 2 {
            return MultivariatePolynomial::gcd(&f[0], &f[1]);
        }

        f.sort_unstable_by_key(|p| p.nterms);

        let mut gcd = f.pop().unwrap();
        for p in f {
            if R::one_is_gcd_unit() && gcd.is_one() {
                return gcd;
            }

            gcd = MultivariatePolynomial::gcd(&gcd, &p);
        }
        gcd
    }

    /// Compute a standard GCD-free basis. The input should not
    /// contain 0 or units.
    pub fn gcd_free_basis(mut polys: Vec<Self>) -> Vec<Self> {
        let mut i = 0;
        while i + 1 < polys.len() {
            if polys[i].is_one() {
                i += 1;
                continue;
            }

            let mut j = i + 1;
            while j < polys.len() {
                if polys[j].is_one() {
                    j += 1;
                    continue;
                }

                let g = MultivariatePolynomial::gcd(&polys[i], &polys[j]);
                if !g.is_one() {
                    polys[i] = &polys[i] / &g;
                    polys[j] = &polys[j] / &g;
                    polys.push(g);
                }

                j += 1;
            }

            i += 1;
        }

        polys.retain(|p| !p.is_one());
        polys
    }

    /// Compute the GCD for simple cases.
    #[inline(always)]
    fn simple_gcd(
        a: &MultivariatePolynomial<R, E>,
        b: &MultivariatePolynomial<R, E>,
    ) -> Option<MultivariatePolynomial<R, E>> {
        if a == b {
            return Some(a.clone());
        }

        if a.is_zero() {
            return Some(b.clone());
        }
        if b.is_zero() {
            return Some(a.clone());
        }

        if a.is_one() {
            return Some(a.clone());
        }

        if b.is_one() {
            return Some(b.clone());
        }

        if a.is_constant() {
            let mut gcd = a.coefficients[0].clone();
            for c in &b.coefficients {
                gcd = a.field.gcd(&gcd, c);
                if R::one_is_gcd_unit() && a.field.is_one(&gcd) {
                    break;
                }
            }
            return Some(MultivariatePolynomial::new_from_constant(a, gcd));
        }

        if b.is_constant() {
            let mut gcd = b.coefficients[0].clone();
            for c in &a.coefficients {
                gcd = a.field.gcd(&gcd, c);
                if R::one_is_gcd_unit() && a.field.is_one(&gcd) {
                    break;
                }
            }
            return Some(MultivariatePolynomial::new_from_constant(a, gcd));
        }

        None
    }

    /// Compute the gcd of two multivariate polynomials.
    #[instrument(skip_all)]
    pub fn gcd(
        a: &MultivariatePolynomial<R, E>,
        b: &MultivariatePolynomial<R, E>,
    ) -> MultivariatePolynomial<R, E> {
        debug_assert_eq!(a.nvars, b.nvars);
        debug!("gcd of {} and {}", a, b);

        if let Some(g) = MultivariatePolynomial::simple_gcd(a, b) {
            debug!("Simple {} ", g);
            return g;
        }

        // a and b are only copied when needed
        let mut a = Cow::Borrowed(a);
        let mut b = Cow::Borrowed(b);

        // determine the maximum shared power of every variable
        let mut shared_degree: SmallVec<[E; INLINED_EXPONENTS]> = a.exponents(0).into();
        for p in [&a, &b] {
            for e in p.exponents.chunks(p.nvars) {
                for (md, v) in shared_degree.iter_mut().zip(e) {
                    *md = (*md).min(*v);
                }
            }
        }

        // divide out the common factors
        if shared_degree.iter().any(|d| *d != E::zero()) {
            let aa = a.to_mut();
            for e in aa.exponents.chunks_mut(aa.nvars) {
                for (v, d) in e.iter_mut().zip(&shared_degree) {
                    *v = *v - *d;
                }
            }

            let bb = b.to_mut();
            for e in bb.exponents.chunks_mut(bb.nvars) {
                for (v, d) in e.iter_mut().zip(&shared_degree) {
                    *v = *v - *d;
                }
            }
        };

        let mut base_degree: SmallVec<[Option<E>; INLINED_EXPONENTS]> = smallvec![None; a.nvars];

        if let Some(g) = MultivariatePolynomial::simple_gcd(&a, &b) {
            return rescale_gcd(
                g,
                &shared_degree,
                &base_degree,
                &a.new_from_constant(a.field.one()),
            );
        }

        // check if the polynomial are functions of x^n, n > 1
        for p in [&a, &b] {
            for t in p.into_iter() {
                for (md, v) in base_degree.iter_mut().zip(t.exponents) {
                    if !v.is_zero() {
                        if let Some(mm) = md.as_mut() {
                            if *mm != E::one() {
                                *mm = mm.gcd(v);
                            }
                        } else {
                            *md = Some(*v);
                        }
                    }
                }
            }
        }

        // rename x^base_deg to x
        if base_degree
            .iter()
            .any(|d| d.is_some() && d.unwrap() > E::one())
        {
            let aa = a.to_mut();
            for e in aa.exponents.chunks_mut(aa.nvars) {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }

            let bb = b.to_mut();
            for e in bb.exponents.chunks_mut(bb.nvars) {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }
        }

        /// Undo simplifications made to the input polynomials and normalize the gcd.
        #[inline(always)]
        fn rescale_gcd<R: EuclideanDomain + PolynomialGCD<E>, E: Exponent>(
            mut g: MultivariatePolynomial<R, E>,
            shared_degree: &[E],
            base_degree: &[Option<E>],
            content: &MultivariatePolynomial<R, E>,
        ) -> MultivariatePolynomial<R, E> {
            if !content.is_one() {
                g = g * content;
            }

            if shared_degree.iter().any(|d| *d > E::from_u32(0))
                || base_degree
                    .iter()
                    .any(|d| d.map(|bd| bd > E::one()).unwrap_or(false))
            {
                for e in g.exponents.chunks_mut(g.nvars) {
                    for ((v, d), s) in e.iter_mut().zip(base_degree).zip(shared_degree) {
                        if let Some(d) = d {
                            *v = *v * *d;
                        }

                        *v += *s;
                    }
                }
            }

            PolynomialGCD::normalize(g)
        }

        if let Some(gcd) = PolynomialGCD::heuristic_gcd(&a, &b) {
            debug!("Heuristic gcd succeeded: {}", gcd.0);
            return rescale_gcd(
                gcd.0,
                &shared_degree,
                &base_degree,
                &a.new_from_constant(a.field.one()),
            );
        }

        // store which variables appear in which expression
        let mut scratch: SmallVec<[i32; INLINED_EXPONENTS]> = smallvec![0i32; a.nvars];
        for (p, inc) in [(&a, 1), (&b, 2)] {
            for t in p.into_iter() {
                for (e, ee) in scratch.iter_mut().zip(t.exponents) {
                    if !ee.is_zero() {
                        *e |= inc;
                    }
                }
            }
        }

        if a == b {
            debug!("Equal {} ", a);
            return rescale_gcd(
                a.into_owned(),
                &shared_degree,
                &base_degree,
                &MultivariatePolynomial::one(&b.field),
            );
        }

        // compute the gcd efficiently if some variables do not occur in both
        // polynomials
        if scratch.iter().any(|x| *x > 0 && *x < 3) {
            let inca: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 1 || *v == 3 { Some(i) } else { None })
                .collect();

            let incb: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 2 || *v == 3 { Some(i) } else { None })
                .collect();

            // extract the variables of b in the coefficient of a and vice versa
            let a1 = a.to_multivariate_polynomial_list(&incb, false);
            let b1 = b.to_multivariate_polynomial_list(&inca, false);

            let f = a1.into_values().chain(b1.into_values()).collect();

            return rescale_gcd(
                PolynomialGCD::gcd_multiple(f),
                &shared_degree,
                &base_degree,
                &MultivariatePolynomial::one(&a.field),
            );
        }

        // try if b divides a or vice versa, doing a heuristical length check first
        if a.nterms >= b.nterms && a.divides(&b).is_some() {
            return rescale_gcd(
                b.into_owned(),
                &shared_degree,
                &base_degree,
                &MultivariatePolynomial::one(&a.field),
            );
        }
        if a.nterms <= b.nterms && b.divides(&a).is_some() {
            return rescale_gcd(
                a.into_owned(),
                &shared_degree,
                &base_degree,
                &MultivariatePolynomial::one(&b.field),
            );
        }

        // check if the polynomial is linear in a variable and compute the gcd using the univariate content
        for (p1, p2) in [(&a, &b), (&b, &a)] {
            if let Some(var) = (0..p1.nvars).find(|v| p1.degree(*v) == E::one()) {
                let mut cont = p1.univariate_content(var);

                let p1_prim = p1.as_ref() / &cont;

                if !cont.is_one() || !R::one_is_gcd_unit() {
                    let cont_p2 = p2.univariate_content(var);
                    cont = MultivariatePolynomial::gcd(&cont, &cont_p2);
                }

                if p2.divides(&p1_prim).is_some() {
                    return rescale_gcd(p1_prim, &shared_degree, &base_degree, &cont);
                } else {
                    return rescale_gcd(
                        cont,
                        &shared_degree,
                        &base_degree,
                        &p1.new_from_constant(p1.field.one()),
                    );
                }
            }
        }

        let mut vars: SmallVec<[_; INLINED_EXPONENTS]> = scratch
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 3 { Some(i) } else { None })
            .collect();

        // determine safe bounds for variables in the gcd
        let mut bounds: SmallVec<[_; INLINED_EXPONENTS]> = (0..a.nvars)
            .map(|i| {
                let da = a.degree(i);
                let db = b.degree(i);
                if da < db {
                    da
                } else {
                    db
                }
            })
            .collect();

        // find better upper bounds for all variables
        // these bounds could actually be wrong due to an unfortunate prime or sampling points
        let mut tight_bounds = R::get_gcd_var_bounds(&a, &b, &vars, &bounds);

        // Determine a good variable ordering based on the estimated degree (decreasing) in the gcd.
        // If it is different from the input, make a copy and rearrange so that the
        // polynomials do not have to be sorted after filling in variables.
        vars.sort_by(|&i, &j| tight_bounds[j].cmp(&tight_bounds[i]));

        // strip the gcd of the univariate contents wrt the new first variable
        let content = if vars.len() > 1 {
            debug!("Starting univariate content computation in {}", vars[0]);
            let content = a.univariate_content_gcd(&b, vars[0]);
            debug!("GCD of content: {}", content);

            if !content.is_one() {
                a = Cow::Owned(a.as_ref() / &content);
                b = Cow::Owned(b.as_ref() / &content);
            }

            // even if variables got removed, benchmarks show that it is not
            // worth it do restart the gcd computation
            content
        } else {
            // get the integer content for univariate polynomials
            let uca = a.content();
            let ucb = b.content();
            let content = a.field.gcd(&a.content(), &b.content());
            let p = MultivariatePolynomial::new_from(&a, Some(1));

            if !a.field.is_one(&uca) {
                a = Cow::Owned(a.into_owned().div_coeff(&uca));
            }
            if !a.field.is_one(&ucb) {
                b = Cow::Owned(b.into_owned().div_coeff(&ucb));
            }

            p.add_monomial(content)
        };

        let rearrange = vars.len() > 1 && vars.windows(2).any(|s| s[0] > s[1]);
        if rearrange {
            debug!("Rearranging variables with map: {:?}", vars);
            a = Cow::Owned(a.rearrange(&vars, false));
            b = Cow::Owned(b.rearrange(&vars, false));

            let mut newbounds: SmallVec<[_; INLINED_EXPONENTS]> =
                smallvec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newbounds[x] = bounds[vars[x]];
            }
            bounds = newbounds;

            let mut newtight_bounds: SmallVec<[_; INLINED_EXPONENTS]> =
                smallvec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newtight_bounds[x] = tight_bounds[vars[x]];
            }
            tight_bounds = newtight_bounds;
        }

        let mut g = PolynomialGCD::gcd(
            &a,
            &b,
            &if rearrange {
                Cow::Owned((0..vars.len()).collect::<SmallVec<[usize; INLINED_EXPONENTS]>>())
            } else {
                Cow::Borrowed(&vars)
            },
            &mut bounds,
            &mut tight_bounds,
        );

        if rearrange {
            g = g.rearrange(&vars, true);
        }

        rescale_gcd(g, &shared_degree, &base_degree, &content)
    }
}

#[derive(Debug)]
pub enum HeuristicGCDError {
    MaxSizeExceeded,
    BadReconstruction,
}

impl<E: Exponent> MultivariatePolynomial<IntegerRing, E> {
    /// Perform a heuristic GCD algorithm.
    #[instrument(level = "debug", skip_all)]
    pub fn heuristic_gcd(&self, b: &Self) -> Result<(Self, Self, Self), HeuristicGCDError> {
        fn interpolate<E: Exponent>(
            mut gamma: MultivariatePolynomial<IntegerRing, E>,
            var: usize,
            xi: &Integer,
        ) -> MultivariatePolynomial<IntegerRing, E> {
            let mut g = MultivariatePolynomial::new_from(&gamma, None);
            let mut i = 0;
            let xi_half = xi / &Integer::Natural(2);
            while !gamma.is_zero() {
                // create xi-adic representation using the symmetric modulus
                let mut g_i = MultivariatePolynomial::new_from(&gamma, Some(gamma.nterms));
                for m in &gamma {
                    let mut c = IntegerRing::new().quot_rem(m.coefficient, xi).1;

                    if c > xi_half {
                        c -= xi;
                    }

                    if !IntegerRing::is_zero(&c) {
                        g_i.append_monomial(c, m.exponents);
                    }
                }

                for c in &mut g_i.coefficients {
                    *c = IntegerRing::new().quot_rem(c, xi).1;

                    if *c > xi_half {
                        *c -= xi;
                    }
                }

                // multiply with var^i
                let mut g_i_2 = g_i.clone();
                for x in g_i_2.exponents.chunks_mut(g_i_2.nvars) {
                    x[var] = E::from_u32(i);
                }

                g = g.add(g_i_2);

                gamma = (gamma - g_i).div_coeff(xi);
                i += 1;
            }
            g
        }

        debug!("a={}; b={}", self, b);

        // do integer GCD
        let content_gcd = self.field.gcd(&self.content(), &b.content());

        debug!("content={}", content_gcd);

        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);

        if !a.field.is_one(&content_gcd) {
            a = Cow::Owned(a.into_owned().div_coeff(&content_gcd));
            b = Cow::Owned(b.into_owned().div_coeff(&content_gcd));
        }

        debug!("a_red={}; b_red={}", a, b);

        if let Some(var) =
            (0..a.nvars).find(|x| a.degree(*x) > E::zero() && b.degree(*x) > E::zero())
        {
            let max_a = a
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Natural(0));

            let max_b = b
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Natural(0));

            let min = if max_a.abs_cmp(max_b) == Ordering::Greater {
                max_b.abs()
            } else {
                max_a.abs()
            };

            let mut xi = &(&min * &Integer::Natural(2)) + &Integer::Natural(29);

            for retry in 0..6 {
                debug!("round {}, xi={}", retry, xi);
                match &xi * &Integer::Natural(a.degree(var).max(b.degree(var)).to_u32() as i64) {
                    Integer::Natural(_) => {}
                    Integer::Large(r) => {
                        if r.as_limbs().len() > 4 {
                            debug!("big num {}", r);
                            return Err(HeuristicGCDError::MaxSizeExceeded);
                        }
                    }
                }

                let aa = a.replace(var, &xi);
                let bb = b.replace(var, &xi);

                let (gamma, co_fac_p, co_fac_q) = match aa.heuristic_gcd(&bb) {
                    Ok(x) => x,
                    Err(HeuristicGCDError::MaxSizeExceeded) => {
                        return Err(HeuristicGCDError::MaxSizeExceeded);
                    }
                    Err(HeuristicGCDError::BadReconstruction) => {
                        xi = IntegerRing::new()
                            .quot_rem(&(&xi * &Integer::Natural(73794)), &Integer::Natural(27011))
                            .0;
                        continue;
                    }
                };

                debug!("gamma={}", gamma);

                let g = interpolate(gamma, var, &xi);
                let g_cont = g.content();

                let gc = g.div_coeff(&g_cont);

                if let Some(q) = a.divides(&gc) {
                    if let Some(q1) = b.divides(&gc) {
                        debug!("match {} {}", q, q1);
                        return Ok((gc.mul_coeff(content_gcd), q, q1));
                    }
                }

                debug!("co_fac_p {}", co_fac_p);

                if !co_fac_p.is_zero() {
                    let a_co_fac = interpolate(co_fac_p, var, &xi);

                    if let Some(q) = a.divides(&a_co_fac) {
                        if let Some(q1) = b.divides(&q) {
                            return Ok((q.mul_coeff(content_gcd), a_co_fac, q1));
                        }
                    }
                }

                if !co_fac_q.is_zero() {
                    let b_co_fac = interpolate(co_fac_q, var, &xi);
                    debug!("cofac b {}", b_co_fac);

                    if let Some(q) = b.divides(&b_co_fac) {
                        if let Some(q1) = a.divides(&q) {
                            return Ok((q.mul_coeff(content_gcd), q1, b_co_fac));
                        }
                    }
                }

                xi = IntegerRing::new()
                    .quot_rem(&(&xi * &Integer::Natural(73794)), &Integer::Natural(27011))
                    .0;
            }

            Err(HeuristicGCDError::BadReconstruction)
        } else {
            Ok((
                MultivariatePolynomial::new_from_constant(self, content_gcd),
                a.into_owned(),
                b.into_owned(),
            ))
        }
    }

    /// Compute the gcd of multiple polynomials efficiently.
    /// `gcd(f0,f1,f2,...)=gcd(f0,f1+k2*f(2)+k3*f(3))`
    /// with high likelihood.
    pub fn gcd_multiple(
        mut f: Vec<MultivariatePolynomial<IntegerRing, E>>,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        assert!(!f.is_empty());

        let mut prime_index = 1; // skip prime 2
        let mut loop_counter = 0;
        loop {
            if f.len() == 1 {
                return f.swap_remove(0);
            }

            if f.len() == 2 {
                return MultivariatePolynomial::gcd(&f[0], &f[1]);
            }

            // check if any entry is a number, as the gcd is then the gcd of the contents
            if let Some(n) = f.iter().find(|x| x.is_constant()) {
                let mut gcd = n.content();
                for x in f.iter() {
                    if x.field.is_one(&gcd) {
                        break;
                    }

                    gcd = x.field.gcd(&gcd, &x.content());
                }
                return MultivariatePolynomial::new_from_constant(n, gcd);
            }

            // take the smallest element
            let index_smallest = f
                .iter()
                .enumerate()
                .min_by_key(|(_, v)| v.nterms)
                .unwrap()
                .0;

            let a = f.swap_remove(index_smallest);

            // add all other polynomials
            let term_bound = f.iter().map(|x| x.nterms).sum();
            let mut b = a.new_from(Some(term_bound));

            // prevent sampling f[i] and f[i+prime_len] with the same
            // prefactor every iteration
            let num_primes = if f.len() % SMALL_PRIMES.len() == 0 {
                SMALL_PRIMES.len() - 1
            } else {
                SMALL_PRIMES.len()
            };

            for p in f.iter() {
                let k = Integer::Natural(SMALL_PRIMES[prime_index % num_primes]);
                prime_index += 1;
                b = b + p.clone().mul_coeff(k);
            }

            let mut gcd = MultivariatePolynomial::gcd(&a, &b);

            if gcd.is_one() {
                return gcd;
            }

            // remove the content from the gcd before the divison test as the odds
            // of an unlucky content are high
            let content = gcd.content();
            gcd = gcd.div_coeff(&content);
            let mut content_gcd = content;

            let old_length = f.len();

            f.retain(|x| {
                if x.divides(&gcd).is_some() {
                    content_gcd = gcd.field.gcd(&content_gcd, &x.content());
                    false
                } else {
                    true
                }
            });

            gcd = gcd.mul_coeff(content_gcd);

            if f.is_empty() {
                return gcd;
            }

            debug!(
                "Multiply GCD not found in one try, current estimate: {}",
                gcd
            );

            f.push(gcd);

            if f.len() == old_length + 1 && loop_counter > 5 {
                debug!("Multiple GCD failed");
                return MultivariatePolynomial::repeated_gcd(f);
            }

            loop_counter += 1;
        }
    }

    /// Compute the gcd of two multivariate polynomials using Zippel's algorithm.
    /// TODO: provide a parallel implementation?
    #[instrument(level = "debug", skip_all)]
    fn gcd_zippel<UField: LargePrimes + FiniteFieldWorkspace + 'static>(
        a: &Self,
        b: &Self,
        vars: &[usize], // variables
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> Self
    where
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Ring>::Element: Copy,
        Integer: ToFiniteField<UField> + FromFiniteField<UField>,
    {
        debug!("Zippel gcd of {} and {}", a, b);
        #[cfg(debug_assertions)]
        {
            a.check_consistency();
            b.check_consistency();
        }

        // compute scaling factor in Z
        let gamma = a
            .field
            .gcd(&a.lcoeff_varorder(vars), &b.lcoeff_varorder(vars));
        debug!("gamma {}", gamma);

        let mut pi = 0;
        let primes = UField::get_primes();

        'newfirstprime: loop {
            pi += 1;

            if pi == primes.len() {
                a.check_consistency();
                b.check_consistency();
                panic!(
                    "Ran out of primes for gcd reconstruction.\ngcd({},{})",
                    a, b
                );
            }

            let mut p = primes[pi];
            let mut finite_field = FiniteField::<UField>::new(p);
            let mut gammap = gamma.to_finite_field(&finite_field);

            if FiniteField::<UField>::is_zero(&gammap) {
                continue 'newfirstprime;
            }

            let ap = a.to_finite_field(&finite_field);
            let bp = b.to_finite_field(&finite_field);

            debug!("New first image: gcd({},{}) mod {}", ap, bp, p);

            // calculate modular gcd image
            let mut gp = match MultivariatePolynomial::gcd_shape_modular(
                &ap,
                &bp,
                vars,
                bounds,
                tight_bounds,
            ) {
                Some(x) => x,
                None => {
                    debug!("Modular GCD failed: getting new prime");
                    continue 'newfirstprime;
                }
            };

            debug!("GCD suggestion: {}", gp);

            bounds[vars[0]] = gp.degree(vars[0]);

            // construct a new assumed form
            // we have to find the proper normalization
            let gfu = gp.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gf that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms > nx {
                    nx = c.nterms;
                }
                if c.nterms == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let gpc = gp.lcoeff_varorder(vars);
            let lcoeff_factor = gp.field.div(&gammap, &gpc);

            // construct the gcd suggestion in Z
            let mut gm = a.new_from(Some(gp.nterms));
            gm.nterms = gp.nterms;
            gm.exponents = gp.exponents.clone();
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|x| Integer::from_finite_field(&gp.field, gp.field.mul(x, &lcoeff_factor)))
                .collect();

            let mut m = Integer::from_prime(&finite_field); // size of finite field

            debug!("GCD suggestion with gamma: {} mod {} ", gm, p);

            let mut old_gm = a.new_from(None);

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                if gm == old_gm {
                    // divide by integer content
                    let gmc = gm.content();
                    let gc = gm.clone().div_coeff(&gmc);

                    debug!("Final suggested gcd: {}", gc);
                    if gc.is_one() || (a.divides(&gc).is_some() && b.divides(&gc).is_some()) {
                        return gc;
                    }

                    // if it does not divide, we need more primes
                    debug!("Does not divide: more primes needed");
                }

                old_gm = gm.clone();

                loop {
                    pi += 1;

                    if pi == LARGE_U32_PRIMES.len() {
                        a.check_consistency();
                        b.check_consistency();
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({},{})\nAttempt: {}\n vars: {:?}, bounds: {:?}; {:?}",
                            a, b, gm, vars, bounds, tight_bounds
                        );
                    }

                    p = primes[pi];
                    finite_field = FiniteField::<UField>::new(p);

                    gammap = gamma.to_finite_field(&finite_field);

                    if !FiniteField::<UField>::is_zero(&gammap) {
                        break;
                    }
                }

                let ap = a.to_finite_field(&finite_field);
                let bp = b.to_finite_field(&finite_field);
                debug!("New image: gcd({},{})", ap, bp);

                // for the univariate case, we don't need to construct an image
                if vars.len() == 1 {
                    gp = MultivariatePolynomial::univariate_gcd(&ap, &bp);
                    if gp.degree(vars[0]) < bounds[vars[0]] {
                        // original image and variable bound unlucky: restart
                        debug!("Unlucky original image: restart");
                        continue 'newfirstprime;
                    }

                    if gp.degree(vars[0]) > bounds[vars[0]] {
                        // prime is probably unlucky
                        debug!("Unlucky current image: try new one");
                        continue 'newprime;
                    }

                    for m in gp.into_iter() {
                        if gfu.iter().all(|(_, pow)| *pow != m.exponents[vars[0]]) {
                            debug!("Bad shape: terms missing");
                            continue 'newfirstprime;
                        }
                    }
                } else {
                    let rec = if let Some(single_scale) = single_scale {
                        construct_new_image_single_scale(
                            &ap,
                            &bp,
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        construct_new_image_multiple_scales(
                            &ap,
                            &bp,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in ap is lower than that of a
                            // which means the sampling will never terminate
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    };

                    match rec {
                        Ok(r) => {
                            gp = r;
                        }
                        Err(GCDError::BadOriginalImage) => continue 'newfirstprime,
                        Err(GCDError::BadCurrentImage) => continue 'newprime,
                    }
                }

                // scale the new image
                let gpc = gp.lcoeff_varorder(vars);
                gp = gp.mul_coeff(ap.field.div(&gammap, &gpc));
                debug!("gp: {} mod {}", gp, gp.field.get_prime());

                // use chinese remainder theorem to merge coefficients and map back to Z
                // terms could be missing in gp, but not in gm (TODO: check this?)
                let mut gpi = 0;
                for t in 0..gm.nterms {
                    let gpc = if gm.exponents(t) == gp.exponents(gpi) {
                        gpi += 1;
                        gp.coefficients[gpi - 1]
                    } else {
                        ap.field.zero()
                    };

                    let gmc = &mut gm.coefficients[t];
                    let coeff = if gmc.is_negative() {
                        a.field.add(gmc, &m)
                    } else {
                        gmc.clone()
                    };

                    *gmc = Integer::chinese_remainder(
                        coeff,
                        Integer::from_finite_field(&gp.field, gpc),
                        m.clone(),
                        Integer::from_prime(&gp.field),
                    );
                }

                a.field.mul_assign(&mut m, &Integer::from_prime(&gp.field));

                debug!("gm: {} from ring {}", gm, m);
            }
        }
    }
}

/// Polynomial GCD functions for a certain coefficient type `Self`.
pub trait PolynomialGCD<E: Exponent>: Ring {
    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )>;
    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E>;
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E>;
    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        loose_bounds: &[E],
    ) -> SmallVec<[E; INLINED_EXPONENTS]>;
    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E>;
}

impl<E: Exponent> PolynomialGCD<E> for IntegerRing {
    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // estimate if the heuristic gcd will overflow
        let mut max_deg_a = 0;
        let mut contains_a: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; a.nvars];
        for t in a {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_a[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_a {
                max_deg_a = deg;
            }
        }

        let mut max_deg_b = 0;
        let mut contains_b: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; b.nvars];
        for t in b {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_b[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_b {
                max_deg_b = deg;
            }
        }

        let num_shared_vars = contains_a
            .iter()
            .zip(&contains_b)
            .filter(|(a, b)| **a && **b)
            .count();

        if max_deg_a < 20 || max_deg_b < 20 || num_shared_vars < 3 && max_deg_a.min(max_deg_b) < 150
        {
            a.heuristic_gcd(b).ok()
        } else {
            None
        }
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::gcd_multiple(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<IntegerRing, E>,
        b: &MultivariatePolynomial<IntegerRing, E>,
        vars: &[usize],
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> MultivariatePolynomial<IntegerRing, E> {
        MultivariatePolynomial::gcd_zippel::<u32>(a, b, vars, bounds, tight_bounds)
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<IntegerRing, E>,
        b: &MultivariatePolynomial<IntegerRing, E>,
        vars: &[usize],
        loose_bounds: &[E],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut tight_bounds: SmallVec<[_; INLINED_EXPONENTS]> = loose_bounds.into();
        let mut i = 0;
        loop {
            let ap = a.to_finite_field(&FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            let bp = b.to_finite_field(&FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            if ap.nterms > 0
                && bp.nterms > 0
                && ap.last_exponents() == a.last_exponents()
                && bp.last_exponents() == b.last_exponents()
            {
                for var in vars.iter() {
                    let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                        vars.iter().filter(|i| *i != var).cloned().collect();
                    tight_bounds[*var] =
                        MultivariatePolynomial::get_gcd_var_bound(&ap, &bp, &vvars, *var);
                }
                break;
            } else {
                debug!("Variable bounds failed due to unlucky prime");
                i += 1;
            }
        }
        tight_bounds
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        if a.lcoeff().is_negative() {
            -a
        } else {
            a
        }
    }
}

impl<E: Exponent> PolynomialGCD<E> for RationalField {
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // TODO: restructure
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        // remove the content so that the polynomials have integer coefficients
        let content = a.field.gcd(&a.content(), &b.content());

        let mut a_int = MultivariatePolynomial::new(
            a.nvars,
            &IntegerRing::new(),
            Some(a.nterms),
            a.var_map.as_ref().map(|x| x.as_slice()),
        );

        for t in a {
            let coeff = a.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            a_int.append_monomial(coeff.numerator(), t.exponents);
        }

        let mut b_int = MultivariatePolynomial::new(
            b.nvars,
            &IntegerRing::new(),
            Some(b.nterms),
            b.var_map.as_ref().map(|x| x.as_slice()),
        );

        for t in b {
            let coeff = b.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            b_int.append_monomial(coeff.numerator(), t.exponents);
        }

        let res_int =
            MultivariatePolynomial::gcd_zippel::<u32>(&a_int, &b_int, vars, bounds, tight_bounds);

        let mut res = a.new_from(Some(res_int.nterms));

        for t in &res_int {
            res.append_monomial(
                a.field.mul(&t.coefficient.to_rational(), &content),
                t.exponents,
            );
        }

        res
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<RationalField, E>,
        b: &MultivariatePolynomial<RationalField, E>,
        vars: &[usize],
        loose_bounds: &[E],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut tight_bounds: SmallVec<[_; INLINED_EXPONENTS]> = loose_bounds.into();
        let mut i = 0;
        loop {
            let f = FiniteField::<u32>::new(LARGE_U32_PRIMES[i]);
            let ap = a.to_finite_field(&f);
            let bp = b.to_finite_field(&f);
            if ap.nterms > 0
                && bp.nterms > 0
                && ap.last_exponents() == a.last_exponents()
                && bp.last_exponents() == b.last_exponents()
            {
                for var in vars.iter() {
                    let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                        vars.iter().filter(|i| *i != var).cloned().collect();
                    tight_bounds[*var] =
                        MultivariatePolynomial::get_gcd_var_bound(&ap, &bp, &vvars, *var);
                }
                break;
            } else {
                debug!("Variable bounds failed due to unlucky prime");
                i += 1;
            }
        }
        tight_bounds
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        if a.lcoeff().is_negative() {
            -a
        } else {
            a
        }
    }
}

impl<UField: FiniteFieldWorkspace, E: Exponent> PolynomialGCD<E> for FiniteField<UField>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Ring>::Element: Copy,
{
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd(
        a: &MultivariatePolynomial<FiniteField<UField>, E>,
        b: &MultivariatePolynomial<FiniteField<UField>, E>,
        vars: &[usize],
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> MultivariatePolynomial<FiniteField<UField>, E> {
        assert!(!a.is_zero() || !b.is_zero());
        MultivariatePolynomial::gcd_shape_modular(a, b, vars, bounds, tight_bounds).unwrap()
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<FiniteField<UField>, E>,
        b: &MultivariatePolynomial<FiniteField<UField>, E>,
        vars: &[usize],
        loose_bounds: &[E],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut tight_bounds: SmallVec<[_; INLINED_EXPONENTS]> = loose_bounds.into();
        for var in vars {
            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            tight_bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(a, b, &vvars, *var);
        }
        tight_bounds
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}
