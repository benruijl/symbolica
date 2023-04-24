use ahash::HashMap;
use rand;
use smallvec::{smallvec, SmallVec};
use std::cmp::{max, min};
use std::mem;
use std::ops::Add;
use tracing::{debug, instrument, trace};

use crate::rings::finite_field::{FiniteField, FiniteFieldElement, ToFiniteField};
use crate::rings::integer::{Integer, IntegerRing};
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

/// The maximum power of a variable that is cached
pub const POW_CACHE_SIZE: usize = 1000;
pub const INITIAL_POW_MAP_SIZE: usize = 1000;

/// The upper bound of the range to be sampled during the computation of multiple gcds
pub const MAX_RNG_PREFACTOR: u32 = 5000;

enum GCDError {
    BadOriginalImage,
    BadCurrentImage,
}

fn newton_interpolation<UField, E: Exponent>(
    a: &[<FiniteField<UField> as Ring>::Element],
    u: &[MultivariatePolynomial<FiniteField<UField>, E>],
    x: usize, // the variable index to extend the polynomial by
) -> MultivariatePolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: Field,
{
    let field = u[0].field;

    // compute inverses
    let mut gammas = vec![];
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
    e[x] = E::from_u32(1);
    let xp = MultivariatePolynomial::from_monomial(field.one(), e, field);
    let mut u = v[v.len() - 1].clone();
    for k in (0..v.len() - 1).rev() {
        // TODO: prevent cloning
        u = u
            * (xp.clone()
                - MultivariatePolynomial::from_constant_with_nvars(a[k].clone(), xp.nvars, field))
            + v[k].clone();
    }
    u
}

#[instrument(level = "trace", fields(%ap, %bp))]
fn construct_new_image<E: Exponent>(
    ap: &MultivariatePolynomial<FiniteField<u32>, E>,
    bp: &MultivariatePolynomial<FiniteField<u32>, E>,
    aldegree: E,
    bldegree: E,
    bounds: &mut [E],
    single_scale: Option<usize>,
    nx: usize,
    vars: &[usize],
    var: usize,
    gfu: &[(MultivariatePolynomial<FiniteField<u32>, E>, E)],
) -> Result<MultivariatePolynomial<FiniteField<u32>, E>, GCDError> {
    let mut rng = rand::thread_rng();

    let mut system = vec![]; // coefficients for the linear system
    let mut ni = 0;
    let mut failure_count = 0;

    let mut rank_failure_count = 0;
    let mut last_rank = (0, 0);

    // store a table for variables raised to a certain power
    let mut cache = (0..ap.nvars)
        .map(|i| {
            vec![
                FiniteField::<u32>::zero();
                min(
                    max(ap.degree(i), bp.degree(i)).to_u32() as usize + 1,
                    POW_CACHE_SIZE
                )
            ]
        })
        .collect::<Vec<_>>();

    let var_bound = max(ap.degree(var).to_u32(), bp.degree(var).to_u32()) as usize + 1;
    let has_small_exp = var_bound < POW_CACHE_SIZE;

    // store a power map for the univariate polynomials that will be sampled
    // the sampling_polynomial routine will set the power to 0 after use.
    // If the exponent is small enough, we use a vec, otherwise we use a hashmap.
    let (mut tm, mut tm_fixed) = if has_small_exp {
        (
            HashMap::with_hasher(Default::default()),
            vec![FiniteField::<u32>::zero(); var_bound],
        )
    } else {
        (
            HashMap::with_capacity_and_hasher(INITIAL_POW_MAP_SIZE, Default::default()),
            vec![],
        )
    };

    'newimage: loop {
        // generate random numbers for all non-leading variables
        // TODO: apply a Horner scheme to speed up the substitution?
        let mut failcount = 0;
        let (r, a1, b1) = loop {
            for v in &mut cache {
                for vi in v {
                    *vi = FiniteField::<u32>::zero();
                }
            }

            let r: SmallVec<[_; 5]> = vars
                .iter()
                .map(|i| {
                    (
                        *i,
                        ap.field.sample(&mut rng, (1, ap.field.get_prime() as i64)),
                    )
                })
                .collect();

            let a1 = if has_small_exp {
                ap.sample_polynomial_small_exponent(var, &r, &mut cache, &mut tm_fixed)
            } else {
                ap.sample_polynomial(var, &r, &mut cache, &mut tm)
            };
            let b1 = if has_small_exp {
                bp.sample_polynomial_small_exponent(var, &r, &mut cache, &mut tm_fixed)
            } else {
                bp.sample_polynomial(var, &r, &mut cache, &mut tm)
            };

            if a1.ldegree(var) == aldegree && b1.ldegree(var) == bldegree {
                break (r, a1, b1);
            }

            failcount += 1;
            if failcount > 10 {
                panic!(
                "Cannot find samples with the right bounds after 10 tries: {} {} {} {}\nap={}\nbp={}\na1={}\nb1={}",
                a1.ldegree(var),
                aldegree,
                b1.ldegree(var),
                bldegree,
                ap,
                bp,
                a1,
                b1
            )
            }
        };

        let g1 = MultivariatePolynomial::univariate_gcd(&a1, &b1);
        trace!("GCD of sample at point {:?}: {}", r, g1);

        if g1.ldegree(var) < bounds[var] {
            // original image and form and degree bounds are unlucky
            // change the bound and try a new prime
            bounds[var] = g1.ldegree(var);
            debug!("Unlucky degree bound");
            return Err(GCDError::BadOriginalImage);
        }

        if g1.ldegree(var) > bounds[var] {
            failure_count += 1;
            if failure_count > 2 || failure_count > ni {
                // p is likely unlucky
                debug!(
                    "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                    ap,
                    bp,
                    ap.field.get_prime(),
                    r,
                    g1
                );
                return Err(GCDError::BadCurrentImage);
            }
            debug!("Degree too high");
            continue;
        }

        // check if the single scaling is there, if we had a single scale
        let mut scale_factor = ap.field.one();
        if let Some(scaling_index) = single_scale {
            // construct the scaling coefficient
            let mut coeff = ap.field.one();
            let (c, d) = &gfu[scaling_index];
            for &(n, v) in r.iter() {
                ap.field.mul_assign(
                    &mut coeff,
                    &ap.field.pow(&v, c.exponents(0)[n].to_u32() as u64),
                );
            }

            let mut found = false;
            for t in 0..g1.nterms {
                if g1.exponents(t)[var] == *d {
                    scale_factor = g1.field.div(&coeff, &g1.coefficients[t]);
                    found = true;
                    break;
                }
            }

            if !found {
                // the scaling term is missing, so the assumed form is wrong
                debug!("Bad original image");
                return Err(GCDError::BadOriginalImage);
            }
        }

        // check if all the monomials of the image appear in the shape
        // if not, the original shape is bad
        for m in g1.into_iter() {
            if gfu.iter().all(|(_, pow)| *pow != m.exponents[var]) {
                debug!("Bad shape: terms missing");
                return Err(GCDError::BadOriginalImage);
            }
        }

        system.push((r, g1, scale_factor));
        ni += 1;

        // make sure we have at least nx images
        if ni < nx {
            continue 'newimage;
        }

        // construct the linear system
        // for single scaling, we split the matrix into (potentially overdetermined) block-submatrices
        if let Some(..) = single_scale {
            // construct the gcd
            let mut gp = MultivariatePolynomial::with_nvars(ap.nvars, ap.field);

            for (i, (c, ex)) in gfu.iter().enumerate() {
                let mut gfm = smallvec![];
                let mut rhs = smallvec![FiniteField::<u32>::zero(); system.len()];

                for (j, (r, g, scale_factor)) in system.iter().enumerate() {
                    let mut row = vec![];

                    // note that we ignore the coefficient of the shape
                    for t in 0..c.nterms {
                        let mut coeff = ap.field.one();
                        for &(n, v) in r.iter() {
                            ap.field.mul_assign(
                                &mut coeff,
                                &ap.field.pow(&v, c.exponents(t)[n].to_u32() as u64),
                            );
                        }
                        row.push(coeff);
                    }

                    // move the coefficients of the image to the rhs
                    if i < g.nterms && g.exponents(i)[var] == *ex {
                        ap.field.sub_assign(
                            &mut rhs[j],
                            &ap.field.mul(&g.coefficients[i], &scale_factor),
                        );
                    } else {
                        // find the matching term if it exists
                        for m in g.into_iter() {
                            if m.exponents[var] == *ex {
                                ap.field.sub_assign(
                                    &mut rhs[j],
                                    &ap.field.mul(m.coefficient, &scale_factor),
                                );
                                break;
                            }
                        }
                    }

                    gfm.extend(row);
                }

                let m = Matrix {
                    shape: (system.len() as u32, c.nterms as u32),
                    data: gfm,
                    field: ap.field,
                };

                let rhs_mat = Matrix {
                    shape: (system.len() as u32, 1),
                    data: rhs,
                    field: ap.field,
                };

                match m.solve(&rhs_mat) {
                    Ok(x) => {
                        debug!("Solution: {}", x);

                        let mut i = 0; // index in the result x
                        for mv in c.into_iter() {
                            let mut ee = mv.exponents.to_vec();
                            ee[var] = *ex;

                            gp.append_monomial(x[(i, 0)], &ee);
                            i += 1;
                        }
                    }
                    Err(LinearSolverError::Underdetermined { min_rank, max_rank }) => {
                        debug!("Underdetermined system 1");

                        if last_rank == (min_rank, max_rank) {
                            rank_failure_count += 1;

                            if rank_failure_count == 3 {
                                debug!("Same degrees of freedom encountered 3 times: assuming bad prime/evaluation point");
                                return Err(GCDError::BadCurrentImage);
                            }
                        } else {
                            // update the rank and get new images
                            rank_failure_count = 0;
                            last_rank = (min_rank, max_rank);
                            gp = MultivariatePolynomial::zero(ap.field);
                            break;
                        }
                    }
                    Err(LinearSolverError::Inconsistent) => {
                        debug!("Inconsistent system");
                        return Err(GCDError::BadOriginalImage);
                    }
                }
            }

            if !gp.is_zero() {
                debug!("Reconstructed {}", gp);
                return Ok(gp);
            }
        } else {
            // multiple scaling case: construct subsystems with augmented
            // columns for the scaling factors
            let mut subsystems = Vec::with_capacity(gfu.len());
            for (i, (c, ex)) in gfu.iter().enumerate() {
                let mut gfm = vec![];

                for (j, (r, g, _scale_factor)) in system.iter().enumerate() {
                    let mut row = Vec::with_capacity(c.nterms + system.len());

                    for t in 0..c.nterms {
                        let mut coeff = ap.field.one();
                        for &(n, v) in r.iter() {
                            ap.field.mul_assign(
                                &mut coeff,
                                &ap.field.pow(&v, c.exponents(t)[n].to_u32() as u64),
                            );
                        }
                        row.push(coeff);
                    }

                    // it could be that some coefficients of g are
                    // 0, so we have to be caul to find the matching monomial
                    for ii in 1..system.len() {
                        if ii == j {
                            if i < g.nterms && g.exponents(i)[var] == *ex {
                                row.push(g.coefficients[i]);
                            } else {
                                // find the matching term or otherwise, push 0
                                let mut found = false;
                                for m in g.into_iter() {
                                    if m.exponents[var] == *ex {
                                        row.push(*m.coefficient);
                                        found = true;
                                        break;
                                    }
                                }
                                if !found {
                                    row.push(FiniteField::<u32>::zero());
                                }
                            }
                        } else {
                            row.push(FiniteField::<u32>::zero());
                        }
                    }

                    // the scaling of the first image is fixed to 1
                    // we add it as a last column, since that is the rhs
                    if j == 0 {
                        if i < g.nterms && g.exponents(i)[var] == *ex {
                            row.push(g.coefficients[i]);
                        } else {
                            // find the matching term or otherwise, push 0
                            let mut found = false;
                            for m in g.into_iter() {
                                if m.exponents[var] == *ex {
                                    row.push(*m.coefficient);
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                row.push(FiniteField::<u32>::zero());
                            }
                        }
                    } else {
                        row.push(FiniteField::<u32>::zero());
                    }

                    gfm.extend(row);
                }

                // bring each subsystem to upper triangular form
                let mut m = Matrix {
                    shape: (system.len() as u32, c.nterms as u32 + system.len() as u32),
                    data: gfm.into(),
                    field: ap.field,
                };

                match m.solve_subsystem(c.nterms as u32) {
                    Ok(..) => {
                        subsystems.push(m);
                    }
                    Err(LinearSolverError::Underdetermined { min_rank, max_rank }) => {
                        debug!("Underdetermined system 2");

                        if last_rank == (min_rank, max_rank) {
                            rank_failure_count += 1;

                            if rank_failure_count == 3 {
                                debug!("Same degrees of freedom encountered 3 times: assuming bad prime/evaluation point");
                                return Err(GCDError::BadCurrentImage);
                            }
                        } else {
                            // update the rank and get new images
                            rank_failure_count = 0;
                            last_rank = (min_rank, max_rank);
                            break;
                        }
                    }
                    Err(LinearSolverError::Inconsistent) => {
                        debug!("Inconsistent system");
                        return Err(GCDError::BadOriginalImage);
                    }
                }
            }

            if subsystems.len() == gfu.len() {
                // construct a system for the scaling constants
                let mut sys = smallvec![];
                let mut rhs = smallvec![];
                for s in &subsystems {
                    for r in s.row_iter() {
                        // only include rows that only depend on scaling constants
                        if r.iter()
                            .take(s.cols() - system.len())
                            .any(|x| !FiniteField::<u32>::is_zero(x))
                        {
                            continue;
                        }

                        // note the last column is the rhs, so we skip it
                        sys.extend(
                            r.iter()
                                .skip(s.cols() - system.len())
                                .take(system.len() - 1)
                                .cloned(),
                        );

                        rhs.push(ap.field.neg(r.iter().last().unwrap()));
                    }
                }

                let m = Matrix {
                    shape: (rhs.len() as u32, system.len() as u32 - 1),
                    data: sys.into(),
                    field: ap.field,
                };
                let rhs_mat = Matrix {
                    shape: (rhs.len() as u32, 1),
                    data: rhs.into(),
                    field: ap.field,
                };

                match m.solve(&rhs_mat) {
                    Ok(x) => {
                        debug!("Solved scaling constants: {:?}", x);

                        let mut gp = MultivariatePolynomial::with_nvars(ap.nvars, ap.field);

                        // now we fill in the constants in the subsystems and solve it
                        let mut si = 0;
                        for s in &mut subsystems {
                            // convert to arrays
                            let mut m = smallvec![];
                            let mut rhs = smallvec![];
                            let k = s.cols() - system.len();
                            for r in s.row_iter() {
                                if r.iter()
                                    .take(s.cols() - system.len())
                                    .all(|x| FiniteField::<u32>::is_zero(x))
                                {
                                    continue;
                                }

                                let mut coeff = FiniteField::<u32>::zero();
                                for (i, xx) in r.iter().enumerate() {
                                    if i < k {
                                        m.push(*xx);
                                    } else {
                                        if i == r.len() - 1 {
                                            ap.field.sub_assign(&mut coeff, xx);
                                        } else {
                                            ap.field.sub_assign(
                                                &mut coeff,
                                                &ap.field.mul(xx, &x[((i - k) as u32, 0)]),
                                            );
                                        }
                                    }
                                }
                                rhs.push(coeff);
                            }

                            // solve the system and plug in the scaling constants
                            let mm = Matrix {
                                shape: (rhs.len() as u32, k as u32),
                                data: m,
                                field: ap.field,
                            };
                            let rhs_mat = Matrix {
                                shape: (rhs.len() as u32, 1),
                                data: rhs,
                                field: ap.field,
                            };

                            match mm.solve(&rhs_mat) {
                                Ok(x) => {
                                    // for every power of the main variable
                                    let mut i = 0; // index in the result x
                                    let (c, ex) = &gfu[si];
                                    for mv in c.into_iter() {
                                        let mut ee = mv.exponents.to_vec();
                                        ee[var] = *ex;

                                        gp.append_monomial(x[(i, 0)], &ee);
                                        i += 1;
                                    }
                                }
                                Err(LinearSolverError::Underdetermined { min_rank, max_rank }) => {
                                    debug!("Underdetermined system 3: {}/{}", ni, nx);

                                    if last_rank == (min_rank, max_rank) {
                                        rank_failure_count += 1;

                                        if rank_failure_count == 3 {
                                            debug!("Same degrees of freedom encountered 3 times: assuming bad prime/evaluation point");
                                            return Err(GCDError::BadCurrentImage);
                                        }
                                    } else {
                                        // update the rank and get new images
                                        rank_failure_count = 0;
                                        last_rank = (min_rank, max_rank);
                                        gp = MultivariatePolynomial::with_nvars(ap.nvars, ap.field);
                                        break;
                                    }
                                }
                                Err(LinearSolverError::Inconsistent) => {
                                    debug!("Inconsistent system");
                                    return Err(GCDError::BadOriginalImage);
                                }
                            }

                            si += 1;
                        }

                        if !gp.is_zero() {
                            debug!("Reconstructed {}", gp);
                            return Ok(gp);
                        }
                    }
                    Err(LinearSolverError::Underdetermined { min_rank, max_rank }) => {
                        debug!(
                            "Underdetermined system 4: {}/{}, rank: {}/{}",
                            ni, nx, min_rank, max_rank
                        );

                        if last_rank == (min_rank, max_rank) {
                            rank_failure_count += 1;

                            if rank_failure_count == 3 {
                                debug!("Same degrees of freedom encountered 3 times: assuming bad prime/evaluation point");
                                return Err(GCDError::BadCurrentImage);
                            }
                        } else {
                            // update the rank and get new images
                            rank_failure_count = 0;
                            last_rank = (min_rank, max_rank);
                        }
                    }
                    Err(LinearSolverError::Inconsistent) => {
                        debug!("Inconsistent system");
                        return Err(GCDError::BadOriginalImage);
                    }
                }
            }
        }
    }
}

impl<E: Exponent> MultivariatePolynomial<FiniteField<u32>, E> {
    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    fn univariate_gcd(
        a: &MultivariatePolynomial<FiniteField<u32>, E>,
        b: &MultivariatePolynomial<FiniteField<u32>, E>,
    ) -> MultivariatePolynomial<FiniteField<u32>, E> {
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
        let mut r = c.fast_divmod(&mut d).1;
        while !r.is_zero() {
            c = d;
            d = r;
            r = c.fast_divmod(&mut d).1;
        }

        // normalize the gcd
        let l = d.coefficients.last().unwrap().clone();
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
        r: &[(usize, FiniteFieldElement<u32>)],
        cache: &mut [Vec<FiniteFieldElement<u32>>],
        tm: &mut HashMap<E, FiniteFieldElement<u32>>,
    ) -> MultivariatePolynomial<FiniteField<u32>, E> {
        for mv in self.into_iter() {
            let mut c = mv.coefficient.clone();
            for &(n, vv) in r {
                let exp = mv.exponents[n].to_u32() as usize;
                if exp > 0 {
                    if n < cache[n].len() {
                        if FiniteField::<u32>::is_zero(&cache[n][exp]) {
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

        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm.drain() {
            if !FiniteField::<u32>::is_zero(&c) {
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
        r: &[(usize, <FiniteField<u32> as Ring>::Element)],
        cache: &mut [Vec<<FiniteField<u32> as Ring>::Element>],
        tm: &mut [<FiniteField<u32> as Ring>::Element],
    ) -> MultivariatePolynomial<FiniteField<u32>, E> {
        for mv in self.into_iter() {
            let mut c = mv.coefficient.clone();
            for &(n, vv) in r {
                let exp = mv.exponents[n].to_u32() as usize;
                if exp > 0 {
                    if n < cache[n].len() {
                        if FiniteField::<u32>::is_zero(&cache[n][exp]) {
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
        let mut res = MultivariatePolynomial::with_nvars(self.nvars, self.field);
        let mut e = vec![E::zero(); self.nvars];
        for (k, c) in tm.iter_mut().enumerate() {
            if !FiniteField::<u32>::is_zero(&c) {
                e[v] = E::from_u32(k as u32);
                res.append_monomial_back(mem::replace(c, FiniteField::<u32>::zero()), &e);
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
    fn get_gcd_var_bound(
        ap: &MultivariatePolynomial<FiniteField<u32>, E>,
        bp: &MultivariatePolynomial<FiniteField<u32>, E>,
        vars: &[usize],
        var: usize,
    ) -> E {
        let mut rng = rand::thread_rng();

        // store a table for variables raised to a certain power
        let mut cache = (0..ap.nvars)
            .map(|i| {
                vec![
                    FiniteField::<u32>::zero();
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
                    *vi = FiniteField::<u32>::zero();
                }
            }

            let r: Vec<_> = vars
                .iter()
                .map(|i| {
                    (
                        *i,
                        ap.field.sample(&mut rng, (1, ap.field.get_prime() as i64)),
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
        return g1.ldegree_max();
    }

    /// Compute the gcd shape of two polynomials in a finite field by filling in random
    /// numbers.
    #[instrument(level = "debug", skip_all)]
    fn gcd_shape_modular(
        a: &MultivariatePolynomial<FiniteField<u32>, E>,
        b: &MultivariatePolynomial<FiniteField<u32>, E>,
        vars: &[usize],         // variables
        bounds: &mut [E],       // degree bounds
        tight_bounds: &mut [E], // tighter degree bounds
    ) -> Option<MultivariatePolynomial<FiniteField<u32>, E>> {
        let lastvar = vars.last().unwrap().clone();

        // if we are in the univariate case, return the univariate gcd
        // TODO: this is a modification of the algorithm!
        if vars.len() == 1 {
            let gg = MultivariatePolynomial::univariate_gcd(&a, &b);
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
            if c != MultivariatePolynomial::from_constant_with_nvars(
                a.field.neg(&a.field.one()),
                a.nvars,
                a.field,
            ) {
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
                let a = a.field.sample(&mut rng, (1, a.field.get_prime() as i64));
                if !gamma.replace(lastvar, a).is_zero() {
                    break a;
                }
            };

            debug!("Chosen variable: {}", v);
            let av = a.replace(lastvar, v);
            let bv = b.replace(lastvar, v);

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
                v, gamma, gv
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
            if single_scale == None {
                let mut nx1 = (gv.nterms() - 1) / (gfu.len() - 1);
                if (gv.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            // we need one extra sample to detect inconsistencies, such
            // as missing terms in the shape.
            // NOTE: not in paper
            nx += 1;

            let mut lc = gv.lcoeff_varorder(vars);

            let mut gseq = vec![gv.clone().mul_coeff(
                gamma
                    .field
                    .div(&gamma.replace(lastvar, v).coefficients[0], &lc),
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
                    let v = a.field.sample(&mut rng, (1, a.field.get_prime() as i64));
                    if !gamma.replace(lastvar, v).is_zero() {
                        // we need unique sampling points
                        if !vseq.contains(&v) {
                            break v;
                        }
                    }
                };

                let av = a.replace(lastvar, v);
                let bv = b.replace(lastvar, v);

                match construct_new_image(
                    &av,
                    &bv,
                    // NOTE: different from paper where they use a.degree(..)
                    // it could be that the degree in av is lower than that of a
                    // which means the sampling will never terminate
                    av.degree(vars[0]),
                    bv.degree(vars[0]),
                    bounds,
                    single_scale,
                    nx,
                    &vars[1..vars.len() - 1],
                    vars[0],
                    &gfu,
                ) {
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
                            .div(&gamma.replace(lastvar, v).coefficients[0], &lc),
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
                let cc = gc.divmod(&cont);
                debug_assert!(cc.1.is_zero());
                gc = cc.0;
            }

            // do a probabilistic division testto_element
            let (g1, a1, b1) = loop {
                // store a table for variables raised to a certain power
                let mut cache = (0..a.nvars)
                    .map(|i| {
                        vec![
                            FiniteField::<u32>::zero();
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
                            a.field.sample(&mut rng, (1, a.field.get_prime() as i64)),
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

            if g1.is_one() || (a1.divmod(&g1).1.is_zero() && b1.divmod(&g1).1.is_zero()) {
                return Some(gc);
            }

            // if the gcd is bad, we had a bad number
            debug!(
                "Division test failed: gcd may be bad or probabilistic division test is unlucky: a1 {} b1 {} g1 {}", a1, b1, g1
            );
        }
    }
}

impl<R: EuclideanDomain, E: Exponent> MultivariatePolynomial<R, E>
where
    MultivariatePolynomial<R, E>: PolynomialGCD<E>,
{
    /// Get the content of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let a = self.to_univariate_polynomial_list(x);

        let mut f = vec![];
        for (c, _) in &a {
            f.push(c.clone());
        }

        MultivariatePolynomial::gcd_multiple(f)
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// multivariate polynomial in all variables except `x`.
    pub fn multivariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        MultivariatePolynomial::gcd_multiple(af.values().cloned().collect())
    }

    /// Compute the gcd of multiple polynomials efficiently.
    /// `gcd(f0,f1,f2,...)=gcd(f0,f1+k2*f(2)+k3*f(3))`
    /// with high likelihood.
    pub fn gcd_multiple(mut f: Vec<MultivariatePolynomial<R, E>>) -> MultivariatePolynomial<R, E>
    where
        MultivariatePolynomial<R, E>: PolynomialGCD<E>,
    {
        assert!(f.len() > 0);

        if f.len() == 1 {
            return f.swap_remove(0);
        }

        if f.len() == 2 {
            return MultivariatePolynomial::gcd(&f[0], &f[1]);
        }

        // check if all entries are numbers
        // in this case summing them may give bad gcds often
        if f.iter().all(|x| x.is_constant()) {
            let mut gcd = f.swap_remove(0);
            for x in f.iter() {
                gcd = MultivariatePolynomial::gcd(&gcd, x);
            }
            return gcd;
        }

        // TODO: extract gcd of content first, since that may easily cause a miss!
        let mut rng = rand::thread_rng();
        let mut k = f[0].field.one(); // counter for scalar multiple
        let mut gcd;

        // take the smallest element
        let mut index_smallest = f
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.nterms)
            .unwrap()
            .0;

        loop {
            let a = f.swap_remove(index_smallest);

            let term_bound = f.iter().map(|x| x.nterms).sum();
            let mut b =
                MultivariatePolynomial::with_nvars_and_capacity(a.nvars, term_bound, a.field);

            // Add all the monomials to a vector, sort them and build a new polynomial.
            // The last step will merge equal monomials.
            {
                let mut c = Vec::with_capacity(f.iter().map(|x| x.nterms).sum());

                for p in f.iter() {
                    for v in p.into_iter() {
                        c.push((p.field.mul(&v.coefficient, &k), v.exponents));
                    }
                    k = p.field.sample(&mut rng, (2, MAX_RNG_PREFACTOR as i64));
                }

                c.sort_unstable_by(|a, b| a.1.cmp(b.1));
                for m in c {
                    b.append_monomial_back(m.0, m.1);
                }
            }

            gcd = MultivariatePolynomial::gcd(&a, &b);

            if gcd.is_one() {
                return gcd;
            }

            let mut newf: Vec<MultivariatePolynomial<R, E>> = Vec::with_capacity(f.len());
            for x in f.drain(..) {
                if !x.divmod(&gcd).1.is_zero() {
                    newf.push(x);
                }
            }

            if newf.len() == 0 {
                return gcd;
            }

            debug!(
                "Multiple-gcd was not found instantly. GCD guess: {}; Terms left: {}",
                gcd,
                newf.len()
            );

            newf.push(gcd);
            index_smallest = newf.len() - 1; // the gcd is the smallest element
            mem::swap(&mut newf, &mut f);
        }
    }

    /// Compute the gcd of the univariate content in `x`.
    pub fn univariate_content_gcd(
        a: &MultivariatePolynomial<R, E>,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = a.to_univariate_polynomial_list(x);
        let bf = b.to_univariate_polynomial_list(x);

        let mut f = vec![];
        for (c, _) in af.iter().chain(bf.iter()) {
            f.push(c.clone());
        }

        MultivariatePolynomial::gcd_multiple(f)
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

        // TODO: drain?
        let f = af.values().cloned().chain(bf.values().cloned()).collect();

        MultivariatePolynomial::gcd_multiple(f)
    }

    /// Compute the gcd of two multivariate polynomials.
    #[instrument(skip_all)]
    pub fn gcd(
        a: &MultivariatePolynomial<R, E>,
        b: &MultivariatePolynomial<R, E>,
    ) -> MultivariatePolynomial<R, E> {
        debug_assert_eq!(a.nvars, b.nvars);

        if a.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return a.clone();
        }

        // if we have two numbers, use the integer gcd
        if a.is_constant() && b.is_constant() {
            return MultivariatePolynomial::from_constant_with_nvars(
                a.field.gcd(&a.coefficients[0], &b.coefficients[0]),
                a.nvars,
                a.field,
            );
        }

        debug!("Compute gcd({}, {})", a, b);

        // compute the gcd efficiently if some variables do not occur in both
        // polynomials
        let mut scratch = vec![0i32; a.nvars];
        for (p, inc) in vec![(a, 1), (b, 2)] {
            for t in 0..p.nterms {
                for (e, ee) in scratch.iter_mut().zip(p.exponents(t)) {
                    if !ee.is_zero() {
                        *e |= inc;
                    }
                }
            }
        }

        if scratch.iter().any(|x| *x > 0 && *x < 3) {
            let inca: Vec<_> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 1 || *v == 3 { Some(i) } else { None })
                .collect();

            let incb: Vec<_> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 2 || *v == 3 { Some(i) } else { None })
                .collect();

            // extract the variables of b in the coefficient of a and vice versa
            let a1 = a.to_multivariate_polynomial_list(&incb, false);
            let b1 = b.to_multivariate_polynomial_list(&inca, false);

            let f = a1.values().cloned().chain(b1.values().cloned()).collect();
            return MultivariatePolynomial::gcd_multiple(f);
        }

        let mut vars: Vec<_> = scratch
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 3 { Some(i) } else { None })
            .collect();

        // remove the gcd of the content wrt the first variable
        // TODO: don't do for univariate poly
        debug!("Starting content computation");
        let c = MultivariatePolynomial::univariate_content_gcd(a, b, vars[0]);
        debug!("GCD of content: {}", c);

        if !c.is_one() {
            let x1 = a.divmod(&c);
            let x2 = b.divmod(&c);

            assert!(x1.1.is_zero());
            assert!(x2.1.is_zero());

            return c * MultivariatePolynomial::gcd(&x1.0, &x2.0);
        }

        // determine safe bounds for variables in the gcd
        let mut bounds: Vec<_> = (0..a.nvars)
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
        let mut tight_bounds = MultivariatePolynomial::get_gcd_var_bounds(&a, &b, &vars, &bounds);

        // Determine a good variable ordering based on the estimated degree (decreasing) in the gcd.
        // If it is different from the input, make a copy and rearrange so that the
        // polynomials do not have to be sorted after filling in variables.
        // TODO: understand better why copying is so much faster (about 10%) than using a map
        vars.sort_by(|&i, &j| tight_bounds[j].cmp(&tight_bounds[i]));

        if vars.len() == 1 || vars.windows(2).all(|s| s[0] < s[1]) {
            debug!("Computing gcd without map");
            PolynomialGCD::gcd(&a, &b, &vars, &mut bounds, &mut tight_bounds)
        } else {
            debug!("Rearranging variables with map: {:?}", vars);
            let aa = a.rearrange(&vars, false);
            let bb = b.rearrange(&vars, false);

            let mut newbounds = vec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newbounds[x] = bounds[vars[x]];
            }

            let mut newtight_bounds = vec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newtight_bounds[x] = tight_bounds[vars[x]];
            }

            // we need to extract the content if the first variable changed
            if vars[1..].iter().any(|&c| c < vars[0]) {
                debug!("Starting new content computation after mapping {:?}", vars);
                let c = MultivariatePolynomial::univariate_content_gcd(&aa, &bb, 0);
                debug!("New content: {}", c);
                if !c.is_one() {
                    let x1 = aa.divmod(&c);
                    let x2 = bb.divmod(&c);

                    assert!(x1.1.is_zero());
                    assert!(x2.1.is_zero());

                    let gcd = c * PolynomialGCD::gcd(
                        &x1.0,
                        &x2.0,
                        &(0..vars.len()).collect::<Vec<_>>(),
                        &mut newbounds,
                        &mut newtight_bounds,
                    );
                    return gcd.rearrange(&vars, true);
                }
            }

            let gcd = PolynomialGCD::gcd(
                &aa,
                &bb,
                &(0..vars.len()).collect::<Vec<_>>(),
                &mut newbounds,
                &mut newtight_bounds,
            );

            gcd.rearrange(&vars, true)
        }
    }
}

impl<R: Ring, E: Exponent> MultivariatePolynomial<R, E>
where
    R::Element: ToFiniteField<u32>,
{
    /// Convert the coefficient from the current field to the given field.
    pub fn to_finite_field_u32(
        &self,
        field: FiniteField<u32>,
    ) -> MultivariatePolynomial<FiniteField<u32>, E> {
        let mut newc = Vec::with_capacity(self.coefficients.len());
        let mut newe = Vec::with_capacity(self.exponents.len());

        for m in self.into_iter() {
            let nc = m.coefficient.to_finite_field(field);
            if !FiniteField::<u32>::is_zero(&nc) {
                newc.push(nc);
                newe.extend(m.exponents);
            }
        }

        let mut a = MultivariatePolynomial::with_nvars(self.nvars, field);
        a.nterms = newc.len();
        a.exponents = newe;
        a.coefficients = newc;
        a
    }
}

impl<E: Exponent> MultivariatePolynomial<IntegerRing, E> {
    /// Compute the gcd of two multivariate polynomials using Zippel's algorithm.
    /// TODO: provide a parallel implementation?
    #[instrument(level = "debug", skip_all)]
    fn gcd_zippel(
        a: &Self,
        b: &Self,
        vars: &[usize], // variables
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> Self {
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

        'newfirstprime: loop {
            pi += 1;

            if pi == LARGE_U32_PRIMES.len() {
                a.check_consistency();
                b.check_consistency();
                panic!(
                    "Ran out of primes for gcd reconstruction.\ngcd({},{})",
                    a, b
                );
            }

            let mut p = LARGE_U32_PRIMES[pi];
            let mut finite_field = FiniteField::<u32>::new(p);
            let mut gammap = gamma.to_finite_field(finite_field);

            if FiniteField::<u32>::is_zero(&gammap) {
                continue 'newfirstprime;
            }

            let ap = a.to_finite_field_u32(finite_field);
            let bp = b.to_finite_field_u32(finite_field);

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
            if single_scale == None {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            // we need one extra sample to detect inconsistencies, such
            // as missing terms in the shape.
            // NOTE: not in paper
            nx += 1;

            let gpc = gp.lcoeff_varorder(vars);
            let lcoeff_factor = gp.field.div(&gammap, &gpc);

            // construct the gcd suggestion in Z
            let mut gm = MultivariatePolynomial::with_nvars(gp.nvars, a.field);
            gm.nterms = gp.nterms;
            gm.exponents = gp.exponents.clone();
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|x| Integer::from_finite_field_u32(gp.field, &gp.field.mul(x, &lcoeff_factor)))
                .collect();

            let mut m = Integer::Natural(p as i64); // size of finite field

            debug!("GCD suggestion with gamma: {} mod {} ", gm, p);

            let mut old_gm = MultivariatePolynomial::with_nvars(a.nvars, a.field);

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                if gm == old_gm {
                    // divide by integer content
                    let gmc = gm.content();
                    let mut gc = gm.clone();
                    gc.coefficients = gc
                        .coefficients
                        .iter()
                        .map(|x| gc.field.quot_rem(x, &gmc).0)
                        .collect();

                    debug!("Final suggested gcd: {}", gc);
                    if gc.is_one() || (a.divmod(&gc).1.is_zero() && b.divmod(&gc).1.is_zero()) {
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

                    p = LARGE_U32_PRIMES[pi];
                    finite_field = FiniteField::<u32>::new(p);

                    gammap = gamma.to_finite_field(finite_field);

                    if !FiniteField::<u32>::is_zero(&gammap) {
                        break;
                    }
                }

                let ap = a.to_finite_field_u32(finite_field);
                let bp = b.to_finite_field_u32(finite_field);
                debug!("New image: gcd({},{}) mod {}", ap, bp, p);

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
                    match construct_new_image(
                        &ap,
                        &bp,
                        // NOTE: different from paper where they use a.degree(..)
                        // it could be that the degree in ap is lower than that of a
                        // which means the sampling will never terminate
                        ap.degree(vars[0]),
                        bp.degree(vars[0]),
                        bounds,
                        single_scale,
                        nx,
                        &vars[1..],
                        vars[0],
                        &gfu,
                    ) {
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
                        gp.coefficients[gpi - 1].clone()
                    } else {
                        FiniteField::<u32>::zero()
                    };

                    let gmc = &mut gm.coefficients[t];
                    let coeff = if gmc.is_negative() {
                        a.field.add(&gmc, &m)
                    } else {
                        gmc.clone()
                    };

                    *gmc = Integer::chinese_remainder(
                        coeff,
                        Integer::Natural(gp.field.from_element(gpc) as i64),
                        m.clone(),
                        Integer::Natural(p as i64),
                    );
                }

                a.field.mul_assign(&mut m, &Integer::Natural(p as i64));

                debug!("gm: {} from ring {}", gm, m);
            }
        }
    }
}

pub trait PolynomialGCD<E: Exponent> {
    fn gcd(a: &Self, b: &Self, vars: &[usize], bounds: &mut [E], tight_bounds: &mut [E]) -> Self;
    fn get_gcd_var_bounds(a: &Self, b: &Self, vars: &[usize], loose_bounds: &[E]) -> Vec<E>;
}

impl<E: Exponent> PolynomialGCD<E> for MultivariatePolynomial<IntegerRing, E> {
    fn gcd(a: &Self, b: &Self, vars: &[usize], bounds: &mut [E], tight_bounds: &mut [E]) -> Self {
        MultivariatePolynomial::gcd_zippel(&a, &b, vars, bounds, tight_bounds)
    }

    fn get_gcd_var_bounds(a: &Self, b: &Self, vars: &[usize], loose_bounds: &[E]) -> Vec<E> {
        let mut tight_bounds = loose_bounds.to_owned();
        let mut i = 0;
        loop {
            let ap = a.to_finite_field_u32(FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            let bp = b.to_finite_field_u32(FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            if ap.nterms > 0
                && bp.nterms > 0
                && ap.last_exponents() == a.last_exponents()
                && bp.last_exponents() == b.last_exponents()
            {
                for var in vars.iter() {
                    let vvars: SmallVec<[usize; 5]> =
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
}

impl<E: Exponent> PolynomialGCD<E> for MultivariatePolynomial<RationalField, E> {
    fn gcd(a: &Self, b: &Self, vars: &[usize], bounds: &mut [E], tight_bounds: &mut [E]) -> Self {
        // remove the content so that the polynomials have integer coefficients
        let content = a.field.gcd(&a.content(), &b.content());

        let mut a_int =
            MultivariatePolynomial::with_nvars_and_capacity(a.nvars, a.nterms, IntegerRing::new());

        for t in a {
            let coeff = a.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            a_int.append_monomial(coeff.numerator(), t.exponents);
        }

        let mut b_int =
            MultivariatePolynomial::with_nvars_and_capacity(a.nvars, a.nterms, IntegerRing::new());

        for t in b {
            let coeff = b.field.div(t.coefficient, &content);
            debug_assert!(coeff.is_integer());
            b_int.append_monomial(coeff.numerator(), t.exponents);
        }

        let res_int =
            MultivariatePolynomial::gcd_zippel(&a_int, &b_int, vars, bounds, tight_bounds);

        let mut res =
            MultivariatePolynomial::with_nvars_and_capacity(res_int.nvars, res_int.nterms, a.field);

        for t in &res_int {
            res.append_monomial(
                a.field.mul(&t.coefficient.to_rational(), &content),
                t.exponents,
            );
        }

        res
    }

    fn get_gcd_var_bounds(a: &Self, b: &Self, vars: &[usize], loose_bounds: &[E]) -> Vec<E> {
        let mut tight_bounds = loose_bounds.to_owned();
        let mut i = 0;
        loop {
            let ap = a.to_finite_field_u32(FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            let bp = b.to_finite_field_u32(FiniteField::<u32>::new(LARGE_U32_PRIMES[i]));
            if ap.nterms > 0
                && bp.nterms > 0
                && ap.last_exponents() == a.last_exponents()
                && bp.last_exponents() == b.last_exponents()
            {
                for var in vars.iter() {
                    let vvars: SmallVec<[usize; 5]> =
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
}

impl<E: Exponent> PolynomialGCD<E> for MultivariatePolynomial<FiniteField<u32>, E> {
    fn gcd(a: &Self, b: &Self, vars: &[usize], bounds: &mut [E], tight_bounds: &mut [E]) -> Self {
        assert!(!a.is_zero() || !b.is_zero());
        MultivariatePolynomial::gcd_shape_modular(&a, &b, vars, bounds, tight_bounds).unwrap()
    }

    fn get_gcd_var_bounds(a: &Self, b: &Self, vars: &[usize], loose_bounds: &[E]) -> Vec<E> {
        let mut tight_bounds = loose_bounds.to_owned();
        for var in vars {
            let vvars: SmallVec<[usize; 5]> = vars.iter().filter(|i| *i != var).cloned().collect();
            tight_bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(&a, &b, &vvars, *var);
        }
        tight_bounds
    }
}
