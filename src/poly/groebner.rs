use std::{cmp::Ordering, rc::Rc};

use ahash::HashMap;

use crate::rings::Field;

use super::{polynomial::MultivariatePolynomial, Exponent, MonomialOrder};

#[derive(Debug)]
pub struct CriticalPair<R: Field, E: Exponent, O: MonomialOrder> {
    lcm_diff_first: Vec<E>,
    poly_first: Rc<MultivariatePolynomial<R, E, O>>,
    lcm_diff_sec: Vec<E>,
    poly_sec: Rc<MultivariatePolynomial<R, E, O>>,
    lcm: Vec<E>,
    degree: E,
    disjoint: bool,
}

impl<'a, R: Field, E: Exponent, O: MonomialOrder> CriticalPair<R, E, O> {
    pub fn new(
        f1: Rc<MultivariatePolynomial<R, E, O>>,
        f2: Rc<MultivariatePolynomial<R, E, O>>,
    ) -> CriticalPair<R, E, O> {
        // determine the lcm of leading monomials
        let lcm: Vec<E> = f1
            .max_exp()
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1.max(e2))
            .collect();

        let lcm_diff_first: Vec<E> = lcm
            .iter()
            .zip(f1.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        let lcm_diff_sec: Vec<E> = lcm
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        CriticalPair {
            disjoint: lcm_diff_first == f2.max_exp(),
            degree: lcm.iter().cloned().sum::<E>(),
            lcm_diff_first,
            poly_first: f1,
            lcm_diff_sec,
            poly_sec: f2,
            lcm,
        }
    }
}

pub struct GroebnerBasis<R: Field, E: Exponent, O: MonomialOrder> {
    pub system: Vec<MultivariatePolynomial<R, E, O>>,
    pub print_stats: bool,
}

impl<R: Field, E: Exponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    /// Construct a Groebner basis for a polynomial ideal.
    ///
    /// Progress can be monitored with `print_stats`.
    pub fn new(
        ideal: &[MultivariatePolynomial<R, E, O>],
        print_stats: bool,
    ) -> GroebnerBasis<R, E, O> {
        let mut ideal = ideal.to_vec();
        for _ in 0..2 {
            let (first, rest) = ideal.split_first_mut().unwrap();
            for x in rest {
                first.unify_var_map(x);
            }
        }

        let mut b = GroebnerBasis {
            system: ideal,
            print_stats,
        };

        b.f4();
        b.reduce_basis()
    }

    /// Add a new polynomial to the basis, updating and filtering the existing
    /// basis and critical pairs, based on Gebauer and Moeller's redundant pair criteria.
    ///
    /// Adapted from "A Computational Approach to Commutative Algebra" by Thomas Becker Volker Weispfenning.
    fn update(
        basis: &mut Vec<Rc<MultivariatePolynomial<R, E, O>>>,
        critical_pairs: &mut Vec<CriticalPair<R, E, O>>,
        f: MultivariatePolynomial<R, E, O>,
    ) {
        let f = Rc::new(f);

        let mut new_pairs: Vec<_> = basis
            .iter()
            .map(|b| (CriticalPair::new(b.clone(), f.clone()), true))
            .collect();

        for i in 0..new_pairs.len() {
            new_pairs[i].1 = false;
            new_pairs[i].1 = new_pairs[i].0.disjoint
                || new_pairs.iter().all(|p2| {
                    !p2.1
                        || new_pairs[i]
                            .0
                            .lcm
                            .iter()
                            .zip(&p2.0.lcm)
                            .any(|(e1, e2)| *e1 < *e2)
                });
        }

        new_pairs.retain(|p| p.1 && !p.0.disjoint);

        critical_pairs.retain(|p| {
            p.lcm.iter().zip(f.max_exp()).any(|(e1, e2)| *e1 < *e2)
                || p.poly_first
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
                || p.poly_sec
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
        });

        critical_pairs.extend(new_pairs.into_iter().map(|np| np.0));

        basis.retain(|b| {
            b.max_exp()
                .iter()
                .zip(f.max_exp())
                .any(|(e1, e2)| *e1 < *e2)
        });

        basis.push(f);
    }

    /// The F4 algorithm for computing a Groebner basis.
    ///
    /// Adapted from [A new efficient algorithm for computing Gröbner bases (F4)](https://doi.org/10.1016/S0022-4049(99)00005-5) by Jean-Charles Faugére.
    ///
    fn f4(&mut self) {
        // TODO: strip content

        let nvars = self.system[0].nvars;
        let field = self.system[0].field.clone();

        let mut basis = vec![];
        let mut critical_pairs = vec![];

        for f in self.system.drain(..) {
            Self::update(&mut basis, &mut critical_pairs, f.clone());
        }

        let mut matrix = vec![];

        struct MonomialData {
            present: bool,
            column: usize,
        }

        let mut all_monomials: HashMap<Vec<E>, MonomialData> = HashMap::default();
        let mut current_monomials = vec![];
        let mut sorted_monomial_indices = vec![];
        let mut exp = vec![E::zero(); nvars];
        let mut new_polys = vec![];
        let mut buffer = vec![];
        let mut selected_polys = vec![];

        let mut iter_count = 1;
        while !critical_pairs.is_empty() {
            // select the critical pairs with the lowest lcm degree
            let lowest_lcm_deg = critical_pairs.iter().map(|x| x.degree).min().unwrap();

            if self.print_stats {
                println!(
                    "Iteration {}:\n\tDegree={}, Basis length={}, Critical pairs={}",
                    iter_count,
                    lowest_lcm_deg,
                    basis.len(),
                    critical_pairs.len(),
                );
            }
            iter_count += 1;

            selected_polys.clear();
            let mut i = critical_pairs.len() - 1;
            loop {
                if critical_pairs[i].degree == lowest_lcm_deg {
                    let pair = critical_pairs.swap_remove(i);

                    let new_f1 = (*pair.poly_first).clone().mul_exp(&pair.lcm_diff_first);
                    selected_polys.push(new_f1);

                    let new_f2 = (*pair.poly_sec).clone().mul_exp(&pair.lcm_diff_sec);
                    selected_polys.push(new_f2);
                }

                if i == 0 {
                    break;
                }

                i -= 1;
            }

            // symbolic preprocessing

            for x in all_monomials.values_mut() {
                x.present = false;
            }

            // flag all head monomials as done
            for p in &selected_polys {
                if let Some(m) = all_monomials.get_mut(p.max_exp()) {
                    m.present = true;
                } else {
                    all_monomials.insert(
                        p.max_exp().to_vec(),
                        MonomialData {
                            present: true,
                            column: 0,
                        },
                    );
                }
            }

            new_polys.clear();
            let mut i = 0;
            while i < selected_polys.len() {
                for monom in selected_polys[i].exponents.chunks(nvars) {
                    if let Some(m) = all_monomials.get_mut(monom) {
                        if m.present {
                            continue;
                        }
                        m.present = true;
                    } else {
                        all_monomials.insert(
                            monom.to_vec(),
                            MonomialData {
                                present: true,
                                column: 0,
                            },
                        );
                    }

                    // search for a reducer and select the smallest for better performance
                    if let Some(g) = basis
                        .iter()
                        .filter(|g| monom.iter().zip(g.max_exp()).all(|(pe, ge)| *pe >= *ge))
                        .min_by_key(|g| g.nterms)
                    {
                        for ((e, pe), ge) in exp.iter_mut().zip(monom).zip(g.max_exp()) {
                            *e = *pe - *ge;
                        }

                        let pp = (**g).clone().mul_exp(&exp);
                        new_polys.push(pp);
                    }
                }

                i += 1;

                selected_polys.extend(new_polys.drain(..));
            }

            // construct a matrix that is sparse in the columns

            current_monomials.clear();
            sorted_monomial_indices.clear();

            for (k, v) in &all_monomials {
                if v.present {
                    current_monomials.extend_from_slice(k);
                }
            }

            for i in 0..(current_monomials.len() / nvars) {
                sorted_monomial_indices.push(i);
            }

            if self.print_stats {
                println!(
                    "\tMonomials in use={}/{}",
                    sorted_monomial_indices.len(),
                    all_monomials.len()
                );
                println!(
                    "\tMatrix shape={}x{}, density={:.2}%",
                    selected_polys.len(),
                    sorted_monomial_indices.len(),
                    selected_polys.iter().map(|i| i.nterms).sum::<usize>() as f64
                        / (sorted_monomial_indices.len() as f64 * selected_polys.len() as f64)
                        * 100.
                );
            }

            // sort monomials in descending order
            sorted_monomial_indices.sort_unstable_by(|e1, e2| {
                O::cmp(
                    &current_monomials[*e2 * nvars..(*e2 + 1) * nvars],
                    &current_monomials[*e1 * nvars..(*e1 + 1) * nvars],
                )
            });

            for (column, index) in sorted_monomial_indices.iter().enumerate() {
                all_monomials
                    .get_mut(&current_monomials[index * nvars..(index + 1) * nvars])
                    .unwrap()
                    .column = column;
            }

            matrix.resize(selected_polys.len(), vec![]);
            for (row, p) in matrix.iter_mut().zip(&mut selected_polys) {
                row.clear();

                for (coeff, exp) in p.coefficients.iter().zip(p.exponents.chunks(nvars)).rev() {
                    row.push((coeff.clone(), all_monomials.get(exp).unwrap().column));
                }
            }

            // sort the matrix rows to make it as upper-triangular as possible
            matrix.sort_unstable_by(|r1, r2| r1[0].1.cmp(&r2[0].1).then(r1.len().cmp(&r2.len())));

            // row-reduce the sparse matrix

            let mut non_empty_pivots = 0;
            for pivot_col in 0..sorted_monomial_indices.len() {
                // find next pivot
                let mut best_pivot: Option<(usize, usize)> = None;
                for (row_index, row) in matrix[non_empty_pivots..].iter().enumerate() {
                    if row[0].1 == pivot_col
                        && (best_pivot.is_none() || best_pivot.unwrap().1 > row.len())
                    {
                        // find the smallest row as a pivot
                        best_pivot = Some((non_empty_pivots + row_index, row.len()));
                    }
                }

                match best_pivot {
                    Some(b) => {
                        matrix.swap(non_empty_pivots, b.0);
                    }
                    None => continue,
                }

                let inv_pivot = field.inv(&matrix[non_empty_pivots][0].0);

                for r in 0..matrix.len() {
                    let row = &matrix[r];
                    let pivot = &matrix[non_empty_pivots];
                    if r == non_empty_pivots || row[0].1 != pivot_col {
                        continue;
                    }

                    let ratio = field.neg(&field.mul(&row[0].0, &inv_pivot));

                    let mut pos_pivot = 0;
                    let mut pos_row = 0;

                    buffer.clear();

                    while pos_row < row.len() && pos_pivot < pivot.len() {
                        match row[pos_row].1.cmp(&pivot[pos_pivot].1) {
                            Ordering::Less => {
                                buffer.push((row[pos_row].0.clone(), row[pos_row].1));
                                pos_row += 1;
                            }
                            Ordering::Greater => {
                                buffer.push((
                                    field.mul(&ratio, &pivot[pos_pivot].0),
                                    pivot[pos_pivot].1,
                                ));
                                pos_pivot += 1;
                            }
                            Ordering::Equal => {
                                let new_coeff = field
                                    .add(&row[pos_row].0, &field.mul(&ratio, &pivot[pos_pivot].0));

                                if !R::is_zero(&new_coeff) {
                                    buffer.push((new_coeff, row[pos_row].1));
                                }

                                pos_row += 1;
                                pos_pivot += 1;
                            }
                        }
                    }

                    for (coeff, col) in &row[pos_row..] {
                        buffer.push((coeff.clone(), *col));
                    }

                    for (coeff, col) in &pivot[pos_pivot..] {
                        buffer.push((field.mul(&ratio, coeff), *col));
                    }

                    std::mem::swap(&mut matrix[r], &mut buffer);
                    buffer.clear();
                }

                matrix.retain(|r| !r.is_empty());

                non_empty_pivots += 1;
            }

            // construct new polynomials
            for m in &matrix {
                let lmi = sorted_monomial_indices[m[0].1];
                let lm = &current_monomials[lmi * nvars..(lmi + 1) * nvars];

                // TODO: update the pivot polynomials, as they have been simplified

                if selected_polys.iter().all(|p| p.max_exp() != lm) {
                    // create the new polynomial in the proper order
                    let mut poly =
                        MultivariatePolynomial::new_from(&selected_polys[0], Some(m.len()));
                    for (coeff, col) in m.iter().rev() {
                        let index = sorted_monomial_indices[*col];
                        let exp = &current_monomials[index * nvars..(index + 1) * nvars];
                        poly.append_monomial(coeff.clone(), exp);
                    }

                    Self::update(&mut basis, &mut critical_pairs, poly);
                }
            }
        }

        self.system = basis.into_iter().map(|x| (*x).clone()).collect();
    }

    /// Completely reduce the polynomial `f` w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    pub fn reduce(
        p: &MultivariatePolynomial<R, E, O>,
        gs: &[MultivariatePolynomial<R, E, O>],
    ) -> MultivariatePolynomial<R, E, O> {
        let mut q = MultivariatePolynomial::new_from(p, Some(p.nterms));
        let mut r = p.clone();

        let mut rest_coeff = vec![];
        let mut rest_exponents = vec![];

        let mut monom = vec![E::zero(); p.nvars];

        'term: while !r.is_zero() {
            // find a divisor that has the least amount of terms
            while let Some(g) = gs
                .iter()
                .filter(|g| {
                    r.max_exp()
                        .iter()
                        .zip(g.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                })
                .min_by_key(|g| g.nterms)
            {
                for ((e, e1), e2) in monom.iter_mut().zip(r.max_exp()).zip(g.max_exp()) {
                    *e = *e1 - *e2;
                }

                let ratio = g.field.div(&r.max_coeff(), &g.max_coeff());
                r = r - g.clone().mul_exp(&monom).mul_coeff(ratio);

                if r.is_zero() {
                    break 'term;
                }
            }

            // strip leading monomial that is not reducible
            rest_exponents.extend_from_slice(r.exponents(r.nterms - 1));
            rest_coeff.push(r.coefficients.pop().unwrap());
            r.nterms -= 1;
        }

        // append in sorted order
        while let Some(c) = rest_coeff.pop() {
            let l = rest_coeff.len();
            q.append_monomial(c, &rest_exponents[l * p.nvars..(l + 1) * p.nvars]);
        }

        q
    }

    pub fn reduce_basis(mut self) -> Self {
        // filter lead-reducible polynomials
        let mut res = vec![true; self.system.len()];
        'l1: for (i, p1) in self.system.iter().enumerate() {
            for (j, p2) in self.system.iter().enumerate() {
                if i != j
                    && res[j]
                    && p1
                        .max_exp()
                        .iter()
                        .zip(p2.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                {
                    res[i] = false;
                    continue 'l1;
                }
            }
        }

        let mut lead_reduced = vec![];
        for (i, p) in self.system.drain(..).enumerate() {
            if res[i] {
                lead_reduced.push(p);
            }
        }

        let mut basis = vec![];
        for i in 0..lead_reduced.len() {
            lead_reduced.swap(0, i);
            let h = Self::reduce(&lead_reduced[0], &lead_reduced[1..]);
            if !h.is_zero() {
                let i = h.field.inv(&h.max_coeff());
                basis.push(h.mul_coeff(i));
            }
        }

        basis.sort_by(|p1, p2| p2.max_exp().cmp(&p1.max_exp()));

        GroebnerBasis {
            system: basis,
            print_stats: self.print_stats,
        }
    }

    pub fn is_groebner_basis(system: &[MultivariatePolynomial<R, E, O>]) -> bool {
        for (i, p1) in system.iter().enumerate() {
            for p2 in &system[i + 1..] {
                let lcm: Vec<E> = p1
                    .max_exp()
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1.max(e2))
                    .collect();

                // construct s-polynomial
                let extra_factor_f1: Vec<E> = lcm
                    .iter()
                    .zip(p1.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();

                let extra_factor_f2: Vec<E> = lcm
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();
                let new_f1 = p1
                    .clone()
                    .mul_exp(&extra_factor_f1)
                    .mul_coeff(p1.field.div(&p2.max_coeff(), &p1.max_coeff()));
                let new_f2 = p2
                    .clone()
                    .mul_exp(&extra_factor_f2)
                    .mul_coeff(p1.field.div(&p1.max_coeff(), &p2.max_coeff()));

                let s = new_f1 - new_f2;

                if !Self::reduce(&s, system).is_zero() {
                    return false;
                }
            }
        }
        true
    }
}
