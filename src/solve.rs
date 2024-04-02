use std::{ops::Neg, sync::Arc};

use crate::{
    domains::{
        integer::{IntegerRing, Z},
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::{Exponent, Variable},
    representations::{Atom, AtomView, Symbol},
    tensors::matrix::Matrix,
};

impl<'a> AtomView<'a> {
    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub fn solve_linear_system<E: Exponent>(
        system: &[AtomView],
        vars: &[Symbol],
    ) -> Result<Vec<Atom>, String> {
        let vars: Vec<_> = vars.iter().map(|v| Variable::Symbol(*v)).collect();

        let mut mat = Vec::with_capacity(system.len() * vars.len());
        let mut row = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); vars.len()];
        let mut rhs = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); vars.len()];

        for (si, a) in system.iter().enumerate() {
            let rat: RationalPolynomial<IntegerRing, E> = a.to_rational_polynomial(&Q, &Z, None);

            let poly = rat.to_polynomial(&vars, true).unwrap();

            for e in &mut row {
                *e = RationalPolynomial::<_, E>::new(&Z, poly.variables.clone());
            }

            // get linear coefficients
            'next_monomial: for e in poly.into_iter() {
                if e.exponents.iter().cloned().sum::<E>() > E::one() {
                    Err("Not a linear system")?;
                }

                for (rv, p) in row.iter_mut().zip(e.exponents) {
                    if !p.is_zero() {
                        *rv = e.coefficient.clone();
                        continue 'next_monomial;
                    }
                }

                // constant term
                rhs[si] = e.coefficient.clone().neg();
            }

            mat.extend_from_slice(&row);
        }

        let Some((first, rest)) = mat.split_first_mut() else {
            return Err("Empty system".to_owned());
        };

        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
            for x in &mut rhs {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z, rhs[0].numerator.get_vars().into());

        let nrows = (mat.len() / rhs.len()) as u32;
        let m = Matrix::from_linear(mat, nrows, rhs.len() as u32, field.clone()).unwrap();
        let b = Matrix::new_vec(rhs, field);

        let sol = match m.solve(&b) {
            Ok(sol) => sol,
            Err(e) => Err(format!("Could not solve {:?}", e))?,
        };

        // replace the temporary variables
        let mut result = Vec::with_capacity(vars.len());

        for s in sol.data {
            result.push(s.to_expression());
        }

        Ok(result)
    }
}
