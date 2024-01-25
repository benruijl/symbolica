use std::ops::Neg;

use ahash::HashMap;

use crate::{
    domains::{
        integer::IntegerRing,
        linear_system::Matrix,
        rational::RationalField,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::{Exponent, Variable},
    representations::{Atom, AtomSet, AtomView, Identifier},
    state::{ResettableBuffer, State, Workspace},
};

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub fn solve_linear_system<E: Exponent>(
        system: &[AtomView<P>],
        vars: &[Identifier],
        workspace: &Workspace<P>,
        state: &State,
    ) -> Result<Vec<Atom<P>>, String> {
        let vars: Vec<_> = vars.iter().map(|v| Variable::Identifier(*v)).collect();
        let mut map = HashMap::default();

        let mut mat = Vec::with_capacity(system.len() * vars.len());
        let mut row = vec![RationalPolynomial::<_, E>::new(&IntegerRing::new(), None); vars.len()];
        let mut rhs = vec![RationalPolynomial::<_, E>::new(&IntegerRing::new(), None); vars.len()];

        for (si, a) in system.iter().enumerate() {
            let rat: RationalPolynomial<IntegerRing, E> = a.to_rational_polynomial_with_map(
                workspace,
                state,
                &RationalField::new(),
                &IntegerRing::new(),
                &mut map,
            );

            let poly = rat.to_polynomial(&vars, true).unwrap();

            for e in &mut row {
                *e = RationalPolynomial::<_, E>::new(&IntegerRing::new(), None);
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
                first.unify_var_map(x);
            }
            for x in &mut rhs {
                first.unify_var_map(x);
            }
        }

        let m = Matrix {
            shape: ((mat.len() / rhs.len()) as u32, rhs.len() as u32),
            data: mat.into(),
            field: RationalPolynomialField::new(IntegerRing::new()),
        };
        let b = Matrix {
            shape: (rhs.len() as u32, 1),
            data: rhs.into(),
            field: RationalPolynomialField::new(IntegerRing::new()),
        };

        let sol = match m.solve(&b) {
            Ok(sol) => sol,
            Err(e) => Err(format!("Could not solve {:?}", e))?,
        };

        // replace the temporary variables
        let mut result = Vec::with_capacity(vars.len());

        let inv_map = map.iter().map(|(k, v)| (*v, k.as_view())).collect();
        for (s, v) in sol.data.iter().zip(&vars) {
            let mut a = Atom::new();
            a.from_rational_polynomial(workspace, state, s, &inv_map);
            let Variable::Identifier(_) = *v else {
                panic!("Temp var left");
            };

            result.push(a);
        }

        Ok(result)
    }
}
