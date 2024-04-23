use std::{ops::Neg, sync::Arc};

use crate::{
    atom::{Atom, AtomView, Symbol},
    domains::{
        integer::{IntegerRing, Z},
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    poly::{Exponent, Variable},
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

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{Atom, AtomView},
        domains::{
            integer::Z,
            rational::Q,
            rational_polynomial::{RationalPolynomial, RationalPolynomialField},
        },
        poly::Variable,
        state::State,
        tensors::matrix::Matrix,
    };

    #[test]
    fn solve() {
        let x = State::get_symbol("v1");
        let y = State::get_symbol("v2");
        let z = State::get_symbol("v3");
        let eqs = [
            "v4*v1 + f1(v4)*v2 + v3 - 1",
            "v1 + v4*v2 + v3/v4 - 2",
            "(v4-1)v1 + v4*v3",
        ];

        let atoms: Vec<_> = eqs.iter().map(|e| Atom::parse(e).unwrap()).collect();
        let system: Vec<_> = atoms.iter().map(|x| x.as_view()).collect();

        let sol = AtomView::solve_linear_system::<u8>(&system, &[x, y, z]).unwrap();

        let res = [
            "(v4^3-2*v4^2*f1(v4))*(v4^2-v4^3+v4^4-f1(v4)+v4*f1(v4)-v4^2*f1(v4))^-1",
            "(v4^2-f1(v4))^-1*(2*v4-1)",
            "(v4^2-v4^3-2*v4*f1(v4)+2*v4^2*f1(v4))*(v4^2-v4^3+v4^4-f1(v4)+v4*f1(v4)-v4^2*f1(v4))^-1",
        ];
        let res = res
            .iter()
            .map(|x| Atom::parse(x).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(sol, res);
    }

    #[test]
    fn solve_from_matrix() {
        let system = [
            ["v4", "v4+1", "v4^2+5"],
            ["1", "v4", "v4+1"],
            ["v4-1", "-1", "v4"],
        ];
        let rhs = ["1", "2", "-1"];

        let var_map = Arc::new(vec![Variable::Symbol(State::get_symbol("v4"))]);

        let system_rat: Vec<RationalPolynomial<_, u8>> = system
            .iter()
            .flatten()
            .map(|s| {
                Atom::parse(s)
                    .unwrap()
                    .to_rational_polynomial(&Q, &Z, Some(var_map.clone()))
            })
            .collect();

        let rhs_rat: Vec<RationalPolynomial<_, u8>> = rhs
            .iter()
            .map(|s| {
                Atom::parse(s)
                    .unwrap()
                    .to_rational_polynomial(&Q, &Z, Some(var_map.clone()))
            })
            .collect();

        let field = RationalPolynomialField::new_from_poly(&rhs_rat[0].numerator);
        let m = Matrix::from_linear(
            system_rat,
            system.len() as u32,
            system.len() as u32,
            field.clone(),
        )
        .unwrap();
        let b = Matrix::new_vec(rhs_rat, field);

        let sol = m.solve(&b).unwrap();

        let res = [
            "(10-2*v4+4*v4^2-v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(-4+10*v4-5*v4^2+2*v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(2-4*v4)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
        ];

        let res = res
            .iter()
            .map(|x| {
                Atom::parse(x).unwrap().to_rational_polynomial(
                    &Z,
                    &Z,
                    m.data[0].get_variables().clone().into(),
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(sol.data, res);
    }
}
