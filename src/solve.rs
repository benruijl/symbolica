use std::{ops::Neg, sync::Arc};

use crate::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    domains::{
        float::{FloatField, Real, SingleFloat},
        integer::Z,
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
        InternalOrdering,
    },
    evaluate::FunctionMap,
    poly::{PositiveExponent, Variable},
    tensors::matrix::Matrix,
};

impl<'a> AtomView<'a> {
    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    pub(crate) fn nsolve<N: SingleFloat + Real + PartialOrd>(
        &self,
        x: Symbol,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<N, String> {
        let v = Atom::new_var(x);
        let f = self
            .to_evaluation_tree(&FunctionMap::new(), std::slice::from_ref(&v))
            .unwrap()
            .optimize(0, 0, None, false);
        let df = self
            .derivative(x)
            .to_evaluation_tree(&FunctionMap::new(), std::slice::from_ref(&v))
            .unwrap()
            .optimize(0, 0, None, false);

        let mut f_e = f.map_coeff(&|x| init.from_rational(x));
        let mut df_e = df.map_coeff(&|x| init.from_rational(x));

        let mut cur = init.clone();

        for _ in 0..max_iterations {
            let df_val = df_e.evaluate_single(std::slice::from_ref(&cur));
            let f_val = f_e.evaluate_single(std::slice::from_ref(&cur));

            if !df_val.is_finite() || df_val.is_zero() {
                return Err("Derivative is zero".to_owned());
            }

            cur = cur - f_val.clone() / df_val;
            if f_val.norm() < prec {
                return Ok(cur);
            }
        }

        Err("Did not converge".to_owned())
    }

    /// Solve a non-linear system numerically over the reals using Newton's method.
    pub(crate) fn nsolve_system<
        N: SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
        T: AtomCore,
    >(
        system: &[T],
        vars: &[Symbol],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        let system = system.iter().map(|v| v.as_atom_view()).collect::<Vec<_>>();
        AtomView::nsolve_system_impl(&system, vars, init, prec, max_iterations)
    }

    fn nsolve_system_impl<
        N: SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
    >(
        system: &[AtomView],
        vars: &[Symbol],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        if system.len() != vars.len() {
            Err("System must have same number of equations as there are unknowns".to_owned())?;
        }

        if vars.len() != init.len() {
            Err("Initial values must be provided for all unknowns".to_owned())?;
        }

        if system.is_empty() {
            return Ok(vec![]);
        }

        if system.len() == 1 {
            return Ok(vec![system[0].nsolve(
                vars[0],
                init[0].clone(),
                prec,
                max_iterations,
            )?]);
        }

        let avars = vars.iter().map(|v| Atom::new_var(*v)).collect::<Vec<_>>();

        let mut fs = system
            .iter()
            .map(|a| {
                a.to_evaluation_tree(&FunctionMap::new(), &avars)
                    .unwrap()
                    .optimize(0, 0, None, false)
                    .map_coeff(&|x| init[0].from_rational(x))
            })
            .collect::<Vec<_>>();

        let mut jacobian = Vec::with_capacity(vars.len() * system.len());
        for a in system {
            let mut row = Vec::with_capacity(vars.len());
            for v in vars {
                let deriv = a.derivative(*v);

                let a = deriv
                    .to_evaluation_tree(&FunctionMap::new(), &avars)
                    .unwrap()
                    .optimize(0, 0, None, false)
                    .map_coeff(&|x| init[0].from_rational(x));

                row.push(a);
            }
            jacobian.extend_from_slice(&row);
        }

        let field = FloatField::from_rep(init[0].clone());
        let mut cur = init.to_vec();

        for _ in 0..max_iterations {
            let f = fs
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();
            let f = Matrix::new_vec(f, field.clone());

            let df = jacobian
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();

            let df = Matrix::from_linear(df, system.len() as u32, vars.len() as u32, field.clone())
                .unwrap();

            let Ok(i) = df.inv() else {
                return Err("Could not invert Jacobian".to_owned());
            };

            let mut ci = Matrix::new_vec(cur.to_vec(), field.clone());

            ci -= &(&i * &f);

            cur = ci.data;

            if f.data.iter().all(|x| x.norm() < prec) {
                return Ok(cur);
            }
        }

        Err("Did not converge".to_owned())
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub(crate) fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, String> {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().into())
            .collect();

        AtomView::solve_linear_system_impl::<E>(&system, &vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    pub(crate) fn system_to_matrix<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().into())
            .collect();

        AtomView::system_to_matrix_impl::<E>(&system, &vars)
    }

    fn system_to_matrix_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[Variable],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        let mut mat = Vec::with_capacity(system.len() * vars.len());
        let mut row = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); vars.len()];
        let mut rhs = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); system.len()];

        for (si, a) in system.iter().enumerate() {
            let rat: RationalPolynomial<Z, E> = a.to_rational_polynomial(&Q, &Z, None);

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

        let field = RationalPolynomialField::new(Z);

        let m = Matrix::from_linear(mat, system.len() as u32, vars.len() as u32, field.clone())
            .unwrap();
        let b = Matrix::new_vec(rhs, field);

        Ok((m, b))
    }

    fn solve_linear_system_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[Variable],
    ) -> Result<Vec<Atom>, String> {
        let (m, b) = Self::system_to_matrix_impl::<E>(system, vars)?;

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
        atom::{representation::InlineVar, Atom, AtomCore, AtomView, Symbol},
        domains::{
            float::{Real, F64},
            integer::Z,
            rational::Q,
            rational_polynomial::{RationalPolynomial, RationalPolynomialField},
        },
        poly::Variable,
        tensors::matrix::Matrix,
    };

    #[test]
    fn solve() {
        let x = Symbol::new("v1").into();
        let y = Symbol::new("v2").into();
        let z = Symbol::new("v3").into();
        let eqs = [
            "v4*v1 + f1(v4)*v2 + v3 - 1",
            "v1 + v4*v2 + v3/v4 - 2",
            "(v4-1)v1 + v4*v3",
        ];

        let system: Vec<_> = eqs.iter().map(|e| Atom::parse(e).unwrap()).collect();

        let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &[x, y, z]).unwrap();

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

        let var_map = Arc::new(vec![Variable::Symbol(Symbol::new("v4"))]);

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

        let field = RationalPolynomialField::from_poly(&rhs_rat[0].numerator);
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

    #[test]
    fn find_root() {
        let x = Symbol::new("x");
        let a = Atom::parse("x^2 - 2").unwrap();
        let a = a.as_view();

        let root = a.nsolve(x, 1.0, 1e-10, 1000).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn solve_system_newton() {
        let a = Atom::parse("5x^2+x*y^2+sin(2y)^2 - 2").unwrap();
        let b = Atom::parse("exp(2x-y)+4y - 3").unwrap();

        let r = AtomView::nsolve_system(
            &[a.as_view(), b.as_view()],
            &[Symbol::new("x"), Symbol::new("y")],
            &[F64::from(1.), F64::from(1.)],
            F64::from(1e-10),
            100,
        )
        .unwrap();

        assert!((r[0].clone() - F64::from(5.6729734993961234e-1)).norm() < 1e-10.into());
        assert!((r[1].clone() - F64::from(-3.0944227920271083e-1)).norm() < 1e-10.into());
    }
}
