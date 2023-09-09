use ahash::HashMap;

use crate::{
    representations::{
        number::BorrowedNumber, Add, AtomSet, AtomView, Fun, Identifier, Mul, Num, Pow, Var,
    },
    rings::{
        float::{NumericalFloatComparison, Real},
        rational::Rational,
    },
    state::{COS, EXP, LOG, SIN},
};

type EvalFnType<T> =
    Box<dyn Fn(&[T], &HashMap<Identifier, T>, &HashMap<Identifier, EvaluationFn<T>>) -> T>;

pub struct EvaluationFn<T>(EvalFnType<T>);

impl<T> EvaluationFn<T> {
    pub fn new(f: EvalFnType<T>) -> EvaluationFn<T> {
        EvaluationFn(f)
    }

    /// Get a reference to the function that can be called to evaluate it.
    pub fn get(&self) -> &EvalFnType<T> {
        &self.0
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Evaluate an expression using a variable map and a function map.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<T: Real + NumericalFloatComparison + for<'b> From<&'b Rational>>(
        &self,
        var_map: &HashMap<Identifier, T>,
        function_map: &HashMap<Identifier, EvaluationFn<T>>,
    ) -> T {
        match self {
            AtomView::Num(n) => match n.get_number_view() {
                BorrowedNumber::Natural(n, d) => (&Rational::Natural(n, d)).into(),
                BorrowedNumber::Large(l) => (&Rational::Large(l.to_rat())).into(),
                BorrowedNumber::FiniteField(_, _) => {
                    unimplemented!("Finite field not yet supported for evaluation")
                }
                BorrowedNumber::RationalPolynomial(_) => unimplemented!(
                    "Rational polynomial coefficient not yet supported for evaluation"
                ),
            },
            AtomView::Var(v) => var_map.get(&v.get_name()).unwrap().clone(),
            AtomView::Fun(f) => {
                let name = f.get_name();
                if [EXP, LOG, SIN, COS].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate(var_map, function_map);

                    return match f.get_name() {
                        EXP => arg_eval.exp(),
                        LOG => arg_eval.log(),
                        SIN => arg_eval.sin(),
                        COS => arg_eval.cos(),
                        _ => unreachable!(),
                    };
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f.iter() {
                    args.push(arg.evaluate(var_map, function_map));
                }

                let fun = function_map.get(&f.get_name()).unwrap();
                fun.get()(&args, var_map, function_map)
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(var_map, function_map);
                let e_eval = e.evaluate(var_map, function_map);

                // FIXME
                b_eval.powf(e_eval.to_f64())
            }
            AtomView::Mul(m) => {
                let mut r = T::one();
                for arg in m.iter() {
                    r *= arg.evaluate(var_map, function_map);
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = T::zero();
                for arg in a.iter() {
                    r += arg.evaluate(var_map, function_map);
                }
                r
            }
        }
    }
}
