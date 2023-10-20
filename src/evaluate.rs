use ahash::HashMap;

use crate::{
    poly::Variable,
    representations::{number::BorrowedNumber, Add, AtomSet, AtomView, Fun, Mul, Num, Pow, Var},
    rings::{
        float::{NumericalFloatComparison, Real},
        rational::Rational,
    },
    state::{COS, EXP, LOG, SIN},
};

type EvalFnType<T, P> = Box<
    dyn Fn(
        &[T],
        &HashMap<Variable, T>,
        &HashMap<Variable, EvaluationFn<T, P>>,
        &mut HashMap<AtomView<'_, P>, T>,
    ) -> T,
>;

pub struct EvaluationFn<T, P: AtomSet>(EvalFnType<T, P>);

impl<T, P: AtomSet> EvaluationFn<T, P> {
    pub fn new(f: EvalFnType<T, P>) -> EvaluationFn<T, P> {
        EvaluationFn(f)
    }

    /// Get a reference to the function that can be called to evaluate it.
    pub fn get(&self) -> &EvalFnType<T, P> {
        &self.0
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Evaluate an expression using a variable map and a function map.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<T: Real + NumericalFloatComparison + for<'b> From<&'b Rational>>(
        &self,
        var_map: &HashMap<Variable, T>,
        function_map: &HashMap<Variable, EvaluationFn<T, P>>,
        cache: &mut HashMap<AtomView<'a, P>, T>,
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
            AtomView::Var(v) => var_map.get(&v.get_name().into()).unwrap().clone(),
            AtomView::Fun(f) => {
                let name = f.get_name();
                if [EXP, LOG, SIN, COS].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate(var_map, function_map, cache);

                    return match f.get_name() {
                        EXP => arg_eval.exp(),
                        LOG => arg_eval.log(),
                        SIN => arg_eval.sin(),
                        COS => arg_eval.cos(),
                        _ => unreachable!(),
                    };
                }

                if let Some(eval) = cache.get(self) {
                    return *eval;
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f.iter() {
                    args.push(arg.evaluate(var_map, function_map, cache));
                }

                let fun = function_map.get(&f.get_name().into()).unwrap();
                let eval = fun.get()(&args, var_map, function_map, cache);

                cache.insert(*self, eval);
                eval
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(var_map, function_map, cache);
                let e_eval = e.evaluate(var_map, function_map, cache);

                // FIXME
                b_eval.powf(e_eval.to_f64())
            }
            AtomView::Mul(m) => {
                let mut r = T::one();
                for arg in m.iter() {
                    r *= arg.evaluate(var_map, function_map, cache);
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = T::zero();
                for arg in a.iter() {
                    r += arg.evaluate(var_map, function_map, cache);
                }
                r
            }
        }
    }
}
