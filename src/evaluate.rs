use ahash::HashMap;

use crate::{
    poly::Variable,
    representations::{number::BorrowedNumber, Add, AtomSet, AtomView, Fun, Mul, Num, Pow, Var},
    rings::{float::Real, rational::Rational},
    state::State,
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
    pub fn evaluate<T: Real + for<'b> From<&'b Rational>>(
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
            AtomView::Var(v) => *var_map.get(&v.get_name().into()).unwrap(),
            AtomView::Fun(f) => {
                let name = f.get_name();
                if [State::EXP, State::LOG, State::SIN, State::COS, State::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate(var_map, function_map, cache);

                    return match f.get_name() {
                        State::EXP => arg_eval.exp(),
                        State::LOG => arg_eval.log(),
                        State::SIN => arg_eval.sin(),
                        State::COS => arg_eval.cos(),
                        State::SQRT => arg_eval.sqrt(),
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

                let Some(fun) = function_map.get(&f.get_name().into()) else {
                    panic!("Missing function with id {:?}", f.get_name()); // TODO: use state to get name
                };
                let eval = fun.get()(&args, var_map, function_map, cache);

                cache.insert(*self, eval);
                eval
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(var_map, function_map, cache);

                if let AtomView::Num(n) = e {
                    if let BorrowedNumber::Natural(num, den) = n.get_number_view() {
                        if den == 1 {
                            if num >= 0 {
                                return b_eval.pow(num as u64);
                            } else {
                                return b_eval.pow(num.unsigned_abs()).inv();
                            }
                        }
                    }
                }

                let e_eval = e.evaluate(var_map, function_map, cache);
                b_eval.powf(e_eval)
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
