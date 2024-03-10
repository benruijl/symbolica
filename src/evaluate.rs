use ahash::HashMap;

use crate::{
    coefficient::CoefficientView,
    domains::{float::Real, rational::Rational},
    representations::{Atom, AtomView, Symbol},
    state::State,
};

type EvalFnType<T> = Box<
    dyn Fn(
        &[T],
        &HashMap<AtomView<'_>, T>,
        &HashMap<Symbol, EvaluationFn<T>>,
        &mut HashMap<AtomView<'_>, T>,
    ) -> T,
>;

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

impl Atom {
    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<'b, T: Real + for<'a> From<&'a Rational>>(
        &'b self,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        cache: &mut HashMap<AtomView<'b>, T>,
    ) -> T {
        self.as_view().evaluate(const_map, function_map, cache)
    }
}

impl<'a> AtomView<'a> {
    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<T: Real + for<'b> From<&'b Rational>>(
        &self,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        cache: &mut HashMap<AtomView<'a>, T>,
    ) -> T {
        if let Some(c) = const_map.get(self) {
            return *c;
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => (&Rational::Natural(n, d)).into(),
                CoefficientView::Large(l) => (&Rational::Large(l.to_rat())).into(),
                CoefficientView::FiniteField(_, _) => {
                    unimplemented!("Finite field not yet supported for evaluation")
                }
                CoefficientView::RationalPolynomial(_) => unimplemented!(
                    "Rational polynomial coefficient not yet supported for evaluation"
                ),
            },
            AtomView::Var(v) => panic!("Variable id {:?} not in constant map", v.get_symbol()),
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [State::EXP, State::LOG, State::SIN, State::COS, State::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate(const_map, function_map, cache);

                    return match f.get_symbol() {
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
                    args.push(arg.evaluate(const_map, function_map, cache));
                }

                let Some(fun) = function_map.get(&f.get_symbol()) else {
                    panic!("Missing function with id {:?}", f.get_symbol()); // TODO: use state to get name
                };
                let eval = fun.get()(&args, const_map, function_map, cache);

                cache.insert(*self, eval);
                eval
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(const_map, function_map, cache);

                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num >= 0 {
                                return b_eval.pow(num as u64);
                            } else {
                                return b_eval.pow(num.unsigned_abs()).inv();
                            }
                        }
                    }
                }

                let e_eval = e.evaluate(const_map, function_map, cache);
                b_eval.powf(e_eval)
            }
            AtomView::Mul(m) => {
                let mut r = T::one();
                for arg in m.iter() {
                    r *= arg.evaluate(const_map, function_map, cache);
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = T::zero();
                for arg in a.iter() {
                    r += arg.evaluate(const_map, function_map, cache);
                }
                r
            }
        }
    }
}
