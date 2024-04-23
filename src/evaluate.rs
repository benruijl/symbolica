use ahash::HashMap;

use crate::{
    atom::{Atom, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::{float::Real, rational::Rational},
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
            AtomView::Var(v) => panic!(
                "Variable {} not in constant map",
                State::get_name(v.get_symbol())
            ),
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
                    panic!("Missing function {}", State::get_name(f.get_symbol()));
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

#[cfg(test)]
mod test {
    use ahash::HashMap;

    use crate::{atom::Atom, evaluate::EvaluationFn, state::State};

    #[test]
    fn evaluate() {
        let x = State::get_symbol("v1");
        let f = State::get_symbol("f1");
        let g = State::get_symbol("f2");
        let p0 = Atom::parse("v2(0)").unwrap();
        let a = Atom::parse("v1*cos(v1) + f1(v1, 1)^2 + f2(f2(v1)) + v2(0)").unwrap();

        let mut const_map = HashMap::default();
        let mut fn_map: HashMap<_, EvaluationFn<_>> = HashMap::default();
        let mut cache = HashMap::default();

        // x = 6 and p(0) = 7
        let v = Atom::new_var(x);
        const_map.insert(v.as_view(), 6.);
        const_map.insert(p0.as_view(), 7.);

        // f(x, y) = x^2 + y
        fn_map.insert(
            f,
            EvaluationFn::new(Box::new(|args: &[f64], _, _, _| {
                args[0] * args[0] + args[1]
            })),
        );

        // g(x) = f(x, 3)
        fn_map.insert(
            g,
            EvaluationFn::new(Box::new(move |args: &[f64], var_map, fn_map, cache| {
                fn_map.get(&f).unwrap().get()(&[args[0], 3.], var_map, fn_map, cache)
            })),
        );

        let r = a.evaluate::<f64>(&const_map, &fn_map, &mut cache);
        assert_eq!(r, 2905.761021719902);
    }
}
