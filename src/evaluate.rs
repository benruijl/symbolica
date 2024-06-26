use ahash::HashMap;

use crate::{
    atom::{representation::InlineVar, Atom, AtomOrView, AtomView, Symbol},
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

pub enum ConstOrExpr<'a, T> {
    Const(T),
    Expr(Vec<Symbol>, AtomView<'a>),
}

impl Atom {
    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<'b, T: Real, F: Fn(&Rational) -> T + Copy>(
        &'b self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        cache: &mut HashMap<AtomView<'b>, T>,
    ) -> T {
        self.as_view()
            .evaluate(coeff_map, const_map, function_map, cache)
    }
}

#[derive(Debug)]
pub enum EvalTree<T> {
    Const(T),
    Parameter(usize),
    Eval(Vec<T>, Vec<EvalTree<T>>, Box<EvalTree<T>>), // first argument is a buffer for the evaluated arguments
    Add(Vec<EvalTree<T>>),
    Mul(Vec<EvalTree<T>>),
    Pow(Box<(EvalTree<T>, i64)>),
    Powf(Box<(EvalTree<T>, EvalTree<T>)>),
    ReadArg(usize),
    BuiltinFun(Symbol, Box<EvalTree<T>>),
}

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    instructions: Vec<Instr>,
    result_index: usize,
}

impl<T: Real> ExpressionEvaluator<T> {
    pub fn evaluate(&mut self, params: &[T]) -> T {
        for (t, p) in self.stack.iter_mut().zip(params) {
            *t = p.clone();
        }

        for i in &self.instructions {
            match i {
                Instr::Add(r, v) => {
                    self.stack[*r] = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        self.stack[*r] += e;
                    }
                }
                Instr::Mul(r, v) => {
                    self.stack[*r] = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        self.stack[*r] *= e;
                    }
                }
                Instr::Pow(r, b, e) => {
                    if *e >= 0 {
                        self.stack[*r] = self.stack[*b].pow(*e as u64);
                    } else {
                        self.stack[*r] = self.stack[*b].pow(e.unsigned_abs()).inv();
                    }
                }
                Instr::Powf(r, b, e) => {
                    self.stack[*r] = self.stack[*b].powf(&self.stack[*e]);
                }
                Instr::BuiltinFun(r, s, arg) => match *s {
                    State::EXP => self.stack[*r] = self.stack[*arg].exp(),
                    State::LOG => self.stack[*r] = self.stack[*arg].log(),
                    State::SIN => self.stack[*r] = self.stack[*arg].sin(),
                    State::COS => self.stack[*r] = self.stack[*arg].cos(),
                    State::SQRT => self.stack[*r] = self.stack[*arg].sqrt(),
                    _ => unreachable!(),
                },
                Instr::Copy(d, s) => {
                    for (o, i) in s.iter().enumerate() {
                        self.stack[*d + o] = self.stack[*i].clone();
                    }
                }
            }
        }

        self.stack[self.result_index].clone()
    }
}

enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, Symbol, usize),
    Copy(usize, Vec<usize>), // copy arguments into an adjacent array
}

impl<T: Clone + Default> EvalTree<T> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(&self, param_len: usize) -> ExpressionEvaluator<T> {
        let mut stack = vec![T::default(); param_len];
        let mut instructions = vec![];
        let result_index = self.linearize_impl(&mut stack, &mut instructions, 0);
        ExpressionEvaluator {
            stack,
            instructions,
            result_index,
        }
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        stack: &mut Vec<T>,
        instr: &mut Vec<Instr>,
        arg_start: usize,
    ) -> usize {
        match self {
            EvalTree::Const(t) => {
                stack.push(t.clone()); // TODO: do once and recycle
                stack.len() - 1
            }
            EvalTree::Parameter(i) => *i,
            EvalTree::Eval(_, args, f) => {
                let dest_pos = stack.len();
                for _ in args {
                    stack.push(T::default());
                }

                let a: Vec<_> = args
                    .iter()
                    .map(|x| x.linearize_impl(stack, instr, arg_start))
                    .collect();

                instr.push(Instr::Copy(dest_pos, a));
                f.linearize_impl(stack, instr, dest_pos)
            }
            EvalTree::Add(a) => {
                stack.push(T::default());
                let res = stack.len() - 1;

                let add = Instr::Add(
                    res,
                    a.iter()
                        .map(|x| x.linearize_impl(stack, instr, arg_start))
                        .collect(),
                );
                instr.push(add);

                res
            }
            EvalTree::Mul(m) => {
                stack.push(T::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(
                    res,
                    m.iter()
                        .map(|x| x.linearize_impl(stack, instr, arg_start))
                        .collect(),
                );
                instr.push(mul);

                res
            }
            EvalTree::Pow(p) => {
                stack.push(T::default());
                let res = stack.len() - 1;
                let b = p.0.linearize_impl(stack, instr, arg_start);

                instr.push(Instr::Pow(res, b, p.1));
                res
            }
            EvalTree::Powf(p) => {
                stack.push(T::default());
                let res = stack.len() - 1;
                let b = p.0.linearize_impl(stack, instr, arg_start);
                let e = p.1.linearize_impl(stack, instr, arg_start);

                instr.push(Instr::Powf(res, b, e));
                res
            }
            EvalTree::ReadArg(a) => arg_start + *a,
            EvalTree::BuiltinFun(s, v) => {
                stack.push(T::default());
                let arg = v.linearize_impl(stack, instr, arg_start);
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push(c);
                stack.len() - 1
            }
        }
    }
}

impl<T: Real> EvalTree<T> {
    /// Evaluate the evaluation tree. Consider converting to a linear form for repeated evaluation.
    pub fn evaluate(&mut self, params: &[T]) -> T {
        self.evaluate_impl(params, &[])
    }

    fn evaluate_impl(&mut self, params: &[T], args: &[T]) -> T {
        match self {
            EvalTree::Const(c) => c.clone(),
            EvalTree::Parameter(p) => params[*p].clone(),
            EvalTree::Eval(arg_buf, e_args, f) => {
                for (b, a) in arg_buf.iter_mut().zip(e_args.iter_mut()) {
                    *b = a.evaluate_impl(params, args);
                }

                f.evaluate_impl(params, &arg_buf)
            }
            EvalTree::Add(a) => {
                let mut r = a[0].evaluate_impl(params, args);
                for arg in &mut a[1..] {
                    r += arg.evaluate_impl(params, args);
                }
                r
            }
            EvalTree::Mul(m) => {
                let mut r = m[0].evaluate_impl(params, args);
                for arg in &mut m[1..] {
                    r *= arg.evaluate_impl(params, args);
                }
                r
            }
            EvalTree::Pow(p) => {
                let (b, e) = &mut **p;
                let b_eval = b.evaluate_impl(params, args);

                if *e >= 0 {
                    b_eval.pow(*e as u64)
                } else {
                    b_eval.pow(e.unsigned_abs()).inv()
                }
            }
            EvalTree::Powf(p) => {
                let (b, e) = &mut **p;
                let b_eval = b.evaluate_impl(params, args);
                let e_eval = e.evaluate_impl(params, args);
                b_eval.powf(&e_eval)
            }
            EvalTree::ReadArg(i) => args[*i].clone(),
            EvalTree::BuiltinFun(s, a) => {
                let arg = a.evaluate_impl(params, args);
                match *s {
                    State::EXP => arg.exp(),
                    State::LOG => arg.log(),
                    State::SIN => arg.sin(),
                    State::COS => arg.cos(),
                    State::SQRT => arg.sqrt(),
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl<'a> AtomView<'a> {
    /// Convert nested expressions to a from suitable for evaluation.
    pub fn evaluator<T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> ExpressionEvaluator<T> {
        let tree = self.to_eval_tree(coeff_map, const_map, params);
        tree.linearize(params.len())
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree<T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> EvalTree<T> {
        self.to_eval_tree_impl(coeff_map, const_map, params, &[])
    }

    fn to_eval_tree_impl<T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
        args: &[Symbol],
    ) -> EvalTree<T> {
        if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
            return EvalTree::Parameter(p);
        }

        if let Some(c) = const_map.get(&self.into()) {
            return match c {
                ConstOrExpr::Const(c) => EvalTree::Const(c.clone()),
                ConstOrExpr::Expr(args, v) => {
                    if !args.is_empty() {
                        panic!(
                            "Function {} called with wrong number of arguments: 0 vs {}",
                            self,
                            args.len()
                        );
                    }

                    let r = v.to_eval_tree_impl(coeff_map, const_map, params, args);
                    EvalTree::Eval(vec![], vec![], Box::new(r))
                }
            };
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => {
                    EvalTree::Const(coeff_map(&Rational::Natural(n, d)))
                }
                CoefficientView::Large(l) => {
                    EvalTree::Const(coeff_map(&Rational::Large(l.to_rat())))
                }
                CoefficientView::Float(f) => {
                    // TODO: converting back to rational is slow
                    EvalTree::Const(coeff_map(&f.to_float().to_rational()))
                }
                CoefficientView::FiniteField(_, _) => {
                    unimplemented!("Finite field not yet supported for evaluation")
                }
                CoefficientView::RationalPolynomial(_) => {
                    unimplemented!(
                        "Rational polynomial coefficient not yet supported for evaluation"
                    )
                }
            },
            AtomView::Var(v) => {
                let name = v.get_symbol();

                if let Some(p) = args.iter().position(|s| *s == name) {
                    return EvalTree::ReadArg(p);
                }

                panic!(
                    "Variable {} not in constant map",
                    State::get_name(v.get_symbol())
                );
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [State::EXP, State::LOG, State::SIN, State::COS, State::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(coeff_map, const_map, params, args);

                    return EvalTree::BuiltinFun(f.get_symbol(), Box::new(arg_eval));
                }

                let symb = InlineVar::new(f.get_symbol());
                let Some(fun) = const_map.get(&symb.as_view().into()) else {
                    panic!("Undefined function {}", State::get_name(f.get_symbol()));
                };

                match fun {
                    ConstOrExpr::Const(t) => EvalTree::Const(t.clone()),
                    ConstOrExpr::Expr(arg_spec, e) => {
                        if f.get_nargs() != arg_spec.len() {
                            panic!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol(),
                                f.get_nargs(),
                                arg_spec.len()
                            );
                        }

                        let eval_args = f
                            .iter()
                            .map(|arg| arg.to_eval_tree_impl(coeff_map, const_map, params, args))
                            .collect();
                        let res = e.to_eval_tree_impl(coeff_map, const_map, params, arg_spec);

                        EvalTree::Eval(vec![T::default(); arg_spec.len()], eval_args, Box::new(res))
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.to_eval_tree_impl(coeff_map, const_map, params, args);

                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            return EvalTree::Pow(Box::new((b_eval, num)));
                        }
                    }
                }

                let e_eval = e.to_eval_tree_impl(coeff_map, const_map, params, args);
                EvalTree::Powf(Box::new((b_eval, e_eval)))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    muls.push(arg.to_eval_tree_impl(coeff_map, const_map, params, args));
                }

                EvalTree::Mul(muls)
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(coeff_map, const_map, params, args));
                }

                EvalTree::Add(adds)
            }
        }
    }

    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomView<'_>, T>,
        function_map: &HashMap<Symbol, EvaluationFn<T>>,
        cache: &mut HashMap<AtomView<'a>, T>,
    ) -> T {
        if let Some(c) = const_map.get(self) {
            return c.clone();
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => coeff_map(&Rational::from_unchecked(n, d)),
                CoefficientView::Large(l) => coeff_map(&l.to_rat()),
                CoefficientView::Float(f) => {
                    // TODO: converting back to rational is slow
                    coeff_map(&f.to_float().to_rational())
                }
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
                    let arg_eval = arg.evaluate(coeff_map, const_map, function_map, cache);

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
                    return eval.clone();
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f {
                    args.push(arg.evaluate(coeff_map, const_map, function_map, cache));
                }

                let Some(fun) = function_map.get(&f.get_symbol()) else {
                    panic!("Missing function {}", State::get_name(f.get_symbol()));
                };
                let eval = fun.get()(&args, const_map, function_map, cache);

                cache.insert(*self, eval.clone());
                eval
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(coeff_map, const_map, function_map, cache);

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

                let e_eval = e.evaluate(coeff_map, const_map, function_map, cache);
                b_eval.powf(&e_eval)
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r = it
                    .next()
                    .unwrap()
                    .evaluate(coeff_map, const_map, function_map, cache);
                for arg in it {
                    r *= arg.evaluate(coeff_map, const_map, function_map, cache);
                }
                r
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r = it
                    .next()
                    .unwrap()
                    .evaluate(coeff_map, const_map, function_map, cache);
                for arg in it {
                    r += arg.evaluate(coeff_map, const_map, function_map, cache);
                }
                r
            }
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::HashMap;

    use crate::{atom::Atom, domains::float::Float, evaluate::EvaluationFn, state::State};

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

        let r = a.evaluate(|x| x.into(), &const_map, &fn_map, &mut cache);
        assert_eq!(r, 2905.761021719902);
    }

    #[test]
    fn arb_prec() {
        let x = State::get_symbol("v1");
        let a = Atom::parse("128731/12893721893721 + v1").unwrap();

        let mut const_map = HashMap::default();

        let v = Atom::new_var(x);
        const_map.insert(v.as_view(), Float::with_val(200, 6));

        let r = a.evaluate(
            |r| r.to_multi_prec_float(200),
            &const_map,
            &HashMap::default(),
            &mut HashMap::default(),
        );

        assert_eq!(
            format!("{}", r),
            "6.00000000998400625211945786243908951675582851493871969158108"
        );
    }
}
