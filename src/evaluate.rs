use std::rc::Rc;

use ahash::{HashMap, HashSet};

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
    Expr(Symbol, Vec<Symbol>, AtomView<'a>),
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

#[derive(Debug, Clone, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub enum EvalTree<T> {
    Const(T),
    Parameter(usize),
    Eval(
        Vec<T>,      // a buffer for the evaluated arguments
        Symbol,      // function name
        Vec<Symbol>, // function argument names
        Vec<EvalTree<T>>,
        Box<EvalTree<T>>,
    ),
    Add(Vec<EvalTree<T>>),
    Mul(Vec<EvalTree<T>>),
    Pow(Box<(EvalTree<T>, i64)>),
    Powf(Box<(EvalTree<T>, EvalTree<T>)>),
    ReadArg(Symbol, usize), // read nth function argument, also store the name for codegen
    BuiltinFun(Symbol, Box<EvalTree<T>>),
    SubExpression(usize, Rc<EvalTree<T>>), // a reference to a subexpression
}

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    reserved_indices: usize,
    instructions: Vec<Instr>,
    result_index: usize,
}

impl<T: Real> ExpressionEvaluator<T> {
    pub fn evaluate(&mut self, params: &[T]) -> T {
        for (t, p) in self.stack.iter_mut().zip(params) {
            *t = p.clone();
        }

        let mut tmp;
        for i in &self.instructions {
            match i {
                Instr::Add(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp += e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
                }
                Instr::Mul(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp *= e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
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
            }
        }

        self.stack[self.result_index].clone()
    }
}

impl<T> ExpressionEvaluator<T> {
    pub fn optimize_stack(&mut self) {
        let mut last_use: Vec<usize> = vec![0; self.stack.len()];

        for (i, x) in self.instructions.iter().enumerate() {
            match x {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    for v in a {
                        last_use[*v] = i;
                    }
                }
                Instr::Pow(_, b, _) | Instr::BuiltinFun(_, _, b) => {
                    last_use[*b] = i;
                }
                Instr::Powf(_, a, b) => {
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
            };
        }

        // prevent init slots from being overwritten
        for i in 0..self.reserved_indices {
            last_use[i] = self.instructions.len();
        }

        let mut rename_map: Vec<_> = (0..self.stack.len()).collect(); // identity map

        let mut max_reg = self.reserved_indices;
        for (i, x) in self.instructions.iter_mut().enumerate() {
            let cur_reg = match x {
                Instr::Add(r, _)
                | Instr::Mul(r, _)
                | Instr::Pow(r, _, _)
                | Instr::Powf(r, _, _)
                | Instr::BuiltinFun(r, _, _) => *r,
            };

            let cur_last_use = last_use[cur_reg];

            let new_reg = if let Some((new_v, lu)) = last_use[..cur_reg]
                .iter_mut()
                .enumerate()
                .find(|(_, r)| **r <= i)
            // <= is ok because we store intermediate results in temp values
            {
                *lu = cur_last_use; // set the last use to the current variable last use
                last_use[cur_reg] = 0; // make the current index available
                rename_map[cur_reg] = new_v; // set the rename map so that every occurrence on the rhs is replaced
                new_v
            } else {
                cur_reg
            };

            max_reg = max_reg.max(new_reg);

            match x {
                Instr::Add(r, a) | Instr::Mul(r, a) => {
                    *r = new_reg;
                    for v in a {
                        *v = rename_map[*v];
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r = new_reg;
                    *b = rename_map[*b];
                }
                Instr::Powf(r, a, b) => {
                    *r = new_reg;
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                }
            };
        }

        self.stack.truncate(max_reg + 1);

        self.result_index = rename_map[self.result_index];
    }
}

#[derive(Debug)]
enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, Symbol, usize), // support function call too? that would be a jump in the instr table here?
}

impl<T: Clone + Default + PartialEq> EvalTree<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTree<T2> {
        match self {
            EvalTree::Const(c) => EvalTree::Const(f(c)),
            EvalTree::Parameter(p) => EvalTree::Parameter(*p),
            EvalTree::Eval(arg_buf, name, arg_names, e_args, ff) => {
                let new_args = e_args.iter().map(|x| x.map_coeff(f)).collect();
                EvalTree::Eval(
                    arg_buf.iter().map(|x| f(x)).collect(),
                    *name,
                    arg_names.clone(),
                    new_args,
                    Box::new(ff.map_coeff(f)),
                )
            }
            EvalTree::Add(a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                EvalTree::Add(new_args)
            }
            EvalTree::Mul(m) => {
                let new_args = m.iter().map(|x| x.map_coeff(f)).collect();
                EvalTree::Mul(new_args)
            }
            EvalTree::Pow(p) => {
                let (b, e) = &**p;
                EvalTree::Pow(Box::new((b.map_coeff(f), *e)))
            }
            EvalTree::Powf(p) => {
                let (b, e) = &**p;
                EvalTree::Powf(Box::new((b.map_coeff(f), e.map_coeff(f))))
            }
            EvalTree::ReadArg(s, i) => EvalTree::ReadArg(*s, *i),
            EvalTree::BuiltinFun(s, a) => EvalTree::BuiltinFun(*s, Box::new(a.map_coeff(f))),
            EvalTree::SubExpression(i, e) => EvalTree::SubExpression(*i, Rc::new(e.map_coeff(f))),
        }
    }

    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(mut self, param_len: usize) -> ExpressionEvaluator<T> {
        let mut stack = vec![T::default(); param_len];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack, param_len);
        let reserved_indices = stack.len();

        let mut sub_expr_pos = HashMap::default();
        let mut instructions = vec![];
        let result_index =
            self.linearize_impl(&mut stack, &mut instructions, &mut sub_expr_pos, &[]);

        let mut e = ExpressionEvaluator {
            stack,
            reserved_indices,
            instructions,
            result_index,
        };

        e.optimize_stack();
        e
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        match self {
            EvalTree::Const(t) => {
                if let Some(p) = stack.iter().skip(param_len).position(|x| x == t) {
                    *self = EvalTree::Parameter(param_len + p);
                } else {
                    stack.push(t.clone());
                    *self = EvalTree::Parameter(stack.len() - 1);
                }
            }
            EvalTree::Parameter(_) => {}
            EvalTree::Eval(_, _, _, e_args, f) => {
                for a in e_args {
                    a.strip_constants(stack, param_len);
                }
                f.strip_constants(stack, param_len);
            }
            EvalTree::Add(a) | EvalTree::Mul(a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            EvalTree::Pow(p) => {
                p.0.strip_constants(stack, param_len);
            }
            EvalTree::Powf(p) => {
                p.0.strip_constants(stack, param_len);
                p.1.strip_constants(stack, param_len);
            }
            EvalTree::ReadArg(_, _) => {}
            EvalTree::BuiltinFun(_, a) => {
                a.strip_constants(stack, param_len);
            }
            EvalTree::SubExpression(_, t) => {
                let mut t2 = t.as_ref().clone();
                t2.strip_constants(stack, param_len);
                *t = Rc::new(t2);
            }
        }
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        stack: &mut Vec<T>,
        instr: &mut Vec<Instr>,
        sub_expr_pos: &mut HashMap<usize, usize>,
        args: &[usize],
    ) -> usize {
        match self {
            EvalTree::Const(t) => {
                stack.push(t.clone()); // TODO: do once and recycle, this messes with the logic as there is no associated instruction
                stack.len() - 1
            }
            EvalTree::Parameter(i) => *i,
            EvalTree::Eval(_, _, _, e_args, f) => {
                // inline the function
                let new_args: Vec<_> = e_args
                    .iter()
                    .map(|x| x.linearize_impl(stack, instr, sub_expr_pos, args))
                    .collect();

                f.linearize_impl(stack, instr, sub_expr_pos, &new_args)
            }
            EvalTree::Add(a) => {
                let args = a
                    .iter()
                    .map(|x| x.linearize_impl(stack, instr, sub_expr_pos, args))
                    .collect();

                stack.push(T::default());
                let res = stack.len() - 1;

                let add = Instr::Add(res, args);
                instr.push(add);

                res
            }
            EvalTree::Mul(m) => {
                let args = m
                    .iter()
                    .map(|x| x.linearize_impl(stack, instr, sub_expr_pos, args))
                    .collect();

                stack.push(T::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(res, args);
                instr.push(mul);

                res
            }
            EvalTree::Pow(p) => {
                let b = p.0.linearize_impl(stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Pow(res, b, p.1));
                res
            }
            EvalTree::Powf(p) => {
                let b = p.0.linearize_impl(stack, instr, sub_expr_pos, args);
                let e = p.1.linearize_impl(stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Powf(res, b, e));
                res
            }
            EvalTree::ReadArg(_, a) => args[*a],
            EvalTree::BuiltinFun(s, v) => {
                let arg = v.linearize_impl(stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push(c);
                stack.len() - 1
            }
            EvalTree::SubExpression(id, s) => {
                if sub_expr_pos.contains_key(id) {
                    *sub_expr_pos.get(id).unwrap()
                } else {
                    let res = s.linearize_impl(stack, instr, sub_expr_pos, args);
                    sub_expr_pos.insert(*id, res);
                    res
                }
            }
        }
    }
}

impl EvalTree<Rational> {
    fn apply_horner_scheme(&mut self, scheme: &[EvalTree<Rational>]) {
        if scheme.is_empty() {
            return;
        }

        let EvalTree::Add(a) = self else {
            return;
        };

        let mut max_pow: Option<i64> = None;
        for x in &*a {
            if let EvalTree::Mul(m) = x {
                let mut pow_counter = 0;
                for y in m {
                    if let EvalTree::Pow(p) = y {
                        if p.0 == scheme[0] {
                            pow_counter += p.1;
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1; // support x*x*x^3 in term
                    }
                }

                if pow_counter > 0 && (max_pow.is_none() || pow_counter < max_pow.unwrap()) {
                    max_pow = Some(pow_counter);
                }
            } else if x == &scheme[0] {
                max_pow = Some(1);
            }
        }

        // TODO: jump to next variable if the current variable only appears in one factor?
        // this will improve the scheme but may hide common subexpressions?

        let Some(max_pow) = max_pow else {
            return self.apply_horner_scheme(&scheme[1..]);
        };

        let mut contains = vec![];
        let mut rest = vec![];

        for x in a {
            let mut found = false;
            if let EvalTree::Mul(m) = x {
                let mut pow_counter = 0;

                m.retain(|y| {
                    if let EvalTree::Pow(p) = y {
                        if p.0 == scheme[0] {
                            pow_counter += p.1;
                            false
                        } else {
                            true
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1;
                        false
                    } else {
                        true
                    }
                });

                if pow_counter > max_pow {
                    if pow_counter > max_pow + 1 {
                        m.push(EvalTree::Pow(Box::new((
                            scheme[0].clone(),
                            pow_counter - max_pow,
                        ))));
                    } else {
                        m.push(scheme[0].clone());
                    }

                    m.sort();
                }

                if m.is_empty() {
                    *x = EvalTree::Const(Rational::one());
                } else if m.len() == 1 {
                    *x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if x == &scheme[0] {
                found = true;
                *x = EvalTree::Const(Rational::one());
            }

            if found {
                contains.push(x.clone());
            } else {
                rest.push(x.clone());
            }
        }

        let extracted = if max_pow == 1 {
            scheme[0].clone()
        } else {
            EvalTree::Pow(Box::new((scheme[0].clone(), max_pow)))
        };

        let mut contains = if contains.len() == 1 {
            contains.pop().unwrap()
        } else {
            EvalTree::Add(contains)
        };

        contains.apply_horner_scheme(&scheme); // keep trying with same variable

        let mut v = vec![contains, extracted];
        v.sort();
        let c = EvalTree::Mul(v);

        if rest.is_empty() {
            *self = c;
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                EvalTree::Add(rest)
            };

            r.apply_horner_scheme(&scheme[1..]);

            let mut v = vec![c, r];
            v.sort();

            *self = EvalTree::Add(v);
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn horner_scheme(&mut self) {
        match self {
            EvalTree::Const(_) | EvalTree::Parameter(_) | EvalTree::ReadArg(_, _) => {}
            EvalTree::Eval(_, _, _, ae, f) => {
                for arg in ae {
                    arg.horner_scheme();
                }
                f.horner_scheme();
            }
            EvalTree::Add(a) => {
                for arg in &mut *a {
                    arg.horner_scheme();
                }

                let mut occurrence = HashMap::default();

                for arg in &*a {
                    match arg {
                        EvalTree::Mul(m) => {
                            for aa in m {
                                if let EvalTree::Pow(p) = aa {
                                    occurrence
                                        .entry(p.0.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                } else {
                                    occurrence
                                        .entry(aa.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let EvalTree::Pow(p) = x {
                                occurrence
                                    .entry(p.0.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            } else {
                                occurrence
                                    .entry(x.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            }
                        }
                    }
                }

                occurrence.retain(|_, v| *v > 1);
                let mut order: Vec<_> = occurrence.into_iter().collect();
                order.sort_by_key(|k| std::cmp::Reverse(k.1)); // occurrence order
                let scheme = order.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

                self.apply_horner_scheme(&scheme);
            }
            EvalTree::Mul(a) => {
                for arg in a {
                    arg.horner_scheme();
                }
            }
            EvalTree::Pow(p) => {
                p.0.horner_scheme();
            }
            EvalTree::Powf(p) => {
                p.0.horner_scheme();
                p.1.horner_scheme();
            }
            EvalTree::BuiltinFun(_, a) => {
                a.horner_scheme();
            }
            EvalTree::SubExpression(_, r) => {
                let mut rr = r.as_ref().clone();
                rr.horner_scheme();
                *r = Rc::new(rr);
            }
        }
    }
}

impl<T: Clone + Default + Eq + std::hash::Hash> EvalTree<T> {
    fn extract_subexpressions(&mut self) {
        let mut h = HashMap::default();
        self.find_subexpression(&mut h, 0, &mut 0);

        h.retain(|_, v| *v > 1);
        for (i, v) in h.values_mut().enumerate() {
            *v = i; // make the second argument a unique index of the subexpression
        }

        self.replace_subexpression(&h, 0, &mut 0, &mut HashMap::default());
    }

    fn replace_subexpression(
        &mut self,
        subexp: &HashMap<(usize, EvalTree<T>), usize>,
        branch_id: usize,
        new_branch_id: &mut usize,
        new_sub_tree: &mut HashMap<usize, EvalTree<T>>,
    ) {
        let key = (branch_id, self.clone()); // key before any replacements
        if let Some(i) = subexp.get(&key) {
            if new_sub_tree.contains_key(i) {
                *self = EvalTree::SubExpression(*i, Rc::new(new_sub_tree[i].clone()));
                return;
            }
        }

        match self {
            EvalTree::Const(_) | EvalTree::Parameter(_) | EvalTree::ReadArg(_, _) => {}
            EvalTree::Eval(_, _, _, ae, f) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, branch_id, new_branch_id, new_sub_tree);
                }

                *new_branch_id += 1;
                f.replace_subexpression(subexp, *new_branch_id, new_branch_id, new_sub_tree);
            }
            EvalTree::Add(a) | EvalTree::Mul(a) => {
                for arg in a {
                    arg.replace_subexpression(subexp, branch_id, new_branch_id, new_sub_tree);
                }
            }
            EvalTree::Pow(p) => {
                p.0.replace_subexpression(subexp, branch_id, new_branch_id, new_sub_tree);
            }
            EvalTree::Powf(p) => {
                p.0.replace_subexpression(subexp, branch_id, new_branch_id, new_sub_tree);
                p.1.replace_subexpression(subexp, branch_id, new_branch_id, new_sub_tree);
            }
            EvalTree::BuiltinFun(_, _) => {}
            EvalTree::SubExpression(_, _) => {
                unimplemented!("The expression should not already have subexpressions")
            }
        }

        if let Some(i) = subexp.get(&key) {
            new_sub_tree.insert(*i, self.clone());
            *self = EvalTree::SubExpression(*i, Rc::new(self.clone()));
        }
    }

    fn find_subexpression(
        &self,
        subexp: &mut HashMap<(usize, EvalTree<T>), usize>,
        branch_id: usize,
        new_branch_id: &mut usize,
    ) {
        if matches!(
            self,
            EvalTree::Const(_) | EvalTree::Parameter(_) | EvalTree::ReadArg(_, _)
        ) {
            return;
        }

        let key = (branch_id, self.clone());
        if let Some(i) = subexp.get_mut(&key) {
            *i += 1;
            return;
        }

        subexp.insert(key, 1);

        match self {
            EvalTree::Const(_) | EvalTree::Parameter(_) | EvalTree::ReadArg(_, _) => {}
            EvalTree::Eval(_, _, _, ae, f) => {
                for arg in ae {
                    arg.find_subexpression(subexp, branch_id, new_branch_id);
                }

                *new_branch_id += 1;
                f.find_subexpression(subexp, *new_branch_id, new_branch_id);
            }
            EvalTree::Add(a) | EvalTree::Mul(a) => {
                for arg in a {
                    arg.find_subexpression(subexp, branch_id, new_branch_id);
                }
            }
            EvalTree::Pow(p) => {
                p.0.find_subexpression(subexp, branch_id, new_branch_id);
            }
            EvalTree::Powf(p) => {
                p.0.find_subexpression(subexp, branch_id, new_branch_id);
                p.1.find_subexpression(subexp, branch_id, new_branch_id);
            }
            EvalTree::BuiltinFun(_, _) => {}
            EvalTree::SubExpression(_, _) => {
                unimplemented!("The expression should not already have subexpressions")
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
            EvalTree::Eval(arg_buf, _, _, e_args, f) => {
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
            EvalTree::ReadArg(_, i) => args[*i].clone(),
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
            EvalTree::SubExpression(_, _) => todo!(),
        }
    }

    pub fn export_cpp(&self) -> String {
        let mut res = String::new();

        let mut out_preamble = Vec::new();
        let mut processed_subexpr = HashSet::default();

        let mut funcs = HashMap::default();
        res += "\treturn ";
        self.export_cpp_impl(
            &mut res,
            &mut out_preamble,
            &mut processed_subexpr,
            &mut funcs,
        );
        res.push_str(";\n}\n");

        let mut fs = funcs.values().cloned().collect::<Vec<_>>();
        fs.sort();
        let mut fs = fs.into_iter().map(|(_, s)| s).collect::<Vec<_>>();

        fs.push(
            format!("template<typename T>\nT eval(T* params) {{\n")
                + out_preamble.join("").as_str()
                + res.as_str(),
        );

        fs.push(
            "extern \"C\" {\n\tdouble eval_double(double* params) {\n\t\t return eval(params);\n\t}\n}\n"
                .to_string()
        );

        fs.push(
            "int main() {\n\tstd::cout << eval(new double[]{5.0,6.0,7.0,8.0,9.0,10.0}) << std::endl;\n\treturn 0;\n}"
                .to_string(),
        );

        let header = "#include <iostream>\n#include <cmath>\n\n";

        header.to_string() + fs.join("\n").as_str()
    }

    fn export_cpp_impl(
        &self,
        out: &mut String,
        out_preamble: &mut Vec<String>,
        processed_subexpr: &mut HashSet<usize>,
        funcs: &mut HashMap<Symbol, (usize, String)>,
    ) {
        match self {
            EvalTree::Const(c) => {
                out.push_str(&format!("T({})", c));
            }
            EvalTree::Parameter(p) => {
                out.push_str(&format!("params[{}]", p));
            }
            EvalTree::Eval(_, name, arg_names, e_args, f) => {
                if funcs.get(name).is_none() {
                    let mut out = String::new();
                    let mut out_preamble = Vec::new();
                    let mut processed_subexpr = HashSet::default();

                    let mut args = arg_names
                        .iter()
                        .map(|x| "T ".to_string() + x.to_string().as_str())
                        .collect::<Vec<_>>();
                    args.insert(0, "T* params".to_string());

                    // our functions are all expressions so we return the expression
                    out.push_str("\treturn ");
                    f.export_cpp_impl(&mut out, &mut out_preamble, &mut processed_subexpr, funcs);

                    out.push_str(";\n}\n");
                    let l = funcs.len();
                    funcs.insert(
                        name.clone(),
                        (
                            l,
                            format!("template<typename T>\nT {}({}) {{\n", name, args.join(","))
                                + out_preamble.join("").as_str()
                                + out.as_str(),
                        ),
                    );
                }

                out.push_str(&format!("{}(params", name));

                for a in e_args {
                    out.push_str(", ");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                }
                out.push_str(")");
            }
            EvalTree::Add(a) => {
                out.push('(');
                a[0].export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                for arg in &a[1..] {
                    out.push_str(" + ");
                    arg.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                }
                out.push_str(")");
            }
            EvalTree::Mul(m) => {
                out.push('(');
                m[0].export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                for arg in &m[1..] {
                    out.push_str(" * ");
                    arg.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                }
                out.push(')');
            }
            EvalTree::Pow(p) => {
                out.push_str("pow(");
                p.0.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                out.push_str(", ");
                out.push_str(&p.1.to_string());
                out.push(')');
            }
            EvalTree::Powf(p) => {
                out.push_str("powf(");
                p.0.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                out.push_str(", ");
                p.1.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                out.push(')');
            }
            EvalTree::ReadArg(s, _) => {
                out.push_str(&format!("{}", s));
            }
            EvalTree::BuiltinFun(s, a) => match *s {
                State::EXP => {
                    out.push_str("exp(");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                    out.push(')');
                }
                State::LOG => {
                    out.push_str("log(");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                    out.push(')');
                }
                State::SIN => {
                    out.push_str("sin(");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                    out.push(')');
                }
                State::COS => {
                    out.push_str("cos(");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                    out.push(')');
                }
                State::SQRT => {
                    out.push_str("sqrt(");
                    a.export_cpp_impl(out, out_preamble, processed_subexpr, funcs);
                    out.push(')');
                }
                _ => unreachable!(),
            },
            EvalTree::SubExpression(id, s) => {
                if processed_subexpr.contains(id) {
                    out.push_str(&format!("s{}_", id));
                } else {
                    processed_subexpr.insert(*id);
                    let mut sub_out = String::new();
                    s.export_cpp_impl(&mut sub_out, out_preamble, processed_subexpr, funcs);

                    out_preamble.push(format!("\tT s{}_ = {};\n", id, sub_out));
                    out.push_str(&format!("s{}_", id));
                }
            }
        }
    }
}

impl<'a> AtomView<'a> {
    /// Convert nested expressions to a tree.
    pub fn to_eval_tree<T: Clone + Default + Eq + std::hash::Hash, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<AtomOrView, ConstOrExpr<'a, T>>,
        params: &[Atom],
    ) -> EvalTree<T> {
        let mut t = self.to_eval_tree_impl(coeff_map, const_map, params, &[]);
        t.extract_subexpressions();
        t
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
                ConstOrExpr::Expr(name, args, v) => {
                    if !args.is_empty() {
                        panic!(
                            "Function {} called with wrong number of arguments: 0 vs {}",
                            self,
                            args.len()
                        );
                    }

                    let r = v.to_eval_tree_impl(coeff_map, const_map, params, args);
                    EvalTree::Eval(vec![], *name, args.clone(), vec![], Box::new(r))
                }
            };
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => EvalTree::Const(coeff_map(&(n, d).into())),
                CoefficientView::Large(l) => EvalTree::Const(coeff_map(&l.to_rat())),
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
                    return EvalTree::ReadArg(name, p);
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
                    ConstOrExpr::Expr(name, arg_spec, e) => {
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

                        EvalTree::Eval(
                            vec![T::default(); arg_spec.len()],
                            *name,
                            arg_spec.clone(),
                            eval_args,
                            Box::new(res),
                        )
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
