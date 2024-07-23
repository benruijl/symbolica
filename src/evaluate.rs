use ahash::HashMap;
use self_cell::self_cell;

use crate::{
    atom::{representation::InlineVar, Atom, AtomOrView, AtomView, Symbol},
    coefficient::CoefficientView,
    domains::{
        float::{Complex, NumericalFloatLike, Real},
        rational::Rational,
    },
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

#[derive(PartialEq, Eq, Hash)]
enum AtomOrTaggedFunction<'a> {
    Atom(AtomOrView<'a>),
    TaggedFunction(Symbol, Vec<AtomOrView<'a>>),
}

pub struct FunctionMap<'a, T = Rational> {
    map: HashMap<AtomOrTaggedFunction<'a>, ConstOrExpr<'a, T>>,
    tag: HashMap<Symbol, usize>,
}

impl<'a, T> FunctionMap<'a, T> {
    pub fn new() -> Self {
        FunctionMap {
            map: HashMap::default(),
            tag: HashMap::default(),
        }
    }

    pub fn add_constant(&mut self, key: AtomOrView<'a>, value: T) {
        self.map
            .insert(AtomOrTaggedFunction::Atom(key), ConstOrExpr::Const(value));
    }

    pub fn add_function(
        &mut self,
        name: Symbol,
        rename: String,
        args: Vec<Symbol>,
        body: AtomView<'a>,
    ) -> Result<(), &str> {
        if let Some(t) = self.tag.insert(name, 0) {
            if t != 0 {
                return Err("Cannot add the same function with a different number of parameters");
            }
        }

        self.map.insert(
            AtomOrTaggedFunction::Atom(Atom::new_var(name).into()),
            ConstOrExpr::Expr(rename, 0, args, body),
        );

        Ok(())
    }

    pub fn add_tagged_function(
        &mut self,
        name: Symbol,
        tags: Vec<AtomOrView>,
        rename: String,
        args: Vec<Symbol>,
        body: AtomView<'a>,
    ) -> Result<(), &str> {
        if let Some(t) = self.tag.insert(name, tags.len()) {
            if t != tags.len() {
                return Err("Cannot add the same function with a different number of parameters");
            }
        }

        self.map.insert(
            AtomOrTaggedFunction::Atom(Atom::new_var(name).into()),
            ConstOrExpr::Expr(rename, tags.len(), args, body),
        );

        Ok(())
    }

    fn get_tag_len(&self, symbol: &Symbol) -> usize {
        self.tag.get(symbol).cloned().unwrap_or(0)
    }

    fn get(&self, a: AtomView<'a>) -> Option<&ConstOrExpr<'a, T>> {
        if let Some(c) = self.map.get(&AtomOrTaggedFunction::Atom(a.into())) {
            return Some(c);
        }

        if let AtomView::Fun(aa) = a {
            let s = aa.get_symbol();
            let tag_len = self.get_tag_len(&s);

            if tag_len != 0 && aa.get_nargs() >= tag_len {
                let tag = aa.iter().take(tag_len).map(|x| x.into()).collect();
                return self.map.get(&AtomOrTaggedFunction::TaggedFunction(s, tag));
            }
        }

        None
    }
}

enum ConstOrExpr<'a, T> {
    Const(T),
    Expr(String, usize, Vec<Symbol>, AtomView<'a>),
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

pub struct SplitExpression<T> {
    pub tree: Vec<Expression<T>>,
    pub subexpressions: Vec<Expression<T>>,
}

pub struct EvalTree<T> {
    functions: Vec<(String, Vec<Symbol>, SplitExpression<T>)>,
    expressions: SplitExpression<T>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Expression<T> {
    Const(T),
    Parameter(usize),
    Eval(usize, Vec<Expression<T>>),
    Add(Vec<Expression<T>>),
    Mul(Vec<Expression<T>>),
    Pow(Box<(Expression<T>, i64)>),
    Powf(Box<(Expression<T>, Expression<T>)>),
    ReadArg(usize), // read nth function argument
    BuiltinFun(Symbol, Box<Expression<T>>),
    SubExpression(usize),
}

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    param_count: usize,
    reserved_indices: usize,
    instructions: Vec<Instr>,
    result_indices: Vec<usize>,
}

impl<T: Real> ExpressionEvaluator<T> {
    pub fn evaluate(&mut self, params: &[T]) -> T {
        let mut res = T::new_zero();
        self.evaluate_multiple(params, std::slice::from_mut(&mut res));
        res
    }

    pub fn evaluate_multiple(&mut self, params: &[T], out: &mut [T]) {
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

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            *o = self.stack[*i].clone();
        }
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

        // prevent the output slots from being overwritten
        for i in &self.result_indices {
            last_use[*i] = self.instructions.len();
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

        for i in &mut self.result_indices {
            *i = rename_map[*i];
        }
    }
}

impl<T: std::fmt::Display> ExpressionEvaluator<T> {
    /// Create a C++ code representation of the evaluation tree.
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
    ) -> Result<ExportedCode, std::io::Error> {
        let cpp = self.export_cpp_str(function_name, include_header);
        std::fs::write(filename, cpp)?;
        Ok(ExportedCode {
            source_filename: filename.to_string(),
            function_name: function_name.to_string(),
        })
    }

    pub fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = if include_header {
            "#include <iostream>\n#include <complex>\n#include <cmath>\n\n".to_string()
        } else {
            String::new()
        };

        res += &format!("\ntemplate<typename T>\nvoid eval(T* params, T* out) {{\n");

        res += &format!(
            "\tT {};\n",
            (0..self.stack.len())
                .map(|x| format!("Z{}", x))
                .collect::<Vec<_>>()
                .join(", ")
        );

        for i in 0..self.param_count {
            res += &format!("\tZ{} = params[{}];\n", i, i);
        }

        for i in self.param_count..self.reserved_indices {
            res += &format!("\tZ{} = {};\n", i, self.stack[i]);
        }

        Self::export_cpp_impl(&self.instructions, &mut res);

        for (i, r) in &mut self.result_indices.iter().enumerate() {
            res += &format!("\tout[{}] = Z{};\n", i, r);
        }

        res += "\treturn;\n}\n";

        res += &format!("\nextern \"C\" {{\n\tvoid {}_double(double* params, double* out) {{\n\t\teval(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);
        res += &format!("\nextern \"C\" {{\n\tvoid {}_complex(std::complex<double>* params, std::complex<double>* out) {{\n\t\teval(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);

        res
    }

    fn export_cpp_impl(instr: &[Instr], out: &mut String) {
        for ins in instr {
            match ins {
                Instr::Add(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| format!("Z{}", x))
                        .collect::<Vec<_>>()
                        .join("+");

                    *out += format!("\tZ{} = {};\n", o, args).as_str();
                }
                Instr::Mul(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| format!("Z{}", x))
                        .collect::<Vec<_>>()
                        .join("*");

                    *out += format!("\tZ{} = {};\n", o, args).as_str();
                }
                Instr::Pow(o, b, e) => {
                    let base = format!("Z{}", b);
                    *out += format!("\tZ{} = pow({}, {});\n", o, base, e).as_str();
                }
                Instr::Powf(o, b, e) => {
                    let base = format!("Z{}", b);
                    let exp = format!("Z{}", e);
                    *out += format!("\tZ{} = pow({}, {});\n", o, base, exp).as_str();
                }
                Instr::BuiltinFun(o, s, a) => match *s {
                    State::EXP => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = exp({});\n", o, arg).as_str();
                    }
                    State::LOG => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = log({});\n", o, arg).as_str();
                    }
                    State::SIN => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = sin({});\n", o, arg).as_str();
                    }
                    State::COS => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = cos({});\n", o, arg).as_str();
                    }
                    State::SQRT => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = sqrt({});\n", o, arg).as_str();
                    }
                    _ => unreachable!(),
                },
            }
        }
    }
}

#[derive(Debug)]
enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, Symbol, usize),
}

impl<T: Clone + Default + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }
}

impl<T: Clone + Default + PartialEq> Expression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> Expression<T2> {
        match self {
            Expression::Const(c) => Expression::Const(f(c)),
            Expression::Parameter(p) => Expression::Parameter(*p),
            Expression::Eval(id, e_args) => {
                Expression::Eval(*id, e_args.iter().map(|x| x.map_coeff(f)).collect())
            }
            Expression::Add(a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Add(new_args)
            }
            Expression::Mul(m) => {
                let new_args = m.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Mul(new_args)
            }
            Expression::Pow(p) => {
                let (b, e) = &**p;
                Expression::Pow(Box::new((b.map_coeff(f), *e)))
            }
            Expression::Powf(p) => {
                let (b, e) = &**p;
                Expression::Powf(Box::new((b.map_coeff(f), e.map_coeff(f))))
            }
            Expression::ReadArg(s) => Expression::ReadArg(*s),
            Expression::BuiltinFun(s, a) => Expression::BuiltinFun(*s, Box::new(a.map_coeff(f))),
            Expression::SubExpression(i) => Expression::SubExpression(*i),
        }
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        match self {
            Expression::Const(t) => {
                if let Some(p) = stack.iter().skip(param_len).position(|x| x == t) {
                    *self = Expression::Parameter(param_len + p);
                } else {
                    stack.push(t.clone());
                    *self = Expression::Parameter(stack.len() - 1);
                }
            }
            Expression::Parameter(_) => {}
            Expression::Eval(_, e_args) => {
                for a in e_args {
                    a.strip_constants(stack, param_len);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::Pow(p) => {
                p.0.strip_constants(stack, param_len);
            }
            Expression::Powf(p) => {
                p.0.strip_constants(stack, param_len);
                p.1.strip_constants(stack, param_len);
            }
            Expression::ReadArg(_) => {}
            Expression::BuiltinFun(_, a) => {
                a.strip_constants(stack, param_len);
            }
            Expression::SubExpression(_) => {}
        }
    }
}

impl<T: Clone + Default + PartialEq> EvalTree<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTree<T2> {
        EvalTree {
            expressions: SplitExpression {
                tree: self
                    .expressions
                    .tree
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
                subexpressions: self
                    .expressions
                    .subexpressions
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
            },
            functions: self
                .functions
                .iter()
                .map(|(s, a, e)| (s.clone(), a.clone(), e.map_coeff(f)))
                .collect(),
        }
    }

    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(mut self, param_count: usize) -> ExpressionEvaluator<T> {
        let mut stack = vec![T::default(); param_count];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack, param_count);
        let reserved_indices = stack.len();

        let mut sub_expr_pos = HashMap::default();
        let mut instructions = vec![];

        let mut result_indices = vec![];

        for t in &self.expressions.tree {
            let result_index = self.linearize_impl(
                &t,
                &self.expressions.subexpressions,
                &mut stack,
                &mut instructions,
                &mut sub_expr_pos,
                &[],
            );
            result_indices.push(result_index);
        }

        let mut e = ExpressionEvaluator {
            stack,
            param_count,
            reserved_indices,
            instructions,
            result_indices,
        };

        e.optimize_stack();
        e
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        for t in &mut self.expressions.tree {
            t.strip_constants(stack, param_len);
        }

        for e in &mut self.expressions.subexpressions {
            e.strip_constants(stack, param_len);
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.strip_constants(stack, param_len);
            }

            for e in &mut e.subexpressions {
                e.strip_constants(stack, param_len);
            }
        }
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        tree: &Expression<T>,
        subexpressions: &[Expression<T>],
        stack: &mut Vec<T>,
        instr: &mut Vec<Instr>,
        sub_expr_pos: &mut HashMap<usize, usize>,
        args: &[usize],
    ) -> usize {
        match tree {
            Expression::Const(t) => {
                stack.push(t.clone()); // TODO: do once and recycle, this messes with the logic as there is no associated instruction
                stack.len() - 1
            }
            Expression::Parameter(i) => *i,
            Expression::Eval(id, e_args) => {
                // inline the function
                let new_args: Vec<_> = e_args
                    .iter()
                    .map(|x| {
                        self.linearize_impl(x, subexpressions, stack, instr, sub_expr_pos, args)
                    })
                    .collect();

                let mut sub_expr_pos = HashMap::default();
                let func = &self.functions[*id].2;
                self.linearize_impl(
                    &func.tree[0],
                    &func.subexpressions,
                    stack,
                    instr,
                    &mut sub_expr_pos,
                    &new_args,
                )
            }
            Expression::Add(a) => {
                let args = a
                    .iter()
                    .map(|x| {
                        self.linearize_impl(x, subexpressions, stack, instr, sub_expr_pos, args)
                    })
                    .collect();

                stack.push(T::default());
                let res = stack.len() - 1;

                let add = Instr::Add(res, args);
                instr.push(add);

                res
            }
            Expression::Mul(m) => {
                let args = m
                    .iter()
                    .map(|x| {
                        self.linearize_impl(x, subexpressions, stack, instr, sub_expr_pos, args)
                    })
                    .collect();

                stack.push(T::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(res, args);
                instr.push(mul);

                res
            }
            Expression::Pow(p) => {
                let b = self.linearize_impl(&p.0, subexpressions, stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Pow(res, b, p.1));
                res
            }
            Expression::Powf(p) => {
                let b = self.linearize_impl(&p.0, subexpressions, stack, instr, sub_expr_pos, args);
                let e = self.linearize_impl(&p.1, subexpressions, stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Powf(res, b, e));
                res
            }
            Expression::ReadArg(a) => args[*a],
            Expression::BuiltinFun(s, v) => {
                let arg = self.linearize_impl(v, subexpressions, stack, instr, sub_expr_pos, args);
                stack.push(T::default());
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push(c);
                stack.len() - 1
            }
            Expression::SubExpression(id) => {
                if sub_expr_pos.contains_key(id) {
                    *sub_expr_pos.get(id).unwrap()
                } else {
                    let res = self.linearize_impl(
                        &subexpressions[*id],
                        subexpressions,
                        stack,
                        instr,
                        sub_expr_pos,
                        args,
                    );
                    sub_expr_pos.insert(*id, res);
                    res
                }
            }
        }
    }
}

impl EvalTree<Rational> {
    pub fn horner_scheme(&mut self) {
        for t in &mut self.expressions.tree {
            t.horner_scheme();
        }

        for e in &mut self.expressions.subexpressions {
            e.horner_scheme();
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.horner_scheme();
            }

            for e in &mut e.subexpressions {
                e.horner_scheme();
            }
        }
    }
}

impl Expression<Rational> {
    fn apply_horner_scheme(&mut self, scheme: &[Expression<Rational>]) {
        if scheme.is_empty() {
            return;
        }

        let Expression::Add(a) = self else {
            return;
        };

        a.sort();

        let mut max_pow: Option<i64> = None;
        for x in &*a {
            if let Expression::Mul(m) = x {
                let mut pow_counter = 0;
                for y in m {
                    if let Expression::Pow(p) = y {
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
            if let Expression::Mul(m) = x {
                let mut pow_counter = 0;

                m.retain(|y| {
                    if let Expression::Pow(p) = y {
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
                        m.push(Expression::Pow(Box::new((
                            scheme[0].clone(),
                            pow_counter - max_pow,
                        ))));
                    } else {
                        m.push(scheme[0].clone());
                    }

                    m.sort();
                }

                if m.is_empty() {
                    *x = Expression::Const(Rational::one());
                } else if m.len() == 1 {
                    *x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if x == &scheme[0] {
                found = true;
                *x = Expression::Const(Rational::one());
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
            Expression::Pow(Box::new((scheme[0].clone(), max_pow)))
        };

        let mut contains = if contains.len() == 1 {
            contains.pop().unwrap()
        } else {
            Expression::Add(contains)
        };

        contains.apply_horner_scheme(&scheme); // keep trying with same variable

        let mut v = vec![contains, extracted];
        v.sort();
        let c = Expression::Mul(v);

        if rest.is_empty() {
            *self = c;
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                Expression::Add(rest)
            };

            r.apply_horner_scheme(&scheme[1..]);

            let mut v = vec![c, r];
            v.sort();

            *self = Expression::Add(v);
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn horner_scheme(&mut self) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.horner_scheme();
                }
            }
            Expression::Add(a) => {
                for arg in &mut *a {
                    arg.horner_scheme();
                }

                let mut occurrence = HashMap::default();

                for arg in &*a {
                    match arg {
                        Expression::Mul(m) => {
                            for aa in m {
                                if let Expression::Pow(p) = aa {
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
                            if let Expression::Pow(p) = x {
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
            Expression::Mul(a) => {
                for arg in a {
                    arg.horner_scheme();
                }
            }
            Expression::Pow(p) => {
                p.0.horner_scheme();
            }
            Expression::Powf(p) => {
                p.0.horner_scheme();
                p.1.horner_scheme();
            }
            Expression::BuiltinFun(_, a) => {
                a.horner_scheme();
            }
            Expression::SubExpression(_) => {}
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> EvalTree<T> {
    pub fn common_subexpression_elimination(&mut self) {
        self.expressions.common_subexpression_elimination();

        for (_, _, e) in &mut self.functions {
            e.common_subexpression_elimination();
        }
    }

    pub fn common_pair_elimination(&mut self) {
        while self.expressions.common_pair_elimination() {}
        for (_, _, e) in &mut self.functions {
            while e.common_pair_elimination() {}
        }
    }

    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.functions {
            let (ea, em) = e.2.count_operations();
            add += ea;
            mul += em;
        }

        let (ea, em) = self.expressions.count_operations();
        (add + ea, mul + em)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> SplitExpression<T> {
    pub fn common_subexpression_elimination(&mut self) {
        let mut h = HashMap::default();

        for t in &mut self.tree {
            t.find_subexpression(&mut h);
        }

        h.retain(|_, v| *v > 1);

        // make the second argument a unique index of the subexpression
        for (i, v) in h.values_mut().enumerate() {
            *v = self.subexpressions.len() + i;
        }

        for t in &mut self.tree {
            t.replace_subexpression(&h);
        }

        let mut v: Vec<_> = h.into_iter().map(|(k, v)| (v, k)).collect();
        v.sort();

        for (_, x) in v {
            self.subexpressions.push(x);
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> Expression<T> {
    fn replace_subexpression(&mut self, subexp: &HashMap<Expression<T>, usize>) {
        if let Some(i) = subexp.get(&self) {
            *self = Expression::SubExpression(*i);
            return;
        }

        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.replace_subexpression(subexp);
                }
            }
            Expression::Pow(p) => {
                p.0.replace_subexpression(subexp);
            }
            Expression::Powf(p) => {
                p.0.replace_subexpression(subexp);
                p.1.replace_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(_) => {
                unimplemented!("The expression should not already have subexpressions")
            }
        }
    }

    fn find_subexpression(&self, subexp: &mut HashMap<Expression<T>, usize>) {
        if matches!(
            self,
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_)
        ) {
            return;
        }

        if let Some(i) = subexp.get_mut(self) {
            *i += 1;
            return;
        }

        subexp.insert(self.clone(), 1);

        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Pow(p) => {
                p.0.find_subexpression(subexp);
            }
            Expression::Powf(p) => {
                p.0.find_subexpression(subexp);
                p.1.find_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(_) => {}
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> SplitExpression<T> {
    /// Find and extract pairs of variables that appear in more than one instruction.
    /// This reduces the number of operations. Returns `true` iff an extraction could be performed.
    ///
    /// This function can be called multiple times such that common subexpressions that
    /// are larger than pairs can also be extracted.
    pub fn common_pair_elimination(&mut self) -> bool {
        let mut pair_count = HashMap::default();

        for e in &self.subexpressions {
            e.find_common_pairs(&mut pair_count);
        }

        for t in &self.tree {
            t.find_common_pairs(&mut pair_count);
        }

        let mut v: Vec<_> = pair_count.into_iter().collect();
        v.retain(|x| x.1 > 1);
        v.sort_by_key(|k| std::cmp::Reverse(k.1));

        let v: Vec<_> = v
            .into_iter()
            .map(|((a, b, c), e)| ((a, b.clone(), c.clone()), e))
            .collect();

        for ((is_add, l, r), _) in &v {
            let id = self.subexpressions.len();

            for t in &mut self.tree {
                t.replace_common_pair(*is_add, l, r, id);
            }

            let mut first_replace = None;
            for (i, e) in &mut self.subexpressions.iter_mut().enumerate() {
                if e.replace_common_pair(*is_add, l, r, id) {
                    if first_replace.is_none() {
                        first_replace = Some(i);
                    }
                }
            }

            let pair = if *is_add {
                Expression::Add(vec![l.clone(), r.clone()])
            } else {
                Expression::Mul(vec![l.clone(), r.clone()])
            };

            if let Some(i) = first_replace {
                // all subexpressions need to be shifted
                for k in i..self.subexpressions.len() {
                    self.subexpressions[k].shift_subexpr(i, id);
                }

                for t in &mut self.tree {
                    t.shift_subexpr(i, id);
                }

                self.subexpressions.insert(i, pair);
            } else {
                self.subexpressions.push(pair);
            }

            // some subexpression could be Z3=Z2 now, remove that
            for i in (0..self.subexpressions.len()).rev() {
                if let Expression::SubExpression(n) = &self.subexpressions[i] {
                    let n = *n;
                    self.subexpressions.remove(i);
                    for e in &mut self.subexpressions[i..] {
                        e.rename_subexpr(i, n);
                    }

                    for t in &mut self.tree {
                        t.rename_subexpr(i, n);
                    }
                }
            }

            return true; // do just one for now
        }

        false
    }

    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.subexpressions {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        for e in &self.tree {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        (add, mul)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> Expression<T> {
    pub fn count_operations(&self) -> (usize, usize) {
        match self {
            Expression::Const(_) => (0, 0),
            Expression::Parameter(_) => (0, 0),
            Expression::Eval(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(p) => {
                let (a, m) = p.0.count_operations();
                (a, m + p.1 as usize - 1)
            }
            Expression::Powf(p) => {
                let (a, m) = p.0.count_operations();
                let (a2, m2) = p.1.count_operations();
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_) => (0, 0),
            Expression::BuiltinFun(_, _) => (0, 0), // not clear how to count this, third arg?
            Expression::SubExpression(_) => (0, 0),
        }
    }

    fn shift_subexpr(&mut self, pos: usize, max: usize) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.shift_subexpr(pos, max);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.shift_subexpr(pos, max);
                }
            }
            Expression::Pow(p) => {
                p.0.shift_subexpr(pos, max);
            }
            Expression::Powf(p) => {
                p.0.shift_subexpr(pos, max);
                p.1.shift_subexpr(pos, max);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(i) => {
                if *i == max {
                    *i = pos;
                } else if *i >= pos {
                    *i += 1;
                }
            }
        }
    }

    fn rename_subexpr(&mut self, old: usize, new: usize) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.rename_subexpr(old, new);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.rename_subexpr(old, new);
                }
            }
            Expression::Pow(p) => {
                p.0.rename_subexpr(old, new);
            }
            Expression::Powf(p) => {
                p.0.rename_subexpr(old, new);
                p.1.rename_subexpr(old, new);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(i) => {
                if *i == old {
                    *i = new;
                } else if *i > old {
                    *i -= 1;
                }
            }
        }
    }

    fn find_common_pairs<'a>(&'a self, subexp: &mut HashMap<(bool, &'a Self, &'a Self), usize>) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.find_common_pairs(subexp);
                }
            }
            x @ Expression::Add(m) | x @ Expression::Mul(m) => {
                for a in m {
                    a.find_common_pairs(subexp);
                }

                let mut d: Vec<_> = m.iter().collect();
                d.dedup();
                let mut rep = vec![0; d.len()];

                for (c, v) in rep.iter_mut().zip(&d) {
                    for v2 in m {
                        if *v == v2 {
                            *c += 1;
                        }
                    }
                }

                for i in 0..d.len() {
                    if rep[i] > 1 {
                        *subexp
                            .entry((matches!(x, Expression::Add(_)), &d[i], &d[i]))
                            .or_insert(0) += rep[i] / 2;
                    }

                    for j in i + 1..d.len() {
                        *subexp
                            .entry((matches!(x, Expression::Add(_)), &d[i], &d[j]))
                            .or_insert(0) += rep[i].min(rep[j]);
                    }
                }
            }
            Expression::Pow(p) => {
                p.0.find_common_pairs(subexp);
            }
            Expression::Powf(p) => {
                p.0.find_common_pairs(subexp);
                p.1.find_common_pairs(subexp);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(_) => {}
        }
    }

    fn replace_common_pair(&mut self, is_add: bool, r: &Self, l: &Self, subexpr_id: usize) -> bool {
        let cur_is_add = matches!(self, Expression::Add(_));

        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => false,
            Expression::Eval(_, ae) => {
                let mut replaced = false;
                for arg in &mut *ae {
                    replaced |= arg.replace_common_pair(is_add, r, l, subexpr_id);
                }
                replaced
            }
            Expression::Add(a) | Expression::Mul(a) => {
                let mut replaced = false;
                for arg in &mut *a {
                    replaced |= arg.replace_common_pair(is_add, r, l, subexpr_id);
                }

                if is_add != cur_is_add {
                    return replaced;
                }

                if l == r {
                    let count = a.iter().filter(|x| *x == l).count();
                    let pairs = count / 2;
                    if pairs > 0 {
                        a.retain(|x| x != l);

                        if count % 2 == 1 {
                            a.push(l.clone());
                        }

                        a.extend(
                            std::iter::repeat(Expression::SubExpression(subexpr_id)).take(pairs),
                        );
                        a.sort();

                        if a.len() == 1 {
                            *self = a.pop().unwrap();
                        }

                        return true;
                    }
                } else {
                    let mut idx1_count = 0;
                    let mut idx2_count = 0;
                    for v in &*a {
                        if v == l {
                            idx1_count += 1;
                        }
                        if v == r {
                            idx2_count += 1;
                        }
                    }

                    let pair_count = idx1_count.min(idx2_count);

                    if pair_count > 0 {
                        a.retain(|x| x != l && x != r);

                        // add back removed indices in cases such as idx1*idx2*idx2
                        if idx1_count > pair_count {
                            a.extend(std::iter::repeat(l.clone()).take(idx1_count - pair_count));
                        }
                        if idx2_count > pair_count {
                            a.extend(std::iter::repeat(r.clone()).take(idx2_count - pair_count));
                        }

                        a.extend(
                            std::iter::repeat(Expression::SubExpression(subexpr_id))
                                .take(pair_count),
                        );
                        a.sort();

                        if a.len() == 1 {
                            *self = a.pop().unwrap();
                        }

                        return true;
                    }
                }

                replaced
            }
            Expression::Pow(p) => p.0.replace_common_pair(is_add, r, l, subexpr_id),
            Expression::Powf(p) => {
                let mut replaced = p.0.replace_common_pair(is_add, r, l, subexpr_id);
                replaced |= p.1.replace_common_pair(is_add, r, l, subexpr_id);
                replaced
            }
            Expression::BuiltinFun(_, _) => false,
            Expression::SubExpression(_) => false,
        }
    }
}

impl<T: Real> EvalTree<T> {
    /// Evaluate the evaluation tree. Consider converting to a linear form for repeated evaluation.
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        for (o, e) in out.iter_mut().zip(&self.expressions.tree) {
            *o = self.evaluate_impl(&e, &self.expressions.subexpressions, params, &[])
        }
    }

    fn evaluate_impl(
        &self,
        expr: &Expression<T>,
        subexpressions: &[Expression<T>],
        params: &[T],
        args: &[T],
    ) -> T {
        match expr {
            Expression::Const(c) => c.clone(),
            Expression::Parameter(p) => params[*p].clone(),
            Expression::Eval(f, e_args) => {
                let mut arg_buf = vec![T::new_zero(); e_args.len()];
                for (b, a) in arg_buf.iter_mut().zip(e_args.iter()) {
                    *b = self.evaluate_impl(a, subexpressions, params, args);
                }

                let func = &self.functions[*f].2;
                self.evaluate_impl(&func.tree[0], &func.subexpressions, params, &arg_buf)
            }
            Expression::Add(a) => {
                let mut r = self.evaluate_impl(&a[0], subexpressions, params, args);
                for arg in &a[1..] {
                    r += self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Mul(m) => {
                let mut r = self.evaluate_impl(&m[0], subexpressions, params, args);
                for arg in &m[1..] {
                    r *= self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Pow(p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);

                if *e >= 0 {
                    b_eval.pow(*e as u64)
                } else {
                    b_eval.pow(e.unsigned_abs()).inv()
                }
            }
            Expression::Powf(p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);
                let e_eval = self.evaluate_impl(e, subexpressions, params, args);
                b_eval.powf(&e_eval)
            }
            Expression::ReadArg(i) => args[*i].clone(),
            Expression::BuiltinFun(s, a) => {
                let arg = self.evaluate_impl(a, subexpressions, params, args);
                match *s {
                    State::EXP => arg.exp(),
                    State::LOG => arg.log(),
                    State::SIN => arg.sin(),
                    State::COS => arg.cos(),
                    State::SQRT => arg.sqrt(),
                    _ => unreachable!(),
                }
            }
            Expression::SubExpression(s) => {
                // TODO: cache
                self.evaluate_impl(&subexpressions[*s], subexpressions, params, args)
            }
        }
    }
}

pub struct ExportedCode {
    source_filename: String,
    function_name: String,
}
pub struct CompiledCode {
    library_filename: String,
    function_name: String,
}

impl CompiledCode {
    /// Load the evaluator from the compiled shared library.
    pub fn load(&self) -> Result<CompiledEvaluator, String> {
        CompiledEvaluator::load(&self.library_filename, &self.function_name)
    }
}

type L = std::sync::Arc<libloading::Library>;

#[derive(Debug)]
struct EvaluatorFunctions<'a> {
    eval_double: libloading::Symbol<'a, unsafe extern "C" fn(params: *const f64, out: *mut f64)>,
    eval_complex: libloading::Symbol<
        'a,
        unsafe extern "C" fn(params: *const Complex<f64>, out: *mut Complex<f64>),
    >,
}

self_cell!(
    pub struct CompiledEvaluator {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctions,
    }

    impl {Debug}
);

/// A floating point type that can be used for compiled evaluation.
pub trait CompiledEvaluatorFloat: Sized {
    fn evaluate(eval: &CompiledEvaluator, args: &[Self], out: &mut [Self]);
}

impl CompiledEvaluatorFloat for f64 {
    #[inline(always)]
    fn evaluate(eval: &CompiledEvaluator, args: &[Self], out: &mut [Self]) {
        eval.evaluate_double(args, out);
    }
}

impl CompiledEvaluatorFloat for Complex<f64> {
    #[inline(always)]
    fn evaluate(eval: &CompiledEvaluator, args: &[Self], out: &mut [Self]) {
        eval.evaluate_complex(args, out);
    }
}

impl CompiledEvaluator {
    /// Load a new function from the same library.
    pub fn load_new_function(&self, function_name: &str) -> Result<CompiledEvaluator, String> {
        unsafe {
            CompiledEvaluator::try_new(self.borrow_owner().clone(), |lib| {
                Ok(EvaluatorFunctions {
                    eval_double: lib
                        .get(format!("{}_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    eval_complex: lib
                        .get(format!("{}_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                })
            })
        }
    }

    /// Load a compiled evaluator from a shared library.
    pub fn load(file: &str, function_name: &str) -> Result<CompiledEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(file) {
                Ok(lib) => lib,
                Err(_) => {
                    libloading::Library::new("./".to_string() + file).map_err(|e| e.to_string())?
                }
            };

            CompiledEvaluator::try_new(std::sync::Arc::new(lib), |lib| {
                Ok(EvaluatorFunctions {
                    eval_double: lib
                        .get(format!("{}_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    eval_complex: lib
                        .get(format!("{}_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                })
            })
        }
    }

    /// Evaluate the compiled code.
    #[inline(always)]
    pub fn evaluate<T: CompiledEvaluatorFloat>(&self, args: &[T], out: &mut [T]) {
        T::evaluate(self, args, out);
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    pub fn evaluate_double(&self, args: &[f64], out: &mut [f64]) {
        unsafe { (self.borrow_dependent().eval_double)(args.as_ptr(), out.as_mut_ptr()) }
    }

    /// Evaluate the compiled code with complex numbers.
    #[inline(always)]
    pub fn evaluate_complex(&self, args: &[Complex<f64>], out: &mut [Complex<f64>]) {
        unsafe { (self.borrow_dependent().eval_complex)(args.as_ptr(), out.as_mut_ptr()) }
    }
}

/// Options for compiling exported code.
pub struct CompileOptions {
    pub optimization_level: usize,
    pub fast_math: bool,
    pub unsafe_math: bool,
    pub compiler: String,
    pub custom: Vec<String>,
}

impl Default for CompileOptions {
    /// Default compile options: `g++ -O3 -ffast-math -funsafe-math-optimizations`.
    fn default() -> Self {
        CompileOptions {
            optimization_level: 3,
            fast_math: true,
            unsafe_math: true,
            compiler: "g++".to_string(),
            custom: vec![],
        }
    }
}

impl ExportedCode {
    /// Create a new exported code object from a source file and function name.
    pub fn new(source_filename: String, function_name: String) -> Self {
        ExportedCode {
            source_filename,
            function_name,
        }
    }

    /// Compile the code to a shared library.
    pub fn compile(
        &self,
        out: &str,
        options: CompileOptions,
    ) -> Result<CompiledCode, std::io::Error> {
        let mut builder = std::process::Command::new(&options.compiler);
        builder
            .arg("-shared")
            .arg("-fPIC")
            .arg(format!("-O{}", options.optimization_level));
        if options.fast_math {
            builder.arg("-ffast-math");
        }
        if options.unsafe_math {
            builder.arg("-funsafe-math-optimizations");
        }
        for c in &options.custom {
            builder.arg(c);
        }

        let r = builder
            .arg("-o")
            .arg(out)
            .arg(&self.source_filename)
            .output()?;

        if !r.status.success() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!(
                    "Could not compile code: {}",
                    String::from_utf8_lossy(&r.stderr)
                ),
            ));
        }

        Ok(CompiledCode {
            library_filename: out.to_string(),
            function_name: self.function_name.clone(),
        })
    }
}

impl<T: NumericalFloatLike> EvalTree<T> {
    /// Create a C++ code representation of the evaluation tree.
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
    ) -> Result<ExportedCode, std::io::Error> {
        let cpp = self.export_cpp_str(function_name, include_header);
        std::fs::write(filename, cpp)?;
        Ok(ExportedCode {
            source_filename: filename.to_string(),
            function_name: function_name.to_string(),
        })
    }

    fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = if include_header {
            "#include <iostream>\n#include <cmath>\n#include <complex>\n\n".to_string()
        } else {
            String::new()
        };

        for (name, arg_names, body) in &self.functions {
            let mut args = arg_names
                .iter()
                .map(|x| " T ".to_string() + x.to_string().as_str())
                .collect::<Vec<_>>();
            args.insert(0, "T* params".to_string());

            res += &format!(
                "\ntemplate<typename T>\nT {}({}) {{\n",
                name,
                args.join(",")
            );

            for (i, s) in body.subexpressions.iter().enumerate() {
                res += &format!("\tT Z{}_ = {};\n", i, self.export_cpp_impl(s, arg_names));
            }

            if body.tree.len() > 1 {
                panic!("Tensor functions not supported yet");
            }

            let ret = self.export_cpp_impl(&body.tree[0], arg_names);
            res += &format!("\treturn {};\n}}\n", ret);
        }

        res += &format!("\ntemplate<typename T>\nvoid eval(T* params, T* out) {{\n");

        for (i, s) in self.expressions.subexpressions.iter().enumerate() {
            res += &format!("\tT Z{}_ = {};\n", i, self.export_cpp_impl(s, &[]));
        }

        for (i, e) in self.expressions.tree.iter().enumerate() {
            res += &format!("\tout[{}] = {};\n", i, self.export_cpp_impl(&e, &[]));
        }

        res += "\treturn;\n}\n";

        res += &format!("\nextern \"C\" {{\n\tvoid {}_double(double* params, double* out) {{\n\t\teval(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);
        res += &format!("\nextern \"C\" {{\n\tvoid {}_complex(std::complex<double>* params, std::complex<double>* out) {{\n\t\teval(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);

        res
    }

    fn export_cpp_impl(&self, expr: &Expression<T>, args: &[Symbol]) -> String {
        match expr {
            Expression::Const(c) => {
                format!("T({})", c)
            }
            Expression::Parameter(p) => {
                format!("params[{}]", p)
            }
            Expression::Eval(id, e_args) => {
                let mut r = format!("{}(params", self.functions[*id].0);

                for a in e_args {
                    r.push_str(", ");
                    r += &self.export_cpp_impl(a, args);
                }
                r.push_str(")");
                r
            }
            Expression::Add(a) => {
                let mut r = "(".to_string();
                r += &self.export_cpp_impl(&a[0], args);
                for arg in &a[1..] {
                    r.push_str(" + ");
                    r += &self.export_cpp_impl(arg, args);
                }
                r.push_str(")");
                r
            }
            Expression::Mul(m) => {
                let mut r = "(".to_string();
                r += &self.export_cpp_impl(&m[0], args);
                for arg in &m[1..] {
                    r.push_str(" * ");
                    r += &self.export_cpp_impl(arg, args);
                }
                r.push_str(")");
                r
            }
            Expression::Pow(p) => {
                let mut r = "pow(".to_string();
                r += &self.export_cpp_impl(&p.0, args);
                r.push_str(", ");
                r.push_str(&p.1.to_string());
                r.push(')');
                r
            }
            Expression::Powf(p) => {
                let mut r = "powf(".to_string();
                r += &self.export_cpp_impl(&p.0, args);
                r.push_str(", ");
                r += &self.export_cpp_impl(&p.1, args);
                r.push(')');
                r
            }
            Expression::ReadArg(s) => args[*s].to_string(),
            Expression::BuiltinFun(s, a) => match *s {
                State::EXP => {
                    let mut r = "exp(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                State::LOG => {
                    let mut r = "log(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                State::SIN => {
                    let mut r = "sin(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                State::COS => {
                    let mut r = "cos(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                State::SQRT => {
                    let mut r = "sqrt(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                _ => unreachable!(),
            },
            Expression::SubExpression(id) => {
                format!("Z{}_", id)
            }
        }
    }
}

impl<'a> AtomView<'a> {
    /// Convert nested expressions to a tree.
    pub fn to_eval_tree<
        T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord,
        F: Fn(&Rational) -> T + Copy,
    >(
        &self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTree<T>, String> {
        Self::to_eval_tree_multiple(std::slice::from_ref(self), coeff_map, fn_map, params)
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree_multiple<
        T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord,
        F: Fn(&Rational) -> T + Copy,
    >(
        exprs: &[Self],
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
    ) -> Result<EvalTree<T>, String> {
        let mut funcs = vec![];
        let tree = exprs
            .iter()
            .map(|t| t.to_eval_tree_impl(coeff_map, fn_map, params, &[], &mut funcs))
            .collect::<Result<_, _>>()?;

        Ok(EvalTree {
            expressions: SplitExpression {
                tree,
                subexpressions: vec![],
            },
            functions: funcs,
        })
    }

    fn to_eval_tree_impl<T: Clone + Default, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        fn_map: &FunctionMap<'a, T>,
        params: &[Atom],
        args: &[Symbol],
        funcs: &mut Vec<(String, Vec<Symbol>, SplitExpression<T>)>,
    ) -> Result<Expression<T>, String> {
        if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
            return Ok(Expression::Parameter(p));
        }

        if let Some(c) = fn_map.get(*self) {
            return match c {
                ConstOrExpr::Const(c) => Ok(Expression::Const(c.clone())),
                ConstOrExpr::Expr(name, tag_len, args, v) => {
                    if args.len() != *tag_len {
                        return Err(format!(
                            "Function {} called with wrong number of arguments: 0 vs {}",
                            self,
                            args.len()
                        ));
                    }

                    if let Some(pos) = funcs.iter().position(|f| f.0 == *name) {
                        Ok(Expression::Eval(pos, vec![]))
                    } else {
                        let r = v.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?;
                        funcs.push((
                            name.clone(),
                            args.clone(),
                            SplitExpression {
                                tree: vec![r.clone()],
                                subexpressions: vec![],
                            },
                        ));
                        Ok(Expression::Eval(funcs.len() - 1, vec![]))
                    }
                }
            };
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => Ok(Expression::Const(coeff_map(&(n, d).into()))),
                CoefficientView::Large(l) => Ok(Expression::Const(coeff_map(&l.to_rat()))),
                CoefficientView::Float(f) => {
                    // TODO: converting back to rational is slow
                    Ok(Expression::Const(coeff_map(&f.to_float().to_rational())))
                }
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => {
                let name = v.get_symbol();

                if let Some(p) = args.iter().position(|s| *s == name) {
                    return Ok(Expression::ReadArg(p));
                }

                Err(format!(
                    "Variable {} not in constant map",
                    State::get_name(v.get_symbol())
                ))
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [State::EXP, State::LOG, State::SIN, State::COS, State::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?;

                    return Ok(Expression::BuiltinFun(f.get_symbol(), Box::new(arg_eval)));
                }

                let symb = InlineVar::new(f.get_symbol());
                let Some(fun) = fn_map.get(symb.as_view()) else {
                    return Err(format!(
                        "Undefined function {}",
                        State::get_name(f.get_symbol())
                    ));
                };

                match fun {
                    ConstOrExpr::Const(t) => Ok(Expression::Const(t.clone())),
                    ConstOrExpr::Expr(name, tag_len, arg_spec, e) => {
                        if f.get_nargs() != arg_spec.len() + *tag_len {
                            return Err(format!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol(),
                                f.get_nargs(),
                                arg_spec.len() + *tag_len
                            ));
                        }

                        let eval_args = f
                            .iter()
                            .skip(*tag_len)
                            .map(|arg| {
                                arg.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)
                            })
                            .collect::<Result<_, _>>()?;

                        if let Some(pos) = funcs.iter().position(|f| f.0 == *name) {
                            Ok(Expression::Eval(pos, eval_args))
                        } else {
                            let r =
                                e.to_eval_tree_impl(coeff_map, fn_map, params, arg_spec, funcs)?;
                            funcs.push((
                                name.clone(),
                                arg_spec.clone(),
                                SplitExpression {
                                    tree: vec![r.clone()],
                                    subexpressions: vec![],
                                },
                            ));
                            Ok(Expression::Eval(funcs.len() - 1, eval_args))
                        }
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?;

                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num > 1 {
                                return Ok(Expression::Mul(vec![b_eval.clone(); num as usize]));
                            }
                            return Ok(Expression::Pow(Box::new((b_eval, num))));
                        }
                    }
                }

                let e_eval = e.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?;
                Ok(Expression::Powf(Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?;
                    if let Expression::Mul(m) = a {
                        muls.extend(m);
                    } else {
                        muls.push(a);
                    }
                }

                Ok(Expression::Mul(muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?);
                }

                Ok(Expression::Add(adds))
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
