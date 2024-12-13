use std::{
    hash::{Hash, Hasher},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use ahash::{AHasher, HashMap};
use rand::{thread_rng, Rng};

use self_cell::self_cell;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    atom::{Atom, AtomOrView, AtomView, Symbol},
    coefficient::CoefficientView,
    combinatorics::unique_permutations,
    domains::{
        float::{
            Complex, ErrorPropagatingFloat, NumericalFloatLike, Real, RealNumberLike, SingleFloat,
        },
        integer::Integer,
        rational::Rational,
    },
    id::ConditionResult,
    state::State,
    LicenseManager,
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

    pub fn add_constant<A: Into<AtomOrView<'a>>>(&mut self, key: A, value: T) {
        self.map.insert(
            AtomOrTaggedFunction::Atom(key.into()),
            ConstOrExpr::Const(value),
        );
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
            AtomOrTaggedFunction::TaggedFunction(name, vec![]),
            ConstOrExpr::Expr(rename, 0, args, body),
        );

        Ok(())
    }

    pub fn add_tagged_function(
        &mut self,
        name: Symbol,
        tags: Vec<AtomOrView<'a>>,
        rename: String,
        args: Vec<Symbol>,
        body: AtomView<'a>,
    ) -> Result<(), &str> {
        if let Some(t) = self.tag.insert(name, tags.len()) {
            if t != tags.len() {
                return Err("Cannot add the same function with a different number of parameters");
            }
        }

        let tag_len = tags.len();
        self.map.insert(
            AtomOrTaggedFunction::TaggedFunction(name, tags),
            ConstOrExpr::Expr(rename, tag_len, args, body),
        );

        Ok(())
    }

    fn get_tag_len(&self, symbol: &Symbol) -> usize {
        self.tag.get(symbol).cloned().unwrap_or(0)
    }

    fn get_constant(&self, a: AtomView<'a>) -> Option<&T> {
        match self.map.get(&AtomOrTaggedFunction::Atom(a.into())) {
            Some(ConstOrExpr::Const(c)) => Some(c),
            _ => None,
        }
    }

    fn get(&self, a: AtomView<'a>) -> Option<&ConstOrExpr<'a, T>> {
        if let Some(c) = self.map.get(&AtomOrTaggedFunction::Atom(a.into())) {
            return Some(c);
        }

        if let AtomView::Fun(aa) = a {
            let s = aa.get_symbol();
            let tag_len = self.get_tag_len(&s);

            if aa.get_nargs() >= tag_len {
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

#[derive(Debug, Clone)]
pub struct OptimizationSettings {
    pub horner_iterations: usize,
    pub n_cores: usize,
    pub cpe_iterations: Option<usize>,
    pub hot_start: Option<Vec<Expression<Rational>>>,
    pub verbose: bool,
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        OptimizationSettings {
            horner_iterations: 10,
            n_cores: 1,
            cpe_iterations: None,
            hot_start: None,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SplitExpression<T> {
    pub tree: Vec<Expression<T>>,
    pub subexpressions: Vec<Expression<T>>,
}

#[derive(Debug, Clone)]
pub struct EvalTree<T> {
    functions: Vec<(String, Vec<Symbol>, SplitExpression<T>)>,
    expressions: SplitExpression<T>,
    param_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuiltinSymbol(Symbol);

impl Serialize for BuiltinSymbol {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.get_id().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BuiltinSymbol {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id: u32 = u32::deserialize(deserializer)?;
        Ok(BuiltinSymbol(unsafe { State::symbol_from_id(id) }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Expression<T> {
    Const(T),
    Parameter(usize),
    Eval(usize, Vec<Expression<T>>),
    Add(Vec<Expression<T>>),
    Mul(Vec<Expression<T>>),
    Pow(Box<(Expression<T>, i64)>),
    Powf(Box<(Expression<T>, Expression<T>)>),
    ReadArg(usize), // read nth function argument
    BuiltinFun(BuiltinSymbol, Box<Expression<T>>),
    SubExpression(usize),
}

type ExpressionHash = u64;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum HashedExpression<T> {
    Const(ExpressionHash, T),
    Parameter(ExpressionHash, usize),
    Eval(ExpressionHash, usize, Vec<HashedExpression<T>>),
    Add(ExpressionHash, Vec<HashedExpression<T>>),
    Mul(ExpressionHash, Vec<HashedExpression<T>>),
    Pow(ExpressionHash, Box<(HashedExpression<T>, i64)>),
    Powf(
        ExpressionHash,
        Box<(HashedExpression<T>, HashedExpression<T>)>,
    ),
    ReadArg(ExpressionHash, usize), // read nth function argument
    BuiltinFun(ExpressionHash, BuiltinSymbol, Box<HashedExpression<T>>),
    SubExpression(ExpressionHash, usize),
}

impl<T> HashedExpression<T> {
    fn get_hash(&self) -> ExpressionHash {
        match self {
            HashedExpression::Const(h, _) => *h,
            HashedExpression::Parameter(h, _) => *h,
            HashedExpression::Eval(h, _, _) => *h,
            HashedExpression::Add(h, _) => *h,
            HashedExpression::Mul(h, _) => *h,
            HashedExpression::Pow(h, _) => *h,
            HashedExpression::Powf(h, _) => *h,
            HashedExpression::ReadArg(h, _) => *h,
            HashedExpression::BuiltinFun(h, _, _) => *h,
            HashedExpression::SubExpression(h, _) => *h,
        }
    }
}

impl<T: Clone> HashedExpression<T> {
    fn to_expression(&self) -> Expression<T> {
        match self {
            HashedExpression::Const(_, c) => Expression::Const(c.clone()),
            HashedExpression::Parameter(_, p) => Expression::Parameter(*p),
            HashedExpression::Eval(_, i, v) => {
                Expression::Eval(*i, v.into_iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Add(_, a) => {
                Expression::Add(a.into_iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Mul(_, a) => {
                Expression::Mul(a.into_iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Pow(_, p) => Expression::Pow(Box::new((p.0.to_expression(), p.1))),
            HashedExpression::Powf(_, p) => {
                Expression::Powf(Box::new((p.0.to_expression(), p.1.to_expression())))
            }
            HashedExpression::ReadArg(_, r) => Expression::ReadArg(*r),
            HashedExpression::BuiltinFun(_, s, a) => {
                Expression::BuiltinFun(*s, Box::new(a.to_expression()))
            }
            HashedExpression::SubExpression(_, s) => Expression::SubExpression(*s),
        }
    }
}

impl<T: Ord> PartialOrd for HashedExpression<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for HashedExpression<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (HashedExpression::Const(_, a), HashedExpression::Const(_, b)) => a.cmp(b),
            (HashedExpression::Parameter(_, a), HashedExpression::Parameter(_, b)) => a.cmp(b),
            (HashedExpression::Eval(_, a, b), HashedExpression::Eval(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (HashedExpression::Add(_, a), HashedExpression::Add(_, b)) => a.cmp(b),
            (HashedExpression::Mul(_, a), HashedExpression::Mul(_, b)) => a.cmp(b),
            (HashedExpression::Pow(_, p1), HashedExpression::Pow(_, p2)) => p1.cmp(p2),
            (HashedExpression::Powf(_, p1), HashedExpression::Powf(_, p2)) => p1.cmp(p2),
            (HashedExpression::ReadArg(_, r1), HashedExpression::ReadArg(_, r2)) => r1.cmp(r2),
            (HashedExpression::BuiltinFun(_, a, b), HashedExpression::BuiltinFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (HashedExpression::SubExpression(_, s1), HashedExpression::SubExpression(_, s2)) => {
                s1.cmp(s2)
            }
            (HashedExpression::Const(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Const(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Parameter(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Parameter(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Eval(_, _, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Eval(_, _, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Add(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Add(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Mul(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Mul(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Pow(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Pow(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Powf(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Powf(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::ReadArg(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::ReadArg(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::BuiltinFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::BuiltinFun(_, _, _)) => std::cmp::Ordering::Greater,
        }
    }
}

impl<T: Eq + Hash> Hash for HashedExpression<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_hash())
    }
}

impl<T: Eq + Hash + Clone + Ord> HashedExpression<T> {
    fn find_subexpression<'a>(
        &'a self,
        subexp: &mut HashMap<&'a HashedExpression<T>, usize>,
    ) -> bool {
        if matches!(
            self,
            HashedExpression::Const(_, _)
                | HashedExpression::Parameter(_, _)
                | HashedExpression::ReadArg(_, _)
        ) {
            return true;
        }

        if let Some(i) = subexp.get_mut(self) {
            *i += 1;
            return true;
        }

        subexp.insert(self, 1);

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp);
                }
            }
            HashedExpression::Add(_, a) | HashedExpression::Mul(_, a) => {
                for arg in a {
                    arg.find_subexpression(subexp);
                }
            }
            HashedExpression::Pow(_, p) => {
                p.0.find_subexpression(subexp);
            }
            HashedExpression::Powf(_, p) => {
                p.0.find_subexpression(subexp);
                p.1.find_subexpression(subexp);
            }
            HashedExpression::BuiltinFun(_, _, _) => {}
            HashedExpression::SubExpression(_, _) => {}
        }

        false
    }

    fn replace_subexpression<'a>(
        &mut self,
        subexp: &HashMap<&'a HashedExpression<T>, usize>,
        skip_root: bool,
    ) {
        if !skip_root {
            if let Some(i) = subexp.get(self) {
                *self = HashedExpression::SubExpression(self.get_hash(), *i); // TODO: do not recyle hash?
                return;
            }
        }

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, false);
                }
            }
            HashedExpression::Add(_, a) | HashedExpression::Mul(_, a) => {
                for arg in a {
                    arg.replace_subexpression(subexp, false);
                }
            }
            HashedExpression::Pow(_, p) => {
                p.0.replace_subexpression(subexp, false);
            }
            HashedExpression::Powf(_, p) => {
                p.0.replace_subexpression(subexp, false);
                p.1.replace_subexpression(subexp, false);
            }
            HashedExpression::BuiltinFun(_, _, _) => {}
            HashedExpression::SubExpression(_, _) => {}
        }
    }

    // Count the number of additions and multiplications in the expression, counting
    // subexpressions only once.
    pub fn count_operations_with_subexpression<'a>(
        &'a self,
        sub_expr: &mut HashMap<&'a Self, usize>,
    ) -> (usize, usize) {
        if matches!(
            self,
            HashedExpression::Const(_, _)
                | HashedExpression::Parameter(_, _)
                | HashedExpression::ReadArg(_, _)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            //println!("SUB {:?}", self);
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            HashedExpression::Const(_, _) => (0, 0),
            HashedExpression::Parameter(_, _) => (0, 0),
            HashedExpression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            HashedExpression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            HashedExpression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            HashedExpression::Pow(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            HashedExpression::Powf(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            HashedExpression::ReadArg(_, _) => (0, 0),
            HashedExpression::BuiltinFun(_, _, b) => {
                b.count_operations_with_subexpression(sub_expr)
            } // not clear how to count this, third arg?
            HashedExpression::SubExpression(_, _) => (0, 0),
        }
    }
}

impl<T: std::hash::Hash + Clone> Expression<T> {
    fn to_hashed_expression(&self) -> (ExpressionHash, HashedExpression<T>) {
        match self {
            Expression::Const(c) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(0);
                c.hash(&mut hasher);
                let h = hasher.finish();
                (h, HashedExpression::Const(h, c.clone()))
            }
            Expression::Parameter(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(1);
                hasher.write_usize(*p);
                let h = hasher.finish();
                (h, HashedExpression::Parameter(h, *p))
            }
            Expression::Eval(i, v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(2);
                hasher.write_usize(*i);
                let mut new_v = vec![];
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    hasher.write_u64(h);
                }
                let h = hasher.finish();
                (h, HashedExpression::Eval(h, *i, new_v))
            }
            Expression::Add(v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(3);
                let mut new_v = vec![];

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    arg_sum = arg_sum.wrapping_add(h);
                }
                hasher.write_u64(arg_sum);
                let h = hasher.finish();
                (h, HashedExpression::Add(h, new_v))
            }
            Expression::Mul(v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(4);
                let mut new_v = vec![];

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    arg_sum = arg_sum.wrapping_add(h);
                }
                hasher.write_u64(arg_sum);
                let h = hasher.finish();
                (h, HashedExpression::Mul(h, new_v))
            }
            Expression::Pow(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(5);
                let (hb, vb) = p.0.to_hashed_expression();
                hasher.write_u64(hb);
                hasher.write_i64(p.1);
                let h = hasher.finish();
                (h, HashedExpression::Pow(h, Box::new((vb, p.1))))
            }
            Expression::Powf(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(6);
                let (hb, vb) = p.0.to_hashed_expression();
                let (he, ve) = p.1.to_hashed_expression();
                hasher.write_u64(hb);
                hasher.write_u64(he);
                let h = hasher.finish();
                (h, HashedExpression::Powf(h, Box::new((vb, ve))))
            }
            Expression::ReadArg(i) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(7);
                hasher.write_usize(*i);
                let h = hasher.finish();
                (h, HashedExpression::ReadArg(h, *i))
            }
            Expression::BuiltinFun(s, a) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(8);
                s.hash(&mut hasher);
                let (ha, va) = a.to_hashed_expression();
                hasher.write_u64(ha);
                let h = hasher.finish();
                (h, HashedExpression::BuiltinFun(h, *s, Box::new(va)))
            }
            Expression::SubExpression(i) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(9);
                hasher.write_usize(*i);
                let h = hasher.finish();
                (h, HashedExpression::SubExpression(h, *i))
            }
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    param_count: usize,
    reserved_indices: usize,
    instructions: Vec<Instr>,
    result_indices: Vec<usize>,
}

impl<T: Real> ExpressionEvaluator<T> {
    pub fn evaluate_single(&mut self, params: &[T]) -> T {
        let mut res = T::new_zero();
        self.evaluate(params, std::slice::from_mut(&mut res));
        res
    }

    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
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
                Instr::BuiltinFun(r, s, arg) => match s.0 {
                    Atom::EXP => self.stack[*r] = self.stack[*arg].exp(),
                    Atom::LOG => self.stack[*r] = self.stack[*arg].log(),
                    Atom::SIN => self.stack[*r] = self.stack[*arg].sin(),
                    Atom::COS => self.stack[*r] = self.stack[*arg].cos(),
                    Atom::SQRT => self.stack[*r] = self.stack[*arg].sqrt(),
                    _ => unreachable!(),
                },
            }
        }

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            *o = self.stack[*i].clone();
        }
    }
}

impl<T: Default> ExpressionEvaluator<T> {
    /// Map the coefficients to a different type.
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(self, f: &F) -> ExpressionEvaluator<T2> {
        ExpressionEvaluator {
            stack: self.stack.iter().map(f).collect(),
            param_count: self.param_count,
            reserved_indices: self.reserved_indices,
            instructions: self.instructions,
            result_indices: self.result_indices,
        }
    }

    pub fn get_input_len(&self) -> usize {
        self.param_count
    }

    pub fn get_output_len(&self) -> usize {
        self.result_indices.len()
    }

    fn remove_common_pairs(&mut self) -> usize {
        let mut pairs: HashMap<_, Vec<usize>> = HashMap::default();

        let mut affected_lines = vec![true; self.instructions.len()];

        for (p, i) in self.instructions.iter().enumerate() {
            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            pairs.entry((is_add, *l, *r)).or_default().push(p);
                        }
                    }
                }
                _ => {}
            }
        }

        // for now, ignore pairs with only occurrences on the same line
        let mut to_remove: Vec<_> = pairs.clone().into_iter().collect();

        to_remove.retain_mut(|(_, v)| {
            v.dedup();
            v.len() > 1
        });

        // sort in other direction since we pop
        to_remove.sort_by_key(|x| x.1.len());

        let total_remove = to_remove.len();

        for x in &mut affected_lines {
            *x = false;
        }

        let old_len = self.instructions.len();

        while let Some(((is_add, l, r), lines)) = to_remove.pop() {
            if lines.iter().any(|x| affected_lines[*x]) {
                continue;
            }

            let new_idx = self.stack.len();
            let new_op = if is_add {
                Instr::Add(new_idx, vec![l, r])
            } else {
                Instr::Mul(new_idx, vec![l, r])
            };

            self.stack.push(T::default());
            self.instructions.push(new_op);

            for line in lines {
                affected_lines[line] = true;
                let is_add = matches!(self.instructions[line], Instr::Add(_, _));

                if let Instr::Add(_, a) | Instr::Mul(_, a) = &mut self.instructions[line] {
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            let pp = pairs.entry((is_add, *l, *r)).or_default();
                            pp.retain(|x| *x != line);
                        }
                    }

                    if l == r {
                        let count = a.iter().filter(|x| **x == l).count();
                        let pairs = count / 2;
                        if pairs > 0 {
                            a.retain(|x| *x != l);

                            if count % 2 == 1 {
                                a.push(l.clone());
                            }

                            a.extend(std::iter::repeat(new_idx).take(pairs));
                            a.sort();
                        }
                    } else {
                        let mut idx1_count = 0;
                        let mut idx2_count = 0;
                        for v in &*a {
                            if *v == l {
                                idx1_count += 1;
                            }
                            if *v == r {
                                idx2_count += 1;
                            }
                        }

                        let pair_count = idx1_count.min(idx2_count);

                        if pair_count > 0 {
                            a.retain(|x| *x != l && *x != r);

                            // add back removed indices in cases such as idx1*idx2*idx2
                            if idx1_count > pair_count {
                                a.extend(
                                    std::iter::repeat(l.clone()).take(idx1_count - pair_count),
                                );
                            }
                            if idx2_count > pair_count {
                                a.extend(
                                    std::iter::repeat(r.clone()).take(idx2_count - pair_count),
                                );
                            }

                            a.extend(std::iter::repeat(new_idx).take(pair_count));
                            a.sort();
                        }
                    }

                    // update the pairs for this line
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            pairs.entry((is_add, *l, *r)).or_default().push(line);
                        }
                    }
                }
            }
        }

        let mut first_use = vec![];
        for i in self.instructions.drain(old_len..) {
            if let Instr::Add(_, a) | Instr::Mul(_, a) = &i {
                let mut last_dep = a[0];
                for v in a {
                    last_dep = last_dep.max(*v);
                }

                let ins = if last_dep + 1 <= self.reserved_indices {
                    0
                } else {
                    last_dep + 1 - self.reserved_indices
                };

                first_use.push((ins, i));
            } else {
                unreachable!()
            }
        }

        first_use.sort_by_key(|x| x.0);

        let mut new_instr = vec![];
        let mut i = 0;
        let mut j = 0;

        let mut sub_rename = HashMap::default();
        let mut rename_map: Vec<_> = (0..self.reserved_indices).collect();

        macro_rules! rename {
            ($i:expr) => {
                if $i >= self.reserved_indices + self.instructions.len() {
                    sub_rename[&$i]
                } else {
                    rename_map[$i]
                }
            };
        }

        while i < self.instructions.len() {
            let new_pos = new_instr.len() + self.reserved_indices;

            if j < first_use.len() && i == first_use[j].0 {
                let (o, a) = match &first_use[j].1 {
                    Instr::Add(o, a) => (*o, a),
                    Instr::Mul(o, a) => (*o, a),
                    _ => unreachable!(),
                };

                let is_add = matches!(&first_use[j].1, Instr::Add(_, _));

                let new_a = a.iter().map(|x| rename!(*x)).collect::<Vec<_>>();

                if is_add {
                    new_instr.push(Instr::Add(new_pos, new_a));
                } else {
                    new_instr.push(Instr::Mul(new_pos, new_a));
                }

                sub_rename.insert(o, new_pos);

                j += 1;
            } else {
                let mut s = self.instructions[i].clone();

                match &mut s {
                    Instr::Add(p, a) | Instr::Mul(p, a) => {
                        for x in &mut *a {
                            *x = rename!(*x);
                        }

                        // remove assignments
                        if a.len() == 1 {
                            rename_map.push(a[0]);
                            i += 1;
                            continue;
                        }

                        *p = new_pos;
                    }
                    Instr::Pow(p, b, _) | Instr::BuiltinFun(p, _, b) => {
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                    Instr::Powf(p, a, b) => {
                        *a = rename!(*a);
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                }

                new_instr.push(s);
                rename_map.push(new_pos);
                i += 1;
            }
        }

        for x in &mut self.result_indices {
            *x = rename!(*x);
        }

        assert!(j == first_use.len());

        self.instructions = new_instr;
        total_remove
    }
}

impl<T: Default + Clone + Eq + Hash> ExpressionEvaluator<T> {
    /// Merge evaluator `other` into `self`. The parameters must be the same.
    pub fn merge(&mut self, mut other: Self, cpe_rounds: Option<usize>) -> Result<(), String> {
        if self.param_count != other.param_count {
            return Err("Parameter count is different".to_owned());
        }

        let mut constants = HashMap::default();

        for (i, c) in self.stack[self.param_count..self.reserved_indices]
            .iter()
            .enumerate()
        {
            constants.insert(c.clone(), i);
        }

        let old_len = self.stack.len() - self.reserved_indices;

        self.stack.truncate(self.reserved_indices);

        for c in &other.stack[self.param_count..other.reserved_indices] {
            if constants.get(c).is_none() {
                let i = constants.len();
                constants.insert(c.clone(), i);
                self.stack.push(c.clone());
            }
        }

        let new_reserved_indices = self.stack.len();
        let mut delta = new_reserved_indices - self.reserved_indices;

        // shift stack indices
        if delta > 0 {
            for i in &mut self.instructions {
                match i {
                    Instr::Add(r, a) | Instr::Mul(r, a) => {
                        *r += delta;
                        for aa in a {
                            if *aa >= self.reserved_indices {
                                *aa += delta;
                            }
                        }
                    }
                    Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                    }
                    Instr::Powf(r, b, e) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                        if *e >= self.reserved_indices {
                            *e += delta;
                        }
                    }
                }
            }

            for x in &mut self.result_indices {
                *x += delta;
            }
        }

        delta = old_len + new_reserved_indices - other.reserved_indices;
        for i in &mut other.instructions {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) => {
                    *r += delta;
                    for aa in a {
                        if *aa >= other.reserved_indices {
                            *aa += delta;
                        } else if *aa >= other.param_count {
                            *aa = self.param_count + constants[&other.stack[*aa]];
                        }
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                }
                Instr::Powf(r, b, e) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                    if *e >= other.reserved_indices {
                        *e += delta;
                    } else if *e >= other.param_count {
                        *e = self.param_count + constants[&other.stack[*e]];
                    }
                }
            }
        }

        for x in &mut other.result_indices {
            if *x >= other.reserved_indices {
                *x += delta;
            } else if *x >= other.param_count {
                *x = self.param_count + constants[&other.stack[*x]];
            }
        }

        self.instructions.extend(other.instructions.drain(..));
        self.result_indices.extend(other.result_indices.drain(..));
        self.reserved_indices = new_reserved_indices;

        // undo the stack optimization
        let mut unfold = HashMap::default();
        for (index, i) in &mut self.instructions.iter_mut().enumerate() {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) => {
                    for aa in a {
                        if *aa >= self.reserved_indices {
                            *aa = unfold[aa];
                        }
                    }

                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Powf(r, b, e) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    if *e >= self.reserved_indices {
                        *e = unfold[e];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
            }
        }

        for i in &mut self.result_indices {
            if *i >= self.reserved_indices {
                *i = unfold[i];
            }
        }

        for _ in 0..self.instructions.len() {
            self.stack.push(T::default());
        }

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if self.remove_common_pairs() == 0 {
                break;
            }
        }

        self.optimize_stack();

        Ok(())
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
    /// With `inline_asm` set to any value other than `None`,
    /// high-performance inline ASM code will be generated for most
    /// evaluation instructions. This often gives better performance than
    /// the `O3` optimization level and results in very fast compilation.
    pub fn export_cpp(
        &self,
        filename: &str,
        function_name: &str,
        include_header: bool,
        inline_asm: InlineASM,
    ) -> Result<ExportedCode, std::io::Error> {
        let mut filename = filename.to_string();
        if !filename.ends_with(".cpp") {
            filename += ".cpp";
        }

        let cpp = match inline_asm {
            InlineASM::X64 => self.export_asm_str(function_name, include_header),
            InlineASM::None => self.export_cpp_str(function_name, include_header),
        };

        let _ = std::fs::write(&filename, cpp)?;
        Ok(ExportedCode {
            source_filename: filename,
            function_name: function_name.to_string(),
        })
    }

    pub fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = String::new();
        if include_header {
            res += &"#include <iostream>\n#include <complex>\n#include <cmath>\n\n";
            res += &"extern \"C\" void drop_buffer_complex(std::complex<double> *buffer)\n{\n\tdelete[] buffer;\n}\n\n";
            res += &"extern \"C\" void drop_buffer_double(double *buffer)\n{\n\tdelete[] buffer;\n}\n\n";
        };

        res += &format!(
            "extern \"C\" std::complex<double> *{}_create_buffer_complex()\n{{\n\treturn new std::complex<double>[{}];\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "extern \"C\" double *{}_create_buffer_double()\n{{\n\treturn new double[{}];\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "\ntemplate<typename T>\nvoid {}(T* params, T* Z, T* out) {{\n",
            function_name
        );

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

        res += &format!("\nextern \"C\" {{\n\tvoid {0}_double(double *params, double *buffer, double *out) {{\n\t\t{0}(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n", function_name);
        res += &format!("\nextern \"C\" {{\n\tvoid {0}_complex(std::complex<double> *params, std::complex<double> *buffer,  std::complex<double> *out) {{\n\t\t{0}(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n", function_name);

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
                Instr::BuiltinFun(o, s, a) => match s.0 {
                    Atom::EXP => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = exp({});\n", o, arg).as_str();
                    }
                    Atom::LOG => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = log({});\n", o, arg).as_str();
                    }
                    Atom::SIN => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = sin({});\n", o, arg).as_str();
                    }
                    Atom::COS => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = cos({});\n", o, arg).as_str();
                    }
                    Atom::SQRT => {
                        let arg = format!("Z{}", a);
                        *out += format!("\tZ{} = sqrt({});\n", o, arg).as_str();
                    }
                    _ => unreachable!(),
                },
            }
        }
    }

    pub fn export_asm_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = String::new();
        if include_header {
            res += &"#include <iostream>\n#include <complex>\n#include <cmath>\n\n";
            res += &"extern \"C\" void drop_buffer_complex(std::complex<double> *buffer)\n{\n\tdelete[] buffer;\n}\n\n";
            res += &"extern \"C\" void drop_buffer_double(double *buffer)\n{\n\tdelete[] buffer;\n}\n\n";
        };

        res += &format!(
            "extern \"C\" std::complex<double> *{}_create_buffer_complex()\n{{\n\treturn new std::complex<double>[{}];\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "static const std::complex<double> {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
            function_name,
            self.reserved_indices - self.param_count + 1,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("std::complex<double>({})", self.stack[i]))
                    .collect::<Vec<_>>();
                nums.push("std::complex<double>(0, -0.)".to_string()); // used for inversion
                nums.join(",")
            }
        );

        res += &format!("extern \"C\" void {}_complex(const std::complex<double> *params, std::complex<double> *Z, std::complex<double> *out)\n{{\n", function_name);

        self.export_asm_complex_impl(&self.instructions, function_name, &mut res);

        res += "\treturn;\n}\n\n";

        res += &format!(
            "extern \"C\" double *{}_create_buffer_double()\n{{\n\treturn new double[{}];\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "static const double {}_CONSTANTS_double[{}] = {{{}}};\n\n",
            function_name,
            self.reserved_indices - self.param_count + 1,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("double({})", self.stack[i]))
                    .collect::<Vec<_>>();
                nums.push("1".to_string()); // used for inversion
                nums.join(",")
            }
        );

        res += &format!(
            "extern \"C\" void {}_double(const double *params, double* Z, double *out)\n{{\n",
            function_name
        );

        self.export_asm_double_impl(&self.instructions, function_name, &mut res);

        res += "\treturn;\n}\n";

        res
    }

    fn export_asm_double_impl(
        &self,
        instr: &[Instr],
        function_name: &str,
        out: &mut String,
    ) -> bool {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_double[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! format_addr {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("{}(%2)", $i * 8)
                } else if $i < self.reserved_indices {
                    format!("{}(%1)", ($i - self.param_count) * 8)
                } else {
                    // TODO: subtract reserved indices
                    format!("{}(%0)", $i * 8)
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                    $in_block = false;
                }
            };
        }

        let mut reg_last_use = vec![self.instructions.len(); self.instructions.len()];
        let mut stack_to_reg = HashMap::default();

        for (i, ins) in instr.iter().enumerate() {
            match ins {
                Instr::Add(r, a) | Instr::Mul(r, a) => {
                    for x in a {
                        if x >= &self.reserved_indices {
                            reg_last_use[stack_to_reg[x]] = i;
                        }
                    }

                    stack_to_reg.insert(r, i);
                }
                Instr::Pow(r, b, _) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::Powf(r, b, e) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    if e >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[e]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::BuiltinFun(r, _, b) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum MemOrReg {
            Mem(usize),
            Reg(usize),
        }

        #[derive(Debug, Clone)]
        enum RegInstr {
            Add(MemOrReg, u16, Vec<MemOrReg>),
            Mul(MemOrReg, u16, Vec<MemOrReg>),
            Pow(MemOrReg, u16, MemOrReg, i64),
            Powf(usize, usize, usize),
            BuiltinFun(usize, BuiltinSymbol, usize),
        }

        let mut new_instr: Vec<RegInstr> = instr
            .iter()
            .map(|i| match i {
                Instr::Add(r, a) => RegInstr::Add(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Mul(r, a) => RegInstr::Mul(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Pow(r, b, e) => {
                    RegInstr::Pow(MemOrReg::Mem(*r), u16::MAX, MemOrReg::Mem(*b), *e)
                }
                Instr::Powf(r, b, e) => RegInstr::Powf(*r, *b, *e),
                Instr::BuiltinFun(r, s, a) => RegInstr::BuiltinFun(*r, *s, *a),
            })
            .collect();

        // sort the list of instructions based on the distance
        let mut reg_list = reg_last_use.iter().enumerate().collect::<Vec<_>>();
        reg_list.sort_by_key(|x| (*x.1 - x.0, x.0));

        'next: for (j, last_use) in reg_list {
            if *last_use == self.instructions.len() {
                continue;
            }

            let old_reg = if let RegInstr::Add(r, _, _)
            | RegInstr::Mul(r, _, _)
            | RegInstr::Pow(r, _, _, -1) = &new_instr[j]
            {
                if let MemOrReg::Mem(r) = r {
                    *r
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // find free registers in the range
            // start at j+1 as we can recycle registers that are last used in iteration j
            let mut free_regs = u16::MAX & !(1 << 15); // leave xmmm15 open

            for k in &new_instr[j + 1..=*last_use] {
                match k {
                    RegInstr::Add(_, f, _)
                    | RegInstr::Mul(_, f, _)
                    | RegInstr::Pow(_, f, _, -1) => {
                        free_regs &= f;
                    }

                    _ => {
                        free_regs = 0; // the current instruction is not allowed to be used outside of ASM blocks
                    }
                }

                if free_regs == 0 {
                    continue 'next;
                }
            }

            if let Some(k) = (0..16).position(|k| free_regs & (1 << k) != 0) {
                if let RegInstr::Add(r, _, _) | RegInstr::Mul(r, _, _) | RegInstr::Pow(r, _, _, _) =
                    &mut new_instr[j]
                {
                    *r = MemOrReg::Reg(k);
                }

                for l in &mut new_instr[j + 1..=*last_use] {
                    match l {
                        RegInstr::Add(_, f, a) | RegInstr::Mul(_, f, a) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            for x in a {
                                if *x == MemOrReg::Mem(old_reg) {
                                    *x = MemOrReg::Reg(k);
                                }
                            }
                        }
                        RegInstr::Pow(_, f, a, -1) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            if *a == MemOrReg::Mem(old_reg) {
                                *a = MemOrReg::Reg(k);
                            }
                        }
                        RegInstr::Pow(_, _, _, _) => {
                            panic!("use outside of ASM block");
                        }
                        RegInstr::Powf(_, a, b) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                            if *b == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::BuiltinFun(_, _, a) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                    }
                }

                // TODO: if last use is not already set to a register, we can set it to the current one
                // this prevents a copy
            }
        }

        let mut in_asm_block = false;
        for ins in &new_instr {
            match ins {
                RegInstr::Add(o, free, a) | RegInstr::Mul(o, free, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let oper = if matches!(ins, RegInstr::Add(_, _, _)) {
                        "add"
                    } else {
                        "mul"
                    };

                    match o {
                        MemOrReg::Reg(out_reg) => {
                            if let Some(j) = a.iter().find(|x| **x == MemOrReg::Reg(*out_reg)) {
                                // we can recycle the register completely
                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || i != j {
                                        match i {
                                            MemOrReg::Reg(k) => {
                                                *out += &format!(
                                                    "\t\t\"{}sd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                                    oper, k, out_reg,
                                                );
                                            }
                                            MemOrReg::Mem(k) => {
                                                *out += &format!(
                                                    "\t\t\"{}sd {}, %%xmm{}\\n\\t\"\n",
                                                    oper,
                                                    format_addr!(*k),
                                                    out_reg
                                                );
                                            }
                                        }
                                    }
                                    first_skipped |= i == j;
                                }
                            } else if let Some(MemOrReg::Reg(j)) =
                                a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                            {
                                *out +=
                                    &format!("\t\t\"movapd %%xmm{}, %%xmm{}\\n\\t\"\n", j, out_reg);

                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || *i != MemOrReg::Reg(*j) {
                                        match i {
                                            MemOrReg::Reg(k) => {
                                                *out += &format!(
                                                    "\t\t\"{}sd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                                    oper, k, out_reg,
                                                );
                                            }
                                            MemOrReg::Mem(k) => {
                                                *out += &format!(
                                                    "\t\t\"{}sd {}, %%xmm{}\\n\\t\"\n",
                                                    oper,
                                                    format_addr!(*k),
                                                    out_reg
                                                );
                                            }
                                        }
                                    }
                                    first_skipped |= *i == MemOrReg::Reg(*j);
                                }
                            } else {
                                if let MemOrReg::Mem(k) = &a[0] {
                                    *out += &format!(
                                        "\t\t\"movsd {}, %%xmm{}\\n\\t\"\n",
                                        format_addr!(*k),
                                        out_reg
                                    );
                                } else {
                                    unreachable!();
                                }

                                for i in &a[1..] {
                                    if let MemOrReg::Mem(k) = i {
                                        *out += &format!(
                                            "\t\t\"{}sd {}, %%xmm{}\\n\\t\"\n",
                                            oper,
                                            format_addr!(*k),
                                            out_reg
                                        );
                                    }
                                }
                            }
                        }
                        MemOrReg::Mem(out_mem) => {
                            // TODO: we would like a last-use check of the free here. Now we need to move
                            if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                if let Some(MemOrReg::Reg(j)) =
                                    a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                                {
                                    *out += &format!(
                                        "\t\t\"movapd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                        j, out_reg
                                    );

                                    let mut first_skipped = false;
                                    for i in a {
                                        if first_skipped || *i != MemOrReg::Reg(*j) {
                                            match i {
                                                MemOrReg::Reg(k) => {
                                                    *out += &format!(
                                                        "\t\t\"{}sd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                                        oper, k, out_reg
                                                    );
                                                }
                                                MemOrReg::Mem(k) => {
                                                    *out += &format!(
                                                        "\t\t\"{}sd {}, %%xmm{}\\n\\t\"\n",
                                                        oper,
                                                        format_addr!(*k),
                                                        out_reg
                                                    );
                                                }
                                            }
                                        }

                                        first_skipped |= *i == MemOrReg::Reg(*j);
                                    }
                                } else {
                                    if let MemOrReg::Mem(k) = &a[0] {
                                        *out += &format!(
                                            "\t\t\"movsd {}, %%xmm{}\\n\\t\"\n",
                                            format_addr!(*k),
                                            out_reg
                                        );
                                    } else {
                                        unreachable!();
                                    }

                                    for i in &a[1..] {
                                        if let MemOrReg::Mem(k) = i {
                                            *out += &format!(
                                                "\t\t\"{}sd {}, %%xmm{}\\n\\t\"\n",
                                                oper,
                                                format_addr!(*k),
                                                out_reg
                                            );
                                        }
                                    }
                                }

                                *out += &format!(
                                    "\t\t\"movsd %%xmm{}, {}\\n\\t\"\n",
                                    out_reg,
                                    format_addr!(*out_mem)
                                );
                            } else {
                                unreachable!("No free registers");
                                // move the value of xmm0 into the memory location of the output register
                                // and then swap later?
                            }
                        }
                    }
                }
                RegInstr::Pow(o, free, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        match o {
                            MemOrReg::Reg(out_reg) => {
                                if *b == MemOrReg::Reg(*out_reg) {
                                    if let Some(tmp_reg) =
                                        (0..16).position(|k| free & (1 << k) != 0)
                                    {
                                        *out += &format!(
                                            "\t\t\"movapd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                            out_reg, tmp_reg
                                        );

                                        *out += &format!(
                                            "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                            (self.reserved_indices - self.param_count) * 8,
                                            out_reg
                                        );

                                        *out += &format!(
                                            "\t\t\"divsd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                            tmp_reg, out_reg
                                        );
                                    } else {
                                        panic!("No free registers for division")
                                    }
                                } else {
                                    *out += &format!(
                                        "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                        (self.reserved_indices - self.param_count) * 8,
                                        out_reg,
                                    );

                                    match b {
                                        MemOrReg::Reg(j) => {
                                            *out += &format!(
                                                "\t\t\"divsd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                                j, out_reg
                                            );
                                        }
                                        MemOrReg::Mem(k) => {
                                            *out += &format!(
                                                "\t\t\"divsd {}, %%xmm{}\\n\\t\"\n",
                                                format_addr!(*k),
                                                out_reg,
                                            );
                                        }
                                    }
                                }
                            }
                            MemOrReg::Mem(out_mem) => {
                                if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                    *out += &format!(
                                        "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                        (self.reserved_indices - self.param_count) * 8,
                                        out_reg
                                    );

                                    match b {
                                        MemOrReg::Reg(j) => {
                                            *out += &format!(
                                                "\t\t\"divsd %%xmm{}, %%xmm{}\\n\\t\"\n",
                                                j, out_reg
                                            );
                                        }
                                        MemOrReg::Mem(k) => {
                                            *out += &format!(
                                                "\t\t\"divsd {}, %%xmm{}\\n\\t\"\n",
                                                format_addr!(*k),
                                                out_reg
                                            );
                                        }
                                    }

                                    *out += &format!(
                                        "\t\t\"movsd %%xmm{}, {}\\n\\t\"\n",
                                        out_reg,
                                        format_addr!(*out_mem)
                                    );
                                } else {
                                    unreachable!("No free registers");
                                    // move the value of xmm0 into the memory location of the output register
                                    // and then swap later?
                                }
                            }
                        }
                    } else {
                        unreachable!(
                            "Powers other than -1 should have been removed at an earlier stage"
                        );
                    }
                }
                RegInstr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);

                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\tZ[{}] = pow({}, {});\n", o, base, exp).as_str();
                }
                RegInstr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.0 {
                        Atom::EXP => {
                            *out += format!("\tZ[{}] = exp({});\n", o, arg).as_str();
                        }
                        Atom::LOG => {
                            *out += format!("\tZ[{}] = log({});\n", o, arg).as_str();
                        }
                        Atom::SIN => {
                            *out += format!("\tZ[{}] = sin({});\n", o, arg).as_str();
                        }
                        Atom::COS => {
                            *out += format!("\tZ[{}] = cos({});\n", o, arg).as_str();
                        }
                        Atom::SQRT => {
                            *out += format!("\tZ[{}] = sqrt({});\n", o, arg).as_str();
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        end_asm_block!(in_asm_block);

        let mut regcount = 0;
        *out += "\t__asm__(\n";
        for (i, r) in self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                *out += &format!("\t\t\"movsd {}(%3), %%xmm{}\\n\\t\"\n", r * 8, regcount);
            } else if *r < self.reserved_indices {
                *out += &format!(
                    "\t\t\"movsd {}(%2), %%xmm{}\\n\\t\"\n",
                    (r - self.param_count) * 8,
                    regcount
                );
            } else {
                *out += &format!("\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n", r * 8, regcount);
            }

            *out += &format!("\t\t\"movsd %%xmm{}, {}(%0)\\n\\t\"\n", regcount, i * 8);
            regcount = (regcount + 1) % 16;
        }

        *out += &format!("\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\");\n",  function_name);
        in_asm_block
    }

    fn export_asm_complex_impl(
        &self,
        instr: &[Instr],
        function_name: &str,
        out: &mut String,
    ) -> bool {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_complex[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! format_addr {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("{}(%2)", $i * 16)
                } else if $i < self.reserved_indices {
                    format!("{}(%1)", ($i - self.param_count) * 16)
                } else {
                    // TODO: subtract reserved indices
                    format!("{}(%0)", $i * 16)
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                    $in_block = false;
                }
            };
        }

        let mut in_asm_block = false;
        for ins in instr {
            match ins {
                Instr::Add(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    *out += &format!("\t\t\"xorpd %%xmm0, %%xmm0\\n\\t\"\n");

                    // TODO: try loading in multiple registers for better instruction-level parallelism?
                    for i in a {
                        *out += &format!("\t\t\"addpd {}, %%xmm0\\n\\t\"\n", format_addr!(*i));
                    }
                    *out += &format!("\t\t\"movupd %%xmm0, {}\\n\\t\"\n", format_addr!(*o));
                }
                Instr::Mul(o, a) => {
                    if a.len() < 15 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        // optimized complex multiplication
                        for (i, r) in a.iter().enumerate() {
                            *out += &format!(
                                "\t\t\"movupd {}, %%xmm{}\\n\\t\"\n",
                                format_addr!(*r),
                                i + 1,
                            );
                        }

                        for i in 1..a.len() {
                            *out += &format!(
                                "\t\t\"movapd %%xmm1, %%xmm0\\n\\t\"
\t\t\"unpckhpd %%xmm0, %%xmm0\\n\\t\"
\t\t\"unpcklpd %%xmm1, %%xmm1\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"
\t\t\"shufpd $1, %%xmm0, %%xmm0\\n\\t\"
\t\t\"addsubpd %%xmm0, %%xmm1\\n\\t\"\n",
                                i + 1
                            );
                        }

                        *out += &format!("\t\t\"movupd %%xmm1, {}\\n\\t\"\n", format_addr!(*o));
                    } else {
                        // TODO: reuse registers

                        end_asm_block!(in_asm_block);

                        let args = a
                            .iter()
                            .map(|x| get_input!(*x))
                            .collect::<Vec<_>>()
                            .join("*");

                        *out += format!("\tZ[{}] = {};\n", o, args).as_str();
                    }
                }
                Instr::Pow(o, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        *out += &format!(
                            "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"movupd {}(%1), %%xmm1\\n\\t\"
\t\t\"movapd %%xmm0, %%xmm2\\n\\t\"
\t\t\"xorpd %%xmm1, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"haddpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"divpd %%xmm2, %%xmm0\\n\\t\"
\t\t\"movupd %%xmm0, {}\\n\\t\"\n",
                            format_addr!(*b),
                            (self.reserved_indices - self.param_count) * 16,
                            format_addr!(*o)
                        );
                    } else {
                        end_asm_block!(in_asm_block);

                        let base = get_input!(*b);
                        *out += format!("\tZ[{}] = pow({}, {});\n", o, base, e).as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);
                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\tZ[{}] = pow({}, {});\n", o, base, exp).as_str();
                }
                Instr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.0 {
                        Atom::EXP => {
                            *out += format!("\tZ[{}] = exp({});\n", o, arg).as_str();
                        }
                        Atom::LOG => {
                            *out += format!("\tZ[{}] = log({});\n", o, arg).as_str();
                        }
                        Atom::SIN => {
                            *out += format!("\tZ[{}] = sin({});\n", o, arg).as_str();
                        }
                        Atom::COS => {
                            *out += format!("\tZ[{}] = cos({});\n", o, arg).as_str();
                        }
                        Atom::SQRT => {
                            *out += format!("\tZ[{}] = sqrt({});\n", o, arg).as_str();
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }

        end_asm_block!(in_asm_block);

        *out += "\t__asm__(\n";
        for (i, r) in &mut self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                *out += &format!("\t\t\"movupd {}(%3), %%xmm0\\n\\t\"\n", r * 16);
            } else if *r < self.reserved_indices {
                *out += &format!(
                    "\t\t\"movupd {}(%2), %%xmm0\\n\\t\"\n",
                    (r - self.param_count) * 16
                );
            } else {
                *out += &format!("\t\t\"movupd {}(%1), %%xmm0\\n\\t\"\n", r * 16);
            }

            *out += &format!("\t\t\"movupd %%xmm0, {}(%0)\\n\\t\"\n", i * 16);
        }

        *out += &format!("\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\");\n", function_name);
        in_asm_block
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, BuiltinSymbol, usize),
}

impl<T: Clone + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }
}

impl<T: Clone + PartialEq> Expression<T> {
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

impl<T: Clone + PartialEq> EvalTree<T> {
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
            param_count: self.param_count,
        }
    }
}

impl<T: Clone + Default + PartialEq> EvalTree<T> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(mut self, cpe_rounds: Option<usize>) -> ExpressionEvaluator<T> {
        let mut stack = vec![T::default(); self.param_count];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack);
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
            param_count: self.param_count,
            reserved_indices,
            instructions,
            result_indices,
        };

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if e.remove_common_pairs() == 0 {
                break;
            }
        }

        e.optimize_stack();
        e
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>) {
        for t in &mut self.expressions.tree {
            t.strip_constants(stack, self.param_count);
        }

        for e in &mut self.expressions.subexpressions {
            e.strip_constants(stack, self.param_count);
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.strip_constants(stack, self.param_count);
            }

            for e in &mut e.subexpressions {
                e.strip_constants(stack, self.param_count);
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

                if p.1 > 1 {
                    instr.push(Instr::Mul(res, vec![b; p.1 as usize]));
                } else {
                    instr.push(Instr::Pow(res, b, p.1));
                }
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
    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize(
        &mut self,
        iterations: usize,
        n_cores: usize,
        start_scheme: Option<Vec<Expression<Rational>>>,
        verbose: bool,
    ) -> ExpressionEvaluator<Rational> {
        let _ = self.optimize_horner_scheme(iterations, n_cores, start_scheme, verbose);
        self.common_subexpression_elimination();
        self.clone().linearize(None)
    }

    /// Write the expressions in a Horner scheme where the variables
    /// are sorted by their occurrence count.
    pub fn horner_scheme(&mut self) {
        for t in &mut self.expressions.tree {
            t.occurrence_order_horner_scheme();
        }

        for e in &mut self.expressions.subexpressions {
            e.occurrence_order_horner_scheme();
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.occurrence_order_horner_scheme();
            }

            for e in &mut e.subexpressions {
                e.occurrence_order_horner_scheme();
            }
        }
    }

    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize_horner_scheme(
        &mut self,
        iterations: usize,
        n_cores: usize,
        start_scheme: Option<Vec<Expression<Rational>>>,
        verbose: bool,
    ) -> Vec<Expression<Rational>> {
        let v = match start_scheme {
            Some(a) => a,
            None => {
                let mut v = HashMap::default();

                for t in &mut self.expressions.tree {
                    t.find_all_variables(&mut v);
                }

                for e in &mut self.expressions.subexpressions {
                    e.find_all_variables(&mut v);
                }

                let mut v: Vec<_> = v.into_iter().collect();

                // for now, limit for parameters only
                v.retain(|(x, _)| matches!(x, Expression::Parameter(_)));
                v.sort_by_key(|k| std::cmp::Reverse(k.1));
                v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
            }
        };

        let scheme = Expression::optimize_horner_scheme_multiple(
            &self.expressions.tree,
            &v,
            iterations,
            n_cores,
            verbose,
        );
        for e in &mut self.expressions.tree {
            e.apply_horner_scheme(&scheme);
        }

        for e in &mut self.expressions.subexpressions {
            e.apply_horner_scheme(&scheme);
        }

        for (_, _, e) in &mut self.functions {
            let mut v = HashMap::default();

            for e in &mut e.subexpressions {
                e.find_all_variables(&mut v);
            }

            let mut v: Vec<_> = v.into_iter().collect();
            v.retain(|(x, _)| matches!(x, Expression::Parameter(_)));
            v.sort_by_key(|k| std::cmp::Reverse(k.1));
            let v = v.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

            let scheme = Expression::optimize_horner_scheme_multiple(
                &e.tree, &v, iterations, n_cores, verbose,
            );

            for t in &mut e.tree {
                t.apply_horner_scheme(&scheme);
            }

            for e in &mut e.subexpressions {
                e.apply_horner_scheme(&scheme);
            }
        }

        scheme
    }
}

impl Expression<Rational> {
    pub fn apply_horner_scheme(&mut self, scheme: &[Expression<Rational>]) {
        if scheme.is_empty() {
            return;
        }

        let a = match self {
            Expression::Add(a) => a,
            Expression::Eval(_, a) => {
                for arg in a {
                    arg.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Mul(m) => {
                for a in m {
                    a.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Pow(b) => {
                b.0.apply_horner_scheme(scheme);
                return;
            }
            Expression::Powf(b) => {
                b.0.apply_horner_scheme(scheme);
                b.1.apply_horner_scheme(scheme);
                return;
            }
            Expression::BuiltinFun(_, b) => {
                b.apply_horner_scheme(scheme);
                return;
            }
            _ => {
                return;
            }
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

        for mut x in a.drain(..) {
            let mut found = false;
            if let Expression::Mul(m) = &mut x {
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
                    x = Expression::Const(Rational::one());
                } else if m.len() == 1 {
                    x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if x == scheme[0] {
                found = true;
                x = Expression::Const(Rational::one());
            }

            if found {
                contains.push(x);
            } else {
                rest.push(x);
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

        let mut v = vec![];
        if let Expression::Mul(a) = contains {
            v.extend(a);
        } else {
            v.push(contains);
        }

        v.push(extracted);
        v.retain(|x| *x != Expression::Const(Rational::one()));
        v.sort();

        let c = if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Expression::Mul(v)
        };

        if rest.is_empty() {
            *self = c;
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                Expression::Add(rest)
            };

            r.apply_horner_scheme(&scheme[1..]);

            a.clear();
            a.push(c);

            if let Expression::Add(aa) = r {
                a.extend(aa);
            } else {
                a.push(r);
            }

            a.sort();
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn occurrence_order_horner_scheme(&mut self) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Add(a) => {
                for arg in &mut *a {
                    arg.occurrence_order_horner_scheme();
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
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Pow(p) => {
                p.0.occurrence_order_horner_scheme();
            }
            Expression::Powf(p) => {
                p.0.occurrence_order_horner_scheme();
                p.1.occurrence_order_horner_scheme();
            }
            Expression::BuiltinFun(_, a) => {
                a.occurrence_order_horner_scheme();
            }
            Expression::SubExpression(_) => {}
        }
    }

    pub fn optimize_horner_scheme(
        &self,
        vars: &[Self],
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) -> Vec<Self> {
        Self::optimize_horner_scheme_multiple(
            std::slice::from_ref(self),
            vars,
            iterations,
            n_cores,
            verbose,
        )
    }

    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Self],
        iterations: usize,
        n_cores: usize,
        verbose: bool,
    ) -> Vec<Self> {
        if vars.len() == 0 {
            return vars.to_vec();
        }

        let horner: Vec<_> = expressions
            .iter()
            .map(|x| {
                let mut h = x.clone();
                h.apply_horner_scheme(&vars);
                h
            })
            .collect();
        let mut subexpr = HashMap::default();
        let mut best_ops = (0, 0);
        for h in &horner {
            let ops = h.count_operations_with_subexpression(&mut subexpr);
            best_ops = (best_ops.0 + ops.0, best_ops.1 + ops.1);
        }

        if verbose {
            println!(
                "Initial ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

        let permutations =
            if vars.len() < 10 && Integer::factorial(vars.len() as u32) <= iterations.max(1) {
                let v: Vec<_> = (0..vars.len()).collect();
                Some(unique_permutations(&v).1)
            } else {
                None
            };
        let p_ref = &permutations;

        let n_cores = if LicenseManager::is_licensed() {
            n_cores
        } else {
            1
        };

        std::thread::scope(|s| {
            for i in 0..n_cores {
                let mut cvars = vars.to_vec();
                let best_scheme = best_scheme.clone();
                let best_mul = best_mul.clone();
                let best_add = best_add.clone();
                s.spawn(move || {
                    let mut r = thread_rng();

                    for j in 0..iterations / n_cores {
                        // try a random swap
                        let mut t1 = 0;
                        let mut t2 = 0;

                        if let Some(p) = p_ref {
                            if j >= p.len() / n_cores {
                                break;
                            }

                            let perm = &p[i * (p.len() / n_cores) + j];
                            cvars = perm.iter().map(|x| vars[*x].clone()).collect();
                        } else {
                            t1 = r.gen_range(0..cvars.len());
                            t2 = r.gen_range(0..cvars.len() - 1);

                            cvars.swap(t1, t2);
                        }

                        let horner: Vec<_> = expressions
                            .iter()
                            .map(|x| {
                                let mut h = x.clone();
                                h.apply_horner_scheme(&cvars);
                                h.to_hashed_expression().1
                            })
                            .collect();
                        let mut subexpr = HashMap::default();
                        let mut cur_ops = (0, 0);

                        for h in &horner {
                            let ops = h.count_operations_with_subexpression(&mut subexpr);
                            cur_ops = (cur_ops.0 + ops.0, cur_ops.1 + ops.1);
                        }

                        // prefer fewer multiplications
                        if cur_ops.1 <= best_mul.load(Ordering::Relaxed)
                            || cur_ops.1 == best_mul.load(Ordering::Relaxed)
                                && cur_ops.0 <= best_add.load(Ordering::Relaxed)
                        {
                            if verbose {
                                println!(
                                    "Accept move at core iteration {}/{}: {} additions and {} multiplications",
                                    j, iterations / n_cores,
                                    cur_ops.0, cur_ops.1
                                );
                            }

                            best_mul.store(cur_ops.1, Ordering::Relaxed);
                            best_add.store(cur_ops.0, Ordering::Relaxed);
                            best_scheme.lock().unwrap().clone_from_slice(&cvars);
                        } else {
                            cvars.swap(t1, t2);
                        }
                    }
                });
            }
        });

        Arc::try_unwrap(best_scheme).unwrap().into_inner().unwrap()
    }

    fn find_all_variables(&self, vars: &mut HashMap<Expression<Rational>, usize>) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Add(a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }

                for arg in a {
                    match arg {
                        Expression::Mul(m) => {
                            for aa in m {
                                if let Expression::Pow(p) = aa {
                                    vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                                } else {
                                    vars.entry(aa.clone()).and_modify(|x| *x += 1).or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(p) = x {
                                vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                            } else {
                                vars.entry(x.clone()).and_modify(|x| *x += 1).or_insert(1);
                            }
                        }
                    }
                }
            }
            Expression::Mul(a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Pow(p) => {
                p.0.find_all_variables(vars);
            }
            Expression::Powf(p) => {
                p.0.find_all_variables(vars);
                p.1.find_all_variables(vars);
            }
            Expression::BuiltinFun(_, a) => {
                a.find_all_variables(vars);
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
    /// Eliminate common subexpressions in the expression, also checking for subexpressions
    /// up to length `max_subexpr_len`.
    pub fn common_subexpression_elimination(&mut self) {
        let mut h = HashMap::default();

        let mut hashed_tree = vec![];
        for t in &self.tree {
            let (_, t) = t.to_hashed_expression();
            hashed_tree.push(t);
        }

        for t in &hashed_tree {
            t.find_subexpression(&mut h);
        }

        h.retain(|_, v| *v > 1);

        // make the second argument a unique index of the subexpression
        for (i, v) in h.values_mut().enumerate() {
            *v = self.subexpressions.len() + i;
        }

        let mut n_hash_tree = hashed_tree.clone();
        for t in &mut n_hash_tree {
            t.replace_subexpression(&h, false);
        }

        self.tree = n_hash_tree.iter().map(|x| x.to_expression()).collect();

        let mut v: Vec<_> = h.clone().into_iter().map(|(k, v)| (v, k)).collect();

        v.sort_by_key(|k| k.0); // not needed

        // replace subexpressions in subexpressions and
        // sort them based on their dependencies
        for (_, x) in v {
            let mut he = x.clone();
            he.replace_subexpression(&h, true);
            self.subexpressions.push(he.to_expression());
        }

        let mut dep_tree = vec![];
        for (i, s) in self.subexpressions.iter().enumerate() {
            let mut deps = vec![];
            s.get_dependent_subexpressions(&mut deps);
            dep_tree.push((i, deps.clone()));
        }

        let mut rename = HashMap::default();
        let mut new_subs = vec![];
        let mut i = 0;
        while !dep_tree.is_empty() {
            if dep_tree[i].1.iter().all(|x| rename.contains_key(x)) {
                rename.insert(dep_tree[i].0, new_subs.len());
                new_subs.push(self.subexpressions[dep_tree[i].0].clone());
                dep_tree.swap_remove(i);
                if i == dep_tree.len() {
                    i = 0;
                }
            } else {
                i = (i + 1) % dep_tree.len();
            }
        }

        for x in &mut new_subs {
            x.rename_subexpression(&rename);
        }
        for t in &mut self.tree {
            t.rename_subexpression(&rename);
        }

        self.subexpressions = new_subs;
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> Expression<T> {
    fn rename_subexpression(&mut self, subexp: &HashMap<usize, usize>) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in &mut *a {
                    arg.rename_subexpression(subexp);
                }

                a.sort();
            }
            Expression::Pow(p) => {
                p.0.rename_subexpression(subexp);
            }
            Expression::Powf(p) => {
                p.0.rename_subexpression(subexp);
                p.1.rename_subexpression(subexp);
            }
            Expression::BuiltinFun(_, a) => {
                a.rename_subexpression(subexp);
            }
            Expression::SubExpression(i) => {
                *self = Expression::SubExpression(*subexp.get(i).unwrap());
            }
        }
    }

    fn get_dependent_subexpressions(&self, dep: &mut Vec<usize>) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Pow(p) => {
                p.0.get_dependent_subexpressions(dep);
            }
            Expression::Powf(p) => {
                p.0.get_dependent_subexpressions(dep);
                p.1.get_dependent_subexpressions(dep);
            }
            Expression::BuiltinFun(_, a) => {
                a.get_dependent_subexpressions(dep);
            }
            Expression::SubExpression(i) => {
                dep.push(*i);
            }
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + Ord> SplitExpression<T> {
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
    // Count the number of additions and multiplications in the expression.
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
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(p) => {
                let (a, m) = p.0.count_operations();
                let (a2, m2) = p.1.count_operations();
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_) => (0, 0),
            Expression::BuiltinFun(_, b) => b.count_operations(), // not clear how to count this, third arg?
            Expression::SubExpression(_) => (0, 0),
        }
    }

    // Count the number of additions and multiplications in the expression, counting
    // subexpressions only once.
    pub fn count_operations_with_subexpression<'a>(
        &'a self,
        sub_expr: &mut HashMap<&'a Self, usize>,
    ) -> (usize, usize) {
        if matches!(
            self,
            Expression::Const(_) | Expression::Parameter(_,) | Expression::ReadArg(_)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            //println!("SUB {:?}", self);
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            Expression::Const(_) => (0, 0),
            Expression::Parameter(_) => (0, 0),
            Expression::Eval(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_) => (0, 0),
            Expression::BuiltinFun(_, b) => b.count_operations_with_subexpression(sub_expr), // not clear how to count this, third arg?
            Expression::SubExpression(_) => (0, 0),
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
                match s.0 {
                    Atom::EXP => arg.exp(),
                    Atom::LOG => arg.log(),
                    Atom::SIN => arg.sin(),
                    Atom::COS => arg.cos(),
                    Atom::SQRT => arg.sqrt(),
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

struct EvaluatorFunctions<'a> {
    eval_double: libloading::Symbol<
        'a,
        unsafe extern "C" fn(params: *const f64, buffer: *mut f64, out: *mut f64),
    >,
    eval_complex: libloading::Symbol<
        'a,
        unsafe extern "C" fn(
            params: *const Complex<f64>,
            buffer: *mut Complex<f64>,
            out: *mut Complex<f64>,
        ),
    >,
    create_buffer_double: libloading::Symbol<'a, unsafe extern "C" fn() -> *mut f64>,
    create_buffer_complex: libloading::Symbol<'a, unsafe extern "C" fn() -> *mut Complex<f64>>,
    drop_buffer_double: libloading::Symbol<'a, unsafe extern "C" fn(buffer: *mut f64)>,
    drop_buffer_complex: libloading::Symbol<'a, unsafe extern "C" fn(buffer: *mut Complex<f64>)>,
}

pub struct CompiledEvaluator {
    fn_name: String,
    library: Library,
    buffer_double: *mut f64,
    buffer_complex: *mut Complex<f64>,
}

self_cell!(
    struct Library {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctions,
    }
);

unsafe impl Send for CompiledEvaluator {}

impl std::fmt::Debug for CompiledEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

/// A floating point type that can be used for compiled evaluation.
pub trait CompiledEvaluatorFloat: Sized {
    fn evaluate(eval: &mut CompiledEvaluator, args: &[Self], out: &mut [Self]);
}

impl CompiledEvaluatorFloat for f64 {
    #[inline(always)]
    fn evaluate(eval: &mut CompiledEvaluator, args: &[Self], out: &mut [Self]) {
        eval.evaluate_double(args, out);
    }
}

impl CompiledEvaluatorFloat for Complex<f64> {
    #[inline(always)]
    fn evaluate(eval: &mut CompiledEvaluator, args: &[Self], out: &mut [Self]) {
        eval.evaluate_complex(args, out);
    }
}

impl Drop for CompiledEvaluator {
    fn drop(&mut self) {
        unsafe {
            (self.library.borrow_dependent().drop_buffer_double)(self.buffer_double);
            (self.library.borrow_dependent().drop_buffer_complex)(self.buffer_complex);
        }
    }
}

impl CompiledEvaluator {
    /// Load a new function from the same library.
    pub fn load_new_function(&self, function_name: &str) -> Result<CompiledEvaluator, String> {
        let library = unsafe {
            Library::try_new::<String>(self.library.borrow_owner().clone(), |lib| {
                Ok(EvaluatorFunctions {
                    eval_double: lib
                        .get(format!("{}_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    eval_complex: lib
                        .get(format!("{}_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    create_buffer_double: lib
                        .get(format!("{}_create_buffer_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    create_buffer_complex: lib
                        .get(format!("{}_create_buffer_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    drop_buffer_double: lib
                        .get("drop_buffer_double".as_bytes())
                        .map_err(|e| e.to_string())?,
                    drop_buffer_complex: lib
                        .get("drop_buffer_complex".as_bytes())
                        .map_err(|e| e.to_string())?,
                })
            })
        }?;

        Ok(CompiledEvaluator {
            fn_name: function_name.to_string(),
            buffer_double: unsafe { (library.borrow_dependent().create_buffer_double)() },
            buffer_complex: unsafe { (library.borrow_dependent().create_buffer_complex)() },
            library,
        })
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

            let library = Library::try_new::<String>(std::sync::Arc::new(lib), |lib| {
                Ok(EvaluatorFunctions {
                    eval_double: lib
                        .get(format!("{}_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    eval_complex: lib
                        .get(format!("{}_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    create_buffer_double: lib
                        .get(format!("{}_create_buffer_double", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    create_buffer_complex: lib
                        .get(format!("{}_create_buffer_complex", function_name).as_bytes())
                        .map_err(|e| e.to_string())?,
                    drop_buffer_double: lib
                        .get("drop_buffer_double".as_bytes())
                        .map_err(|e| e.to_string())?,
                    drop_buffer_complex: lib
                        .get("drop_buffer_complex".as_bytes())
                        .map_err(|e| e.to_string())?,
                })
            })?;

            Ok(CompiledEvaluator {
                fn_name: function_name.to_string(),
                buffer_double: (library.borrow_dependent().create_buffer_double)(),
                buffer_complex: (library.borrow_dependent().create_buffer_complex)(),
                library,
            })
        }
    }

    /// Evaluate the compiled code.
    #[inline(always)]
    pub fn evaluate<T: CompiledEvaluatorFloat>(&mut self, args: &[T], out: &mut [T]) {
        T::evaluate(self, args, out);
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    pub fn evaluate_double(&mut self, args: &[f64], out: &mut [f64]) {
        unsafe {
            (self.library.borrow_dependent().eval_double)(
                args.as_ptr(),
                self.buffer_double,
                out.as_mut_ptr(),
            )
        }
    }

    /// Evaluate the compiled code with complex numbers.
    #[inline(always)]
    pub fn evaluate_complex(&mut self, args: &[Complex<f64>], out: &mut [Complex<f64>]) {
        unsafe {
            (self.library.borrow_dependent().eval_complex)(
                args.as_ptr(),
                self.buffer_complex,
                out.as_mut_ptr(),
            )
        }
    }
}

/// Options for compiling exported code.
#[derive(Clone)]
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

/// The inline assembly mode used to generate fast
/// assembly instructions for mathematical operations.
/// Set to `None` to disable inline assembly.
#[derive(Copy, Clone)]
pub enum InlineASM {
    /// Use instructions suitable for x86_64 machines.
    X64,
    /// Do not generate inline assembly.
    None,
}

impl Default for InlineASM {
    /// Set the assembly mode suitable for the current
    /// architecture.
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            return InlineASM::X64;
        } else {
            InlineASM::None
        }
    }
}

impl<T: NumericalFloatLike> EvalTree<T> {
    /// Export the evaluation tree to C++ code. For much improved performance,
    /// optimize the tree instead.
    pub fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
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

        res += &format!(
            "\ntemplate<typename T>\nvoid {}(T* params, T* out) {{\n",
            function_name
        );

        for (i, s) in self.expressions.subexpressions.iter().enumerate() {
            res += &format!("\tT Z{}_ = {};\n", i, self.export_cpp_impl(s, &[]));
        }

        for (i, e) in self.expressions.tree.iter().enumerate() {
            res += &format!("\tout[{}] = {};\n", i, self.export_cpp_impl(&e, &[]));
        }

        res += "\treturn;\n}\n";

        res += &format!("\nextern \"C\" {{\n\tvoid {0}_double(double* params, double* out) {{\n\t\t{0}(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);
        res += &format!("\nextern \"C\" {{\n\tvoid {0}_complex(std::complex<double>* params, std::complex<double>* out) {{\n\t\t{0}(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);

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
            Expression::BuiltinFun(s, a) => match s.0 {
                Atom::EXP => {
                    let mut r = "exp(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Atom::LOG => {
                    let mut r = "log(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Atom::SIN => {
                    let mut r = "sin(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Atom::COS => {
                    let mut r = "cos(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Atom::SQRT => {
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
    pub fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTree<Rational>, String> {
        Self::to_eval_tree_multiple(std::slice::from_ref(self), fn_map, params)
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree_multiple(
        exprs: &[Self],
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
    ) -> Result<EvalTree<Rational>, String> {
        let mut funcs = vec![];
        let tree = exprs
            .iter()
            .map(|t| t.to_eval_tree_impl(fn_map, params, &[], &mut funcs))
            .collect::<Result<_, _>>()?;

        Ok(EvalTree {
            expressions: SplitExpression {
                tree,
                subexpressions: vec![],
            },
            functions: funcs,
            param_count: params.len(),
        })
    }

    fn to_eval_tree_impl(
        &self,
        fn_map: &FunctionMap<'a, Rational>,
        params: &[Atom],
        args: &[Symbol],
        funcs: &mut Vec<(String, Vec<Symbol>, SplitExpression<Rational>)>,
    ) -> Result<Expression<Rational>, String> {
        if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
            return Ok(Expression::Parameter(p));
        }

        if let Some(c) = fn_map.get_constant(*self) {
            return Ok(Expression::Const(c.clone()));
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => Ok(Expression::Const((n, d).into())),
                CoefficientView::Large(l) => Ok(Expression::Const(l.to_rat())),
                CoefficientView::Float(f) => {
                    // TODO: converting back to rational is slow
                    Ok(Expression::Const(f.to_float().to_rational()))
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
                if [Atom::EXP, Atom::LOG, Atom::SIN, Atom::COS, Atom::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(fn_map, params, args, funcs)?;

                    return Ok(Expression::BuiltinFun(
                        BuiltinSymbol(f.get_symbol()),
                        Box::new(arg_eval),
                    ));
                }

                let Some(fun) = fn_map.get(*self) else {
                    return Err(format!("Undefined function {}", self));
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
                            .map(|arg| arg.to_eval_tree_impl(fn_map, params, args, funcs))
                            .collect::<Result<_, _>>()?;

                        if let Some(pos) = funcs.iter().position(|f| f.0 == *name) {
                            Ok(Expression::Eval(pos, eval_args))
                        } else {
                            let r = e.to_eval_tree_impl(fn_map, params, arg_spec, funcs)?;
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
                let b_eval = b.to_eval_tree_impl(fn_map, params, args, funcs)?;

                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num > 1 {
                                return Ok(Expression::Mul(vec![b_eval.clone(); num as usize]));
                            } else {
                                return Ok(Expression::Pow(Box::new((
                                    Expression::Mul(vec![
                                        b_eval.clone();
                                        num.unsigned_abs() as usize
                                    ]),
                                    -1,
                                ))));
                            }
                        }
                    }
                }

                let e_eval = e.to_eval_tree_impl(fn_map, params, args, funcs)?;
                Ok(Expression::Powf(Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(fn_map, params, args, funcs)?;
                    if let Expression::Mul(m) = a {
                        muls.extend(m);
                    } else {
                        muls.push(a);
                    }
                }

                muls.sort();

                Ok(Expression::Mul(muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(fn_map, params, args, funcs)?);
                }

                adds.sort();

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
    ) -> Result<T, String> {
        if let Some(c) = const_map.get(self) {
            return Ok(c.clone());
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d) => Ok(coeff_map(&Rational::from_unchecked(n, d))),
                CoefficientView::Large(l) => Ok(coeff_map(&l.to_rat())),
                CoefficientView::Float(f) => {
                    // TODO: converting back to rational is slow
                    Ok(coeff_map(&f.to_float().to_rational()))
                }
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => match v.get_symbol() {
                Atom::E => Ok(coeff_map(&1.into()).e()),
                Atom::PI => Ok(coeff_map(&1.into()).pi()),
                Atom::I => coeff_map(&1.into())
                    .i()
                    .ok_or_else(|| "Numerical type does not support imaginary unit".to_string()),
                _ => Err(format!(
                    "Variable {} not in constant map",
                    State::get_name(v.get_symbol())
                )),
            },
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [Atom::EXP, Atom::LOG, Atom::SIN, Atom::COS, Atom::SQRT].contains(&name) {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate(coeff_map, const_map, function_map, cache)?;

                    return Ok(match f.get_symbol() {
                        Atom::EXP => arg_eval.exp(),
                        Atom::LOG => arg_eval.log(),
                        Atom::SIN => arg_eval.sin(),
                        Atom::COS => arg_eval.cos(),
                        Atom::SQRT => arg_eval.sqrt(),
                        _ => unreachable!(),
                    });
                }

                if let Some(eval) = cache.get(self) {
                    return Ok(eval.clone());
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f {
                    args.push(arg.evaluate(coeff_map, const_map, function_map, cache)?);
                }

                let Some(fun) = function_map.get(&f.get_symbol()) else {
                    Err(format!(
                        "Missing function {}",
                        State::get_name(f.get_symbol())
                    ))?
                };
                let eval = fun.get()(&args, const_map, function_map, cache);

                cache.insert(*self, eval.clone());
                Ok(eval)
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate(coeff_map, const_map, function_map, cache)?;

                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, den) = n.get_coeff_view() {
                        if den == 1 {
                            if num >= 0 {
                                return Ok(b_eval.pow(num as u64));
                            } else {
                                return Ok(b_eval.pow(num.unsigned_abs()).inv());
                            }
                        }
                    }
                }

                let e_eval = e.evaluate(coeff_map, const_map, function_map, cache)?;
                Ok(b_eval.powf(&e_eval))
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r *= arg.evaluate(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r += arg.evaluate(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
        }
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    pub fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        match self {
            AtomView::Num(num_view) => {
                if num_view.is_zero() {
                    ConditionResult::True
                } else {
                    ConditionResult::False
                }
            }
            AtomView::Var(_) => ConditionResult::False,
            AtomView::Fun(_) => ConditionResult::False,
            AtomView::Pow(p) => p.get_base().zero_test(iterations, tolerance),
            AtomView::Mul(mul_view) => {
                let mut is_zero = ConditionResult::False;
                for arg in mul_view {
                    match arg.zero_test(iterations, tolerance) {
                        ConditionResult::True => return ConditionResult::True,
                        ConditionResult::False => {}
                        ConditionResult::Inconclusive => {
                            is_zero = ConditionResult::Inconclusive;
                        }
                    }
                }

                is_zero
            }
            AtomView::Add(_) => {
                // an expanded polynomial is only zero if it is a literal zero
                if self.is_polynomial(false, true).is_some() {
                    ConditionResult::False
                } else {
                    self.zero_test_impl(iterations, tolerance)
                }
            }
        }
    }

    fn zero_test_impl(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        // collect all variables and functions and fill in random variables

        let mut rng = rand::thread_rng();

        if self.contains_symbol(State::I) {
            let mut vars: HashMap<_, _> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_builtin(s) || s == Atom::DERIVATIVE {
                        Some((x, Complex::new(0f64.into(), 0f64.into())))
                    } else {
                        None
                    }
                })
                .collect();

            let mut cache = HashMap::default();

            for _ in 0..iterations {
                cache.clear();

                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            Complex::new(
                                ErrorPropagatingFloat::new(
                                    0f64.from_rational(x),
                                    -0f64.get_epsilon().log10(),
                                ),
                                ErrorPropagatingFloat::new(
                                    0f64.zero(),
                                    -0f64.get_epsilon().log10(),
                                ),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                        &mut cache,
                    )
                    .unwrap();

                let res_re = r.re.get_num().to_f64();
                let res_im = r.im.get_num().to_f64();
                if res_re.is_finite()
                    && (res_re - r.re.get_absolute_error() > 0.
                        || res_re + r.re.get_absolute_error() < 0.)
                    || res_im.is_finite()
                        && (res_im - r.im.get_absolute_error() > 0.
                            || res_im + r.im.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.len() == 0 && r.re.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        } else {
            let mut vars: HashMap<_, ErrorPropagatingFloat<f64>> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_builtin(s) || s == Atom::DERIVATIVE {
                        Some((x, 0f64.into()))
                    } else {
                        None
                    }
                })
                .collect();

            let mut cache = HashMap::default();

            for _ in 0..iterations {
                cache.clear();

                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            ErrorPropagatingFloat::new(
                                0f64.from_rational(x),
                                -0f64.get_epsilon().log10(),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                        &mut cache,
                    )
                    .unwrap();

                let res = r.get_num().to_f64();
                if res.is_finite()
                    && (res - r.get_absolute_error() > 0. || res + r.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.len() == 0 && r.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::HashMap;

    use crate::{
        atom::{Atom, AtomCore},
        domains::{float::Float, rational::Rational},
        evaluate::{EvaluationFn, FunctionMap, OptimizationSettings},
        id::ConditionResult,
        state::State,
    };

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

        let r = a
            .evaluate(|x| x.into(), &const_map, &fn_map, &mut cache)
            .unwrap();
        assert_eq!(r, 2905.761021719902);
    }

    #[test]
    fn arb_prec() {
        let x = State::get_symbol("v1");
        let a = Atom::parse("128731/12893721893721 + v1").unwrap();

        let mut const_map = HashMap::default();

        let v = Atom::new_var(x);
        const_map.insert(v.as_view(), Float::with_val(200, 6));

        let r = a
            .evaluate(
                |r| r.to_multi_prec_float(200),
                &const_map,
                &HashMap::default(),
                &mut HashMap::default(),
            )
            .unwrap();

        assert_eq!(
            format!("{}", r),
            "6.00000000998400625211945786243908951675582851493871969158108"
        );
    }

    #[test]
    fn nested() {
        let e1 = Atom::parse("x + pi + cos(x) + f(g(x+1),h(x*2)) + p(1,x)").unwrap();
        let e2 = Atom::parse("x + h(x*2) + cos(x)").unwrap();
        let f = Atom::parse("y^2 + z^2*y^2").unwrap();
        let g = Atom::parse("i(y+7)+x*i(y+7)*(y-1)").unwrap();
        let h = Atom::parse("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)").unwrap();
        let i = Atom::parse("y - 1").unwrap();
        let p1 = Atom::parse("3*z^3 + 4*z^2 + 6*z +8").unwrap();

        let mut fn_map = FunctionMap::new();

        fn_map.add_constant(
            Atom::new_var(State::get_symbol("pi")),
            Rational::from((22, 7)).into(),
        );
        fn_map
            .add_tagged_function(
                State::get_symbol("p"),
                vec![Atom::new_num(1).into()],
                "p1".to_string(),
                vec![State::get_symbol("z")],
                p1.as_view(),
            )
            .unwrap();
        fn_map
            .add_function(
                State::get_symbol("f"),
                "f".to_string(),
                vec![State::get_symbol("y"), State::get_symbol("z")],
                f.as_view(),
            )
            .unwrap();
        fn_map
            .add_function(
                State::get_symbol("g"),
                "g".to_string(),
                vec![State::get_symbol("y")],
                g.as_view(),
            )
            .unwrap();
        fn_map
            .add_function(
                State::get_symbol("h"),
                "h".to_string(),
                vec![State::get_symbol("y")],
                h.as_view(),
            )
            .unwrap();
        fn_map
            .add_function(
                State::get_symbol("i"),
                "i".to_string(),
                vec![State::get_symbol("y")],
                i.as_view(),
            )
            .unwrap();

        let params = vec![Atom::parse("x").unwrap()];

        let evaluator = Atom::evaluator_multiple(
            &[e1.as_view(), e2.as_view()],
            &fn_map,
            &params,
            OptimizationSettings::default(),
        )
        .unwrap();

        let mut e_f64 = evaluator.map_coeff(&|x| x.into());
        let r = e_f64.evaluate_single(&[1.1]);
        assert!((r - 1622709.2254269677).abs() / 1622709.2254269677 < 1e-10);
    }

    #[test]
    fn zero_test() {
        let e = Atom::parse("(sin(v1)^2-sin(v1))(sin(v1)^2+sin(v1))^2 - (1/4 sin(2v1)^2-1/2 sin(2v1)cos(v1)-2 cos(v1)^2+1/2 sin(2v1)cos(v1)^3+3 cos(v1)^4-cos(v1)^6)").unwrap();
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::Inconclusive);

        let e = Atom::parse("x + (1+x)^2 + (x+2)*5").unwrap();
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::False);
    }
}
