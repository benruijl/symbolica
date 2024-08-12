use std::hash::{Hash, Hasher};

use ahash::{AHasher, HashMap};
use self_cell::self_cell;

use crate::{
    atom::{representation::InlineVar, Atom, AtomOrView, AtomView, Symbol},
    coefficient::CoefficientView,
    combinatorics::CombinationIterator,
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
    BuiltinFun(ExpressionHash, Symbol, Box<HashedExpression<T>>),
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

#[derive(Debug, Eq, Clone)]
pub struct HashedSubExpression<'a, T> {
    pub hash: u64,
    pub op: u8, // should be 0 when it's a single item, else 3 for add, 4 for mul; used for EQ!
    pub expression: Vec<&'a HashedExpression<T>>,
}

impl<T: PartialEq> PartialEq for HashedSubExpression<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        if self.hash != other.hash {
            return false;
        }

        if self.op == other.op {
            return self.expression.len() == other.expression.len()
                && self
                    .expression
                    .iter()
                    .zip(&other.expression)
                    .all(|x| **x.0 == **x.1);
        }

        if self.op != 0 {
            return other.eq(self);
        }

        if other.op == 3 {
            if let HashedExpression::Add(_, v) = &self.expression[0] {
                return self.expression.iter().zip(v).all(|(a, b)| *a == b);
            }
        } else if other.op == 4 {
            if let HashedExpression::Mul(_, v) = &self.expression[0] {
                return self.expression.iter().zip(v).all(|(a, b)| *a == b);
            }
        }

        false
    }
}

impl<'a, T: Clone> HashedSubExpression<'a, T> {
    fn to_hashed_expression(&self) -> HashedExpression<T> {
        match self.op {
            3 => HashedExpression::Add(
                self.hash,
                self.expression.iter().cloned().cloned().collect(),
            ),
            4 => HashedExpression::Mul(
                self.hash,
                self.expression.iter().cloned().cloned().collect(),
            ),
            _ => self.expression[0].clone(),
        }
    }
}

impl<'a, T: Eq + Hash> Hash for HashedSubExpression<'a, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash)
    }
}
impl<T: Eq + Hash + Clone + Ord> HashedExpression<T> {
    fn find_subexpression<'a>(
        &'a self,
        subexp: &mut HashMap<HashedSubExpression<'a, T>, usize>,
        max_subexpr_len: usize,
    ) -> bool {
        if matches!(
            self,
            HashedExpression::Const(_, _)
                | HashedExpression::Parameter(_, _)
                | HashedExpression::ReadArg(_, _)
        ) {
            return true;
        }

        let complete_node = HashedSubExpression {
            hash: self.get_hash(),
            op: 0,
            expression: vec![self],
        };

        if let Some(i) = subexp.get_mut(&complete_node) {
            *i += 1;
            return true;
        }

        subexp.insert(complete_node.clone(), 1);

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp, max_subexpr_len);
                }
            }
            HashedExpression::Add(_, a) | HashedExpression::Mul(_, a) => {
                let mut unused_indices = (0..a.len()).collect::<Vec<_>>();
                let mut k = max_subexpr_len.min(unused_indices.len() - 1);

                /*let op: u64 = if let HashedExpression::Add(_, _) = self {
                    3
                } else {
                    4
                };

                let min = if op == 3 { 2 } else { 1 };*/

                //k = 0;
                'big_loop: while k > 1 {
                    let mut it = CombinationIterator::new(unused_indices.len(), k);
                    while let Some(x) = it.next() {
                        let op: u64 = if let HashedExpression::Add(_, _) = self {
                            3
                        } else {
                            4
                        };

                        let mut hash = op;
                        for i in x {
                            hash = hash.wrapping_add(a[unused_indices[*i]].get_hash());
                        }

                        // FIXME: the op does not play well!
                        // we need to construct a new
                        let complete_node = HashedSubExpression {
                            hash,
                            op: op as u8,
                            expression: x.iter().map(|i| &a[unused_indices[*i]]).collect(),
                        };

                        if let Some(i) = subexp.get_mut(&complete_node) {
                            *i += 1;
                            for j in x.iter().rev() {
                                unused_indices.remove(*j);
                            }

                            k = unused_indices.len().min(k);
                            continue 'big_loop;
                        } else {
                            subexp.insert(complete_node.clone(), 1);
                        }
                    }

                    k -= 1;
                }

                for arg in unused_indices {
                    a[arg].find_subexpression(subexp, max_subexpr_len);
                }
            }
            HashedExpression::Pow(_, p) => {
                p.0.find_subexpression(subexp, max_subexpr_len);
            }
            HashedExpression::Powf(_, p) => {
                p.0.find_subexpression(subexp, max_subexpr_len);
                p.1.find_subexpression(subexp, max_subexpr_len);
            }
            HashedExpression::BuiltinFun(_, _, _) => {}
            HashedExpression::SubExpression(_, _) => {}
        }

        false
    }

    fn replace_subexpression<'a>(
        &mut self,
        subexp: &HashMap<HashedSubExpression<'a, T>, usize>,
        max_subexpr_len: usize,
        skip_root: bool,
    ) {
        if !skip_root {
            let complete_node = HashedSubExpression {
                hash: self.get_hash(),
                op: 0,
                expression: vec![self],
            };

            if let Some(i) = subexp.get(&complete_node) {
                *self = HashedExpression::SubExpression(self.get_hash(), *i); // recycle hash!
                return;
            }
        }

        let op: u64 = if let HashedExpression::Add(_, _) = self {
            3
        } else {
            4
        };

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, max_subexpr_len, false);
                }
            }
            HashedExpression::Add(_, a) | HashedExpression::Mul(_, a) => {
                let mut unused_indices = (0..a.len()).collect::<Vec<_>>();
                let mut k = max_subexpr_len.min(unused_indices.len() - 1);

                let mut res = vec![];

                /*if op == 3 {
                    // k = 0;
                }

                let min = if op == 3 { 2 } else { 1 };*/

                //k = 0;
                'big_loop: while k > 1 {
                    let mut it = CombinationIterator::new(unused_indices.len(), k);
                    while let Some(x) = it.next() {
                        let mut hash = op;
                        for i in x {
                            hash = hash.wrapping_add(a[unused_indices[*i]].get_hash());
                        }

                        // FIXME: the op does not play well!
                        // we need to construct a new
                        let complete_node = HashedSubExpression {
                            hash,
                            op: op as u8,
                            expression: x.iter().map(|i| &a[unused_indices[*i]]).collect(),
                        };

                        if let Some(i) = subexp.get(&complete_node) {
                            res.push(HashedExpression::SubExpression(hash, *i)); // recycle hash???

                            for j in x.iter().rev() {
                                unused_indices.remove(*j);
                            }

                            k = unused_indices.len().min(k);
                            continue 'big_loop;
                        }
                    }

                    k -= 1;
                }

                for arg in unused_indices {
                    a[arg].replace_subexpression(subexp, max_subexpr_len, false);
                    res.push(a[arg].clone());
                }

                res.sort();
                *a = res;
            }
            HashedExpression::Pow(_, p) => {
                p.0.replace_subexpression(subexp, max_subexpr_len, false);
            }
            HashedExpression::Powf(_, p) => {
                p.0.replace_subexpression(subexp, max_subexpr_len, false);
                p.1.replace_subexpression(subexp, max_subexpr_len, false);
            }
            HashedExpression::BuiltinFun(_, _, _) => {}
            HashedExpression::SubExpression(_, _) => {}
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

#[derive(Clone)]

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

impl<T: Default> ExpressionEvaluator<T> {
    pub fn remove_common_pairs(&mut self) -> usize {
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
        let cpp = match inline_asm {
            InlineASM::Intel => self.export_asm_str(function_name, include_header),
            InlineASM::None => self.export_cpp_str(function_name, include_header),
        };

        let _ = std::fs::write(filename, cpp)?;
        Ok(ExportedCode {
            source_filename: filename.to_string(),
            function_name: function_name.to_string(),
            inline_asm,
        })
    }

    pub fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = if include_header {
            "#include <iostream>\n#include <complex>\n#include <cmath>\n\n".to_string()
        } else {
            String::new()
        };

        res += &format!(
            "\ntemplate<typename T>\nvoid {}(T* params, T* out) {{\n",
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

        res += &format!("\nextern \"C\" {{\n\tvoid {0}_double(double* params, double* out) {{\n\t\t{0}(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);
        res += &format!("\nextern \"C\" {{\n\tvoid {0}_complex(std::complex<double>* params, std::complex<double>* out) {{\n\t\t{0}(params, out);\n\t\treturn;\n\t}}\n}}\n", function_name);

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

    pub fn export_asm_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = if include_header {
            "#include <iostream>\n#include <complex>\n#include <cmath>\n\n".to_string()
        } else {
            String::new()
        };

        res += &format!(
            "static const std::complex<double> CONSTANTS_complex[{}] = {{{}}};\n\n",
            self.reserved_indices - self.param_count + 1,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("std::complex<double>({})", self.stack[i]))
                    .collect::<Vec<_>>();
                nums.push("std::complex<double>(0, -0.)".to_string()); // used for inversion
                nums.join(",")
            }
        );

        res += &format!("extern \"C\" void {}_complex(const std::complex<double> *params, std::complex<double> *out)\n{{\n", function_name);

        // TODO: pass as argument to prevent stack reallocation
        res += &format!("\tstd::complex<double> Z[{}];\n", self.stack.len());

        self.export_asm_complex_impl(&self.instructions, &mut res);

        res += "\treturn;\n}\n\n";

        res += &format!(
            "static const double CONSTANTS_double[{}] = {{{}}};\n\n",
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
            "extern \"C\" void {}_double(const double *params, double *out)\n{{\n",
            function_name
        );

        res += &format!("\tdouble Z[{}];\n", self.stack.len());

        self.export_asm_double_impl(&self.instructions, &mut res);

        res += "\treturn;\n}\n";

        res
    }

    fn export_asm_double_impl(&self, instr: &[Instr], out: &mut String) -> bool {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!("CONSTANTS_double[{}]", $i - self.param_count)
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! format_addr {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("PTR [%2+{}]", $i * 8)
                } else if $i < self.reserved_indices {
                    format!("PTR [%1+{}]", ($i - self.param_count) * 8)
                } else {
                    // TODO: subtract reserved indices
                    format!("PTR [%0+{}]", $i * 8)
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    *out += "\t\t:\n\t\t: \"r\"(Z), \"r\"(CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n";
                    $in_block = false;
                }
            };
        }

        let mut in_asm_block = false;
        let mut regcount = 0;

        fn reg_unused(reg: usize, instr: &[Instr], out: &[usize]) -> bool {
            if out.contains(&reg) {
                return false;
            }

            for ins in instr {
                match ins {
                    Instr::Add(r, a) | Instr::Mul(r, a) => {
                        if a.iter().any(|x| *x == reg) {
                            return false;
                        }

                        if r == &reg {
                            return true;
                        }
                    }
                    Instr::Pow(r, b, _) => {
                        if *b == reg {
                            return false;
                        }
                        if r == &reg {
                            return true;
                        }
                    }
                    Instr::Powf(r, b, e) => {
                        if *b == reg || *e == reg {
                            return false;
                        }
                        if r == &reg {
                            return true;
                        }
                    }
                    Instr::BuiltinFun(r, _, b) => {
                        if *b == reg {
                            return false;
                        }
                        if r == &reg {
                            return true;
                        }
                    }
                }
            }

            true
        }

        let mut recycle_register: (Option<(usize, u32)>, Option<(usize, u32)>) = (None, None); // old and current register
        for (i, ins) in instr.iter().enumerate() {
            // keep results in xmm registers if the last use is in the next instruction
            if let Some(ii) = instr.get(i + 1) {
                match ins {
                    Instr::Add(r, _) | Instr::Mul(r, _) => match ii {
                        Instr::Add(j, _) | Instr::Mul(j, _) => {
                            if r == j || reg_unused(*r, &instr[i + 2..], &self.result_indices) {
                                if let Some(old) = recycle_register.0 {
                                    recycle_register.1 = Some((*r, old.1));
                                } else {
                                    recycle_register.1 = Some((*r, regcount));
                                }
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            match ins {
                Instr::Add(o, a) | Instr::Mul(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let oper = if matches!(ins, Instr::Add(_, _)) {
                        "add"
                    } else {
                        "mul"
                    };

                    if let Some(old) = recycle_register.0 {
                        assert!(a.iter().any(|rr| *rr == old.0)); // the last value must be used

                        for i in a {
                            if *i != old.0 {
                                *out += &format!(
                                    "\t\t\"{}sd xmm{}, QWORD {}\\n\\t\"\n",
                                    oper,
                                    old.1,
                                    format_addr!(*i)
                                );
                            }
                        }

                        if recycle_register.1.is_none() {
                            *out += &format!(
                                "\t\t\"movsd QWORD {}, xmm{}\\n\\t\"\n",
                                format_addr!(*o),
                                old.1,
                            );
                        }
                    } else if let Some(new) = recycle_register.1 {
                        *out += &format!(
                            "\t\t\"movsd xmm{}, QWORD {}\\n\\t\"\n",
                            new.1,
                            format_addr!(a[0])
                        );

                        for i in &a[1..] {
                            *out += &format!(
                                "\t\t\"{}sd xmm{}, QWORD {}\\n\\t\"\n",
                                oper,
                                new.1,
                                format_addr!(*i)
                            );
                        }
                    } else {
                        *out += &format!(
                            "\t\t\"movsd xmm{}, QWORD {}\\n\\t\"\n",
                            regcount,
                            format_addr!(a[0])
                        );

                        for i in &a[1..] {
                            *out += &format!(
                                "\t\t\"{}sd xmm{}, QWORD {}\\n\\t\"\n",
                                oper,
                                regcount,
                                format_addr!(*i)
                            );
                        }

                        *out += &format!(
                            "\t\t\"movsd QWORD {}, xmm{}\\n\\t\"\n",
                            format_addr!(*o),
                            regcount,
                        );
                    }
                }
                Instr::Pow(o, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        *out += &format!(
                            "\t\t\"movsd xmm{0}, QWORD PTR [%1+{1}]\\n\\t\"
\t\t\"divsd xmm{0}, QWORD {2}\\n\\t\"
\t\t\"movsd QWORD {3}, xmm{0}\\n\\t\"\n",
                            regcount,
                            (self.reserved_indices - self.param_count) * 8,
                            format_addr!(*b),
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

                    match *s {
                        State::EXP => {
                            *out += format!("\tZ[{}] = exp({});\n", o, arg).as_str();
                        }
                        State::LOG => {
                            *out += format!("\tZ[{}] = log({});\n", o, arg).as_str();
                        }
                        State::SIN => {
                            *out += format!("\tZ[{}] = sin({});\n", o, arg).as_str();
                        }
                        State::COS => {
                            *out += format!("\tZ[{}] = cos({});\n", o, arg).as_str();
                        }
                        State::SQRT => {
                            *out += format!("\tZ[{}] = sqrt({});\n", o, arg).as_str();
                        }
                        _ => unreachable!(),
                    }
                }
            }

            recycle_register.0 = recycle_register.1.take();
        }

        end_asm_block!(in_asm_block);

        *out += "\t__asm__(\n";
        for (i, r) in &mut self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                *out += &format!(
                    "\t\t\"movsd xmm{}, QWORD PTR[%3+{}]\\n\\t\"\n",
                    regcount,
                    r * 8
                );
            } else if *r < self.reserved_indices {
                *out += &format!(
                    "\t\t\"movsd xmm{}, QWORD PTR[%2+{}]\\n\\t\"\n",
                    regcount,
                    (r - self.param_count) * 8
                );
            } else {
                *out += &format!(
                    "\t\t\"movsd xmm{}, QWORD PTR[%1+{}]\\n\\t\"\n",
                    regcount,
                    r * 8
                );
            }

            *out += &format!(
                "\t\t\"movsd QWORD PTR[%0+{}], xmm{}\\n\\t\"\n",
                i * 8,
                regcount
            );
            regcount = (regcount + 1) % 16;
        }

        *out += "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"(CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\");\n";
        in_asm_block
    }

    fn export_asm_complex_impl(&self, instr: &[Instr], out: &mut String) -> bool {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!("CONSTANTS_complex[{}]", $i - self.param_count)
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! format_addr {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("PTR [%2+{}]", $i * 16)
                } else if $i < self.reserved_indices {
                    format!("PTR [%1+{}]", ($i - self.param_count) * 16)
                } else {
                    // TODO: subtract reserved indices
                    format!("PTR [%0+{}]", $i * 16)
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    *out += "\t\t:\n\t\t: \"r\"(Z), \"r\"(CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n";
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

                    *out += &format!("\t\t\"xorpd xmm0, xmm0\\n\\t\"\n");

                    // TODO: try loading in multiple registers for better instruction-level parallelism?
                    for i in a {
                        *out +=
                            &format!("\t\t\"addpd xmm0, XMMWORD {}\\n\\t\"\n", format_addr!(*i));
                    }
                    *out += &format!("\t\t\"movupd XMMWORD {}, xmm0\\n\\t\"\n", format_addr!(*o),);
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
                                "\t\t\"movupd xmm{}, XMMWORD {}\\n\\t\"\n",
                                i + 1,
                                format_addr!(*r)
                            );
                        }

                        for i in 1..a.len() {
                            *out += &format!(
                                "\t\t\"movapd xmm0, xmm1\\n\\t\"
\t\t\"unpckhpd xmm0, xmm0\\n\\t\"
\t\t\"unpcklpd xmm1, xmm1\\n\\t\"
\t\t\"mulpd xmm0, xmm{0}\\n\\t\"
\t\t\"mulpd xmm1, xmm{0}\\n\\t\"
\t\t\"shufpd xmm0, xmm0, 1\\n\\t\"
\t\t\"addsubpd xmm1, xmm0\\n\\t\"\n",
                                i + 1
                            );
                        }

                        *out +=
                            &format!("\t\t\"movupd XMMWORD {}, xmm1\\n\\t\"\n", format_addr!(*o));
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
                            "\t\t\"movupd xmm0, XMMWORD {}\\n\\t\"
\t\t\"movupd xmm1, XMMWORD PTR [%1+{}]\\n\\t\"
\t\t\"movapd xmm2, xmm0\\n\\t\"
\t\t\"xorpd xmm0, xmm1\\n\\t\"
\t\t\"mulpd xmm2, xmm2\\n\\t\"
\t\t\"haddpd xmm2, xmm2\\n\\t\"
\t\t\"divpd xmm0, xmm2\\n\\t\"
\t\t\"movupd XMMWORD {}, xmm0\\n\\t\"",
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

                    match *s {
                        State::EXP => {
                            *out += format!("\tZ[{}] = exp({});\n", o, arg).as_str();
                        }
                        State::LOG => {
                            *out += format!("\tZ[{}] = log({});\n", o, arg).as_str();
                        }
                        State::SIN => {
                            *out += format!("\tZ[{}] = sin({});\n", o, arg).as_str();
                        }
                        State::COS => {
                            *out += format!("\tZ[{}] = cos({});\n", o, arg).as_str();
                        }
                        State::SQRT => {
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
                *out += &format!("\t\t\"movupd xmm0, XMMWORD PTR[%3+{}]\\n\\t\"\n", r * 16);
            } else if *r < self.reserved_indices {
                *out += &format!(
                    "\t\t\"movupd xmm0, XMMWORD PTR[%2+{}]\\n\\t\"\n",
                    (r - self.param_count) * 16
                );
            } else {
                *out += &format!("\t\t\"movupd xmm0, XMMWORD PTR[%1+{}]\\n\\t\"\n", r * 16);
            }

            *out += &format!("\t\t\"movupd XMMWORD PTR[%0+{}], xmm0\\n\\t\"\n", i * 16);
        }

        *out += "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"(CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\");\n";
        in_asm_block
    }
}

#[derive(Debug, Clone)]
enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, Symbol, usize),
}

impl<T: Clone + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }

    pub fn unnest(&mut self, max_depth: usize) {
        // TODO: also unnest subexpressions
        for t in &mut self.tree {
            Self::unnest_impl(t, &mut self.subexpressions, 0, max_depth);
        }
    }

    fn unnest_impl(
        expr: &mut Expression<T>,
        subs: &mut Vec<Expression<T>>,
        depth: usize,
        max_depth: usize,
    ) {
        match expr {
            Expression::Add(a) | Expression::Mul(a) => {
                if depth == max_depth {
                    // split off into new subexpression

                    Self::unnest_impl(expr, subs, 0, max_depth);

                    let mut r = Expression::SubExpression(subs.len());
                    std::mem::swap(expr, &mut r);
                    subs.push(r);
                    return;
                }

                for x in a {
                    Self::unnest_impl(x, subs, depth + 1, max_depth);
                }
            }
            Expression::Eval(_, _) => {} // TODO: count the arg evals! always bring to base level?
            Expression::BuiltinFun(_, _) => {}
            _ => {} // TODO: count pow levels too?
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

    pub fn unnest(&mut self, max_depth: usize) {
        for (_, _, e) in &mut self.functions {
            e.unnest(max_depth);
        }

        self.expressions.unnest(max_depth);
    }
}

impl<T: Clone + Default + PartialEq> EvalTree<T> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(mut self, cpe_rounds: usize) -> ExpressionEvaluator<T> {
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

        for _ in 0..cpe_rounds {
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

        let mut v = vec![];
        if let Expression::Mul(a) = contains {
            v.extend(a);
        } else {
            v.push(contains);
        }

        v.push(extracted);
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

            let mut v = vec![c];
            if let Expression::Add(a) = r {
                v.extend(a);
            } else {
                v.push(r);
            }

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
    pub fn common_subexpression_elimination(&mut self, max_subexpr_len: usize) {
        self.expressions
            .common_subexpression_elimination(max_subexpr_len);

        for (_, _, e) in &mut self.functions {
            e.common_subexpression_elimination(max_subexpr_len);
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
    /// Eliminate common subexpressions in the expression, also checking for subexpressions
    /// up to length `max_subexpr_len`.
    pub fn common_subexpression_elimination(&mut self, max_subexpr_len: usize) {
        let mut h = HashMap::default();

        let mut hashed_tree = vec![];
        for t in &self.tree {
            let (_, t) = t.to_hashed_expression();
            hashed_tree.push(t);
        }

        for t in &hashed_tree {
            t.find_subexpression(&mut h, max_subexpr_len);
        }

        h.retain(|_, v| *v > 1);

        // make the second argument a unique index of the subexpression
        for (i, v) in h.values_mut().enumerate() {
            *v = self.subexpressions.len() + i;
        }

        let mut n_hash_tree = hashed_tree.clone();
        for t in &mut n_hash_tree {
            t.replace_subexpression(&h, max_subexpr_len, false);
        }

        self.tree = n_hash_tree.iter().map(|x| x.to_expression()).collect();

        let mut v: Vec<_> = h.clone().into_iter().map(|(k, v)| (v, k)).collect();

        v.sort_by_key(|k| k.0); // not needed

        // replace subexpressions in subexpressions and
        // sort them based on their dependencies
        for (_, x) in v {
            let mut he = x.to_hashed_expression();
            he.replace_subexpression(&h, max_subexpr_len, true);
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

    /*
    fn replace_subexpression(&mut self, subexp: &HashMap<Expression<T>, usize>, skip_root: bool) {
        if !skip_root {
            if let Some(i) = subexp.get(&self) {
                *self = Expression::SubExpression(*i);
                return;
            }
        }

        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, false);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in &mut *a {
                    arg.replace_subexpression(subexp, false);
                }

                a.sort();
            }
            Expression::Pow(p) => {
                p.0.replace_subexpression(subexp, false);
            }
            Expression::Powf(p) => {
                p.0.replace_subexpression(subexp, false);
                p.1.replace_subexpression(subexp, false);
            }
            Expression::BuiltinFun(_, _) => {}
            Expression::SubExpression(_) => {}
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
    */
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
    inline_asm: InlineASM,
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
    fn_name: String,
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

impl Clone for CompiledEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.with_dependent(|_, d| &d.fn_name))
            .unwrap()
    }
}

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
                    fn_name: function_name.to_string(),
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
                    fn_name: function_name.to_string(),
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
    pub fn new(source_filename: String, function_name: String, inline_asm: InlineASM) -> Self {
        ExportedCode {
            source_filename,
            function_name,
            inline_asm,
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

        match self.inline_asm {
            InlineASM::Intel => {
                builder.arg("-masm=intel");
            }
            InlineASM::None => {}
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
    Intel,
    /// Do not generate inline assembly.
    None,
}

impl Default for InlineASM {
    /// Set the assembly mode suitable for the current
    /// architecture.
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            return InlineASM::Intel;
        } else {
            InlineASM::None
        }
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
            inline_asm: InlineASM::None,
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
            param_count: params.len(),
        })
    }

    fn to_eval_tree_impl<T: Clone + Default + Ord, F: Fn(&Rational) -> T + Copy>(
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

                muls.sort();

                Ok(Expression::Mul(muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(coeff_map, fn_map, params, args, funcs)?);
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
