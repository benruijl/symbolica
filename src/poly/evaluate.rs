use std::{
    cmp::Reverse,
    hash::{Hash, Hasher},
};

use ahash::{AHasher, HashMap, HashSet, HashSetExt};
use rand::{thread_rng, Rng};

use crate::{
    representations::Identifier,
    rings::{
        integer::{Integer, IntegerRing},
        Ring,
    },
    state::State,
};

use super::{polynomial::MultivariatePolynomial, Exponent};

/// A borrowed version of a Horner node, suitable as a key in a
/// hashmap. It uses precomputed hashes for the complete node
/// `var^pow*content+rest` and for its children `var^pow*content` and `var^pow`.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedHornerNode<'a, R: Ring>
where
    R::Element: Hash + Eq,
{
    pub var: usize,
    pub pow: usize,
    pub content: Option<&'a HornerScheme<R>>,
    pub rest: Option<&'a HornerScheme<R>>,
    pub hash: (u64, u64, u64),
}

impl<'a, R: Ring> PartialEq for BorrowedHornerNode<'a, R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash is skipped
        self.var == other.var
            && self.pow == other.pow
            && self.content == other.content
            && self.rest == other.rest
    }
}

impl<'a, R: Ring> Eq for BorrowedHornerNode<'a, R> where R::Element: Hash + Eq {}

impl<'a> Hash for BorrowedHornerNode<'a, IntegerRing> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash = if let Some(_) = self.content {
            if let Some(_) = self.rest {
                self.hash.2
            } else {
                self.hash.1
            }
        } else {
            self.hash.0
        };

        state.write_u64(hash);
    }
}

impl<'a, R: Ring> From<&'a HornerNode<R>> for BorrowedHornerNode<'a, R>
where
    R::Element: Hash + Eq,
{
    fn from(n: &'a HornerNode<R>) -> Self {
        BorrowedHornerNode {
            var: n.var,
            pow: n.pow,
            content: n.content_rest.0.as_ref(),
            rest: n.content_rest.1.as_ref(),
            hash: n.hash,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BorrowedHornerScheme<'a, R: Ring>
where
    R::Element: Hash + Eq,
{
    Node(BorrowedHornerNode<'a, R>),
    Leaf(&'a R::Element),
}

impl<'a, R: Ring> From<&'a HornerScheme<R>> for BorrowedHornerScheme<'a, R>
where
    R::Element: Hash + Eq,
{
    fn from(value: &'a HornerScheme<R>) -> Self {
        match value {
            HornerScheme::Node(n) => BorrowedHornerScheme::Node(BorrowedHornerNode {
                var: n.var,
                pow: n.pow,
                content: n.content_rest.0.as_ref(),
                rest: n.content_rest.1.as_ref(),
                hash: n.hash,
            }),
            HornerScheme::Leaf(_, l) => BorrowedHornerScheme::Leaf(l),
        }
    }
}

impl<'a> BorrowedHornerScheme<'a, IntegerRing> {
    /// Determine the number of operations required to evaluate the Horner scheme.
    /// Common subexpressions are only counted once.
    pub fn op_count_cse(&self) -> usize {
        let mut h = HashSet::default();
        self.op_count_cse_impl(&mut h)
    }

    fn op_count_cse_impl(&self, set: &mut HashSet<BorrowedHornerNode<'a, IntegerRing>>) -> usize {
        match self {
            BorrowedHornerScheme::Node(n) => {
                // check if n = var^pow*a+b is seen before
                if set.contains(n) {
                    return 0;
                }

                // check if var^pow*a is seen before
                let mut b = n.clone();
                b.rest = None;

                if set.contains(&b) {
                    set.insert(n.clone());
                    n.rest
                        .map(|x| 1 + BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                        .unwrap()
                } else {
                    // add var^pow to seen list
                    let instr = if n.pow > 1 {
                        let mut c = b.clone();
                        c.content = None;

                        if set.contains(&c) {
                            1
                        } else {
                            set.insert(c.clone());
                            n.pow
                        }
                    } else {
                        1
                    };

                    set.insert(b.clone());
                    set.insert(n.clone());
                    instr
                        + n.content
                            .map(|x| BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap_or(0)
                        + n.rest
                            .map(|x| 1 + BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap_or(0)
                }
            }
            BorrowedHornerScheme::Leaf(_) => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HornerNode<R: Ring>
where
    R::Element: Hash + Eq,
{
    pub var: usize,
    pub pow: usize,
    pub content_rest: Box<(Option<HornerScheme<R>>, Option<HornerScheme<R>>)>,
    pub hash: (u64, u64, u64),
}

impl<R: Ring> PartialEq for HornerNode<R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash is skipped
        self.var == other.var && self.pow == other.pow && self.content_rest == other.content_rest
    }
}

/// A polynomial written in a Horner scheme, i.e.:
/// `x^2*(x*(y*(z + 1) + y) + 5) + z^2`.
#[derive(Debug, Clone)]
pub enum HornerScheme<R: Ring>
where
    R::Element: Hash + Eq,
{
    Node(HornerNode<R>),
    Leaf(u64, R::Element),
}

impl<R: Ring> PartialEq for HornerScheme<R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Node(l0), Self::Node(r0)) => l0 == r0,
            (Self::Leaf(_, l1), Self::Leaf(_, r1)) => l1 == r1,
            _ => false,
        }
    }
}

impl HornerScheme<IntegerRing> {
    /// Evaluate a polynomial written in a Horner scheme. For faster
    /// evaluation, convert the Horner scheme into an `InstructionSet`.
    pub fn evaluate(&self, samples: &[Integer]) -> Integer {
        let field = IntegerRing::new();
        match self {
            HornerScheme::Node(n) => match &n.content_rest.0 {
                Some(s) => match &n.content_rest.1 {
                    Some(s1) => field.add(
                        &field.mul(
                            &field.pow(&samples[n.var], n.pow as u64),
                            &s.evaluate(samples),
                        ),
                        &s1.evaluate(samples),
                    ),
                    None => field.mul(
                        &field.pow(&samples[n.var], n.pow as u64),
                        &s.evaluate(samples),
                    ),
                },
                None => field.pow(&samples[n.var], n.pow as u64),
            },
            HornerScheme::Leaf(_, l) => l.clone(),
        }
    }

    /// Clean up the Horner scheme and save the allocations in `boxes`.
    fn cleanup(
        &mut self,
        boxes: &mut Vec<
            Box<(
                Option<HornerScheme<IntegerRing>>,
                Option<HornerScheme<IntegerRing>>,
            )>,
        >,
    ) {
        let private = std::mem::replace(self, HornerScheme::Leaf(0, Integer::Natural(0)));
        match private {
            HornerScheme::Node(mut n) => {
                if let Some(c) = &mut n.content_rest.0 {
                    c.cleanup(boxes);
                }
                if let Some(c) = &mut n.content_rest.1 {
                    c.cleanup(boxes);
                }

                *n.content_rest = (None, None);

                boxes.push(n.content_rest);
            }
            HornerScheme::Leaf(_, _) => {}
        }
    }
}

impl<R: Ring> std::fmt::Display for HornerScheme<R>
where
    R::Element: Hash + Eq,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HornerScheme::Leaf(_, l) => f.write_fmt(format_args!("{:?}", l)),
            HornerScheme::Node(n) => {
                if n.pow == 1 {
                    f.write_fmt(format_args!("+x{}*(", n.var))?;
                } else {
                    f.write_fmt(format_args!("+x{}^{}*(", n.var, n.pow))?;
                }
                if let Some(s) = &n.content_rest.0 {
                    s.fmt(f)?;
                }

                f.write_str(")")?;

                if let Some(s) = &n.content_rest.1 {
                    if let HornerScheme::Leaf(_, _) = &s {
                        f.write_str("+")?;
                        s.fmt(f)?;
                    } else {
                        s.fmt(f)?;
                    }
                }

                Ok(())
            }
        }
    }
}

impl<E: Exponent> MultivariatePolynomial<IntegerRing, E> {
    /// Write the polynomial in a Horner scheme with the variable ordering
    /// defined in `order`.
    pub fn to_horner_scheme(&self, order: &[usize]) -> HornerScheme<IntegerRing> {
        let mut indices: Vec<_> = (0..self.nterms).collect();
        let mut power_sub = vec![E::zero(); self.nvars];
        let mut horner_boxes = vec![];

        self.to_horner_scheme_impl(order, &mut indices, 0, &mut power_sub, &mut horner_boxes)
    }

    /// Create a Horner scheme using the variable order specified in `order`,
    /// using the terms from `self` indexed by `indices[index_start..]`.
    /// The exponents of each term should be reduced by `power_sub`
    /// and allocations from previous Horner scheme constructions can
    /// be provided using `boxes`.
    ///
    /// This function is optimized, as it is called many times during
    /// the Horner scheme optimization.
    fn to_horner_scheme_impl(
        &self,
        order: &[usize],
        indices: &mut Vec<usize>,
        index_start: usize,
        power_sub: &mut [E],
        boxes: &mut Vec<
            Box<(
                Option<HornerScheme<IntegerRing>>,
                Option<HornerScheme<IntegerRing>>,
            )>,
        >,
    ) -> HornerScheme<IntegerRing> {
        if order.is_empty() {
            debug_assert!(indices.len() <= index_start + 1);

            let num = if indices.len() == index_start {
                self.field.zero()
            } else {
                self.coefficients[indices[index_start]].clone()
            };

            let mut h = AHasher::default();
            h.write_u8(1);
            num.hash(&mut h);
            return HornerScheme::Leaf(h.finish(), num);
        }

        // find the lowest non-zero power of the next variable in the ordering
        let var = order[0];
        let mut min_pow = E::zero();

        for i in &indices[index_start..] {
            let e = self.exponents(*i)[var];
            if e > power_sub[var] {
                let p = e - power_sub[var];
                if min_pow == E::zero() || p < min_pow {
                    min_pow = p;
                }
            }
        }

        if min_pow == E::zero() {
            return self.to_horner_scheme_impl(&order[1..], indices, index_start, power_sub, boxes);
        }

        let new_index_start = indices.len();

        // create the branch for terms that do not contain var^min_pow
        for t in index_start..new_index_start {
            if self.exponents(indices[t])[var] <= power_sub[var] {
                indices.push(indices[t]);
            }
        }

        let rest =
            self.to_horner_scheme_impl(&order[1..], indices, new_index_start, power_sub, boxes);

        // create the branch for terms that do contain var^min_pow and lower the power
        indices.truncate(new_index_start);
        for t in index_start..new_index_start {
            if self.exponents(indices[t])[var] > power_sub[var] {
                indices.push(indices[t]);
            }
        }

        power_sub[var] = power_sub[var] + min_pow;

        let content =
            self.to_horner_scheme_impl(&order, indices, new_index_start, power_sub, boxes);

        power_sub[var] = power_sub[var] - min_pow;

        indices.truncate(new_index_start);

        let mut h = AHasher::default();
        h.write_u8(0);
        var.hash(&mut h);
        (min_pow.to_u32() as usize).hash(&mut h);

        let pow_hash = h.finish(); // hash var^pow

        match &content {
            HornerScheme::Node(n) => h.write_u64(n.hash.2),
            HornerScheme::Leaf(n, _) => h.write_u64(*n),
        }

        let pow_content_hash = h.finish(); // hash var^pow*content

        match &rest {
            HornerScheme::Node(n) => h.write_u64(n.hash.2),
            HornerScheme::Leaf(n, _) => h.write_u64(*n),
        }

        let full_hash = h.finish(); // hash var^pow*content+rest

        let children = (
            if let HornerScheme::Leaf(_, Integer::Natural(1)) = content {
                None
            } else {
                Some(content)
            },
            if let HornerScheme::Leaf(_, Integer::Natural(0)) = rest {
                None
            } else {
                Some(rest)
            },
        );

        // recycle the box if it is available
        let boxed_children = if let Some(mut b) = boxes.pop() {
            *b = children;
            b
        } else {
            Box::new(children)
        };

        HornerScheme::Node(HornerNode {
            var,
            pow: min_pow.to_u32() as usize,
            hash: (pow_hash, pow_content_hash, full_hash),
            content_rest: boxed_children,
        })
    }

    /// Find the Horner scheme that minimizes the number of operations,
    /// taking common subexpressions into account.
    pub fn optimize_horner_scheme(
        &self,
        num_tries: usize,
    ) -> (HornerScheme<IntegerRing>, Vec<usize>) {
        // the starting scheme is the descending order of occurrence of variables
        let mut occurrence: Vec<_> = (0..self.nvars).map(|x| (x, 0)).collect();
        for es in self.exponents.chunks(self.nvars) {
            for ((_, o), e) in occurrence.iter_mut().zip(es) {
                if *e > E::zero() {
                    *o += 1;
                }
            }
        }
        occurrence.sort_by_key(|e| Reverse(e.1));

        let mut scheme: Vec<_> = occurrence.into_iter().map(|(v, _)| v).collect();

        let mut indices: Vec<_> = (0..self.nterms).collect();
        let mut power_sub = vec![E::zero(); self.nvars];

        let mut horner_boxes = vec![];

        let mut best =
            self.to_horner_scheme_impl(&scheme, &mut indices, 0, &mut power_sub, &mut horner_boxes);
        let mut best_score = BorrowedHornerScheme::from(&best).op_count_cse();

        let mut best_scheme = scheme.clone();
        let mut rng = thread_rng();

        // TODO: for few variables, test all permutations
        for i in 0..num_tries {
            let a = rng.gen_range(0..self.nvars);
            let b = rng.gen_range(0..self.nvars);
            scheme.swap(a, b);

            let mut new = self.to_horner_scheme_impl(
                &scheme,
                &mut indices,
                0,
                &mut power_sub,
                &mut horner_boxes,
            );

            // TODO: recycle
            let mut hash_set = HashSet::with_capacity(best_score * 2);
            let new_oc = BorrowedHornerScheme::from(&new).op_count_cse_impl(&mut hash_set);

            if new_oc <= best_score {
                // accept move
                best = new;
                best_score = new_oc;
                best_scheme.copy_from_slice(&scheme);
            } else {
                new.cleanup(&mut horner_boxes);
            }

            // see if we reject the change
            if new_oc as f64 / best_score as f64 > 1. + 0.5 * (1. - i as f64 / num_tries as f64) {
                //if new_oc as f64 > best_score as f64 {
                scheme.swap(a, b);
            }
        }

        (best, best_scheme)
    }
}

impl HornerScheme<IntegerRing> {
    /// Convert the Horner scheme to a list of instructions, suitable for numerical evaluation.
    pub fn to_instr<'a>(&'a self) -> InstructionList {
        let mut instr = vec![];
        let mut seen: HashMap<BorrowedHornerNode<'a, IntegerRing>, usize> = HashMap::default();
        let v = self.to_instr_rec(&mut instr, &mut seen); // variable can be ignored, should point to last
        instr.push(Instruction::Yield(v));
        InstructionList { instr }
    }

    fn to_instr_rec<'a>(
        &'a self,
        instr: &mut Vec<Instruction>,
        seen: &mut HashMap<BorrowedHornerNode<'a, IntegerRing>, usize>,
    ) -> Variable {
        match self {
            HornerScheme::Node(n) => {
                let nb = BorrowedHornerNode::from(n);

                if let Some(v) = seen.get(&nb) {
                    return Variable::Index(*v);
                }

                let mut v2 = Variable::Constant(Integer::zero());
                if let Some(s) = &n.content_rest.1 {
                    v2 = s.to_instr_rec(instr, seen);
                }

                let v1 = if let Some(s) = &n.content_rest.0 {
                    let mut v1 = s.to_instr_rec(instr, seen);

                    // check if var^pow*content is seen before
                    let mut b = BorrowedHornerNode::from(n);
                    b.rest = None;

                    if let Some(v) = seen.get(&b) {
                        v1 = Variable::Index(*v);
                    } else {
                        // check if var^pow is seen before
                        if n.pow > 1 {
                            let mut c = b.clone();
                            c.content = None;
                            let vp = if let Some(v) = seen.get(&c) {
                                Variable::Index(*v)
                            } else {
                                instr.push(Instruction::Mul(vec![Variable::Var(n.var, n.pow)]));
                                seen.insert(c, instr.len() - 1);
                                Variable::Index(instr.len() - 1)
                            };
                            instr.push(Instruction::Mul(vec![vp, v1]));
                        } else {
                            instr.push(Instruction::Mul(vec![Variable::Var(n.var, n.pow), v1]));
                        };

                        seen.insert(b, instr.len() - 1);
                        v1 = Variable::Index(instr.len() - 1);
                    }
                    v1
                } else {
                    if n.pow > 1 {
                        instr.push(Instruction::Mul(vec![Variable::Var(n.var, n.pow)]));
                        Variable::Index(instr.len() - 1)
                    } else {
                        Variable::Var(n.var, n.pow)
                    }
                };

                if v2 != Variable::Constant(Integer::zero()) {
                    instr.push(Instruction::Add(vec![v1, v2]));
                    seen.insert(nb, instr.len() - 1);
                    Variable::Index(instr.len() - 1)
                } else {
                    if let Variable::Index(i) = v1 {
                        seen.insert(nb, i);
                        Variable::Index(i)
                    } else {
                        v1
                    }
                }
            }
            HornerScheme::Leaf(_, l) => Variable::Constant(l.clone()),
        }
    }
}

// An arithmetical instruction that is part of an `InstructionList`.
#[derive(Debug)]
pub enum Instruction {
    Add(Vec<Variable>),
    Mul(Vec<Variable>),
    Yield(Variable),
    Empty,
}

// An variable that is part of an `InstructionList`,
// which may refer to another instruction in the instruction list.
#[derive(Debug, PartialEq, Eq)]
pub enum Variable {
    Index(usize),
    Var(usize, usize), // var^pow
    Constant(Integer),
}

impl Variable {
    fn to_pretty_string(&self, var_map: &[Identifier], state: &State) -> String {
        match self {
            Variable::Index(i) => format!("Z{}", i),
            Variable::Var(v, p) => {
                let var = state.get_name(var_map[*v]).unwrap();
                if *p == 1 {
                    format!("{}", var)
                } else {
                    format!("{}^{}", var, p)
                }
            }

            Variable::Constant(c) => format!("{}", c),
        }
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Index(i) => f.write_fmt(format_args!("Z{}", i)),
            Variable::Var(v, p) => {
                if *p == 1 {
                    f.write_fmt(format_args!("x{}", v))
                } else {
                    f.write_fmt(format_args!("x{}^{}", v, p))
                }
            }

            Variable::Constant(c) => c.fmt(f),
        }
    }
}

impl Variable {
    pub fn evaluate(&self, samples: &[Integer], instr_eval: &[Integer]) -> Integer {
        match self {
            Variable::Index(i) => instr_eval[*i].clone(),
            Variable::Var(v, p) => samples[*v].pow(*p as u64),
            Variable::Constant(c) => c.clone(),
        }
    }

    pub fn op_count(&self, is_mul: bool) -> isize {
        match self {
            Variable::Index(_) => 1,
            Variable::Var(_, p) => *p as isize,
            Variable::Constant(l) => {
                if is_mul && l.abs().is_one() {
                    0
                } else {
                    1
                }
            }
        }
    }
}

/// A list of instructions suitable for fast numerical evaluation.
pub struct InstructionList {
    instr: Vec<Instruction>,
}

impl InstructionList {
    /// Evaluate the instructions and yield the result.
    pub fn evaluate(&self, samples: &[Integer]) -> Integer {
        let mut eval: Vec<Integer> = Vec::with_capacity(self.instr.len());

        for (_i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => {
                    let mut r = Integer::zero();
                    for x in a {
                        r += &x.evaluate(samples, &eval);
                    }
                    eval.push(r);
                }
                Instruction::Mul(m) => {
                    let mut r = Integer::one();
                    for x in m {
                        r *= &x.evaluate(samples, &eval);
                    }
                    eval.push(r);
                }
                Instruction::Yield(y) => return y.evaluate(samples, &eval),
                Instruction::Empty => panic!("Evaluation of empty instruction requested"),
            }
        }
        unreachable!()
    }

    /// Return the number of arithmetical operations required for evaluation.
    pub fn op_count(&self) -> isize {
        let mut sum = 0;
        for (_i, x) in self.instr.iter().enumerate() {
            sum += match x {
                Instruction::Add(a) => a.iter().map(|x| x.op_count(false)).sum::<isize>() - 1,
                Instruction::Mul(m) => m.iter().map(|x| x.op_count(true)).sum::<isize>() - 1,
                Instruction::Yield(_) => 0,
                Instruction::Empty => 0,
            };
        }
        sum
    }
}

impl std::fmt::Display for InstructionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    i,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    i,
                    m.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => f.write_fmt(format_args!("Z{} = {};\n", i, y))?,
                Instruction::Empty => f.write_fmt(format_args!("Z{} = NOP;\n", i))?,
            }
        }

        Ok(())
    }
}

pub struct InstructionSetPrinter<'a> {
    instr: &'a InstructionList,
    var_map: &'a [Identifier],
    state: &'a State,
}

impl<'a> std::fmt::Display for InstructionSetPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, x) in self.instr.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    i,
                    a.iter()
                        .map(|x| x.to_pretty_string(self.var_map, self.state))
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    i,
                    m.iter()
                        .map(|x| x.to_pretty_string(self.var_map, self.state))
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => f.write_fmt(format_args!("Z{} = {};\n", i, y))?,
                Instruction::Empty => f.write_fmt(format_args!("Z{} = NOP;\n", i))?,
            }
        }

        Ok(())
    }
}
