use std::{
    cmp::Reverse,
    hash::{Hash, Hasher},
};

use ahash::{AHasher, HashMap, HashSet, HashSetExt};
use rand::{thread_rng, Rng};

use crate::rings::{
    float::NumericalFloatLike,
    rational::{Rational, RationalField},
    EuclideanDomain,
};
use crate::{representations::Identifier, rings::Ring, state::State};

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
    pub gcd: R::Element, // only used for counting number of operations
    pub content: Option<&'a HornerScheme<R>>,
    pub rest: Option<&'a HornerScheme<R>>,
    pub hash: (u64, u64, u64),
}

impl<'a, R: Ring> PartialEq for BorrowedHornerNode<'a, R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash and gcd is skipped
        self.var == other.var
            && self.pow == other.pow
            && self.content == other.content
            && self.rest == other.rest
    }
}

impl<'a, R: Ring> Eq for BorrowedHornerNode<'a, R> where R::Element: Hash + Eq {}

impl<'a> Hash for BorrowedHornerNode<'a, RationalField> {
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
            gcd: n.gcd.clone(),
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
            HornerScheme::Node(n) => BorrowedHornerScheme::Node(BorrowedHornerNode::from(n)),
            HornerScheme::Leaf(_, l) => BorrowedHornerScheme::Leaf(l),
        }
    }
}

impl<'a> BorrowedHornerScheme<'a, RationalField> {
    /// Determine the number of operations required to evaluate the Horner scheme.
    /// Common subexpressions are only counted once.
    pub fn op_count_cse(&self) -> usize {
        let mut h = HashSet::default();
        self.op_count_cse_impl(&mut h)
    }

    fn op_count_cse_impl(&self, set: &mut HashSet<BorrowedHornerNode<'a, RationalField>>) -> usize {
        match self {
            BorrowedHornerScheme::Node(n) => {
                let gcd_op = if n.gcd.abs() != Rational::one() { 1 } else { 0 };

                // check if n = var^pow*a+b is seen before
                if set.contains(n) {
                    return gcd_op;
                }

                // check if var^pow*a is seen before
                let mut b = n.clone();
                b.rest = None;

                if set.contains(&b) {
                    set.insert(n.clone());
                    gcd_op
                        + 1
                        + n.rest
                            .map(|x| BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap() // can fail now because of gcd?
                } else {
                    // add var^pow to seen list
                    let instr = if n.pow > 1 {
                        let mut c = b.clone();
                        c.content = None;

                        if set.contains(&c) {
                            0
                        } else {
                            set.insert(c.clone());
                            n.pow - 1
                        }
                    } else {
                        0
                    };

                    set.insert(b.clone());
                    set.insert(n.clone());

                    gcd_op
                        + instr
                        + n.content
                            .map(|x| 1 + BorrowedHornerScheme::from(x).op_count_cse_impl(set))
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
    pub gcd: R::Element,
    pub content_rest: Box<(Option<HornerScheme<R>>, Option<HornerScheme<R>>)>,
    pub hash: (u64, u64, u64),
}

impl<R: Ring> PartialEq for HornerNode<R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash is skipped, but the gcd is included
        self.var == other.var
            && self.pow == other.pow
            && self.gcd == other.gcd
            && self.content_rest == other.content_rest
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
    Leaf(u64, R::Element), // hash and number
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

impl HornerScheme<RationalField> {
    /// Evaluate a polynomial written in a Horner scheme. For faster
    /// evaluation, convert the Horner scheme into an `InstructionList`.
    pub fn evaluate(&self, samples: &[Rational]) -> Rational {
        let field = RationalField::new();
        match self {
            HornerScheme::Node(n) => {
                let e = match &n.content_rest.0 {
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
                    None => match &n.content_rest.1 {
                        Some(s1) => field.add(
                            &field.pow(&samples[n.var], n.pow as u64),
                            &s1.evaluate(samples),
                        ),
                        None => field.pow(&samples[n.var], n.pow as u64),
                    },
                };
                &e * &n.gcd
            }
            HornerScheme::Leaf(_, l) => l.clone(),
        }
    }

    /// Clean up the Horner scheme and save the allocations in `boxes`.
    fn cleanup(
        &mut self,
        boxes: &mut Vec<
            Box<(
                Option<HornerScheme<RationalField>>,
                Option<HornerScheme<RationalField>>,
            )>,
        >,
    ) {
        let private = std::mem::replace(self, HornerScheme::Leaf(0, Rational::zero()));
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
                f.write_fmt(format_args!("+{:?}*(", n.gcd))?;

                if n.pow == 1 {
                    f.write_fmt(format_args!("x{}", n.var))?;
                } else {
                    f.write_fmt(format_args!("x{}^{}", n.var, n.pow))?;
                }
                if let Some(s) = &n.content_rest.0 {
                    f.write_fmt(format_args!("*("))?;
                    s.fmt(f)?;
                    f.write_fmt(format_args!(")"))?;
                }

                if let Some(s) = &n.content_rest.1 {
                    if let HornerScheme::Leaf(_, _) = &s {
                        f.write_str("+")?;
                        s.fmt(f)?;
                    } else {
                        s.fmt(f)?;
                    }
                }

                f.write_str(")")?;

                Ok(())
            }
        }
    }
}

impl<E: Exponent> MultivariatePolynomial<RationalField, E> {
    /// Write the polynomial in a Horner scheme with the variable ordering
    /// defined in `order`.
    pub fn to_horner_scheme(&self, order: &[usize]) -> HornerScheme<RationalField> {
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
                Option<HornerScheme<RationalField>>,
                Option<HornerScheme<RationalField>>,
            )>,
        >,
    ) -> HornerScheme<RationalField> {
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

        let mut rest =
            self.to_horner_scheme_impl(&order[1..], indices, new_index_start, power_sub, boxes);

        // create the branch for terms that do contain var^min_pow and lower the power
        indices.truncate(new_index_start);
        for t in index_start..new_index_start {
            if self.exponents(indices[t])[var] > power_sub[var] {
                indices.push(indices[t]);
            }
        }

        power_sub[var] = power_sub[var] + min_pow;

        let mut content =
            self.to_horner_scheme_impl(&order, indices, new_index_start, power_sub, boxes);

        power_sub[var] = power_sub[var] - min_pow;

        indices.truncate(new_index_start);

        // compute the gcd of both branches
        // normalize such that the first branch is positive
        let mut gcd = match &content {
            HornerScheme::Node(n) => n.gcd.clone(),
            HornerScheme::Leaf(_, l) => l.clone(),
        };

        let gcd_norm = match &content {
            HornerScheme::Node(n) => {
                if n.gcd.is_negative() {
                    Rational::Natural(-1, 1)
                } else {
                    Rational::Natural(1, 1)
                }
            }
            HornerScheme::Leaf(_, l) => {
                if l.is_negative() {
                    Rational::Natural(-1, 1)
                } else {
                    Rational::Natural(1, 1)
                }
            }
        };

        gcd = RationalField::new().gcd(
            &gcd,
            match &rest {
                HornerScheme::Node(n) => &n.gcd,
                HornerScheme::Leaf(_, l) => l,
            },
        );

        gcd *= &gcd_norm;

        if !gcd.is_one() {
            for s in [&mut content, &mut rest] {
                match s {
                    HornerScheme::Node(n) => n.gcd = &n.gcd / &gcd,
                    HornerScheme::Leaf(n, l) => {
                        *l = &*l / &gcd;

                        // overwrite the hash of the number
                        let mut h = AHasher::default();
                        h.write_u8(1);
                        l.hash(&mut h);
                        *n = h.finish();
                    }
                };
            }
        }

        // compute the hash of the node and its components
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
            if let HornerScheme::Leaf(_, Rational::Natural(1, 1)) = content {
                None
            } else {
                Some(content)
            },
            if let HornerScheme::Leaf(_, Rational::Natural(0, 1)) = rest {
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
            gcd,
            hash: (pow_hash, pow_content_hash, full_hash),
            content_rest: boxed_children,
        })
    }

    /// Find the Horner scheme that minimizes the number of operations,
    /// taking common subexpressions into account.
    pub fn optimize_horner_scheme(
        &self,
        num_tries: usize,
    ) -> (HornerScheme<RationalField>, Vec<usize>) {
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
                best.cleanup(&mut horner_boxes);
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

impl HornerScheme<RationalField> {
    /// Convert the Horner scheme to a list of instructions, suitable for numerical evaluation.
    pub fn to_instr<'a>(&'a self) -> InstructionList {
        let mut instr = vec![];
        let mut seen: HashMap<BorrowedHornerNode<'a, RationalField>, usize> = HashMap::default();
        let v = self.to_instr_rec(&mut instr, &mut seen); // variable can be ignored, should point to last
        instr.push(Instruction::Yield(v));
        InstructionList { instr }
    }

    fn to_instr_rec<'a>(
        &'a self,
        instr: &mut Vec<Instruction<Rational>>,
        seen: &mut HashMap<BorrowedHornerNode<'a, RationalField>, usize>,
    ) -> Variable<Rational> {
        match self {
            HornerScheme::Node(n) => {
                let nb = BorrowedHornerNode::from(n);

                // check if n = var^pow*content+rest is seen before
                if let Some(v) = seen.get(&nb) {
                    return if n.gcd != Rational::one() {
                        instr.push(Instruction::Mul(vec![
                            Variable::Index(*v),
                            Variable::Constant(n.gcd.clone()),
                        ]));
                        Variable::Index(instr.len() - 1)
                    } else {
                        Variable::Index(*v)
                    };
                }

                // check if var^pow*content is seen before
                let mut b = BorrowedHornerNode::from(n);
                b.rest = None;

                let v1 = if let Some(v) = seen.get(&b) {
                    Variable::Index(*v)
                } else {
                    // check if var^pow is seen before
                    let vp = if n.pow > 1 {
                        let mut c = b.clone();
                        c.content = None;
                        if let Some(v) = seen.get(&c) {
                            Variable::Index(*v)
                        } else {
                            instr.push(Instruction::Mul(vec![Variable::Var(n.var, n.pow)]));
                            seen.insert(c, instr.len() - 1);
                            Variable::Index(instr.len() - 1)
                        }
                    } else {
                        Variable::Var(n.var, n.pow)
                    };

                    if let Some(s) = &n.content_rest.0 {
                        let v1 = s.to_instr_rec(instr, seen);
                        instr.push(Instruction::Mul(vec![vp, v1]));
                        seen.insert(b, instr.len() - 1);
                        Variable::Index(instr.len() - 1)
                    } else {
                        vp
                    }
                };

                let vr = if let Some(s) = &n.content_rest.1 {
                    let v2 = s.to_instr_rec(instr, seen);
                    instr.push(Instruction::Add(vec![v1, v2]));
                    seen.insert(nb, instr.len() - 1);
                    Variable::Index(instr.len() - 1)
                } else {
                    v1
                };

                if n.gcd != Rational::one() {
                    instr.push(Instruction::Mul(vec![
                        vr,
                        Variable::Constant(n.gcd.clone()),
                    ]));
                    Variable::Index(instr.len() - 1)
                } else {
                    vr
                }
            }
            HornerScheme::Leaf(_, l) => Variable::Constant(l.clone()),
        }
    }
}

// An arithmetical instruction that is part of an `InstructionList`.
#[derive(Debug)]
pub enum Instruction<N: NumericalFloatLike> {
    Add(Vec<Variable<N>>),
    Mul(Vec<Variable<N>>),
    Yield(Variable<N>),
    Empty,
}

// An variable that is part of an `InstructionList`,
// which may refer to another instruction in the instruction list.
#[derive(Debug, Clone, PartialEq)]
pub enum Variable<N: NumericalFloatLike> {
    Index(usize),
    Var(usize, usize), // var^pow
    Constant(N),
}

impl<N: NumericalFloatLike> PartialOrd for Variable<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Variable::Index(i1), Variable::Index(i2)) => i1.partial_cmp(i2),
            (Variable::Index(_), _) => Some(std::cmp::Ordering::Less),
            (_, Variable::Index(_)) => Some(std::cmp::Ordering::Greater),
            (Variable::Var(v1, _), Variable::Var(v2, _)) => v1.partial_cmp(v2),
            (Variable::Var(_, _), _) => Some(std::cmp::Ordering::Less),
            (_, Variable::Var(_, _)) => Some(std::cmp::Ordering::Greater),
            (Variable::Constant(_), Variable::Constant(_)) => None, // should never be needed
        }
    }
}

impl<N: NumericalFloatLike + Eq> Eq for Variable<N> {}

impl<N: NumericalFloatLike + Eq + Hash> Ord for Variable<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<N: NumericalFloatLike + Eq + Hash> Hash for Variable<N> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Variable::Index(i) => i.hash(state),
            Variable::Var(v, p) => {
                v.hash(state);
                p.hash(state);
            }
            Variable::Constant(c) => c.hash(state),
        }
    }
}

impl Variable<Rational> {
    fn to_pretty_string(
        &self,
        var_map: &[Identifier],
        state: &State,
        mode: InstructionSetMode,
    ) -> String {
        match self {
            Variable::Index(i) => format!("Z{}", i),
            Variable::Var(v, p) => {
                let var = state.get_name(var_map[*v]).unwrap();
                if *p == 1 {
                    format!("{}", var.as_str())
                } else if *p == 2 {
                    format!("{0}*{0}", var)
                } else {
                    match mode {
                        InstructionSetMode::Plain => format!("{}^{}", v, p),
                        InstructionSetMode::CPP(_) => format!("std::pow({},{})", v, p),
                    }
                }
            }
            Variable::Constant(c) => match mode {
                InstructionSetMode::Plain => format!("{}", c),
                InstructionSetMode::CPP(_) => {
                    if c.is_integer() {
                        format!("T({})", c.numerator())
                    } else {
                        format!("T({})/T({})", c.numerator(), c.denominator())
                    }
                }
            },
        }
    }
}

impl<N: NumericalFloatLike> std::fmt::Display for Variable<N> {
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

impl<N: NumericalFloatLike> Variable<N> {
    pub fn convert<'a, NO: NumericalFloatLike + From<&'a N>>(&'a self) -> Variable<NO> {
        match self {
            Variable::Index(i) => Variable::Index(*i),
            Variable::Var(v, p) => Variable::Var(*v, *p),
            Variable::Constant(c) => Variable::Constant(NO::from(c)),
        }
    }

    #[inline(always)]
    pub fn evaluate(&self, samples: &[N], instr_eval: &[N]) -> N {
        // TODO: write everything in registers to avoid branching
        match self {
            Variable::Index(i) => instr_eval[*i].clone(),
            Variable::Var(v, p) => {
                if *p == 2 {
                    samples[*v].mul(&samples[*v])
                } else if *p == 3 {
                    samples[*v].mul(&samples[*v]).mul(&samples[*v])
                } else {
                    samples[*v].pow(*p as u64)
                }
            }
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
    instr: Vec<Instruction<Rational>>,
}

impl InstructionList {
    /// Evaluate the instructions and yield the result.
    /// For a more efficient evaluation, call `to_output()` first.
    pub fn evaluate(&self, samples: &[Rational]) -> Rational {
        let mut eval: Vec<Rational> = vec![Rational::zero(); self.instr.len() + 1];

        for (reg, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => {
                    let mut r = Rational::zero();
                    for x in a {
                        r += &x.evaluate(samples, &eval);
                    }
                    eval[reg] = r;
                }
                Instruction::Mul(m) => {
                    let mut r = Rational::one();
                    for x in m {
                        r *= &x.evaluate(samples, &eval);
                    }
                    eval[reg] = r;
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
        for x in &self.instr {
            sum += match x {
                Instruction::Add(a) => a.iter().map(|x| x.op_count(false)).sum::<isize>() - 1,
                Instruction::Mul(m) => m.iter().map(|x| x.op_count(true)).sum::<isize>() - 1,
                Instruction::Yield(_) => 0,
                Instruction::Empty => 0,
            };
        }
        sum
    }

    /// Fuse `Z1=a+b`, `Z2=Z1+c` to `Z1=a+b+c` if `Z1` is only used in `Z2`
    pub fn fuse_operations(&mut self) {
        let mut use_count: Vec<usize> = vec![0; self.instr.len()];

        for x in &self.instr {
            match x {
                Instruction::Add(a) => {
                    for v in a {
                        if let Variable::Index(ii) = v {
                            if let Instruction::Mul(_) = self.instr[*ii] {
                                use_count[*ii] = 2; // different type, so disable
                            } else {
                                use_count[*ii] += 1;
                            }
                        }
                    }
                }
                Instruction::Mul(m) => {
                    for v in m {
                        if let Variable::Index(ii) = v {
                            if let Instruction::Add(_) = self.instr[*ii] {
                                use_count[*ii] = 2;
                            } else {
                                use_count[*ii] += 1;
                            }
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        use_count[*ii] = 2; // always different type
                    }
                }
                Instruction::Empty => {}
            };
        }

        for i in 0..self.instr.len() {
            // we could be in chain of single use -> single use -> etc so work from the start
            if let Instruction::Add(a) | Instruction::Mul(a) = &self.instr[i] {
                if a.iter().any(|v| {
                    if let Variable::Index(ii) = v {
                        use_count[*ii] == 1
                    } else {
                        false
                    }
                }) {
                    let mut instr = std::mem::replace(&mut self.instr[i], Instruction::Empty);

                    if let Instruction::Add(a) | Instruction::Mul(a) = &mut instr {
                        let mut new_a = Vec::with_capacity(a.len());
                        for v in a.drain(..) {
                            if let Variable::Index(ii) = v {
                                if use_count[ii] == 1 {
                                    if let Instruction::Add(aa) | Instruction::Mul(aa) =
                                        &self.instr[ii]
                                    {
                                        for x in aa {
                                            new_a.push(x.clone());
                                        }
                                        self.instr[ii] = Instruction::Empty;
                                    } else {
                                        unreachable!()
                                    }
                                } else {
                                    new_a.push(v);
                                }
                            } else {
                                new_a.push(v);
                            }
                        }
                        new_a.sort();
                        *a = new_a;
                    }

                    self.instr[i] = instr;
                }
            }
        }

        self.remove_empty_ops();
    }

    /// Remove empty instructions from the list.
    fn remove_empty_ops(&mut self) {
        // now remove the old labels and renumber all
        let mut num_removed_entries = 0;
        let mut new_instr = Vec::with_capacity(self.instr.len());
        let mut cum_step = vec![0; self.instr.len()];
        for (i, mut x) in std::mem::take(&mut self.instr).into_iter().enumerate() {
            cum_step[i] = num_removed_entries;
            if let Instruction::Empty = x {
                num_removed_entries += 1;
                continue;
            }

            match &mut x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        if let Variable::Index(ii) = v {
                            *ii -= cum_step[*ii];
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        *ii -= cum_step[*ii];
                    }
                }
                Instruction::Empty => {}
            };
            new_instr.push(x);
        }

        self.instr = new_instr;
    }

    /// Find and extract pairs of variables that appear in more than one instruction.
    /// This reduces the number of operations. Returns `true` iff an extraction could be performed.
    ///
    /// This function can be called multiple times such that common subexpressions that
    /// are larger than pairs can also be extracted.
    pub fn common_pair_elimination(&mut self) -> bool {
        let mut pair_count = HashMap::default();

        for x in &self.instr {
            match x {
                Instruction::Add(m) | Instruction::Mul(m) => {
                    for i in 0..m.len() - 1 {
                        for j in &m[i + 1..] {
                            *pair_count
                                .entry((matches!(x, Instruction::Add(_)), m[i].clone(), j.clone()))
                                .or_insert(0) += 1;
                        }
                    }
                }
                Instruction::Yield(_) | Instruction::Empty => {}
            };
        }

        let mut v: Vec<_> = pair_count.into_iter().collect();
        v.sort_by_key(|k| Reverse(k.1));

        if v.len() == 0 || v[0].1 < 2 {
            return false;
        }

        // TODO: instead of only doing the first, do all non-overlapping replacements

        let insert_index = match (&v[0].0 .1, &v[0].0 .2) {
            (Variable::Index(i1), Variable::Index(i2)) => i1.max(i2) + 1,
            (Variable::Index(i1), _) => i1 + 1,
            (_, Variable::Index(i2)) => i2 + 1,
            _ => 0,
        };

        self.instr.insert(
            insert_index,
            if v[0].0 .0 {
                Instruction::Add(vec![v[0].0 .1.clone(), v[0].0 .2.clone()])
            } else {
                Instruction::Mul(vec![v[0].0 .1.clone(), v[0].0 .2.clone()])
            },
        );

        // now remove the old labels and renumber all
        for (i, x) in self.instr.iter_mut().enumerate() {
            if i <= insert_index {
                continue;
            }

            match x {
                Instruction::Add(a) => {
                    let hit = if v[0].0 .0 && a.contains(&v[0].0 .1) && a.contains(&v[0].0 .2) {
                        a.retain(|x| x != &v[0].0 .1 && x != &v[0].0 .2);
                        true
                    } else {
                        false
                    };

                    for v in &mut *a {
                        if let Variable::Index(ii) = v {
                            if *ii >= insert_index {
                                *ii += 1;
                            }
                        }
                    }

                    if hit {
                        a.push(Variable::Index(insert_index));
                    }
                }
                Instruction::Mul(m) => {
                    let hit = if !v[0].0 .0 && m.contains(&v[0].0 .1) && m.contains(&v[0].0 .2) {
                        m.retain(|x| x != &v[0].0 .1 && x != &v[0].0 .2);
                        true
                    } else {
                        false
                    };

                    for v in &mut *m {
                        if let Variable::Index(ii) = v {
                            if *ii >= insert_index {
                                *ii += 1;
                            }
                        }
                    }

                    if hit {
                        m.push(Variable::Index(insert_index));
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        if *ii >= insert_index {
                            *ii += 1;
                        }
                    }
                }
                Instruction::Empty => {}
            };
        }

        // remove Z2=Z1
        let mut map: Vec<_> = (0..self.instr.len()).collect();
        for (i, x) in self.instr.iter_mut().enumerate() {
            match x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    if a.len() == 1 {
                        if let Variable::Index(ii) = a[0] {
                            map[i] = ii;
                            *x = Instruction::Empty;
                            continue;
                        }
                    }

                    for v in a {
                        if let Variable::Index(ii) = v {
                            *ii = map[*ii];
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        *ii = map[*ii];
                    }
                }
                Instruction::Empty => {}
            };
        }

        self.remove_empty_ops();

        true
    }

    /// Convert the instruction list into its output format, where
    /// intermediate registers can be recycled. This vastly improves
    /// the computational memory needed for evaluations.
    pub fn to_output(self, recycle_registers: bool) -> InstructionListOutput<Rational> {
        if !recycle_registers {
            return InstructionListOutput {
                instr: self.instr.into_iter().enumerate().collect(),
            };
        }

        let mut last_use: Vec<usize> = (0..self.instr.len()).collect();

        for (i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        if let Variable::Index(ii) = v {
                            last_use[*ii] = i;
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        last_use[*ii] = i;
                    }
                }
                Instruction::Empty => {}
            };
        }

        let mut rename_map: Vec<_> = (0..self.instr.len()).collect();

        let mut output = Vec::with_capacity(self.instr.len());

        for (i, mut x) in self.instr.into_iter().enumerate() {
            let cur_last_use = last_use[i];
            // find first free variable
            let reg = if let Some((new_v, lu)) =
                last_use[..i].iter_mut().enumerate().find(|(_, r)| **r < i)
            {
                *lu = cur_last_use; // set the last use to the current variable last use
                last_use[i] = 0; // make the current index available
                rename_map[i] = new_v; // set the rename map so that every occurrence on the rhs is replaced
                new_v
            } else {
                i
            };

            match &mut x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        if let Variable::Index(ii) = v {
                            *ii = rename_map[*ii];
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if let Variable::Index(ii) = v {
                        *ii = rename_map[*ii];
                    }
                }
                Instruction::Empty => {}
            };

            output.push((reg, x));
        }

        InstructionListOutput { instr: output }
    }
}

/// A list of instructions suitable for fast numerical evaluation.
pub struct InstructionListOutput<N: NumericalFloatLike> {
    instr: Vec<(usize, Instruction<N>)>,
}

impl<N: NumericalFloatLike> InstructionListOutput<N> {
    /// Convert all numbers in the instruction list from the field `N` to the field `NO`.
    pub fn convert<'a, NO: NumericalFloatLike + From<&'a N>>(
        &'a self,
    ) -> InstructionListOutput<NO> {
        let mut instr = Vec::with_capacity(self.instr.len());

        for (reg, inst) in &self.instr {
            let new_instr = match inst {
                Instruction::Add(a) => Instruction::Add(a.iter().map(|v| v.convert()).collect()),
                Instruction::Mul(a) => Instruction::Mul(a.iter().map(|v| v.convert()).collect()),
                Instruction::Yield(y) => Instruction::Yield(y.convert()),
                Instruction::Empty => unreachable!("No empty slots allowed in output"),
            };
            instr.push((*reg, new_instr));
        }
        InstructionListOutput { instr }
    }

    /// Evaluate the instructions and yield the result.
    pub fn evaluate(&self, samples: &[N]) -> N {
        let max_register = self.instr.iter().map(|r| r.0).max().unwrap_or(0);
        let mut eval: Vec<N> = vec![N::zero(); max_register + 1];

        for (reg, x) in &self.instr {
            match x {
                Instruction::Add(a) => {
                    let mut r = N::zero();
                    for x in a {
                        r += &x.evaluate(samples, &eval);
                    }
                    eval[*reg] = r;
                }
                Instruction::Mul(m) => {
                    let mut r = N::one();
                    for x in m {
                        r *= &x.evaluate(samples, &eval);
                    }
                    eval[*reg] = r;
                }
                Instruction::Yield(y) => return y.evaluate(samples, &eval),
                Instruction::Empty => panic!("Evaluation of empty instruction requested"),
            }
        }
        unreachable!()
    }

    pub fn evaluate_faster(&self, samples: &[N], eval: &mut Vec<N>) -> N {
        let max_register = self.instr.iter().map(|r| r.0).max().unwrap_or(0);
        eval.clear();
        eval.resize(max_register + 1, N::zero());

        let mut tmp;
        for (reg, x) in &self.instr {
            match x {
                Instruction::Add(a) => {
                    tmp = N::zero();
                    for x in a {
                        tmp += &x.evaluate(samples, &eval);
                    }
                    eval[*reg] = tmp;
                }
                Instruction::Mul(m) => {
                    tmp = N::one();
                    for x in m {
                        tmp *= &x.evaluate(samples, &eval);
                    }
                    eval[*reg] = tmp;
                }
                Instruction::Yield(y) => return y.evaluate(samples, &eval),
                Instruction::Empty => {}
            }
        }
        N::zero() // unreachable
    }
}

impl std::fmt::Display for InstructionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (reg, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    reg,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    reg,
                    m.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => f.write_fmt(format_args!("Z{} = {};\n", reg, y))?,
                Instruction::Empty => f.write_fmt(format_args!("Z{} = NOP;\n", reg))?,
            }
        }

        Ok(())
    }
}

pub struct CPPPrinter {}

impl CPPPrinter {
    pub fn format_number(num: &Rational) -> String {
        if num.is_integer() {
            format!("T({})", num.numerator())
        } else {
            format!("T({})/T({})", num.numerator(), num.denominator())
        }
    }
}

#[derive(Clone, Copy)]
pub struct InstructionSetModeCPPSettings {
    pub write_header_and_test: bool,
}

#[derive(Clone, Copy)]
pub enum InstructionSetMode {
    Plain,
    CPP(InstructionSetModeCPPSettings),
}

pub struct InstructionSetPrinter<'a> {
    pub instr: &'a InstructionListOutput<Rational>,
    pub var_map: &'a [Identifier],
    pub state: &'a State,
    pub mode: InstructionSetMode,
}

impl<'a> InstructionSetPrinter<'a> {
    /*pub fn new(inst) -> InstructionSetPrinter<'a, N> {
        InstructionSetPrinter { instr: (), var_map: (), state: (), mode: () }
    }*/
}

impl<'a> std::fmt::Display for InstructionSetPrinter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let InstructionSetMode::CPP(s) = self.mode {
            if s.write_header_and_test {
                f.write_str("#include <cstdlib>\n")?;
                f.write_str("#include <cmath>\n")?;
                f.write_str("#include <iostream>\n")?;
                f.write_str("\n")?;
            }

            f.write_str("template<typename T>\n")?;
            f.write_fmt(format_args!(
                "T evaluate({}) {{\n",
                self.var_map
                    .iter()
                    .map(|x| format!("T {}", self.state.get_name(*x).unwrap()))
                    .collect::<Vec<_>>()
                    .join(","),
            ))?;

            let max_register = self.instr.instr.iter().map(|r| r.0).max().unwrap_or(0);
            f.write_fmt(format_args!(
                "\tT {};\n",
                (0..=max_register)
                    .map(|x| format!("Z{}", x))
                    .collect::<Vec<_>>()
                    .join(","),
            ))?;
        }

        for (reg, x) in &self.instr.instr {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "\tZ{} = {};\n",
                    reg,
                    a.iter()
                        .map(|x| x.to_pretty_string(self.var_map, self.state, self.mode))
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "\tZ{} = {};\n",
                    reg,
                    m.iter()
                        .map(|x| x.to_pretty_string(self.var_map, self.state, self.mode))
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => match self.mode {
                    InstructionSetMode::Plain => {
                        f.write_fmt(format_args!("Z{} = {};\n", reg, y))?
                    }
                    InstructionSetMode::CPP(_) => f.write_fmt(format_args!("return {};\n", y))?,
                },
                Instruction::Empty => f.write_fmt(format_args!("Z{} = NOP;\n", reg))?,
            }
        }

        if let InstructionSetMode::CPP(s) = self.mode {
            f.write_str("}\n")?;

            if s.write_header_and_test {
                let points: Vec<_> = (0..self.var_map.len())
                    .map(|i| ((i + 1) as f64 / (self.var_map.len() + 2) as f64).to_string())
                    .collect();

                f.write_fmt(format_args!(
                    "\nint main() {{\n\tstd::cout << evaluate({}) << std::endl;\n}}",
                    points.join(",")
                ))?;
            }
        }

        Ok(())
    }
}
