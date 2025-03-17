//! Defines transformations uses to manipulate expressions.
//!
//! Used primarily in the Python API.

use std::{ops::ControlFlow, sync::Arc, time::Instant};

use crate::{
    atom::{representation::FunView, Atom, AtomView, Fun, Symbol},
    coefficient::{Coefficient, CoefficientView},
    combinatorics::{partitions, unique_permutations},
    domains::rational::Rational,
    id::{
        Condition, Evaluate, MatchSettings, Pattern, PatternRestriction, Relation, ReplaceWith,
        Replacement,
    },
    printer::{AtomPrinter, PrintOptions},
    state::{RecycledAtom, Workspace},
};
use ahash::HashMap;
use colored::Colorize;
use dyn_clone::DynClone;
use rayon::ThreadPool;

/// A function that maps an atom(view) to another atom.
pub trait Map:
    Fn(AtomView, &mut Atom) -> Result<(), TransformerError> + DynClone + Send + Sync
{
}
dyn_clone::clone_trait_object!(Map);
impl<T: Clone + Send + Sync + Fn(AtomView<'_>, &mut Atom) -> Result<(), TransformerError>> Map
    for T
{
}

/// Options for printing statistics of operations used in
/// [Transformer::Stats].
#[derive(Clone, Debug)]
pub struct StatsOptions {
    pub tag: String,
    pub color_medium_change_threshold: Option<f64>,
    pub color_large_change_threshold: Option<f64>,
}

impl StatsOptions {
    pub fn format_size(&self, size: usize) -> String {
        let mut s = size as f64;
        let kb = 1024.;
        let tag = [" ", "K", "M", "G", "T"];

        for t in tag {
            if s < kb {
                return format!("{:.2}{}B", s, t);
            }

            s /= kb;
        }

        format!("{:.2}EB", s)
    }

    pub fn format_count(&self, count: usize) -> String {
        format!("{}", count)
    }

    pub fn print(&self, input: AtomView, output: AtomView, dt: std::time::Duration) {
        let in_nterms = if let AtomView::Add(a) = input {
            a.get_nargs()
        } else {
            1
        };
        let in_size = input.get_byte_size();

        let out_nterms = if let AtomView::Add(a) = output {
            a.get_nargs()
        } else {
            1
        };
        let out_size = output.get_byte_size();

        let in_nterms_s = self.format_count(in_nterms);
        let out_nterms_s = self.format_count(out_nterms);

        println!(
            "Stats for {}:
\tIn  │ {:>width$} │ {:>8} │
\tOut │ {:>width$} │ {:>8} │ ⧗ {:#.2?}",
            self.tag.bold(),
            in_nterms_s,
            self.format_size(in_size),
            if out_nterms as f64 / in_nterms as f64
                > self.color_medium_change_threshold.unwrap_or(f64::INFINITY)
            {
                if out_nterms as f64 / in_nterms as f64
                    > self.color_large_change_threshold.unwrap_or(f64::INFINITY)
                {
                    out_nterms_s.red()
                } else {
                    out_nterms_s.bright_magenta()
                }
            } else {
                out_nterms_s.as_str().into()
            },
            self.format_size(out_size),
            dt,
            width = in_nterms_s.len().max(out_nterms_s.len()).min(6),
        );
    }
}

/// Errors that can occur during transformations.
#[derive(Clone, Debug)]
pub enum TransformerError {
    ValueError(String),
    Interrupt,
}

/// Operations that take an expression as the input and produce a new expression.
#[derive(Clone)]
pub enum Transformer {
    IfElse(Condition<Relation>, Vec<Transformer>, Vec<Transformer>),
    IfChanged(Vec<Transformer>, Vec<Transformer>, Vec<Transformer>),
    BreakChain,
    /// Expand the rhs.
    Expand(Option<Atom>, bool),
    /// Distribute numbers.
    ExpandNum,
    /// Derive the rhs w.r.t a variable.
    Derivative(Symbol),
    /// Perform a series expansion.
    Series(Symbol, Atom, Rational, bool),
    ///Collect all terms in powers of a variable.
    Collect(Vec<Atom>, Vec<Transformer>, Vec<Transformer>),
    ///Collect all terms in powers of a variable name.
    CollectSymbol(Symbol, Vec<Transformer>, Vec<Transformer>),
    /// Collect common factors from (nested) sums.
    CollectFactors,
    /// Collect numbers.
    CollectNum,
    /// Apply find-and-replace on the lhs.
    ReplaceAll(
        Pattern,
        ReplaceWith<'static>,
        Condition<PatternRestriction>,
        MatchSettings,
    ),
    /// Apply multiple find-and-replace on the lhs.
    ReplaceAllMultiple(Vec<Replacement>),
    /// Take the product of a list of arguments in the rhs.
    Product,
    /// Take the sum of a list of arguments in the rhs.
    Sum,
    /// Return the number of arguments of a function in the input.
    /// If the argument of `ArgCount` is `true`, only the number
    /// of arguments of `arg()` is returned and 1 is returned otherwise.
    /// If the argument is `false`, 0 is returned for non-functions.
    ArgCount(bool),
    /// Linearize a function, optionally extracting `symbols` as well.
    Linearize(Option<Vec<Symbol>>),
    /// Map the rhs with a user-specified function.
    Map(Box<dyn Map>),
    /// Apply a transformation to each argument of the `arg()` function.
    /// If the input is not `arg()`, map the current input.
    ForEach(Vec<Transformer>),
    /// Map the transformers over the terms, potentially in parallel
    MapTerms(Vec<Transformer>, Option<Arc<ThreadPool>>),
    /// Split a `Mul` or `Add` into a list of arguments.
    Split,
    Partition(Vec<(Symbol, usize)>, bool, bool),
    Sort,
    CycleSymmetrize,
    Deduplicate,
    Permutations(Symbol),
    Repeat(Vec<Transformer>),
    Print(PrintOptions),
    Stats(StatsOptions, Vec<Transformer>),
    FromNumber,
}

impl std::fmt::Debug for Transformer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transformer::IfElse(_, _, _) => f.debug_tuple("IfElse").finish(),
            Transformer::IfChanged(_, _, _) => f.debug_tuple("IfChanged").finish(),
            Transformer::BreakChain => f.debug_tuple("BreakChain").finish(),
            Transformer::Expand(s, _) => f.debug_tuple("Expand").field(s).finish(),
            Transformer::ExpandNum => f.debug_tuple("ExpandNum").finish(),
            Transformer::Derivative(x) => f.debug_tuple("Derivative").field(x).finish(),
            Transformer::Collect(x, a, b) => {
                f.debug_tuple("Collect").field(x).field(a).field(b).finish()
            }
            Transformer::CollectFactors => f.debug_tuple("CollectFactors").finish(),
            Transformer::CollectSymbol(x, a, b) => f
                .debug_tuple("CollectSymbol")
                .field(x)
                .field(a)
                .field(b)
                .finish(),
            Transformer::CollectNum => f.debug_tuple("CollectNum").finish(),
            Transformer::ReplaceAll(pat, rhs, ..) => {
                f.debug_tuple("ReplaceAll").field(pat).field(rhs).finish()
            }
            Transformer::ReplaceAllMultiple(pats) => {
                f.debug_tuple("ReplaceAllMultiple").field(pats).finish()
            }
            Transformer::Product => f.debug_tuple("Product").finish(),
            Transformer::Sum => f.debug_tuple("Sum").finish(),
            Transformer::ArgCount(p) => f.debug_tuple("ArgCount").field(p).finish(),
            Transformer::Linearize(s) => f.debug_tuple("Linearize").field(s).finish(),
            Transformer::Map(_) => f.debug_tuple("Map").finish(),
            Transformer::MapTerms(v, c) => f.debug_tuple("Map").field(v).field(c).finish(),
            Transformer::ForEach(t) => f.debug_tuple("ForEach").field(t).finish(),
            Transformer::Split => f.debug_tuple("Split").finish(),
            Transformer::Partition(g, b1, b2) => f
                .debug_tuple("Partition")
                .field(g)
                .field(b1)
                .field(b2)
                .finish(),
            Transformer::Sort => f.debug_tuple("Sort").finish(),
            Transformer::CycleSymmetrize => f.debug_tuple("CycleSymmetrize").finish(),
            Transformer::Deduplicate => f.debug_tuple("Deduplicate").finish(),
            Transformer::Permutations(i) => f.debug_tuple("Permutations").field(i).finish(),
            Transformer::Series(x, point, d, depth_is_absolute) => f
                .debug_tuple("TaylorSeries")
                .field(x)
                .field(point)
                .field(d)
                .field(depth_is_absolute)
                .finish(),
            Transformer::Repeat(r) => f.debug_tuple("Repeat").field(r).finish(),
            Transformer::Print(p) => f.debug_tuple("Print").field(p).finish(),
            Transformer::Stats(o, r) => f.debug_tuple("Timing").field(o).field(r).finish(),
            Transformer::FromNumber => f.debug_tuple("FromNumber").finish(),
        }
    }
}

impl FunView<'_> {
    /// Linearize a function, optionally extracting `symbols` as well.
    pub fn linearize(&self, symbols: Option<&[Symbol]>) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            self.linearize_impl(symbols, ws, &mut out);
        });
        out
    }

    fn linearize_impl(&self, symbols: Option<&[Symbol]>, workspace: &Workspace, out: &mut Atom) {
        /// Add an argument `a` to `f` and flatten nested `arg`s.
        #[inline(always)]
        fn add_arg(f: &mut Fun, a: AtomView) {
            if let AtomView::Fun(fa) = a {
                if fa.get_symbol() == Atom::ARG {
                    // flatten f(arg(...)) = f(...)
                    for aa in fa.iter() {
                        f.add_arg(aa);
                    }

                    return;
                }
            }

            f.add_arg(a);
        }

        /// Take Cartesian product of arguments
        #[inline(always)]
        fn cartesian_product<'b>(
            workspace: &Workspace,
            list: &[Vec<AtomView<'b>>],
            fun_name: Symbol,
            cur: &mut Vec<AtomView<'b>>,
            acc: &mut Vec<RecycledAtom>,
        ) {
            if list.is_empty() {
                let mut h = workspace.new_atom();
                let f = h.to_fun(fun_name);
                for a in cur.iter() {
                    add_arg(f, *a);
                }
                acc.push(h);
                return;
            }

            for a in &list[0] {
                cur.push(*a);
                cartesian_product(workspace, &list[1..], fun_name, cur, acc);
                cur.pop();
            }
        }

        if self.iter().any(|a| matches!(a, AtomView::Add(_))) {
            let mut arg_buf = Vec::with_capacity(self.get_nargs());

            for a in self.iter() {
                let mut vec = vec![];
                if let AtomView::Add(aa) = a {
                    for a in aa.iter() {
                        vec.push(a);
                    }
                } else {
                    vec.push(a);
                }
                arg_buf.push(vec);
            }

            let mut acc = Vec::new();
            cartesian_product(
                workspace,
                &arg_buf,
                self.get_symbol(),
                &mut vec![],
                &mut acc,
            );

            let mut add_h = workspace.new_atom();
            let add = add_h.to_add();

            let mut h = workspace.new_atom();
            for a in acc {
                a.as_view().normalize(workspace, &mut h);

                if let AtomView::Fun(ff) = h.as_view() {
                    let mut h2 = workspace.new_atom();
                    ff.linearize_impl(symbols, workspace, &mut h2);
                    add.extend(h2.as_view());
                } else {
                    add.extend(h.as_view());
                }
            }

            add_h.as_view().normalize(workspace, out);
            return;
        }

        // linearize products
        if self.iter().any(|a| {
            symbols.is_some()
                || if let AtomView::Mul(m) = a {
                    m.has_coefficient()
                } else {
                    false
                }
        }) {
            let mut new_term = workspace.new_atom();
            let t = new_term.to_mul();
            let mut new_fun = workspace.new_atom();
            let nf = new_fun.to_fun(self.get_symbol());

            let mut coeff = workspace.new_atom();
            let c = coeff.to_mul();
            for a in self.iter() {
                if let AtomView::Mul(m) = a {
                    if m.has_coefficient() || symbols.is_some() {
                        let mut stripped = workspace.new_atom();
                        let mul = stripped.to_mul();

                        for a in m {
                            if let AtomView::Num(_) = a {
                                c.extend(a);
                            } else if let AtomView::Var(v) = a {
                                let s = v.get_symbol();
                                if symbols.map(|x| x.contains(&s)).unwrap_or(false) {
                                    c.extend(a);
                                } else {
                                    mul.extend(a);
                                }
                            } else if let AtomView::Pow(p) = a {
                                if let AtomView::Var(v) = p.get_base() {
                                    let s = v.get_symbol();
                                    if symbols.map(|x| x.contains(&s)).unwrap_or(false) {
                                        c.extend(a);
                                    } else {
                                        mul.extend(a);
                                    }
                                } else {
                                    mul.extend(a);
                                }
                            } else {
                                mul.extend(a);
                            }
                        }

                        nf.add_arg(stripped.as_view());
                    } else {
                        nf.add_arg(a);
                    }
                } else {
                    nf.add_arg(a);
                }
            }

            t.extend(new_fun.as_view());
            t.extend(coeff.as_view());
            t.as_view().normalize(workspace, out);
        } else {
            out.set_from_view(&self.as_view());
        }
    }
}

impl Transformer {
    /// Create a new partition transformer that must exactly fit the input.
    pub fn new_partition_exact(partitions: Vec<(Symbol, usize)>) -> Transformer {
        Transformer::Partition(partitions, false, false)
    }

    /// Create a new partition transformer that collects all left-over
    /// atoms in the last bin.
    pub fn new_partition_collect_in_last(
        mut partitions: Vec<(Symbol, usize)>,
        rest: Symbol,
    ) -> Transformer {
        partitions.push((rest, 0));
        Transformer::Partition(partitions, true, false)
    }

    /// Create a new partition transformer that repeats the partitions so that it can fit
    /// the input.
    pub fn new_partition_repeat(partition: (Symbol, usize)) -> Transformer {
        Transformer::Partition(vec![partition], false, true)
    }

    /// Apply the transformer to `input`.
    pub fn execute(&self, input: AtomView<'_>) -> Result<Atom, TransformerError> {
        let mut a = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(input, std::slice::from_ref(self), ws, &mut a).map_err(|e| e)
        })?;
        Ok(a)
    }

    /// Apply the transformer to `input`.
    pub fn execute_with_ws(
        &self,
        input: AtomView<'_>,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<ControlFlow<()>, TransformerError> {
        Transformer::execute_chain(input, std::slice::from_ref(self), workspace, out)
    }

    /// Apply a chain of transformers to `input`.
    pub fn execute_chain(
        input: AtomView<'_>,
        chain: &[Transformer],
        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<ControlFlow<()>, TransformerError> {
        out.set_from_view(&input);
        let mut tmp = workspace.new_atom();
        for t in chain {
            std::mem::swap(out, &mut tmp);
            let cur_input = tmp.as_view();

            match t {
                Transformer::IfElse(cond, t1, t2) => {
                    if cond
                        .evaluate(&Some(cur_input))
                        .map_err(|e| TransformerError::ValueError(e))?
                        .is_true()
                    {
                        if Transformer::execute_chain(cur_input, t1, workspace, out)?.is_break() {
                            return Ok(ControlFlow::Break(()));
                        }
                    } else if Transformer::execute_chain(cur_input, t2, workspace, out)?.is_break()
                    {
                        return Ok(ControlFlow::Break(()));
                    }
                }
                Transformer::IfChanged(cond, t1, t2) => {
                    Transformer::execute_chain(cur_input, cond, workspace, out)?;
                    std::mem::swap(out, &mut tmp);

                    if tmp.as_view() != out.as_view() {
                        if Transformer::execute_chain(tmp.as_view(), t1, workspace, out)?.is_break()
                        {
                            return Ok(ControlFlow::Break(()));
                        }
                    } else if Transformer::execute_chain(tmp.as_view(), t2, workspace, out)?
                        .is_break()
                    {
                        return Ok(ControlFlow::Break(()));
                    }
                }
                Transformer::BreakChain => {
                    std::mem::swap(out, &mut tmp);
                    return Ok(ControlFlow::Break(()));
                }
                Transformer::Map(f) => {
                    f(cur_input, out)?;
                }
                Transformer::MapTerms(t, p) => {
                    if let Some(p) = p {
                        *out = cur_input.map_terms_with_pool(
                            |arg| {
                                Workspace::get_local().with(|ws| {
                                    let mut a = Atom::new();
                                    Self::execute_chain(arg, t, ws, &mut a).unwrap();
                                    a
                                })
                            },
                            p,
                        );
                    } else {
                        *out = cur_input.map_terms_single_core(|arg| {
                            Workspace::get_local().with(|ws| {
                                let mut a = Atom::new();
                                Self::execute_chain(arg, t, ws, &mut a).unwrap();
                                a
                            })
                        })
                    }
                }
                Transformer::ForEach(t) => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let mut ff = workspace.new_atom();
                            let ff = ff.to_fun(Atom::ARG);

                            let mut a = workspace.new_atom();
                            for arg in f {
                                Self::execute_chain(arg, t, workspace, &mut a)?;
                                ff.add_arg(a.as_view());
                            }

                            ff.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    Self::execute_chain(cur_input, t, workspace, out)?;
                }
                Transformer::Expand(s, via_poly) => {
                    if *via_poly {
                        *out = cur_input.expand_via_poly::<u16>(s.as_ref().map(|x| x.as_view()));
                    } else {
                        cur_input.expand_with_ws_into(
                            workspace,
                            s.as_ref().map(|x| x.as_view()),
                            out,
                        );
                    }
                }
                Transformer::ExpandNum => {
                    cur_input.expand_num_into(out);
                }
                Transformer::Derivative(x) => {
                    cur_input.derivative_with_ws_into(*x, workspace, out);
                }
                Transformer::Collect(x, key_map, coeff_map) => cur_input
                    .collect_multiple_impl::<i16, _>(
                        x,
                        workspace,
                        if key_map.is_empty() {
                            None
                        } else {
                            let key_map = key_map.clone();
                            Some(Box::new(move |i, o| {
                                Workspace::get_local()
                                    .with(|ws| Self::execute_chain(i, &key_map, ws, o).unwrap());
                            }))
                        },
                        if coeff_map.is_empty() {
                            None
                        } else {
                            let coeff_map = coeff_map.clone();
                            Some(Box::new(move |i, o| {
                                Workspace::get_local()
                                    .with(|ws| Self::execute_chain(i, &coeff_map, ws, o).unwrap());
                            }))
                        },
                        out,
                    ),
                Transformer::CollectSymbol(x, key_map, coeff_map) => {
                    *out = cur_input.collect_symbol::<i16>(
                        *x,
                        if key_map.is_empty() {
                            None
                        } else {
                            let key_map = key_map.clone();
                            Some(Box::new(move |i, o| {
                                Workspace::get_local()
                                    .with(|ws| Self::execute_chain(i, &key_map, ws, o).unwrap());
                            }))
                        },
                        if coeff_map.is_empty() {
                            None
                        } else {
                            let coeff_map = coeff_map.clone();
                            Some(Box::new(move |i, o| {
                                Workspace::get_local()
                                    .with(|ws| Self::execute_chain(i, &coeff_map, ws, o).unwrap());
                            }))
                        },
                    );
                }
                Transformer::CollectFactors => {
                    *out = cur_input.collect_factors();
                }
                Transformer::CollectNum => {
                    *out = cur_input.collect_num();
                }
                Transformer::Series(x, expansion_point, depth, depth_is_absolute) => {
                    if let Ok(s) = cur_input.series(
                        *x,
                        expansion_point.as_view(),
                        depth.clone(),
                        *depth_is_absolute,
                    ) {
                        s.to_atom_into(out);
                    } else {
                        std::mem::swap(out, &mut tmp);
                    }
                }
                Transformer::ReplaceAll(pat, rhs, cond, settings) => {
                    cur_input.replace_with_ws_into(
                        pat,
                        rhs,
                        workspace,
                        cond.into(),
                        settings.into(),
                        out,
                    );
                }
                Transformer::ReplaceAllMultiple(replacements) => {
                    cur_input.replace_multiple_into(&replacements, out);
                }
                Transformer::Product => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let mut mul_h = workspace.new_atom();
                            let mul = mul_h.to_mul();

                            for arg in f {
                                mul.extend(arg);
                            }

                            mul_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::Sum => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let mut add_h = workspace.new_atom();
                            let add = add_h.to_add();

                            for arg in f {
                                add.extend(arg);
                            }

                            add_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::ArgCount(only_for_arg_fun) => {
                    if let AtomView::Fun(f) = cur_input {
                        if !*only_for_arg_fun || f.get_symbol() == Atom::ARG {
                            let n_args = f.get_nargs();
                            out.to_num((n_args as i64).into());
                        } else {
                            out.to_num(1.into());
                        }
                    } else if !only_for_arg_fun {
                        out.to_num(1.into());
                    } else {
                        out.to_num(Coefficient::zero());
                    }
                }
                Transformer::Linearize(symbols) => {
                    if let AtomView::Fun(f) = cur_input {
                        f.linearize_impl(symbols.as_ref().map(|x| x.as_slice()), workspace, out);
                    } else {
                        std::mem::swap(out, &mut tmp);
                    }
                }
                Transformer::Split => match cur_input {
                    AtomView::Mul(m) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun(Atom::ARG);

                        for factor in m {
                            arg.add_arg(factor);
                        }

                        arg_h.as_view().normalize(workspace, out);
                    }
                    AtomView::Add(a) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun(Atom::ARG);

                        for summand in a {
                            arg.add_arg(summand);
                        }

                        arg_h.as_view().normalize(workspace, out);
                    }
                    _ => {
                        std::mem::swap(out, &mut tmp);
                    }
                },
                Transformer::Partition(bins, fill_last, repeat) => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let args: Vec<_> = f.iter().collect();

                            let mut sum_h = workspace.new_atom();
                            let sum = sum_h.to_add();

                            let partitions = partitions(&args, bins, *fill_last, *repeat);

                            if partitions.is_empty() {
                                out.set_from_view(&workspace.new_num(0).as_view());
                                continue;
                            }

                            for (p, args) in partitions {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();

                                if !p.is_one() {
                                    mul.extend(workspace.new_num(p).as_view());
                                }

                                for (name, f_args) in args {
                                    let mut fun_h = workspace.new_atom();
                                    let fun = fun_h.to_fun(name);
                                    for x in f_args {
                                        fun.add_arg(x);
                                    }

                                    mul.extend(fun_h.as_view());
                                }

                                sum.extend(mul_h.as_view());
                            }

                            sum_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::Sort => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let mut args: Vec<_> = f.iter().collect();
                            args.sort();

                            let mut fun_h = workspace.new_atom();
                            let fun = fun_h.to_fun(Atom::ARG);

                            for arg in args {
                                fun.add_arg(arg);
                            }

                            fun_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::CycleSymmetrize => {
                    if let AtomView::Fun(f) = cur_input {
                        let args: Vec<_> = f.iter().collect();

                        let mut best_shift = 0;
                        'shift: for shift in 1..args.len() {
                            for i in 0..args.len() {
                                match args[(i + best_shift) % args.len()]
                                    .cmp(&args[(i + shift) % args.len()])
                                {
                                    std::cmp::Ordering::Equal => {}
                                    std::cmp::Ordering::Less => {
                                        continue 'shift;
                                    }
                                    std::cmp::Ordering::Greater => break,
                                }
                            }

                            best_shift = shift;
                        }

                        let mut fun_h = workspace.new_atom();
                        let fun = fun_h.to_fun(f.get_symbol());

                        for arg in args[best_shift..].iter().chain(&args[..best_shift]) {
                            fun.add_arg(*arg);
                        }

                        fun_h.as_view().normalize(workspace, out);
                    } else {
                        std::mem::swap(out, &mut tmp);
                    }
                }
                Transformer::Deduplicate => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let args: Vec<_> = f.iter().collect();
                            let mut args_dedup: Vec<_> = Vec::with_capacity(args.len());

                            for a in args {
                                // check last argument first, so that the sorted list case is fast
                                if args_dedup.last() != Some(&a) && !args_dedup.contains(&a) {
                                    args_dedup.push(a);
                                }
                            }

                            let mut fun_h = workspace.new_atom();
                            let fun = fun_h.to_fun(Atom::ARG);

                            for arg in args_dedup {
                                fun.add_arg(arg);
                            }

                            fun_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::Permutations(f_name) => {
                    if let AtomView::Fun(f) = cur_input {
                        if f.get_symbol() == Atom::ARG {
                            let args: Vec<_> = f.iter().collect();

                            let mut sum_h = workspace.new_atom();
                            let sum = sum_h.to_add();

                            let (prefactor, permutations) = unique_permutations(&args);

                            if permutations.is_empty() {
                                out.set_from_view(&workspace.new_num(0).as_view());
                                continue;
                            }

                            for a in permutations {
                                let mut fun_h = workspace.new_atom();
                                let fun = fun_h.to_fun(*f_name);
                                for x in a {
                                    fun.add_arg(x);
                                }

                                if !prefactor.is_one() {
                                    let mut mul_h = workspace.new_atom();
                                    let mul = mul_h.to_mul();
                                    mul.extend(fun_h.as_view());
                                    mul.extend(workspace.new_num(prefactor.clone()).as_view());
                                    sum.extend(mul_h.as_view());
                                } else {
                                    sum.extend(fun_h.as_view());
                                }
                            }

                            sum_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    std::mem::swap(out, &mut tmp);
                }
                Transformer::Repeat(r) => loop {
                    Self::execute_chain(tmp.as_view(), r, workspace, out)?;

                    if tmp.as_view() == out.as_view() {
                        break;
                    }

                    std::mem::swap(out, &mut tmp);
                },
                Transformer::Print(o) => {
                    println!("{}", AtomPrinter::new_with_options(cur_input, *o));
                    std::mem::swap(out, &mut tmp);
                }
                Transformer::Stats(o, r) => {
                    let t = Instant::now();
                    Self::execute_chain(cur_input, r, workspace, out)?;
                    let dt = t.elapsed();
                    o.print(cur_input, out.as_view(), dt);
                }
                Transformer::FromNumber => {
                    if let AtomView::Num(n) = cur_input {
                        if let CoefficientView::RationalPolynomial(r) = n.get_coeff_view() {
                            r.deserialize().to_expression_with_map(
                                workspace,
                                &HashMap::default(),
                                out,
                            );
                            continue;
                        }
                    }
                    std::mem::swap(out, &mut tmp);
                }
            }
        }

        Ok(ControlFlow::Continue(()))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore, FunctionBuilder},
        id::{Condition, Match, MatchSettings, WildcardRestriction},
        parse,
        printer::PrintOptions,
        state::Workspace,
        symbol,
        transformer::StatsOptions,
    };

    use super::Transformer;

    #[test]
    fn expand_derivative() {
        let p = parse!("(1+v1)^2").unwrap();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(
                p.as_view(),
                &[
                    Transformer::Expand(Some(Atom::new_var(symbol!("v1"))), false),
                    Transformer::Derivative(symbol!("v1")),
                ],
                ws,
                &mut out,
            )
            .unwrap()
        });

        let r = parse!("2+2*v1").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn split_argcount() {
        let p = parse!("v1+v2+v3").unwrap();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(
                p.as_view(),
                &[Transformer::Split, Transformer::ArgCount(true)],
                ws,
                &mut out,
            )
            .unwrap()
        });

        let r = parse!("3").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn product_series() {
        let p = parse!("arg(v1,v1+1,3)").unwrap();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(
                p.as_view(),
                &[
                    Transformer::Product,
                    Transformer::Series(symbol!("v1"), Atom::new_num(1), 3.into(), true),
                ],
                ws,
                &mut out,
            )
            .unwrap()
        });

        let r = parse!("3*(v1-1)^2+9*(v1-1)+6").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn sort_deduplicate() {
        let p = parse!("f1(3,2,1,3)").unwrap();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(
                p.as_view(),
                &[
                    Transformer::ReplaceAll(
                        parse!("f1(x__)").unwrap().to_pattern(),
                        parse!("x__").unwrap().to_pattern().into(),
                        Condition::default(),
                        MatchSettings::default(),
                    ),
                    Transformer::Sort,
                    Transformer::Deduplicate,
                    Transformer::Map(Box::new(|x, out| {
                        let mut f = FunctionBuilder::new(symbol!("f1"));
                        f = f.add_arg(x);
                        *out = f.finish();
                        Ok(())
                    })),
                ],
                ws,
                &mut out,
            )
            .unwrap()
        });

        let r = parse!("f1(1,2,3)").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn deep_nesting() {
        let p = parse!("arg(3,2,1,3)").unwrap();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            Transformer::execute_chain(
                p.as_view(),
                &[Transformer::Repeat(vec![Transformer::Stats(
                    StatsOptions {
                        tag: "test".to_owned(),
                        color_medium_change_threshold: Some(10.),
                        color_large_change_threshold: Some(100.),
                    },
                    vec![Transformer::ForEach(vec![
                        Transformer::Print(PrintOptions::default()),
                        Transformer::ReplaceAll(
                            parse!("x_").unwrap().to_pattern(),
                            parse!("x_-1").unwrap().to_pattern().into(),
                            (
                                symbol!("x_"),
                                WildcardRestriction::Filter(Box::new(|x| {
                                    x != &Match::Single(Atom::new_num(0).as_view())
                                })),
                            )
                                .into(),
                            MatchSettings::default(),
                        ),
                    ])],
                )])],
                ws,
                &mut out,
            )
            .unwrap()
        });

        let r = parse!("arg(0,0,0,0)").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn linearize() {
        let p = parse!("f1(v1+v2,4*v3*v4+3*v4/v3)").unwrap();

        let out = Transformer::Linearize(Some(vec![symbol!("v3")]))
            .execute(p.as_view())
            .unwrap();

        let r =
            parse!("4*v3*f1(v1,v4)+4*v3*f1(v2,v4)+3*v3^-1*f1(v1,v4)+3*v3^-1*f1(v2,v4)").unwrap();
        assert_eq!(out, r);
    }

    #[test]
    fn cycle_symmetrize() {
        let p = parse!("f1(1,2,3,5,1,2,3,4)").unwrap();

        let out = Transformer::CycleSymmetrize.execute(p.as_view()).unwrap();

        let r = parse!("f1(1,2,3,4,1,2,3,5)").unwrap();
        assert_eq!(out, r);
    }
}
