use std::time::Instant;

use crate::{
    coefficient::{Coefficient, CoefficientView},
    combinatorics::{partitions, unique_permutations},
    id::{Condition, MatchSettings, Pattern, WildcardAndRestriction},
    printer::{AtomPrinter, PrintOptions},
    representations::{Atom, AtomView, Symbol},
    state::{State, Workspace},
};
use ahash::HashMap;
use colored::Colorize;
use dyn_clone::DynClone;

pub trait Map:
    Fn(AtomView, &mut Atom) -> Result<(), TransformerError> + DynClone + Send + Sync
{
}
dyn_clone::clone_trait_object!(Map);
impl<T: Clone + Send + Sync + Fn(AtomView<'_>, &mut Atom) -> Result<(), TransformerError>> Map
    for T
{
}

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
}

#[derive(Clone, Debug)]
pub enum TransformerError {
    ValueError(String),
    Interrupt,
}

/// Operations that take a pattern as the input and produce an expression
#[derive(Clone)]
pub enum Transformer {
    /// Expand the rhs.
    Expand,
    /// Derive the rhs w.r.t a variable.
    Derivative(Symbol),
    /// Derive the rhs w.r.t a variable.
    TaylorSeries(Symbol, Atom, u32),
    /// Apply find-and-replace on the rhs.
    ReplaceAll(
        Pattern,
        Pattern,
        Condition<WildcardAndRestriction>,
        MatchSettings,
    ),
    /// Take the product of a list of arguments in the rhs.
    Product,
    /// Take the sum of a list of arguments in the rhs.
    Sum,
    /// Return the number of arguments of a function in the input.
    /// If the argument of `ArgCount` is `true`, only the number
    /// of arguments of `arg()` is returned and 1 is returned otherwise.
    /// If the argument is `false`, 0 is returned for non-functions.
    ArgCount(bool),
    /// Map the rhs with a user-specified function.
    Map(Box<dyn Map>),
    /// Split a `Mul` or `Add` into a list of arguments.
    Split,
    Partition(Vec<(Symbol, usize)>, bool, bool),
    Sort,
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
            Transformer::Expand => f.debug_tuple("Expand").finish(),
            Transformer::Derivative(x) => f.debug_tuple("Derivative").field(x).finish(),
            Transformer::ReplaceAll(pat, rhs, ..) => {
                f.debug_tuple("ReplaceAll").field(pat).field(rhs).finish()
            }
            Transformer::Product => f.debug_tuple("Product").finish(),
            Transformer::Sum => f.debug_tuple("Sum").finish(),
            Transformer::ArgCount(p) => f.debug_tuple("ArgCount").field(p).finish(),
            Transformer::Map(_) => f.debug_tuple("Map").finish(),
            Transformer::Split => f.debug_tuple("Split").finish(),
            Transformer::Partition(g, b1, b2) => f
                .debug_tuple("Partition")
                .field(g)
                .field(b1)
                .field(b2)
                .finish(),
            Transformer::Sort => f.debug_tuple("Sort").finish(),
            Transformer::Deduplicate => f.debug_tuple("Deduplicate").finish(),
            Transformer::Permutations(i) => f.debug_tuple("Permutations").field(i).finish(),
            Transformer::TaylorSeries(x, point, d) => f
                .debug_tuple("TaylorSeries")
                .field(x)
                .field(point)
                .field(d)
                .finish(),
            Transformer::Repeat(r) => f.debug_tuple("Repeat").field(r).finish(),
            Transformer::Print(p) => f.debug_tuple("Print").field(p).finish(),
            Transformer::Stats(o, r) => f.debug_tuple("Timing").field(o).field(r).finish(),
            Transformer::FromNumber => f.debug_tuple("FromNumber").finish(),
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

    pub fn execute(
        orig_input: AtomView<'_>,
        chain: &[Transformer],

        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<(), TransformerError> {
        out.set_from_view(&orig_input);
        let mut tmp = workspace.new_atom();
        for t in chain {
            std::mem::swap(out, &mut tmp);
            let input = tmp.as_view();

            match t {
                Transformer::Map(f) => {
                    f(input, out)?;
                }
                Transformer::Expand => {
                    input.expand_with_ws_into(workspace, out);
                }
                Transformer::Derivative(x) => {
                    input.derivative_with_ws_into(*x, workspace, out);
                }
                Transformer::TaylorSeries(x, expansion_point, depth) => {
                    input.taylor_series_with_ws_into(
                        *x,
                        expansion_point.as_view(),
                        *depth,
                        workspace,
                        out,
                    );
                }
                Transformer::ReplaceAll(pat, rhs, cond, settings) => {
                    pat.replace_all_with_ws_into(
                        input,
                        rhs,
                        workspace,
                        cond.into(),
                        settings.into(),
                        out,
                    );
                }
                Transformer::Product => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
                            let mut mul_h = workspace.new_atom();
                            let mul = mul_h.to_mul();

                            for arg in f.iter() {
                                mul.extend(arg);
                            }

                            mul_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    out.set_from_view(&input);
                }
                Transformer::Sum => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
                            let mut add_h = workspace.new_atom();
                            let add = add_h.to_add();

                            for arg in f.iter() {
                                add.extend(arg);
                            }

                            add_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    out.set_from_view(&input);
                }
                Transformer::ArgCount(only_for_arg_fun) => {
                    if let AtomView::Fun(f) = input {
                        if !*only_for_arg_fun || f.get_symbol() == State::ARG {
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
                Transformer::Split => match input {
                    AtomView::Mul(m) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun(State::ARG);

                        for factor in m.iter() {
                            arg.add_arg(factor);
                        }

                        arg_h.as_view().normalize(workspace, out);
                        continue;
                    }
                    AtomView::Add(a) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun(State::ARG);

                        for summand in a.iter() {
                            arg.add_arg(summand);
                        }

                        arg_h.as_view().normalize(workspace, out);
                        continue;
                    }
                    _ => {
                        out.set_from_view(&input);
                    }
                },
                Transformer::Partition(bins, fill_last, repeat) => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
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

                    out.set_from_view(&input);
                }
                Transformer::Sort => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
                            let mut args: Vec<_> = f.iter().collect();
                            args.sort();

                            let mut fun_h = workspace.new_atom();
                            let fun = fun_h.to_fun(State::ARG);

                            for arg in args {
                                fun.add_arg(arg);
                            }

                            fun_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    out.set_from_view(&input);
                }
                Transformer::Deduplicate => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
                            let args: Vec<_> = f.iter().collect();
                            let mut args_dedup: Vec<_> = Vec::with_capacity(args.len());

                            for a in args {
                                // check last argument first, so that the sorted list case is fast
                                if args_dedup.last() != Some(&a) && !args_dedup.contains(&a) {
                                    args_dedup.push(a);
                                }
                            }

                            let mut fun_h = workspace.new_atom();
                            let fun = fun_h.to_fun(State::ARG);

                            for arg in args_dedup {
                                fun.add_arg(arg);
                            }

                            fun_h.as_view().normalize(workspace, out);
                            continue;
                        }
                    }

                    out.set_from_view(&input);
                }
                Transformer::Permutations(f_name) => {
                    if let AtomView::Fun(f) = input {
                        if f.get_symbol() == State::ARG {
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

                    out.set_from_view(&input);
                }
                Transformer::Repeat(r) => loop {
                    Self::execute(tmp.as_view(), r, workspace, out)?;

                    if tmp.as_view() == out.as_view() {
                        break;
                    }

                    std::mem::swap(out, &mut tmp);
                },
                Transformer::Print(o) => {
                    println!("{}", AtomPrinter::new_with_options(input, *o));
                    out.set_from_view(&input);
                }
                Transformer::Stats(o, r) => {
                    let in_nterms = if let AtomView::Add(a) = input {
                        a.get_nargs()
                    } else {
                        1
                    };
                    let in_size = input.get_byte_size();

                    let t = Instant::now();
                    Self::execute(input, r, workspace, out)?;

                    let out_nterms = if let AtomView::Add(a) = out.as_view() {
                        a.get_nargs()
                    } else {
                        1
                    };
                    let out_size = out.as_view().get_byte_size();

                    let in_nterms_s = o.format_count(in_nterms);
                    let out_nterms_s = o.format_count(out_nterms);

                    println!(
                        "Stats for {}:
\tIn  │ {:>width$} │ {:>8} │
\tOut │ {:>width$} │ {:>8} │ ⧗ {:#.2?}",
                        o.tag.bold(),
                        in_nterms_s,
                        o.format_size(in_size),
                        if out_nterms as f64 / in_nterms as f64
                            > o.color_medium_change_threshold.unwrap_or(f64::INFINITY)
                        {
                            if out_nterms as f64 / in_nterms as f64
                                > o.color_large_change_threshold.unwrap_or(f64::INFINITY)
                            {
                                out_nterms_s.red()
                            } else {
                                out_nterms_s.bright_magenta()
                            }
                        } else {
                            out_nterms_s.as_str().into()
                        },
                        o.format_size(out_size),
                        Instant::now().duration_since(t),
                        width = in_nterms_s.len().max(out_nterms_s.len()).min(6),
                    );
                }
                Transformer::FromNumber => {
                    if let AtomView::Num(n) = input {
                        if let CoefficientView::RationalPolynomial(r) = n.get_coeff_view() {
                            r.to_expression_with_map(workspace, &HashMap::default(), out);
                            continue;
                        }
                    }

                    out.set_from_view(&input);
                }
            }
        }

        Ok(())
    }
}
