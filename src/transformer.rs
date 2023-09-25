use ahash::HashMap;
use dyn_clone::DynClone;

use crate::{
    combinatorics::partitions,
    id::{MatchStack, Pattern, PatternRestriction},
    representations::{
        Add, Atom, AtomSet, AtomView, Fun, Identifier, Mul, OwnedAdd, OwnedFun, OwnedMul,
    },
    state::{State, Workspace, ARG, INPUT_ID},
};

pub trait Map<P: AtomSet>: Fn(AtomView<P>, &mut Atom<P>) + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<P: AtomSet> Map<P>);
impl<P: AtomSet, T: Clone + Send + Sync + Fn(AtomView<'_, P>, &mut Atom<P>)> Map<P> for T {}

/// Operations that take a pattern as the input and produce an expression
#[derive(Clone)]
pub enum Transformer<P: AtomSet + 'static> {
    Input,
    /// Expand the rhs.
    Expand(Pattern<P>),
    /// Derive the rhs w.r.t a variable.
    Derivative(Pattern<P>, Identifier),
    /// Apply find-and-replace on the rhs.
    ReplaceAll(
        Pattern<P>,
        Pattern<P>,
        Pattern<P>,
        HashMap<Identifier, Vec<PatternRestriction<P>>>,
    ),
    /// Take the product of a list of arguments in the rhs.
    Product(Pattern<P>),
    /// Take the sum of a list of arguments in the rhs.
    Sum(Pattern<P>),
    /// Map the rhs with a user-specified function.
    Map(Pattern<P>, Box<dyn Map<P>>),
    /// Split a `Mul` or `Add` into a list of arguments.
    Split(Pattern<P>),
    Partition(Pattern<P>, Vec<(Identifier, usize)>, bool, bool),
    Sort(Pattern<P>),
}

impl<P: AtomSet> std::fmt::Debug for Transformer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Transformer::Input => f.debug_tuple("Input").finish(),
            Transformer::Expand(e) => f.debug_tuple("Expand").field(e).finish(),
            Transformer::Derivative(e, x) => f.debug_tuple("Derivative").field(e).field(x).finish(),
            Transformer::ReplaceAll(pat, input, rhs, ..) => f
                .debug_tuple("ReplaceAll")
                .field(pat)
                .field(input)
                .field(rhs)
                .finish(),
            Transformer::Product(p) => f.debug_tuple("Product").field(p).finish(),
            Transformer::Sum(p) => f.debug_tuple("Sum").field(p).finish(),
            Transformer::Map(p, _) => f.debug_tuple("Map").field(p).finish(),
            Transformer::Split(p) => f.debug_tuple("Split").field(p).finish(),
            Transformer::Partition(p, g, b1, b2) => f
                .debug_tuple("Partition")
                .field(p)
                .field(g)
                .field(b1)
                .field(b2)
                .finish(),
            Transformer::Sort(p) => f.debug_tuple("Sort").field(p).finish(),
        }
    }
}

impl<P: AtomSet> Transformer<P> {
    /// Create a new partition transformer that must exactly fit the input.
    pub fn new_partition_exact(
        pat: Pattern<P>,
        partitions: Vec<(Identifier, usize)>,
    ) -> Transformer<P> {
        Transformer::Partition(pat, partitions, false, false)
    }

    /// Create a new partition transformer that collects all left-over
    /// atoms in the last bin.
    pub fn new_partition_collect_in_last(
        pat: Pattern<P>,
        mut partitions: Vec<(Identifier, usize)>,
        rest: Identifier,
    ) -> Transformer<P> {
        partitions.push((rest, 0));
        Transformer::Partition(pat, partitions, true, false)
    }

    /// Create a new partition transformer that repeats the partitions so that it can fit
    /// the input.
    pub fn new_partition_repeat(pat: Pattern<P>, partition: (Identifier, usize)) -> Transformer<P> {
        Transformer::Partition(pat, vec![partition], false, true)
    }

    pub fn execute(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        match_stack: &MatchStack<P>,
        out: &mut Atom<P>,
    ) {
        match self {
            Transformer::Input => {
                assert!(match_stack.len() == 1);
                match match_stack.get(INPUT_ID).unwrap() {
                    crate::id::Match::Single(s) => {
                        out.set_from_view(s);
                    }
                    _ => unreachable!("Bad pattern match on input"),
                }
            }
            Transformer::Map(e, f) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);
                f(h.as_view(), out);
            }
            Transformer::Expand(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);
                h.as_view().expand(workspace, state, out);
            }
            Transformer::Derivative(e, x) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);
                h.as_view().derivative(*x, workspace, state, out);
            }
            Transformer::ReplaceAll(pat, input, rhs, cond) => {
                // prepare the target by executing the transformers
                let mut h = workspace.new_atom();
                input.substitute_wildcards(state, workspace, &mut h, match_stack);
                pat.replace_all(h.as_view(), rhs, state, workspace, cond, out);
            }
            Transformer::Product(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);

                if let AtomView::Fun(f) = h.as_view() {
                    if f.get_name() == ARG {
                        let mut mul_h = workspace.new_atom();
                        let mul = mul_h.to_mul();

                        for arg in f.iter() {
                            mul.extend(arg);
                        }

                        mul.set_dirty(true);
                        mul_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                }

                out.set_from_view(&h.as_view());
            }
            Transformer::Sum(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);

                if let AtomView::Fun(f) = h.as_view() {
                    if f.get_name() == ARG {
                        let mut add_h = workspace.new_atom();
                        let add = add_h.to_add();

                        for arg in f.iter() {
                            add.extend(arg);
                        }

                        add.set_dirty(true);
                        add_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                }

                out.set_from_view(&h.as_view());
            }
            Transformer::Split(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);

                match h.as_view() {
                    AtomView::Mul(m) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun();
                        arg.set_from_name(ARG);

                        for factor in m.iter() {
                            arg.add_arg(factor);
                        }

                        arg.set_dirty(true);
                        arg_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                    AtomView::Add(a) => {
                        let mut arg_h = workspace.new_atom();
                        let arg = arg_h.to_fun();
                        arg.set_from_name(ARG);

                        for summand in a.iter() {
                            arg.add_arg(summand);
                        }

                        arg.set_dirty(true);
                        arg_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                    _ => {
                        out.set_from_view(&h.as_view());
                    }
                }
            }
            Transformer::Partition(e, bins, fill_last, repeat) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);

                if let AtomView::Fun(f) = h.as_view() {
                    if f.get_name() == ARG {
                        let args: Vec<_> = f.iter().collect();

                        let mut sum_h = workspace.new_atom();
                        let sum = sum_h.to_add();

                        let partitions = partitions(&args, bins, *fill_last, *repeat);

                        if partitions.is_empty() {
                            out.set_from_view(&workspace.new_num(0).as_view());
                            return;
                        }

                        for (p, args) in partitions {
                            let mut mul_h = workspace.new_atom();
                            let mul = mul_h.to_mul();

                            if !p.is_one() {
                                mul.extend(workspace.new_num(p).as_view());
                            }

                            for (name, f_args) in args {
                                let mut fun_h = workspace.new_atom();
                                let fun = fun_h.to_fun();
                                fun.set_from_name(name);
                                for x in f_args {
                                    fun.add_arg(x);
                                }
                                fun.set_dirty(true);

                                mul.extend(fun_h.as_view());
                            }

                            mul.set_dirty(true);
                            sum.extend(mul_h.as_view());
                        }

                        sum.set_dirty(true);
                        sum_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                }

                out.set_from_view(&h.as_view());
            }
            Transformer::Sort(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);

                if let AtomView::Fun(f) = h.as_view() {
                    if f.get_name() == ARG {
                        let mut args: Vec<_> = f.iter().collect();
                        args.sort();

                        let mut fun_h = workspace.new_atom();
                        let fun = fun_h.to_fun();
                        fun.set_from_name(ARG);

                        for arg in args {
                            fun.add_arg(arg);
                        }

                        fun.set_dirty(true);
                        fun_h.as_view().normalize(workspace, state, out);
                        return;
                    }
                }

                out.set_from_view(&h.as_view());
            }
        }
    }
}
