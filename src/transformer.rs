use ahash::HashMap;
use dyn_clone::DynClone;

use crate::{
    id::{MatchStack, Pattern, PatternRestriction},
    representations::{Atom, AtomSet, AtomView, Fun, Identifier, OwnedAdd, OwnedMul},
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
}

impl<P: AtomSet> std::fmt::Debug for Transformer<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => f.debug_tuple("Input").finish(),
            Self::Expand(e) => f.debug_tuple("Expand").field(e).finish(),
            Self::Derivative(e, x) => f.debug_tuple("Derivative").field(e).field(x).finish(),
            Self::ReplaceAll(pat, input, rhs, ..) => f
                .debug_tuple("ReplaceAll")
                .field(pat)
                .field(input)
                .field(rhs)
                .finish(),
            Self::Product(p) => f.debug_tuple("Product").field(p).finish(),
            Self::Sum(p) => f.debug_tuple("Sum").field(p).finish(),
            Self::Map(p, _) => f.debug_tuple("Map").field(p).finish(),
        }
    }
}

impl<P: AtomSet> Transformer<P> {
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
        }
    }
}
