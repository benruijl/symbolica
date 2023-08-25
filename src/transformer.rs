use ahash::HashMap;

use crate::{
    id::{MatchStack, Pattern, PatternRestriction},
    representations::{Atom, Identifier, OwnedAtom},
    state::{State, Workspace, INPUT_ID},
};

/// Operations that take a pattern as the input and produce an expression
#[derive(Clone)]
pub enum Transformer<P: Atom + 'static> {
    Input,
    Expand(Pattern<P>),
    Derivative(Pattern<P>, Identifier),
    ReplaceAll(
        Pattern<P>,
        Pattern<P>,
        Pattern<P>,
        HashMap<Identifier, Vec<PatternRestriction<P>>>,
    ),
}

impl<P: Atom> std::fmt::Debug for Transformer<P> {
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
        }
    }
}

impl<P: Atom> Transformer<P> {
    pub fn execute(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        match_stack: &MatchStack<P>,
        out: &mut OwnedAtom<P>,
    ) {
        match self {
            Transformer::Input => {
                assert!(match_stack.len() == 1);
                match match_stack.get(INPUT_ID).unwrap() {
                    crate::id::Match::Single(s) => {
                        out.from_view(s);
                    }
                    _ => unreachable!("Bad pattern match on input"),
                }
            }
            Transformer::Expand(e) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);
                h.to_view().expand(workspace, state, out);
            }
            Transformer::Derivative(e, x) => {
                let mut h = workspace.new_atom();
                e.substitute_wildcards(state, workspace, &mut h, match_stack);
                h.to_view().derivative(*x, workspace, state, out);
            }
            Transformer::ReplaceAll(pat, input, rhs, cond) => {
                // prepare the target by executing the transformers
                let mut h = workspace.new_atom();
                input.substitute_wildcards(state, workspace, &mut h, match_stack);
                pat.replace_all(h.to_view(), rhs, state, workspace, cond, out);
            }
        }
    }
}
