pub mod default;
pub mod number;
pub mod tree;

use std::cmp::Ordering;

use crate::state::ResettableBuffer;

use self::tree::Atom;

/// An identifier, for example for a variable or function.
/// Should be created using `get_or_insert` of `State`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier(u32);

impl std::fmt::Debug for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

impl From<u32> for Identifier {
    fn from(value: u32) -> Self {
        Identifier(value)
    }
}

impl Identifier {
    pub fn to_u32(&self) -> u32 {
        self.0
    }
}

// TODO: rename Expr to Plus and Term to Times ala Mathematica
pub trait AtomT: PartialEq {
    type N<'a>: NumberT<'a, P = Self>;
    type V<'a>: VarT<'a, P = Self>;
    type F<'a>: FunctionT<'a, P = Self>;
    type P<'a>: PowT<'a, P = Self>;
    type T<'a>: TermT<'a, P = Self>;
    type E<'a>: ExprT<'a, P = Self>;
    type O: OwnedAtomT<P = Self>;
    type ON: OwnedNumberT<P = Self>;
    type OV: OwnedVarT<P = Self>;
    type OF: OwnedFunctionT<P = Self>;
    type OP: OwnedPowT<P = Self>;
    type OT: OwnedTermT<P = Self>;
    type OE: OwnedExprT<P = Self>;
}

pub trait OwnedAtomT: ResettableBuffer {
    type P: AtomT;

    fn from_num(source: <Self::P as AtomT>::ON) -> Self;
    fn write<'a>(&mut self, source: &AtomView<'a, Self::P>);
    fn write_tree(&mut self, source: &Atom);
    fn from_tree(a: &Atom) -> Self;
    fn to_tree(&self) -> Atom;
    fn to_view<'a>(&'a self) -> AtomView<'a, Self::P>;
    fn len(&self) -> usize;
}

pub trait OwnedNumberT: ResettableBuffer {
    type P: AtomT;

    fn from_u64_frac(&mut self, num: u64, den: u64);
    fn from_view<'a>(&mut self, a: <Self::P as AtomT>::N<'a>);
    fn add<'a>(&mut self, other: &<Self::P as AtomT>::N<'a>);
    fn to_num_view<'a>(&'a self) -> <Self::P as AtomT>::N<'a>;
}

pub trait OwnedVarT: ResettableBuffer {
    type P: AtomT;

    fn from_id(&mut self, id: Identifier);
    fn to_var_view<'a>(&'a self) -> <Self::P as AtomT>::V<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait OwnedFunctionT: ResettableBuffer {
    type P: AtomT;

    fn from_name(&mut self, id: Identifier);
    fn set_dirty(&mut self, dirty: bool);
    fn add_arg(&mut self, other: AtomView<Self::P>);
    fn to_func_view<'a>(&'a self) -> <Self::P as AtomT>::F<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait OwnedPowT: ResettableBuffer {
    type P: AtomT;

    fn from_base_and_exp(&mut self, base: AtomView<Self::P>, exp: AtomView<Self::P>);
    fn set_dirty(&mut self, dirty: bool);
    fn to_pow_view<'a>(&'a self) -> <Self::P as AtomT>::P<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait OwnedTermT: ResettableBuffer {
    type P: AtomT;

    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_term_view<'a>(&'a self) -> <Self::P as AtomT>::T<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait OwnedExprT: ResettableBuffer {
    type P: AtomT;

    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_expr_view<'a>(&'a self) -> <Self::P as AtomT>::E<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait NumberT<'a>: Clone + PartialEq {
    type P: AtomT;

    fn is_one(&self) -> bool;
    fn add<'b>(&self, other: &Self, out: &mut <Self::P as AtomT>::O);
    fn get_numden(&self) -> (u64, u64);
}

pub trait VarT<'a>: Clone + PartialEq {
    type P: AtomT;

    fn get_name(&self) -> Identifier;
}

pub trait FunctionT<'a>: Clone + PartialEq {
    type P: AtomT;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_name(&self) -> Identifier;
    fn get_nargs(&self) -> usize;
    fn is_dirty(&self) -> bool;
    fn cmp(&self, other: &Self) -> Ordering;
    fn into_iter(&self) -> Self::I;
}

pub trait PowT<'a>: Clone + PartialEq {
    type P: AtomT;

    fn get_base(&self) -> AtomView<Self::P>;
    fn get_exp(&self) -> AtomView<Self::P>;
    fn get_base_exp(&self) -> (AtomView<Self::P>, AtomView<Self::P>);
}

pub trait TermT<'a>: Clone + PartialEq {
    type P: AtomT;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
}

pub trait ExprT<'a>: Clone + PartialEq {
    type P: AtomT;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
}

pub trait ListIteratorT<'a>: Clone {
    type P: AtomT;
    fn next(&mut self) -> Option<AtomView<'a, Self::P>>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AtomView<'a, P: AtomT> {
    Number(P::N<'a>),
    Var(P::V<'a>),
    Function(P::F<'a>),
    Pow(P::P<'a>),
    Term(P::T<'a>),
    Expression(P::E<'a>),
}

impl<'a, P: AtomT> PartialOrd for AtomView<'a, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (AtomView::Number(_), AtomView::Number(_)) => Some(Ordering::Equal),
            (AtomView::Number(_), _) => Some(Ordering::Greater),
            (_, AtomView::Number(_)) => Some(Ordering::Less),

            (AtomView::Var(_), AtomView::Var(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Var(_), _) => Some(Ordering::Less),
            (_, AtomView::Var(_)) => Some(Ordering::Greater),

            (AtomView::Pow(_), AtomView::Pow(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Pow(_), _) => Some(Ordering::Less),
            (_, AtomView::Pow(_)) => Some(Ordering::Greater),

            (AtomView::Term(_), AtomView::Term(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Term(_), _) => Some(Ordering::Less),
            (_, AtomView::Term(_)) => Some(Ordering::Greater),

            (AtomView::Expression(_), AtomView::Expression(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Expression(_), _) => Some(Ordering::Less),
            (_, AtomView::Expression(_)) => Some(Ordering::Greater),

            (AtomView::Function(_), AtomView::Function(_)) => Some(Ordering::Equal), // FIXME
        }
    }
}

impl<'a, P: AtomT> AtomView<'a, P> {
    pub fn dbg_print_tree(&self, level: usize) {
        let mut owned_atom = P::O::new();
        owned_atom.write(self);

        match &self {
            AtomView::Var(_) => println!("{}{:?}", " ".repeat(level), owned_atom.to_tree()),
            AtomView::Function(f) => {
                println!(
                    "{}entering func {:?}",
                    " ".repeat(level),
                    owned_atom.to_tree()
                );
                let mut f_it = f.into_iter();
                while let Some(arg) = f_it.next() {
                    arg.dbg_print_tree(level + 1);
                }
            }
            AtomView::Number(_) => println!("{}{:?}", " ".repeat(level), owned_atom.to_tree()),
            AtomView::Term(t) => {
                println!(
                    "{}entering term {:?}",
                    " ".repeat(level),
                    owned_atom.to_tree()
                );
                let mut t_it = t.into_iter();
                while let Some(arg) = t_it.next() {
                    arg.dbg_print_tree(level + 1);
                }
            }
            AtomView::Pow(p) => {
                println!(
                    "{}entering pow {:?}",
                    " ".repeat(level),
                    owned_atom.to_tree()
                );

                let b = p.get_base();
                b.dbg_print_tree(level + 1);
                let e = p.get_exp();
                e.dbg_print_tree(level + 1);
            }
            AtomView::Expression(t) => {
                println!(
                    "{}entering expr {:?}",
                    " ".repeat(level),
                    owned_atom.to_tree()
                );
                let mut t_it = t.into_iter();
                while let Some(arg) = t_it.next() {
                    arg.dbg_print_tree(level + 1);
                }
            }
        }
    }
}
