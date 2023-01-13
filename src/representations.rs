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

pub trait AtomT {
    type N<'a>: NumberT<'a, P = Self>;
    type V<'a>: VarT<'a, P = Self>;
    type F<'a>: FunctionT<'a, P = Self>;
    type T<'a>: TermT<'a, P = Self>;
    type O: OwnedAtomT<P = Self>;
    type ON: OwnedNumberT<P = Self>;
    type OV: OwnedVarT<P = Self>;
    type OF: OwnedFunctionT<P = Self>;
    type OT: OwnedTermT<P = Self>;
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

    fn from_view<'a>(a: <Self::P as AtomT>::N<'a>) -> Self;
    fn add<'a>(&mut self, other: &<Self::P as AtomT>::N<'a>);
    fn to_num_view<'a>(&'a self) -> <Self::P as AtomT>::N<'a>;
}

pub trait OwnedVarT: ResettableBuffer {
    type P: AtomT;

    fn from_id_pow(&mut self, id: Identifier, pow: <Self::P as AtomT>::ON);
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

pub trait OwnedTermT: ResettableBuffer {
    type P: AtomT;

    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_term_view<'a>(&'a self) -> <Self::P as AtomT>::T<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as AtomT>::O);
}

pub trait NumberT<'a>: Clone {
    type P: AtomT;

    fn is_one(&self) -> bool;
    fn add<'b>(&self, other: &Self, out: &mut <Self::P as AtomT>::O);
    fn get_numden(&self) -> (u64, u64);
}

pub trait VarT<'a>: Clone {
    type P: AtomT;

    fn get_name(&self) -> Identifier;
    fn get_pow(&self) -> <Self::P as AtomT>::N<'a>;
}

pub trait FunctionT<'a>: Clone {
    type P: AtomT;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_name(&self) -> Identifier;
    fn get_nargs(&self) -> usize;
    fn is_dirty(&self) -> bool;
    fn cmp(&self, other: &Self) -> Ordering;
    fn into_iter(&self) -> Self::I;
}

pub trait TermT<'a>: Clone {
    type P: AtomT;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
}

pub trait ListIteratorT<'a>: Clone {
    type P: AtomT;
    fn next(&mut self) -> Option<AtomView<'a, Self::P>>;
}

#[derive(Debug, Copy, Clone)]
pub enum AtomView<'a, P: AtomT> {
    Number(P::N<'a>),
    Var(P::V<'a>),
    Function(P::F<'a>),
    Term(P::T<'a>),
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
        }
    }
}
