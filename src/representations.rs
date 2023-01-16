pub mod default;
pub mod number;
pub mod tree;

use std::cmp::Ordering;

use crate::state::ResettableBuffer;

use self::tree::AtomTree;

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

pub trait Atom: PartialEq {
    type N<'a>: Num<'a, P = Self>;
    type V<'a>: Var<'a, P = Self>;
    type F<'a>: Fn<'a, P = Self>;
    type P<'a>: Pow<'a, P = Self>;
    type M<'a>: Mul<'a, P = Self>;
    type A<'a>: Add<'a, P = Self>;
    type O: OwnedAtom<P = Self>;
    type ON: OwnedNum<P = Self>;
    type OV: OwnedVar<P = Self>;
    type OF: OwnedFn<P = Self>;
    type OP: OwnedPow<P = Self>;
    type OM: OwnedMul<P = Self>;
    type OA: OwnedAdd<P = Self>;
}

pub trait OwnedAtom: ResettableBuffer {
    type P: Atom;

    fn from_num(source: <Self::P as Atom>::ON) -> Self;
    fn write<'a>(&mut self, source: &AtomView<'a, Self::P>);
    fn write_tree(&mut self, source: &AtomTree);
    fn from_tree(a: &AtomTree) -> Self;
    fn to_tree(&self) -> AtomTree;
    fn to_view<'a>(&'a self) -> AtomView<'a, Self::P>;
    fn len(&self) -> usize;
}

pub trait OwnedNum: ResettableBuffer {
    type P: Atom;

    fn from_u64_frac(&mut self, num: u64, den: u64);
    fn from_view<'a>(&mut self, a: <Self::P as Atom>::N<'a>);
    fn add<'a>(&mut self, other: &<Self::P as Atom>::N<'a>);
    fn to_num_view<'a>(&'a self) -> <Self::P as Atom>::N<'a>;
}

pub trait OwnedVar: ResettableBuffer {
    type P: Atom;

    fn from_id(&mut self, id: Identifier);
    fn to_var_view<'a>(&'a self) -> <Self::P as Atom>::V<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as Atom>::O);
}

pub trait OwnedFn: ResettableBuffer {
    type P: Atom;

    fn from_name(&mut self, id: Identifier);
    fn set_dirty(&mut self, dirty: bool);
    fn add_arg(&mut self, other: AtomView<Self::P>);
    fn to_fn_view<'a>(&'a self) -> <Self::P as Atom>::F<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as Atom>::O);
}

pub trait OwnedPow: ResettableBuffer {
    type P: Atom;

    fn from_base_and_exp(&mut self, base: AtomView<Self::P>, exp: AtomView<Self::P>);
    fn set_dirty(&mut self, dirty: bool);
    fn to_pow_view<'a>(&'a self) -> <Self::P as Atom>::P<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as Atom>::O);
}

pub trait OwnedMul: ResettableBuffer {
    type P: Atom;

    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_mul_view<'a>(&'a self) -> <Self::P as Atom>::M<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as Atom>::O);
}

pub trait OwnedAdd: ResettableBuffer {
    type P: Atom;

    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_add_view<'a>(&'a self) -> <Self::P as Atom>::A<'a>;
    fn to_atom(&mut self, out: &mut <Self::P as Atom>::O);
}

pub trait Num<'a>: Clone + PartialEq {
    type P: Atom;

    fn is_one(&self) -> bool;
    fn add<'b>(&self, other: &Self, out: &mut <Self::P as Atom>::O);
    fn get_numden(&self) -> (u64, u64);
}

pub trait Var<'a>: Clone + PartialEq {
    type P: Atom;

    fn get_name(&self) -> Identifier;
}

pub trait Fn<'a>: Clone + PartialEq {
    type P: Atom;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_name(&self) -> Identifier;
    fn get_nargs(&self) -> usize;
    fn is_dirty(&self) -> bool;
    fn cmp(&self, other: &Self) -> Ordering;
    fn into_iter(&self) -> Self::I;
}

pub trait Pow<'a>: Clone + PartialEq {
    type P: Atom;

    fn get_base(&self) -> AtomView<Self::P>;
    fn get_exp(&self) -> AtomView<Self::P>;
    fn get_base_exp(&self) -> (AtomView<Self::P>, AtomView<Self::P>);
}

pub trait Mul<'a>: Clone + PartialEq {
    type P: Atom;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
}

pub trait Add<'a>: Clone + PartialEq {
    type P: Atom;
    type I: ListIteratorT<'a, P = Self::P>;

    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
}

pub trait ListIteratorT<'a>: Clone {
    type P: Atom;
    fn next(&mut self) -> Option<AtomView<'a, Self::P>>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum AtomView<'a, P: Atom> {
    Num(P::N<'a>),
    Var(P::V<'a>),
    Fn(P::F<'a>),
    Pow(P::P<'a>),
    Mul(P::M<'a>),
    Add(P::A<'a>),
}

impl<'a, P: Atom> PartialOrd for AtomView<'a, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Some(Ordering::Equal),
            (AtomView::Num(_), _) => Some(Ordering::Greater),
            (_, AtomView::Num(_)) => Some(Ordering::Less),

            (AtomView::Var(_), AtomView::Var(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Var(_), _) => Some(Ordering::Less),
            (_, AtomView::Var(_)) => Some(Ordering::Greater),

            (AtomView::Pow(_), AtomView::Pow(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Pow(_), _) => Some(Ordering::Less),
            (_, AtomView::Pow(_)) => Some(Ordering::Greater),

            (AtomView::Mul(_), AtomView::Mul(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Mul(_), _) => Some(Ordering::Less),
            (_, AtomView::Mul(_)) => Some(Ordering::Greater),

            (AtomView::Add(_), AtomView::Add(_)) => Some(Ordering::Equal), // FIXME
            (AtomView::Add(_), _) => Some(Ordering::Less),
            (_, AtomView::Add(_)) => Some(Ordering::Greater),

            (AtomView::Fn(_), AtomView::Fn(_)) => Some(Ordering::Equal), // FIXME
        }
    }
}

impl<'a, P: Atom> AtomView<'a, P> {
    pub fn dbg_print_tree(&self, level: usize) {
        let mut owned_atom = P::O::new();
        owned_atom.write(self);

        match &self {
            AtomView::Var(_) => println!("{}{:?}", " ".repeat(level), owned_atom.to_tree()),
            AtomView::Fn(f) => {
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
            AtomView::Num(_) => println!("{}{:?}", " ".repeat(level), owned_atom.to_tree()),
            AtomView::Mul(t) => {
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
            AtomView::Add(t) => {
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
