pub mod default;
pub mod number;
pub mod tree;

use crate::state::{ResettableBuffer, State, Workspace};
use std::{cmp::Ordering, ops::Range};

use self::{
    number::{BorrowedNumber, Number},
    tree::AtomTree,
};

/// An identifier, for example for a variable or function.
/// Should be created using `get_or_insert` of `State`.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier(u32);

impl std::fmt::Debug for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

impl From<u32> for Identifier {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Identifier {
    pub fn to_u32(&self) -> u32 {
        self.0
    }
}

/// Represents all atoms of a mathematical expression.
pub trait Atom: PartialEq {
    type N<'a>: Num<'a, P = Self>;
    type V<'a>: Var<'a, P = Self>;
    type F<'a>: Fun<'a, P = Self>;
    type P<'a>: Pow<'a, P = Self>;
    type M<'a>: Mul<'a, P = Self>;
    type A<'a>: Add<'a, P = Self>;
    type ON: OwnedNum<P = Self>;
    type OV: OwnedVar<P = Self>;
    type OF: OwnedFun<P = Self>;
    type OP: OwnedPow<P = Self>;
    type OM: OwnedMul<P = Self>;
    type OA: OwnedAdd<P = Self>;
    type S<'a>: ListSlice<'a, P = Self>;

    fn from_tree(tree: &AtomTree, state: &State, workspace: &Workspace<Self>) -> OwnedAtom<Self>
    where
        Self: Sized;
}

/// Convert the owned atoms by recycling and clearing their interal buffers.
pub trait Convert<P: Atom> {
    fn to_owned_var(self) -> P::OV;
    fn to_owned_num(self) -> P::ON;
    fn to_owned_fun(self) -> P::OF;
    fn to_owned_pow(self) -> P::OP;
    fn to_owned_add(self) -> P::OA;
    fn to_owned_mul(self) -> P::OM;
}

pub trait OwnedNum: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_from_number(&mut self, num: Number);
    fn set_from_view(&mut self, a: &<Self::P as Atom>::N<'_>);
    fn add(&mut self, other: &<Self::P as Atom>::N<'_>, state: &State);
    fn mul(&mut self, other: &<Self::P as Atom>::N<'_>, state: &State);
    fn to_num_view(&self) -> <Self::P as Atom>::N<'_>;
}

pub trait OwnedVar: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_from_id(&mut self, id: Identifier);
    fn set_from_view(&mut self, view: &<Self::P as Atom>::V<'_>);
    fn to_var_view(&self) -> <Self::P as Atom>::V<'_>;
}

pub trait OwnedFun: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_from_view(&mut self, view: &<Self::P as Atom>::F<'_>);
    fn set_from_name(&mut self, id: Identifier);
    fn set_dirty(&mut self, dirty: bool);
    fn add_arg(&mut self, other: AtomView<Self::P>);
    fn to_fun_view(&self) -> <Self::P as Atom>::F<'_>;
}

pub trait OwnedPow: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_from_view(&mut self, view: &<Self::P as Atom>::P<'_>);
    fn set_from_base_and_exp(&mut self, base: AtomView<'_, Self::P>, exp: AtomView<'_, Self::P>);
    fn set_dirty(&mut self, dirty: bool);
    fn to_pow_view(&self) -> <Self::P as Atom>::P<'_>;
}

pub trait OwnedMul: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_dirty(&mut self, dirty: bool);
    fn set_from_view(&mut self, view: &<Self::P as Atom>::M<'_>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn replace_last(&mut self, other: AtomView<Self::P>);
    fn to_mul_view(&self) -> <Self::P as Atom>::M<'_>;
}

pub trait OwnedAdd: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_dirty(&mut self, dirty: bool);
    fn set_from_view(&mut self, view: &<Self::P as Atom>::A<'_>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_add_view(&self) -> <Self::P as Atom>::A<'_>;
}

pub trait Num<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::N<'b>> {
    type P: Atom;

    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_dirty(&self) -> bool;
    fn get_number_view(&self) -> BorrowedNumber<'_>;
    fn to_view(&self) -> AtomView<'a, Self::P>;
}

pub trait Var<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::V<'b>> {
    type P: Atom;

    fn get_name(&self) -> Identifier;
    fn to_view(&self) -> AtomView<'a, Self::P>;
}

pub trait Fun<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::F<'b>> {
    type P: Atom;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn get_name(&self) -> Identifier;
    fn get_nargs(&self) -> usize;
    fn is_dirty(&self) -> bool;
    fn cmp(&self, other: &Self) -> Ordering;
    fn iter(&self) -> Self::I;
    fn to_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as Atom>::S<'a>;
}

pub trait Pow<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::P<'b>> {
    type P: Atom;

    fn get_base(&self) -> AtomView<'a, Self::P>;
    fn get_exp(&self) -> AtomView<'a, Self::P>;
    fn is_dirty(&self) -> bool;
    fn get_base_exp(&self) -> (AtomView<'a, Self::P>, AtomView<'a, Self::P>);
    fn to_view(&self) -> AtomView<'a, Self::P>;

    /// Interpret `x^y` as slice `[x,y]`.
    fn to_slice(&self) -> <Self::P as Atom>::S<'a>;
}

pub trait Mul<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::M<'b>> {
    type P: Atom;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn is_dirty(&self) -> bool;
    fn get_nargs(&self) -> usize;
    fn iter(&self) -> Self::I;
    fn to_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as Atom>::S<'a>;
}

pub trait Add<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::A<'b>> {
    type P: Atom;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn is_dirty(&self) -> bool;
    fn get_nargs(&self) -> usize;
    fn iter(&self) -> Self::I;
    fn to_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as Atom>::S<'a>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceType {
    Add,
    Mul,
    Arg,
    One,
    Pow,
    Empty,
}

pub trait ListSlice<'a>: Clone {
    type P: Atom;
    type ListSliceIterator: Iterator<Item = AtomView<'a, Self::P>>;

    fn iter(&self) -> Self::ListSliceIterator;
    fn from_one(view: AtomView<'a, Self::P>) -> Self;
    fn get_type(&self) -> SliceType;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> AtomView<'a, Self::P>;
    fn get_subslice(&self, range: Range<usize>) -> Self;
    fn eq(&self, other: &Self) -> bool;
}

pub enum AtomView<'a, P: Atom> {
    Num(P::N<'a>),
    Var(P::V<'a>),
    Fun(P::F<'a>),
    Pow(P::P<'a>),
    Mul(P::M<'a>),
    Add(P::A<'a>),
}

impl<'a, P: Atom> Clone for AtomView<'a, P> {
    fn clone(&self) -> Self {
        match self {
            Self::Num(arg0) => Self::Num(*arg0),
            Self::Var(arg0) => Self::Var(*arg0),
            Self::Fun(arg0) => Self::Fun(*arg0),
            Self::Pow(arg0) => Self::Pow(*arg0),
            Self::Mul(arg0) => Self::Mul(*arg0),
            Self::Add(arg0) => Self::Add(*arg0),
        }
    }
}

impl<'a, P: Atom> Copy for AtomView<'a, P> {}

impl<'a, 'b, P: Atom> PartialEq<AtomView<'b, P>> for AtomView<'a, P> {
    fn eq(&self, other: &AtomView<P>) -> bool {
        match (self, other) {
            (Self::Num(n1), AtomView::Num(n2)) => n1 == n2,
            (Self::Var(v1), AtomView::Var(v2)) => v1 == v2,
            (Self::Fun(f1), AtomView::Fun(f2)) => f1 == f2,
            (Self::Pow(p1), AtomView::Pow(p2)) => p1 == p2,
            (Self::Mul(m1), AtomView::Mul(m2)) => m1 == m2,
            (Self::Add(a1), AtomView::Add(a2)) => a1 == a2,
            _ => false,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum OwnedAtom<P: Atom> {
    Num(P::ON),
    Var(P::OV),
    Fun(P::OF),
    Pow(P::OP),
    Mul(P::OM),
    Add(P::OA),
    Empty, // for internal use only
}

impl<P: Atom> std::fmt::Debug for OwnedAtom<P> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_view().fmt(fmt)
    }
}

impl<P: Atom> OwnedAtom<P> {
    pub fn transform_to_num(&mut self) -> &mut P::ON {
        let mut ov = std::mem::replace(self, Self::Empty);

        *self = match ov {
            Self::Num(_) => {
                ov.reset();
                ov
            }
            Self::Var(v) => Self::Num(v.to_owned_num()),
            Self::Fun(f) => Self::Num(f.to_owned_num()),
            Self::Pow(p) => Self::Num(p.to_owned_num()),
            Self::Mul(m) => Self::Num(m.to_owned_num()),
            Self::Add(a) => Self::Num(a.to_owned_num()),
            Self::Empty => unreachable!(),
        };

        let Self::Num(n) = self else { unreachable!() };
        n
    }

    pub fn transform_to_pow(&mut self) -> &mut P::OP {
        let mut ov = std::mem::replace(self, Self::Empty);

        *self = match ov {
            Self::Pow(_) => {
                ov.reset();
                ov
            }
            Self::Num(n) => Self::Pow(n.to_owned_pow()),
            Self::Var(v) => Self::Pow(v.to_owned_pow()),
            Self::Fun(f) => Self::Pow(f.to_owned_pow()),
            Self::Mul(m) => Self::Pow(m.to_owned_pow()),
            Self::Add(a) => Self::Pow(a.to_owned_pow()),
            Self::Empty => unreachable!(),
        };

        let Self::Pow(p) = self else { unreachable!() };
        p
    }

    pub fn transform_to_var(&mut self) -> &mut P::OV {
        let mut ov = std::mem::replace(self, Self::Empty);

        *self = match ov {
            Self::Var(_) => {
                ov.reset();
                ov
            }
            Self::Num(n) => Self::Var(n.to_owned_var()),
            Self::Pow(p) => Self::Var(p.to_owned_var()),
            Self::Fun(f) => Self::Var(f.to_owned_var()),
            Self::Mul(m) => Self::Var(m.to_owned_var()),
            Self::Add(a) => Self::Var(a.to_owned_var()),
            Self::Empty => unreachable!(),
        };

        let Self::Var(v) = self else { unreachable!() };
        v
    }

    pub fn transform_to_fun(&mut self) -> &mut P::OF {
        let mut of = std::mem::replace(self, Self::Empty);

        *self = match of {
            Self::Fun(_) => {
                of.reset();
                of
            }
            Self::Num(n) => Self::Fun(n.to_owned_fun()),
            Self::Pow(p) => Self::Fun(p.to_owned_fun()),
            Self::Var(v) => Self::Fun(v.to_owned_fun()),
            Self::Mul(m) => Self::Fun(m.to_owned_fun()),
            Self::Add(a) => Self::Fun(a.to_owned_fun()),
            Self::Empty => unreachable!(),
        };

        let Self::Fun(f) = self else { unreachable!() };
        f
    }

    pub fn transform_to_mul(&mut self) -> &mut P::OM {
        let mut om = std::mem::replace(self, Self::Empty);

        *self = match om {
            Self::Mul(_) => {
                om.reset();
                om
            }
            Self::Num(n) => Self::Mul(n.to_owned_mul()),
            Self::Pow(p) => Self::Mul(p.to_owned_mul()),
            Self::Var(v) => Self::Mul(v.to_owned_mul()),
            Self::Fun(f) => Self::Mul(f.to_owned_mul()),
            Self::Add(a) => Self::Mul(a.to_owned_mul()),
            Self::Empty => unreachable!(),
        };

        let Self::Mul(m) = self else { unreachable!() };
        m
    }

    pub fn transform_to_add(&mut self) -> &mut P::OA {
        let mut oa = std::mem::replace(self, Self::Empty);

        *self = match oa {
            Self::Add(_) => {
                oa.reset();
                oa
            }
            Self::Num(n) => Self::Add(n.to_owned_add()),
            Self::Pow(p) => Self::Add(p.to_owned_add()),
            Self::Var(v) => Self::Add(v.to_owned_add()),
            Self::Fun(f) => Self::Add(f.to_owned_add()),
            Self::Mul(m) => Self::Add(m.to_owned_add()),
            Self::Empty => unreachable!(),
        };

        let Self::Add(a) = self else { unreachable!() };
        a
    }

    /// This function allocates a new OwnedAtom with the same content as `view`.
    pub fn new_from_view(view: &AtomView<P>) -> Self {
        let mut owned = Self::new();
        owned.from_view(view);
        owned
    }

    pub fn from_view(&mut self, view: &AtomView<P>) {
        match view {
            AtomView::Num(n) => {
                let num = self.transform_to_num();
                num.set_from_view(n);
            }
            AtomView::Var(v) => {
                let var = self.transform_to_var();
                var.set_from_view(v);
            }
            AtomView::Fun(f) => {
                let fun = self.transform_to_fun();
                fun.set_from_view(f);
            }
            AtomView::Pow(p) => {
                let pow = self.transform_to_pow();
                pow.set_from_view(p);
            }
            AtomView::Mul(m) => {
                let mul = self.transform_to_mul();
                mul.set_from_view(m);
            }
            AtomView::Add(a) => {
                let add = self.transform_to_add();
                add.set_from_view(a);
            }
        }
    }

    pub fn to_view(&self) -> AtomView<'_, P> {
        match self {
            Self::Num(n) => AtomView::Num(n.to_num_view()),
            Self::Var(v) => AtomView::Var(v.to_var_view()),
            Self::Fun(f) => AtomView::Fun(f.to_fun_view()),
            Self::Pow(p) => AtomView::Pow(p.to_pow_view()),
            Self::Mul(m) => AtomView::Mul(m.to_mul_view()),
            Self::Add(a) => AtomView::Add(a.to_add_view()),
            Self::Empty => unreachable!(),
        }
    }
}

impl<P: Atom> ResettableBuffer for OwnedAtom<P> {
    fn new() -> Self {
        Self::Num(P::ON::new())
    }

    fn reset(&mut self) {
        match self {
            Self::Num(n) => n.reset(),
            Self::Var(v) => v.reset(),
            Self::Fun(f) => f.reset(),
            Self::Pow(p) => p.reset(),
            Self::Mul(m) => m.reset(),
            Self::Add(a) => a.reset(),
            Self::Empty => {}
        }
    }
}
