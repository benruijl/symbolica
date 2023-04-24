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
        Identifier(value)
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

    fn from_number(&mut self, num: Number);
    fn from_view(&mut self, a: &<Self::P as Atom>::N<'_>);
    fn add<'a>(&mut self, other: &<Self::P as Atom>::N<'a>, state: &State);
    fn mul<'a>(&mut self, other: &<Self::P as Atom>::N<'a>, state: &State);
    fn to_num_view<'a>(&'a self) -> <Self::P as Atom>::N<'a>;
}

pub trait OwnedVar: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn from_id(&mut self, id: Identifier);
    fn from_view<'a>(&mut self, view: &<Self::P as Atom>::V<'a>);
    fn to_var_view<'a>(&'a self) -> <Self::P as Atom>::V<'a>;
}

pub trait OwnedFun: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn from_view<'a>(&mut self, view: &<Self::P as Atom>::F<'a>);
    fn from_name(&mut self, id: Identifier);
    fn set_dirty(&mut self, dirty: bool);
    fn add_arg(&mut self, other: AtomView<Self::P>);
    fn to_fun_view<'a>(&'a self) -> <Self::P as Atom>::F<'a>;
}

pub trait OwnedPow: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn from_view<'a>(&mut self, view: &<Self::P as Atom>::P<'a>);
    fn from_base_and_exp(&mut self, base: AtomView<'_, Self::P>, exp: AtomView<'_, Self::P>);
    fn set_dirty(&mut self, dirty: bool);
    fn to_pow_view(&self) -> <Self::P as Atom>::P<'_>;
}

pub trait OwnedMul: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_dirty(&mut self, dirty: bool);
    fn from_view<'a>(&mut self, view: &<Self::P as Atom>::M<'a>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn replace_last(&mut self, other: AtomView<Self::P>);
    fn to_mul_view<'a>(&'a self) -> <Self::P as Atom>::M<'a>;
}

pub trait OwnedAdd: Clone + ResettableBuffer + Convert<Self::P> {
    type P: Atom;

    fn set_dirty(&mut self, dirty: bool);
    fn from_view<'a>(&mut self, view: &<Self::P as Atom>::A<'a>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_add_view<'a>(&'a self) -> <Self::P as Atom>::A<'a>;
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
    fn into_iter(&self) -> Self::I;
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
    fn into_iter(&self) -> Self::I;
    fn to_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as Atom>::S<'a>;
}

pub trait Add<'a>: Copy + Clone + for<'b> PartialEq<<Self::P as Atom>::A<'b>> {
    type P: Atom;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn is_dirty(&self) -> bool;
    fn get_nargs(&self) -> usize;
    fn into_iter(&self) -> Self::I;
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

    fn into_iter(&self) -> Self::ListSliceIterator;
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
            Self::Num(arg0) => Self::Num(arg0.clone()),
            Self::Var(arg0) => Self::Var(arg0.clone()),
            Self::Fun(arg0) => Self::Fun(arg0.clone()),
            Self::Pow(arg0) => Self::Pow(arg0.clone()),
            Self::Mul(arg0) => Self::Mul(arg0.clone()),
            Self::Add(arg0) => Self::Add(arg0.clone()),
        }
    }
}

impl<'a, P: Atom> Copy for AtomView<'a, P> {}

impl<'a, 'b, P: Atom> PartialEq<AtomView<'b, P>> for AtomView<'a, P> {
    fn eq(&self, other: &AtomView<P>) -> bool {
        match (self, other) {
            (AtomView::Num(n1), AtomView::Num(n2)) => n1 == n2,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1 == v2,
            (AtomView::Fun(f1), AtomView::Fun(f2)) => f1 == f2,
            (AtomView::Pow(p1), AtomView::Pow(p2)) => p1 == p2,
            (AtomView::Mul(m1), AtomView::Mul(m2)) => m1 == m2,
            (AtomView::Add(a1), AtomView::Add(a2)) => a1 == a2,
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
        let mut ov = std::mem::replace(self, OwnedAtom::Empty);

        *self = match ov {
            OwnedAtom::Num(_) => {
                ov.reset();
                ov
            }
            OwnedAtom::Var(v) => OwnedAtom::Num(v.to_owned_num()),
            OwnedAtom::Fun(f) => OwnedAtom::Num(f.to_owned_num()),
            OwnedAtom::Pow(p) => OwnedAtom::Num(p.to_owned_num()),
            OwnedAtom::Mul(m) => OwnedAtom::Num(m.to_owned_num()),
            OwnedAtom::Add(a) => OwnedAtom::Num(a.to_owned_num()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Num(n) => n,
            _ => unreachable!(),
        }
    }

    pub fn transform_to_pow(&mut self) -> &mut P::OP {
        let mut ov = std::mem::replace(self, OwnedAtom::Empty);

        *self = match ov {
            OwnedAtom::Pow(_) => {
                ov.reset();
                ov
            }
            OwnedAtom::Num(n) => OwnedAtom::Pow(n.to_owned_pow()),
            OwnedAtom::Var(v) => OwnedAtom::Pow(v.to_owned_pow()),
            OwnedAtom::Fun(f) => OwnedAtom::Pow(f.to_owned_pow()),
            OwnedAtom::Mul(m) => OwnedAtom::Pow(m.to_owned_pow()),
            OwnedAtom::Add(a) => OwnedAtom::Pow(a.to_owned_pow()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Pow(p) => p,
            _ => unreachable!(),
        }
    }

    pub fn transform_to_var(&mut self) -> &mut P::OV {
        let mut ov = std::mem::replace(self, OwnedAtom::Empty);

        *self = match ov {
            OwnedAtom::Var(_) => {
                ov.reset();
                ov
            }
            OwnedAtom::Num(n) => OwnedAtom::Var(n.to_owned_var()),
            OwnedAtom::Pow(p) => OwnedAtom::Var(p.to_owned_var()),
            OwnedAtom::Fun(f) => OwnedAtom::Var(f.to_owned_var()),
            OwnedAtom::Mul(m) => OwnedAtom::Var(m.to_owned_var()),
            OwnedAtom::Add(a) => OwnedAtom::Var(a.to_owned_var()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Var(v) => v,
            _ => unreachable!(),
        }
    }

    pub fn transform_to_fun(&mut self) -> &mut P::OF {
        let mut of = std::mem::replace(self, OwnedAtom::Empty);

        *self = match of {
            OwnedAtom::Fun(_) => {
                of.reset();
                of
            }
            OwnedAtom::Num(n) => OwnedAtom::Fun(n.to_owned_fun()),
            OwnedAtom::Pow(p) => OwnedAtom::Fun(p.to_owned_fun()),
            OwnedAtom::Var(v) => OwnedAtom::Fun(v.to_owned_fun()),
            OwnedAtom::Mul(m) => OwnedAtom::Fun(m.to_owned_fun()),
            OwnedAtom::Add(a) => OwnedAtom::Fun(a.to_owned_fun()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Fun(f) => f,
            _ => unreachable!(),
        }
    }

    pub fn transform_to_mul(&mut self) -> &mut P::OM {
        let mut om = std::mem::replace(self, OwnedAtom::Empty);

        *self = match om {
            OwnedAtom::Mul(_) => {
                om.reset();
                om
            }
            OwnedAtom::Num(n) => OwnedAtom::Mul(n.to_owned_mul()),
            OwnedAtom::Pow(p) => OwnedAtom::Mul(p.to_owned_mul()),
            OwnedAtom::Var(v) => OwnedAtom::Mul(v.to_owned_mul()),
            OwnedAtom::Fun(f) => OwnedAtom::Mul(f.to_owned_mul()),
            OwnedAtom::Add(a) => OwnedAtom::Mul(a.to_owned_mul()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Mul(m) => m,
            _ => unreachable!(),
        }
    }

    pub fn transform_to_add(&mut self) -> &mut P::OA {
        let mut oa = std::mem::replace(self, OwnedAtom::Empty);

        *self = match oa {
            OwnedAtom::Add(_) => {
                oa.reset();
                oa
            }
            OwnedAtom::Num(n) => OwnedAtom::Add(n.to_owned_add()),
            OwnedAtom::Pow(p) => OwnedAtom::Add(p.to_owned_add()),
            OwnedAtom::Var(v) => OwnedAtom::Add(v.to_owned_add()),
            OwnedAtom::Fun(f) => OwnedAtom::Add(f.to_owned_add()),
            OwnedAtom::Mul(m) => OwnedAtom::Add(m.to_owned_add()),
            OwnedAtom::Empty => unreachable!(),
        };

        match self {
            OwnedAtom::Add(a) => a,
            _ => unreachable!(),
        }
    }

    /// This function allocates a new OwnedAtom with the same content as `view`.
    pub fn new_from_view(view: &AtomView<P>) -> OwnedAtom<P> {
        let mut owned = OwnedAtom::new();
        owned.from_view(view);
        owned
    }

    pub fn from_view(&mut self, view: &AtomView<P>) {
        match view {
            AtomView::Num(n) => {
                let num = self.transform_to_num();
                num.from_view(n);
            }
            AtomView::Var(v) => {
                let var = self.transform_to_var();
                var.from_view(v);
            }
            AtomView::Fun(f) => {
                let fun = self.transform_to_fun();
                fun.from_view(f);
            }
            AtomView::Pow(p) => {
                let pow = self.transform_to_pow();
                pow.from_view(p);
            }
            AtomView::Mul(m) => {
                let mul = self.transform_to_mul();
                mul.from_view(m);
            }
            AtomView::Add(a) => {
                let add = self.transform_to_add();
                add.from_view(a);
            }
        }
    }

    pub fn to_view<'a>(&'a self) -> AtomView<'a, P> {
        match self {
            OwnedAtom::Num(n) => AtomView::Num(n.to_num_view()),
            OwnedAtom::Var(v) => AtomView::Var(v.to_var_view()),
            OwnedAtom::Fun(f) => AtomView::Fun(f.to_fun_view()),
            OwnedAtom::Pow(p) => AtomView::Pow(p.to_pow_view()),
            OwnedAtom::Mul(m) => AtomView::Mul(m.to_mul_view()),
            OwnedAtom::Add(a) => AtomView::Add(a.to_add_view()),
            OwnedAtom::Empty => unreachable!(),
        }
    }
}

impl<P: Atom> ResettableBuffer for OwnedAtom<P> {
    fn new() -> Self {
        Self::Num(P::ON::new())
    }

    fn reset(&mut self) {
        match self {
            OwnedAtom::Num(n) => n.reset(),
            OwnedAtom::Var(v) => v.reset(),
            OwnedAtom::Fun(f) => f.reset(),
            OwnedAtom::Pow(p) => p.reset(),
            OwnedAtom::Mul(m) => m.reset(),
            OwnedAtom::Add(a) => a.reset(),
            OwnedAtom::Empty => {}
        }
    }
}
