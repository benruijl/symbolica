mod coefficient;
pub mod default;

use crate::{
    coefficient::{Coefficient, CoefficientView},
    parser::Token,
    printer::AtomPrinter,
    state::{BufferHandle, ResettableBuffer, State, Workspace},
};
use std::{
    cmp::Ordering,
    hash::Hash,
    ops::{DerefMut, Range},
};

use self::default::Linear;

/// An identifier, for example for a variable or function.
/// Should be created using `get_or_insert` of `State`.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier(u32);

impl Identifier {
    pub(crate) const fn init(value: u32) -> Self {
        Identifier(value)
    }
}

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

/// Represents the collection of all types appearing in a mathematical expression, where
/// each type has a compatible memory representation.
pub trait AtomSet: Copy + Clone + PartialEq + Eq + Hash + Send + 'static {
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
}

/// Convert the owned atoms by recycling and clearing their internal buffers.
pub trait Convert<P: AtomSet> {
    fn to_owned_var(self) -> P::OV;
    fn to_owned_num(self) -> P::ON;
    fn to_owned_fun(self) -> P::OF;
    fn to_owned_pow(self) -> P::OP;
    fn to_owned_add(self) -> P::OA;
    fn to_owned_mul(self) -> P::OM;
}

pub trait OwnedNum: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_from_coeff(&mut self, num: Coefficient);
    fn set_from_view(&mut self, a: &<Self::P as AtomSet>::N<'_>);
    fn add(&mut self, other: &<Self::P as AtomSet>::N<'_>, state: &State);
    fn mul(&mut self, other: &<Self::P as AtomSet>::N<'_>, state: &State);
    fn to_num_view(&self) -> <Self::P as AtomSet>::N<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait OwnedVar: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_from_id(&mut self, id: Identifier);
    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::V<'_>);
    fn to_var_view(&self) -> <Self::P as AtomSet>::V<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait OwnedFun: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::F<'_>);
    fn set_from_name(&mut self, id: Identifier);
    fn set_dirty(&mut self, dirty: bool);
    fn add_arg(&mut self, other: AtomView<Self::P>);
    fn to_fun_view(&self) -> <Self::P as AtomSet>::F<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait OwnedPow: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::P<'_>);
    fn set_from_base_and_exp(&mut self, base: AtomView<'_, Self::P>, exp: AtomView<'_, Self::P>);
    fn set_dirty(&mut self, dirty: bool);
    fn to_pow_view(&self) -> <Self::P as AtomSet>::P<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait OwnedMul: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_dirty(&mut self, dirty: bool);
    fn set_has_coefficient(&mut self, has_coeff: bool);
    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::M<'_>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn replace_last(&mut self, other: AtomView<Self::P>);
    fn to_mul_view(&self) -> <Self::P as AtomSet>::M<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait OwnedAdd: Clone + PartialEq + Hash + Send + ResettableBuffer + Convert<Self::P> {
    type P: AtomSet;

    fn set_dirty(&mut self, dirty: bool);
    fn set_from_view(&mut self, view: &<Self::P as AtomSet>::A<'_>);
    fn extend(&mut self, other: AtomView<Self::P>);
    fn to_add_view(&self) -> <Self::P as AtomSet>::A<'_>;
    fn as_view(&self) -> AtomView<Self::P>;
}

pub trait Num<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::N<'b>> {
    type P: AtomSet;

    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_dirty(&self) -> bool;
    fn get_coeff_view(&self) -> CoefficientView<'_>;
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn get_byte_size(&self) -> usize;
}

pub trait Var<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::V<'b>> {
    type P: AtomSet;

    fn get_name(&self) -> Identifier;
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn get_byte_size(&self) -> usize;
}

pub trait Fun<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::F<'b>> {
    type P: AtomSet;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn get_name(&self) -> Identifier;
    fn get_nargs(&self) -> usize;
    fn is_dirty(&self) -> bool;
    fn iter(&self) -> Self::I;
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as AtomSet>::S<'a>;
    fn get_byte_size(&self) -> usize;

    /// Perform a fast comparison between two functions. This function may use
    /// conditions that rely on the underlying data format and is not suitable for human interpretation.
    fn fast_cmp(&self, other: <Self::P as AtomSet>::F<'_>) -> Ordering;
}

pub trait Pow<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::P<'b>> {
    type P: AtomSet;

    fn get_base(&self) -> AtomView<'a, Self::P>;
    fn get_exp(&self) -> AtomView<'a, Self::P>;
    fn is_dirty(&self) -> bool;
    fn get_base_exp(&self) -> (AtomView<'a, Self::P>, AtomView<'a, Self::P>);
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn get_byte_size(&self) -> usize;

    /// Interpret `x^y` as slice `[x,y]`.
    fn to_slice(&self) -> <Self::P as AtomSet>::S<'a>;
}

pub trait Mul<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::M<'b>> {
    type P: AtomSet;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn is_dirty(&self) -> bool;
    fn get_nargs(&self) -> usize;
    fn iter(&self) -> Self::I;
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as AtomSet>::S<'a>;
    fn get_byte_size(&self) -> usize;

    /// Return true if the multiplication has a coefficient that is not 1
    fn has_coefficient(&self) -> bool;
}

pub trait Add<'a>: Copy + Clone + Hash + for<'b> PartialEq<<Self::P as AtomSet>::A<'b>> {
    type P: AtomSet;
    type I: Iterator<Item = AtomView<'a, Self::P>>;

    fn is_dirty(&self) -> bool;
    fn get_nargs(&self) -> usize;
    fn iter(&self) -> Self::I;
    fn as_view(&self) -> AtomView<'a, Self::P>;
    fn to_slice(&self) -> <Self::P as AtomSet>::S<'a>;
    fn get_byte_size(&self) -> usize;
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
    type P: AtomSet;
    type ListSliceIterator: Iterator<Item = AtomView<'a, Self::P>>;

    fn iter(&self) -> Self::ListSliceIterator;
    fn from_one(view: AtomView<'a, Self::P>) -> Self;
    fn get_type(&self) -> SliceType;
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> AtomView<'a, Self::P>;
    fn get_subslice(&self, range: Range<usize>) -> Self;
    fn eq(&self, other: &<Self::P as AtomSet>::S<'_>) -> bool;
}

pub enum AtomView<'a, P: AtomSet = Linear> {
    Num(P::N<'a>),
    Var(P::V<'a>),
    Fun(P::F<'a>),
    Pow(P::P<'a>),
    Mul(P::M<'a>),
    Add(P::A<'a>),
}

impl<'a, P: AtomSet> Clone for AtomView<'a, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, P: AtomSet> Copy for AtomView<'a, P> {}

impl<'a, 'b, P: AtomSet> PartialEq<AtomView<'b, P>> for AtomView<'a, P> {
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

impl<'a, P: AtomSet> Eq for AtomView<'a, P> {}

impl<'a, P: AtomSet> PartialOrd for AtomView<'a, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, P: AtomSet> Ord for AtomView<'a, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl<'a, P: AtomSet> Hash for AtomView<'a, P> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AtomView::Num(a) => a.hash(state),
            AtomView::Var(a) => a.hash(state),
            AtomView::Fun(a) => a.hash(state),
            AtomView::Pow(a) => a.hash(state),
            AtomView::Mul(a) => a.hash(state),
            AtomView::Add(a) => a.hash(state),
        }
    }
}

/// A trait for any type that can be converted into an `AtomView`.
/// To be used for functions that accept any argument that can be
/// converted to an `AtomView`.
pub trait AsAtomView<'a, P: AtomSet>: Sized {
    fn as_atom_view(self) -> AtomView<'a, P>;

    /// Create a builder of an atom. Can be used for easy
    /// construction of terms.
    fn builder<'b>(
        self,
        state: &'b State,
        workspace: &'b Workspace<P>,
    ) -> AtomBuilder<'b, BufferHandle<'b, Atom<P>>, P> {
        AtomBuilder::new(self, state, workspace, workspace.new_atom())
    }

    fn add<'b, T: AsAtomView<'b, P>>(
        self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: T,
        out: &mut Atom<P>,
    ) {
        AtomView::add(
            &self.as_atom_view(),
            state,
            workspace,
            rhs.as_atom_view(),
            out,
        )
    }

    fn mul<'b, T: AsAtomView<'b, P>>(
        self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: T,
        out: &mut Atom<P>,
    ) {
        AtomView::mul(
            &self.as_atom_view(),
            state,
            workspace,
            rhs.as_atom_view(),
            out,
        )
    }

    fn div<'b, T: AsAtomView<'b, P>>(
        self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: T,
        out: &mut Atom<P>,
    ) {
        AtomView::div(
            &self.as_atom_view(),
            state,
            workspace,
            rhs.as_atom_view(),
            out,
        )
    }

    fn pow<'b, T: AsAtomView<'b, P>>(
        self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: T,
        out: &mut Atom<P>,
    ) {
        AtomView::pow(
            &self.as_atom_view(),
            state,
            workspace,
            rhs.as_atom_view(),
            out,
        )
    }

    fn neg(self, state: &State, workspace: &Workspace<P>, out: &mut Atom<P>) {
        AtomView::neg(&self.as_atom_view(), state, workspace, out)
    }
}

impl<'a, P: AtomSet> AsAtomView<'a, P> for AtomView<'a, P> {
    fn as_atom_view(self) -> AtomView<'a, P> {
        self
    }
}

impl<'a, P: AtomSet> AsAtomView<'a, P> for &'a Atom<P> {
    fn as_atom_view(self) -> AtomView<'a, P> {
        self.as_view()
    }
}

impl<'a, P: AtomSet> From<AtomView<'a, P>> for Atom<P> {
    /// Convert an `AtomView` into an `Atom` by allocating.
    fn from(val: AtomView<'a, P>) -> Self {
        Atom::new_from_view(&val)
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Create a pretty-printer for an atom.
    pub fn printer<'b>(&self, state: &'b State) -> AtomPrinter<'a, 'b, P> {
        AtomPrinter::new(*self, state)
    }

    /// Add two atoms and return the buffer that contains the unnormalized result.
    fn add_no_norm<'b>(
        &self,
        workspace: &'b Workspace<P>,
        rhs: AtomView<'_, P>,
    ) -> BufferHandle<'b, Atom<P>> {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs);
        a.set_dirty(true);
        e
    }

    /// Subtract two atoms and return the buffer that contains the unnormalized result.
    fn sub_no_norm<'b>(
        &self,
        workspace: &'b Workspace<P>,
        rhs: AtomView<'_, P>,
    ) -> BufferHandle<'b, Atom<P>> {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs.neg_no_norm(workspace).as_atom_view());
        a.set_dirty(true);
        e
    }

    /// Multiply two atoms and return the buffer that contains the unnormalized result.
    fn mul_no_norm<'b>(
        &self,
        workspace: &'b Workspace<P>,
        rhs: AtomView<'_, P>,
    ) -> BufferHandle<'b, Atom<P>> {
        let mut e = workspace.new_atom();
        let a = e.to_mul();

        // TODO: check if self or rhs is mul
        a.extend(*self);
        a.extend(rhs);
        a.set_dirty(true);
        e
    }

    /// Construct `self^exp` and return the buffer that contains the unnormalized result.
    fn pow_no_norm<'b>(
        &self,
        workspace: &'b Workspace<P>,
        exp: AtomView<'_, P>,
    ) -> BufferHandle<'b, Atom<P>> {
        let mut e = workspace.new_atom();
        let a = e.to_pow();
        a.set_from_base_and_exp(*self, exp);
        a.set_dirty(true);
        e
    }

    /// Divide `self` by `div` and return the buffer that contains the unnormalized result.
    fn div_no_norm<'b>(
        &self,
        workspace: &'b Workspace<P>,
        div: AtomView<'_, P>,
    ) -> BufferHandle<'b, Atom<P>> {
        self.mul_no_norm(
            workspace,
            div.pow_no_norm(workspace, workspace.new_num(-1).as_view())
                .as_view(),
        )
    }

    /// Negate `self` and return the buffer that contains the unnormalized result.
    fn neg_no_norm<'b>(&self, workspace: &'b Workspace<P>) -> BufferHandle<'b, Atom<P>> {
        self.mul_no_norm(workspace, workspace.new_num(-1).as_view())
    }

    /// Add `self` and `rhs`, writing the result in `out`.
    pub fn add(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: AtomView<'_, P>,
        out: &mut Atom<P>,
    ) {
        self.add_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, state, out);
    }

    /// Multiply `self` and `rhs`, writing the result in `out`.
    pub fn mul(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        rhs: AtomView<'_, P>,
        out: &mut Atom<P>,
    ) {
        self.mul_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, state, out);
    }

    /// Construct `self^exp`, writing the result in `out`.
    pub fn pow(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        exp: AtomView<'_, P>,
        out: &mut Atom<P>,
    ) {
        self.pow_no_norm(workspace, exp)
            .as_view()
            .normalize(workspace, state, out);
    }

    /// Divide `self` by `div`, writing the result in `out`.
    pub fn div(
        &self,
        state: &State,
        workspace: &Workspace<P>,
        div: AtomView<'_, P>,
        out: &mut Atom<P>,
    ) {
        self.div_no_norm(workspace, div)
            .as_view()
            .normalize(workspace, state, out);
    }

    /// Negate `self`, writing the result in `out`.
    pub fn neg(&self, state: &State, workspace: &Workspace<P>, out: &mut Atom<P>) {
        self.neg_no_norm(workspace)
            .as_view()
            .normalize(workspace, state, out);
    }

    pub fn get_byte_size(&self) -> usize {
        match self {
            AtomView::Num(n) => n.get_byte_size(),
            AtomView::Var(v) => v.get_byte_size(),
            AtomView::Fun(f) => f.get_byte_size(),
            AtomView::Pow(p) => p.get_byte_size(),
            AtomView::Mul(m) => m.get_byte_size(),
            AtomView::Add(a) => a.get_byte_size(),
        }
    }
}

#[derive(Copy, Clone)]
pub enum Atom<P: AtomSet = Linear> {
    Num(P::ON),
    Var(P::OV),
    Fun(P::OF),
    Pow(P::OP),
    Mul(P::OM),
    Add(P::OA),
    Empty, // for internal use only
}

impl<P: AtomSet> PartialEq for Atom<P> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Num(l0), Self::Num(r0)) => l0 == r0,
            (Self::Var(l0), Self::Var(r0)) => l0 == r0,
            (Self::Fun(l0), Self::Fun(r0)) => l0 == r0,
            (Self::Pow(l0), Self::Pow(r0)) => l0 == r0,
            (Self::Mul(l0), Self::Mul(r0)) => l0 == r0,
            (Self::Add(l0), Self::Add(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<P: AtomSet> Eq for Atom<P> {}

impl<P: AtomSet> Hash for Atom<P> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Atom::Num(a) => a.hash(state),
            Atom::Var(a) => a.hash(state),
            Atom::Fun(a) => a.hash(state),
            Atom::Pow(a) => a.hash(state),
            Atom::Mul(a) => a.hash(state),
            Atom::Add(a) => a.hash(state),
            Atom::Empty => 1.hash(state),
        }
    }
}

impl<P: AtomSet> std::fmt::Debug for Atom<P> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl<P: AtomSet> Atom<P> {
    /// Parse and atom from a string.
    pub fn parse(
        input: &str,
        state: &mut State,
        workspace: &Workspace<P>,
    ) -> Result<Atom<P>, String> {
        Token::parse(input)?.to_atom(state, workspace)
    }

    pub fn new_var(id: Identifier) -> Atom<P> {
        let mut owned = Self::new();
        owned.to_var().set_from_id(id);
        owned
    }

    pub fn new_num<T: Into<Coefficient>>(num: T) -> Atom<P> {
        let mut owned = Self::new();
        owned.to_num().set_from_coeff(num.into());
        owned
    }

    /// Create a pretty-printer for an atom.
    pub fn printer<'a, 'b>(&'a self, state: &'b State) -> AtomPrinter<'a, 'b, P> {
        AtomPrinter::new(self.as_view(), state)
    }

    /// Convert the owned atom to a `OwnedAtom::Num(n)`, returning a reference to `n`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_num(&mut self) -> &mut P::ON {
        let mut ov = std::mem::replace(self, Atom::Empty);

        *self = match ov {
            Atom::Num(_) => {
                ov.reset();
                ov
            }
            Atom::Var(v) => Atom::Num(v.to_owned_num()),
            Atom::Fun(f) => Atom::Num(f.to_owned_num()),
            Atom::Pow(p) => Atom::Num(p.to_owned_num()),
            Atom::Mul(m) => Atom::Num(m.to_owned_num()),
            Atom::Add(a) => Atom::Num(a.to_owned_num()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Num(n) => n,
            _ => unreachable!(),
        }
    }

    /// Convert the owned atom to a `OwnedAtom::Pow(p)`, returning a reference to `p`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_pow(&mut self) -> &mut P::OP {
        let mut ov = std::mem::replace(self, Atom::Empty);

        *self = match ov {
            Atom::Pow(_) => {
                ov.reset();
                ov
            }
            Atom::Num(n) => Atom::Pow(n.to_owned_pow()),
            Atom::Var(v) => Atom::Pow(v.to_owned_pow()),
            Atom::Fun(f) => Atom::Pow(f.to_owned_pow()),
            Atom::Mul(m) => Atom::Pow(m.to_owned_pow()),
            Atom::Add(a) => Atom::Pow(a.to_owned_pow()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Pow(p) => p,
            _ => unreachable!(),
        }
    }

    /// Convert the owned atom to a `OwnedAtom::Var(v)`, returning a reference to `v`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_var(&mut self) -> &mut P::OV {
        let mut ov = std::mem::replace(self, Atom::Empty);

        *self = match ov {
            Atom::Var(_) => {
                ov.reset();
                ov
            }
            Atom::Num(n) => Atom::Var(n.to_owned_var()),
            Atom::Pow(p) => Atom::Var(p.to_owned_var()),
            Atom::Fun(f) => Atom::Var(f.to_owned_var()),
            Atom::Mul(m) => Atom::Var(m.to_owned_var()),
            Atom::Add(a) => Atom::Var(a.to_owned_var()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Var(v) => v,
            _ => unreachable!(),
        }
    }

    /// Convert the owned atom to a `OwnedAtom::Fun(f)`, returning a reference to `f`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_fun(&mut self) -> &mut P::OF {
        let mut of = std::mem::replace(self, Atom::Empty);

        *self = match of {
            Atom::Fun(_) => {
                of.reset();
                of
            }
            Atom::Num(n) => Atom::Fun(n.to_owned_fun()),
            Atom::Pow(p) => Atom::Fun(p.to_owned_fun()),
            Atom::Var(v) => Atom::Fun(v.to_owned_fun()),
            Atom::Mul(m) => Atom::Fun(m.to_owned_fun()),
            Atom::Add(a) => Atom::Fun(a.to_owned_fun()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Fun(f) => f,
            _ => unreachable!(),
        }
    }

    /// Convert the owned atom to a `OwnedAtom::Mul(m)`, returning a reference to `m`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_mul(&mut self) -> &mut P::OM {
        let mut om = std::mem::replace(self, Atom::Empty);

        *self = match om {
            Atom::Mul(_) => {
                om.reset();
                om
            }
            Atom::Num(n) => Atom::Mul(n.to_owned_mul()),
            Atom::Pow(p) => Atom::Mul(p.to_owned_mul()),
            Atom::Var(v) => Atom::Mul(v.to_owned_mul()),
            Atom::Fun(f) => Atom::Mul(f.to_owned_mul()),
            Atom::Add(a) => Atom::Mul(a.to_owned_mul()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Mul(m) => m,
            _ => unreachable!(),
        }
    }

    /// Convert the owned atom to a `OwnedAtom::Add(a)`, returning a reference to `a`.
    /// This destroys any previous content of the owned atom, but reuses the memory.
    #[inline]
    pub fn to_add(&mut self) -> &mut P::OA {
        let mut oa = std::mem::replace(self, Atom::Empty);

        *self = match oa {
            Atom::Add(_) => {
                oa.reset();
                oa
            }
            Atom::Num(n) => Atom::Add(n.to_owned_add()),
            Atom::Pow(p) => Atom::Add(p.to_owned_add()),
            Atom::Var(v) => Atom::Add(v.to_owned_add()),
            Atom::Fun(f) => Atom::Add(f.to_owned_add()),
            Atom::Mul(m) => Atom::Add(m.to_owned_add()),
            Atom::Empty => unreachable!(),
        };

        match self {
            Atom::Add(a) => a,
            _ => unreachable!(),
        }
    }

    /// This function allocates a new OwnedAtom with the same content as `view`.
    pub fn new_from_view(view: &AtomView<P>) -> Atom<P> {
        let mut owned = Atom::new();
        owned.set_from_view(view);
        owned
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &AtomView<P>) {
        match view {
            AtomView::Num(n) => {
                let num = self.to_num();
                num.set_from_view(n);
            }
            AtomView::Var(v) => {
                let var = self.to_var();
                var.set_from_view(v);
            }
            AtomView::Fun(f) => {
                let fun = self.to_fun();
                fun.set_from_view(f);
            }
            AtomView::Pow(p) => {
                let pow = self.to_pow();
                pow.set_from_view(p);
            }
            AtomView::Mul(m) => {
                let mul = self.to_mul();
                mul.set_from_view(m);
            }
            AtomView::Add(a) => {
                let add = self.to_add();
                add.set_from_view(a);
            }
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_, P> {
        match self {
            Atom::Num(n) => AtomView::Num(n.to_num_view()),
            Atom::Var(v) => AtomView::Var(v.to_var_view()),
            Atom::Fun(f) => AtomView::Fun(f.to_fun_view()),
            Atom::Pow(p) => AtomView::Pow(p.to_pow_view()),
            Atom::Mul(m) => AtomView::Mul(m.to_mul_view()),
            Atom::Add(a) => AtomView::Add(a.to_add_view()),
            Atom::Empty => unreachable!(),
        }
    }
}

impl<P: AtomSet> ResettableBuffer for Atom<P> {
    fn new() -> Self {
        Self::Num(P::ON::new())
    }

    #[inline(always)]
    fn reset(&mut self) {
        match self {
            Atom::Num(n) => n.reset(),
            Atom::Var(v) => v.reset(),
            Atom::Fun(f) => f.reset(),
            Atom::Pow(p) => p.reset(),
            Atom::Mul(m) => m.reset(),
            Atom::Add(a) => a.reset(),
            Atom::Empty => {}
        }
    }
}

/// A constructor of a function, that wraps the state and workspace
///
/// For example:
/// ```
/// # use symbolica::{
/// #     representations::{AsAtomView, FunctionBuilder},
/// #     state::{FunctionAttribute, State, Workspace},
/// # };
/// # fn main() {
/// let mut state = State::new();
/// let ws: Workspace = Workspace::new();
///
/// let f_id = state.get_or_insert_fn("f", Some(vec![FunctionAttribute::Symmetric]));
/// let fb = FunctionBuilder::new(f_id, &state, &ws);
/// let a = fb
///     .add_arg(&ws.new_num(3))
///     .add_arg(&ws.new_num(2))
///     .add_arg(&ws.new_num(1))
///     .finish();
///
/// println!("{}", a.as_atom_view().printer(&state));
/// # }
/// ```
pub struct FunctionBuilder<'a, P: AtomSet = Linear> {
    state: &'a State,
    workspace: &'a Workspace<P>,
    handle: BufferHandle<'a, Atom<P>>,
}

impl<'a, P: AtomSet> FunctionBuilder<'a, P> {
    /// Create a new `FunctionBuilder`.
    pub fn new(
        name: Identifier,
        state: &'a State,
        workspace: &'a Workspace<P>,
    ) -> FunctionBuilder<'a, P> {
        let mut a = workspace.new_atom();
        let f = a.to_fun();
        f.set_from_name(name);
        f.set_dirty(true);
        FunctionBuilder {
            state,
            workspace,
            handle: a,
        }
    }

    /// Add an argument to the function.
    pub fn add_arg<'b, T: AsAtomView<'b, P>>(mut self, arg: T) -> FunctionBuilder<'a, P> {
        if let Atom::Fun(f) = self.handle.get_mut() {
            f.add_arg(arg.as_atom_view());
        }

        self
    }

    /// Finish the function construction and return an `AtomBuilder`.
    pub fn finish(self) -> AtomBuilder<'a, BufferHandle<'a, Atom<P>>, P> {
        let mut out = self.workspace.new_atom();
        self.handle
            .as_view()
            .normalize(self.workspace, self.state, &mut out);

        AtomBuilder {
            state: self.state,
            workspace: self.workspace,
            out,
        }
    }
}

/// A wrapper around an atom, the state and workspace
/// that contains all the necessary information to do
/// arithmetic. To construct a function, see [`FunctionBuilder`].
///
/// For example:
/// ```
/// # use symbolica::{
/// # representations::{AsAtomView, Atom},
/// # state::{State, Workspace},
/// # };
/// # fn main() {
/// let mut state = State::new();
/// let ws: Workspace = Workspace::new();
///
/// let x = Atom::parse("x", &mut state, &ws).unwrap();
/// let y = Atom::parse("y", &mut state, &ws).unwrap();
///
/// let mut xb = x.builder(&state, &ws);
/// xb = (-(xb + &y + &x) * &y * &ws.new_num(6)).pow(&ws.new_num(5)) / &y;
///
/// println!("{}", xb.as_atom_view().printer(&state));
/// # }
/// ```
pub struct AtomBuilder<'a, A: DerefMut<Target = Atom<P>>, P: AtomSet = Linear> {
    state: &'a State,
    workspace: &'a Workspace<P>,
    out: A,
}

impl<'a, P: AtomSet, A: DerefMut<Target = Atom<P>>> AtomBuilder<'a, A, P> {
    /// Create a new `AtomBuilder`.
    pub fn new<'b, T: AsAtomView<'b, P>>(
        atom: T,
        state: &'a State,
        workspace: &'a Workspace<P>,
        mut out: A,
    ) -> AtomBuilder<'a, A, P> {
        out.set_from_view(&atom.as_atom_view());
        AtomBuilder {
            state,
            workspace,
            out,
        }
    }

    /// Yield the mutable reference to the output atom.
    pub fn as_atom_mut(&mut self) -> &mut Atom<P> {
        &mut self.out
    }

    /// Take the `self` to the power `exp`. Use [`AtomBuilder:rpow()`] for the reverse operation.
    pub fn pow<'b, T: AsAtomView<'b, P>>(mut self, exp: T) -> AtomBuilder<'a, A, P> {
        self.out
            .as_view()
            .pow_no_norm(self.workspace, exp.as_atom_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }

    /// Take base` to the power `self`.
    pub fn rpow<'b, T: AsAtomView<'b, P>>(mut self, base: T) -> AtomBuilder<'a, A, P> {
        base.as_atom_view()
            .pow_no_norm(self.workspace, self.out.as_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, P: AtomSet, A: DerefMut<Target = Atom<P>>> From<&AtomBuilder<'a, A, P>>
    for AtomBuilder<'a, BufferHandle<'a, Atom<P>>, P>
{
    fn from(value: &AtomBuilder<'a, A, P>) -> Self {
        let mut h = value.workspace.new_atom();
        h.set_from_view(&value.as_atom_view());
        AtomBuilder {
            state: value.state,
            workspace: value.workspace,
            out: h,
        }
    }
}

impl<'a, P: AtomSet> Clone for AtomBuilder<'a, BufferHandle<'a, Atom<P>>, P> {
    fn clone(&self) -> Self {
        let mut h = self.workspace.new_atom();
        h.set_from_view(&self.as_atom_view());
        AtomBuilder {
            state: self.state,
            workspace: self.workspace,
            out: h,
        }
    }
}

impl<'a, 'b, P: AtomSet, T: AsAtomView<'b, P>, A: DerefMut<Target = Atom<P>>> std::ops::Add<T>
    for AtomBuilder<'a, A, P>
{
    type Output = AtomBuilder<'a, A, P>;

    fn add(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .add_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, 'b, P: AtomSet, T: AsAtomView<'b, P>, A: DerefMut<Target = Atom<P>>> std::ops::Sub<T>
    for AtomBuilder<'a, A, P>
{
    type Output = AtomBuilder<'a, A, P>;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .sub_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, 'b, P: AtomSet, T: AsAtomView<'b, P>, A: DerefMut<Target = Atom<P>>> std::ops::Mul<T>
    for AtomBuilder<'a, A, P>
{
    type Output = AtomBuilder<'a, A, P>;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .mul_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, 'b, P: AtomSet, T: AsAtomView<'b, P>, A: DerefMut<Target = Atom<P>>> std::ops::Div<T>
    for AtomBuilder<'a, A, P>
{
    type Output = AtomBuilder<'a, A, P>;

    fn div(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .div_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, P: AtomSet, A: DerefMut<Target = Atom<P>>> std::ops::Neg for AtomBuilder<'a, A, P> {
    type Output = AtomBuilder<'a, A, P>;

    fn neg(mut self) -> Self::Output {
        self.out
            .as_view()
            .neg_no_norm(self.workspace)
            .as_view()
            .normalize(self.workspace, self.state, &mut self.out);
        self
    }
}

impl<'a, 'b, P: AtomSet, A: DerefMut<Target = Atom<P>>> AsAtomView<'b, P>
    for &'b AtomBuilder<'a, A, P>
{
    fn as_atom_view(self) -> AtomView<'b, P> {
        self.out.as_atom_view()
    }
}
