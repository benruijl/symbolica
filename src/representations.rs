mod coefficient;
pub mod default;

use crate::{
    coefficient::Coefficient,
    parser::Token,
    printer::AtomPrinter,
    state::{BufferHandle, ResettableBuffer, State, Workspace},
};
use std::{cmp::Ordering, hash::Hash, ops::DerefMut};

pub use self::default::{
    Add, AddView, Fun, ListIterator, ListSlice, Mul, MulView, Num, NumView, Pow, PowView, Var,
    VarView,
};
use self::default::{FunView, RawAtom};

/// An identifier, for example for a variable or function.
/// Should be created using `get_or_insert` of `State`.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Identifier {
    id: u32,
    wildcard_level: u8,
    is_symmetric: bool,
    is_antisymmetric: bool,
    is_linear: bool,
}

impl Identifier {
    /// Create a new identifier for a variable. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn init_var(id: u32, wildcard_level: u8) -> Self {
        Identifier {
            id,
            wildcard_level,
            is_symmetric: false,
            is_antisymmetric: false,
            is_linear: false,
        }
    }

    /// Create a new identifier for a function. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn init_fn(
        id: u32,
        wildcard_level: u8,
        is_symmetric: bool,
        is_antisymmetric: bool,
        is_linear: bool,
    ) -> Self {
        Identifier {
            id,
            wildcard_level,
            is_symmetric,
            is_antisymmetric,
            is_linear,
        }
    }

    pub fn get_id(&self) -> u32 {
        self.id
    }

    pub fn get_wildcard_level(&self) -> u8 {
        self.wildcard_level
    }

    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    pub fn is_antisymmetric(&self) -> bool {
        self.is_antisymmetric
    }

    pub fn is_linear(&self) -> bool {
        self.is_linear
    }
}

impl std::fmt::Debug for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", self.id))?;
        for _ in 0..self.wildcard_level {
            f.write_str("_")?;
        }
        Ok(())
    }
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

pub enum AtomView<'a> {
    Num(NumView<'a>),
    Var(VarView<'a>),
    Fun(FunView<'a>),
    Pow(PowView<'a>),
    Mul(MulView<'a>),
    Add(AddView<'a>),
}

impl Clone for AtomView<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AtomView<'_> {}

impl PartialEq<AtomView<'_>> for AtomView<'_> {
    fn eq(&self, other: &AtomView) -> bool {
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

impl Eq for AtomView<'_> {}

impl PartialOrd for AtomView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl Hash for AtomView<'_> {
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

impl std::fmt::Display for AtomView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        AtomPrinter::new(*self).fmt(f)
    }
}

impl<'a> From<NumView<'a>> for AtomView<'a> {
    fn from(n: NumView<'a>) -> AtomView<'a> {
        AtomView::Num(n)
    }
}

impl<'a> From<VarView<'a>> for AtomView<'a> {
    fn from(n: VarView<'a>) -> AtomView<'a> {
        AtomView::Var(n)
    }
}

impl<'a> From<FunView<'a>> for AtomView<'a> {
    fn from(n: FunView<'a>) -> AtomView<'a> {
        AtomView::Fun(n)
    }
}

impl<'a> From<MulView<'a>> for AtomView<'a> {
    fn from(n: MulView<'a>) -> AtomView<'a> {
        AtomView::Mul(n)
    }
}

impl<'a> From<AddView<'a>> for AtomView<'a> {
    fn from(n: AddView<'a>) -> AtomView<'a> {
        AtomView::Add(n)
    }
}

/// A trait for any type that can be converted into an `AtomView`.
/// To be used for functions that accept any argument that can be
/// converted to an `AtomView`.
pub trait AsAtomView<'a>: Sized {
    fn as_atom_view(self) -> AtomView<'a>;

    /// Create a builder of an atom. Can be used for easy
    /// construction of terms.
    fn builder<'b>(self, workspace: &'b Workspace) -> AtomBuilder<'b, BufferHandle<'b, Atom>> {
        AtomBuilder::new(self, workspace, workspace.new_atom())
    }

    fn add<'b, T: AsAtomView<'b>>(self, workspace: &Workspace, rhs: T, out: &mut Atom) {
        AtomView::add(&self.as_atom_view(), workspace, rhs.as_atom_view(), out)
    }

    fn mul<'b, T: AsAtomView<'b>>(self, workspace: &Workspace, rhs: T, out: &mut Atom) {
        AtomView::mul(&self.as_atom_view(), workspace, rhs.as_atom_view(), out)
    }

    fn div<'b, T: AsAtomView<'b>>(self, workspace: &Workspace, rhs: T, out: &mut Atom) {
        AtomView::div(&self.as_atom_view(), workspace, rhs.as_atom_view(), out)
    }

    fn pow<'b, T: AsAtomView<'b>>(self, workspace: &Workspace, rhs: T, out: &mut Atom) {
        AtomView::pow(&self.as_atom_view(), workspace, rhs.as_atom_view(), out)
    }

    fn neg(self, workspace: &Workspace, out: &mut Atom) {
        AtomView::neg(&self.as_atom_view(), workspace, out)
    }
}

impl<'a> AsAtomView<'a> for AtomView<'a> {
    fn as_atom_view(self) -> AtomView<'a> {
        self
    }
}

impl<'a> AsAtomView<'a> for &'a Atom {
    fn as_atom_view(self) -> AtomView<'a> {
        self.as_view()
    }
}

impl<'a> AtomView<'a> {
    pub fn to_owned(&self) -> Atom {
        let mut a = Atom::new();
        a.set_from_view(self);
        a
    }

    pub fn clone_into(&self, target: &mut Atom) {
        target.set_from_view(self);
    }

    /// Add two atoms and return the buffer that contains the unnormalized result.
    fn add_no_norm<'b>(
        &self,
        workspace: &'b Workspace,
        rhs: AtomView<'_>,
    ) -> BufferHandle<'b, Atom> {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs);
        e
    }

    /// Subtract two atoms and return the buffer that contains the unnormalized result.
    fn sub_no_norm<'b>(
        &self,
        workspace: &'b Workspace,
        rhs: AtomView<'_>,
    ) -> BufferHandle<'b, Atom> {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs.neg_no_norm(workspace).as_atom_view());
        e
    }

    /// Multiply two atoms and return the buffer that contains the unnormalized result.
    fn mul_no_norm<'b>(
        &self,
        workspace: &'b Workspace,
        rhs: AtomView<'_>,
    ) -> BufferHandle<'b, Atom> {
        let mut e = workspace.new_atom();
        let a = e.to_mul();

        // TODO: check if self or rhs is mul
        a.extend(*self);
        a.extend(rhs);
        e
    }

    /// Construct `self^exp` and return the buffer that contains the unnormalized result.
    fn pow_no_norm<'b>(
        &self,
        workspace: &'b Workspace,
        exp: AtomView<'_>,
    ) -> BufferHandle<'b, Atom> {
        let mut e = workspace.new_atom();
        e.to_pow(*self, exp);
        e
    }

    /// Divide `self` by `div` and return the buffer that contains the unnormalized result.
    fn div_no_norm<'b>(
        &self,
        workspace: &'b Workspace,
        div: AtomView<'_>,
    ) -> BufferHandle<'b, Atom> {
        self.mul_no_norm(
            workspace,
            div.pow_no_norm(workspace, workspace.new_num(-1).as_view())
                .as_view(),
        )
    }

    /// Negate `self` and return the buffer that contains the unnormalized result.
    fn neg_no_norm<'b>(&self, workspace: &'b Workspace) -> BufferHandle<'b, Atom> {
        self.mul_no_norm(workspace, workspace.new_num(-1).as_view())
    }

    /// Add `self` and `rhs`, writing the result in `out`.
    pub fn add(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.add_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Multiply `self` and `rhs`, writing the result in `out`.
    pub fn mul(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.mul_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Construct `self^exp`, writing the result in `out`.
    pub fn pow(&self, workspace: &Workspace, exp: AtomView<'_>, out: &mut Atom) {
        self.pow_no_norm(workspace, exp)
            .as_view()
            .normalize(workspace, out);
    }

    /// Divide `self` by `div`, writing the result in `out`.
    pub fn div(&self, workspace: &Workspace, div: AtomView<'_>, out: &mut Atom) {
        self.div_no_norm(workspace, div)
            .as_view()
            .normalize(workspace, out);
    }

    /// Negate `self`, writing the result in `out`.
    pub fn neg(&self, workspace: &Workspace, out: &mut Atom) {
        self.neg_no_norm(workspace)
            .as_view()
            .normalize(workspace, out);
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Atom {
    Num(Num),
    Var(Var),
    Fun(Fun),
    Pow(Pow),
    Mul(Mul),
    Add(Add),
    Empty, // for internal use, TODO: deprecate
}

impl Default for Atom {
    #[inline]
    fn default() -> Self {
        Num::zero(RawAtom::new()).into()
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        AtomPrinter::new(self.as_view()).fmt(f)
    }
}

impl std::fmt::Debug for Atom {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl ResettableBuffer for Atom {
    #[inline]
    fn new() -> Self {
        Atom::default()
    }

    #[inline(always)]
    fn reset(&mut self) {
        *self = Num::zero(std::mem::replace(self, Atom::Empty).into_raw()).into();
    }
}

impl From<Num> for Atom {
    fn from(n: Num) -> Atom {
        Atom::Num(n)
    }
}

impl From<Var> for Atom {
    fn from(n: Var) -> Atom {
        Atom::Var(n)
    }
}

impl From<Add> for Atom {
    fn from(n: Add) -> Atom {
        Atom::Add(n)
    }
}

impl From<Mul> for Atom {
    fn from(n: Mul) -> Atom {
        Atom::Mul(n)
    }
}

impl From<Fun> for Atom {
    fn from(n: Fun) -> Atom {
        Atom::Fun(n)
    }
}

impl Atom {
    /// Parse and atom from a string.
    pub fn parse(input: &str, state: &mut State, workspace: &Workspace) -> Result<Atom, String> {
        Token::parse(input)?.to_atom(state, workspace)
    }

    /// Create a pretty-printer for an atom.
    pub fn printer<'a>(&'a self) -> AtomPrinter<'a> {
        AtomPrinter::new(self.as_view())
    }

    #[inline]
    pub fn new_var(id: Identifier) -> Atom {
        Var::new(id).into()
    }

    #[inline]
    pub fn new_num<T: Into<Coefficient>>(num: T) -> Atom {
        Num::new(num.into()).into()
    }

    #[inline]
    pub fn to_num(&mut self, coeff: Coefficient) -> &mut Num {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Num(Num::new_into(coeff, buffer));
        if let Atom::Num(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_var(&mut self, id: Identifier) -> &mut Var {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Var(Var::new_into(id, buffer));
        if let Atom::Var(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_fun(&mut self, id: Identifier) -> &mut Fun {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Fun(Fun::new_into(id, buffer));
        if let Atom::Fun(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_pow(&mut self, base: AtomView, exp: AtomView) -> &mut Pow {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Pow(Pow::new_into(base, exp, buffer));
        if let Atom::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_mul(&mut self) -> &mut Mul {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Mul(Mul::new_into(buffer));
        if let Atom::Mul(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_add(&mut self) -> &mut Add {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Add(Add::new_into(buffer));
        if let Atom::Add(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        match self {
            Atom::Num(n) => n.into_raw(),
            Atom::Var(v) => v.into_raw(),
            Atom::Fun(f) => f.into_raw(),
            Atom::Pow(p) => p.into_raw(),
            Atom::Mul(m) => m.into_raw(),
            Atom::Add(a) => a.into_raw(),
            Atom::Empty => unreachable!("Empty atom"),
        }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &AtomView) {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        match view {
            AtomView::Num(n) => *self = Atom::Num(Num::from_view_into(n, buffer)),
            AtomView::Var(v) => *self = Atom::Var(Var::from_view_into(v, buffer)),
            AtomView::Fun(f) => *self = Atom::Fun(Fun::from_view_into(f, buffer)),
            AtomView::Pow(p) => *self = Atom::Pow(Pow::from_view_into(p, buffer)),
            AtomView::Mul(m) => *self = Atom::Mul(Mul::from_view_into(m, buffer)),
            AtomView::Add(a) => *self = Atom::Add(Add::from_view_into(a, buffer)),
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        match self {
            Atom::Num(n) => AtomView::Num(n.to_num_view()),
            Atom::Var(v) => AtomView::Var(v.to_var_view()),
            Atom::Fun(f) => AtomView::Fun(f.to_fun_view()),
            Atom::Pow(p) => AtomView::Pow(p.to_pow_view()),
            Atom::Mul(m) => AtomView::Mul(m.to_mul_view()),
            Atom::Add(a) => AtomView::Add(a.to_add_view()),
            Atom::Empty => unreachable!("Empty atom"),
        }
    }

    #[inline(always)]
    pub fn set_normalized(&mut self, normalized: bool) {
        match self {
            Atom::Num(_) => {}
            Atom::Var(_) => {}
            Atom::Fun(a) => a.set_normalized(normalized),
            Atom::Pow(a) => a.set_normalized(normalized),
            Atom::Mul(a) => a.set_normalized(normalized),
            Atom::Add(a) => a.set_normalized(normalized),
            Atom::Empty => unreachable!("Empty atom"),
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
/// let fb = FunctionBuilder::new(f_id, &ws);
/// let a = fb
///     .add_arg(&ws.new_num(3))
///     .add_arg(&ws.new_num(2))
///     .add_arg(&ws.new_num(1))
///     .finish();
///
/// println!("{}", a.as_atom_view());
/// # }
/// ```
pub struct FunctionBuilder<'a> {
    workspace: &'a Workspace,
    handle: BufferHandle<'a, Atom>,
}

impl<'a> FunctionBuilder<'a> {
    /// Create a new `FunctionBuilder`.
    pub fn new(name: Identifier, workspace: &'a Workspace) -> FunctionBuilder<'a> {
        let mut a = workspace.new_atom();
        a.to_fun(name);
        FunctionBuilder {
            workspace,
            handle: a,
        }
    }

    /// Add an argument to the function.
    pub fn add_arg<'b, T: AsAtomView<'b>>(mut self, arg: T) -> FunctionBuilder<'a> {
        if let Atom::Fun(f) = self.handle.get_mut() {
            f.add_arg(arg.as_atom_view());
        }

        self
    }

    /// Finish the function construction and return an `AtomBuilder`.
    pub fn finish(self) -> AtomBuilder<'a, BufferHandle<'a, Atom>> {
        let mut out = self.workspace.new_atom();
        self.handle.as_view().normalize(self.workspace, &mut out);

        AtomBuilder {
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
/// let mut xb = x.builder(&ws);
/// xb = (-(xb + &y + &x) * &y * &ws.new_num(6)).pow(&ws.new_num(5)) / &y;
///
/// println!("{}", xb.as_atom_view());
/// # }
/// ```
pub struct AtomBuilder<'a, A: DerefMut<Target = Atom>> {
    workspace: &'a Workspace,
    out: A,
}

impl<'a, A: DerefMut<Target = Atom>> AtomBuilder<'a, A> {
    /// Create a new `AtomBuilder`.
    pub fn new<'b, T: AsAtomView<'b>>(
        atom: T,
        workspace: &'a Workspace,
        mut out: A,
    ) -> AtomBuilder<'a, A> {
        out.set_from_view(&atom.as_atom_view());
        AtomBuilder { workspace, out }
    }

    /// Yield the mutable reference to the output atom.
    pub fn as_atom_mut(&mut self) -> &mut Atom {
        &mut self.out
    }

    /// Take the `self` to the power `exp`. Use [`AtomBuilder:rpow()`] for the reverse operation.
    pub fn pow<'b, T: AsAtomView<'b>>(mut self, exp: T) -> AtomBuilder<'a, A> {
        self.out
            .as_view()
            .pow_no_norm(self.workspace, exp.as_atom_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }

    /// Take base` to the power `self`.
    pub fn rpow<'b, T: AsAtomView<'b>>(mut self, base: T) -> AtomBuilder<'a, A> {
        base.as_atom_view()
            .pow_no_norm(self.workspace, self.out.as_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, A: DerefMut<Target = Atom>> From<&AtomBuilder<'a, A>>
    for AtomBuilder<'a, BufferHandle<'a, Atom>>
{
    fn from(value: &AtomBuilder<'a, A>) -> Self {
        let mut h = value.workspace.new_atom();
        h.set_from_view(&value.as_atom_view());
        AtomBuilder {
            workspace: value.workspace,
            out: h,
        }
    }
}

impl<'a> Clone for AtomBuilder<'a, BufferHandle<'a, Atom>> {
    fn clone(&self) -> Self {
        let mut h = self.workspace.new_atom();
        h.set_from_view(&self.as_atom_view());
        AtomBuilder {
            workspace: self.workspace,
            out: h,
        }
    }
}

impl<'a, 'b, T: AsAtomView<'b>, A: DerefMut<Target = Atom>> std::ops::Add<T>
    for AtomBuilder<'a, A>
{
    type Output = AtomBuilder<'a, A>;

    fn add(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .add_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, 'b, T: AsAtomView<'b>, A: DerefMut<Target = Atom>> std::ops::Sub<T>
    for AtomBuilder<'a, A>
{
    type Output = AtomBuilder<'a, A>;

    fn sub(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .sub_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, 'b, T: AsAtomView<'b>, A: DerefMut<Target = Atom>> std::ops::Mul<T>
    for AtomBuilder<'a, A>
{
    type Output = AtomBuilder<'a, A>;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .mul_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, 'b, T: AsAtomView<'b>, A: DerefMut<Target = Atom>> std::ops::Div<T>
    for AtomBuilder<'a, A>
{
    type Output = AtomBuilder<'a, A>;

    fn div(mut self, rhs: T) -> Self::Output {
        self.out
            .as_view()
            .div_no_norm(self.workspace, rhs.as_atom_view())
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, A: DerefMut<Target = Atom>> std::ops::Neg for AtomBuilder<'a, A> {
    type Output = AtomBuilder<'a, A>;

    fn neg(mut self) -> Self::Output {
        self.out
            .as_view()
            .neg_no_norm(self.workspace)
            .as_view()
            .normalize(self.workspace, &mut self.out);
        self
    }
}

impl<'a, 'b, A: DerefMut<Target = Atom>> AsAtomView<'b> for &'b AtomBuilder<'a, A> {
    fn as_atom_view(self) -> AtomView<'b> {
        self.out.as_atom_view()
    }
}
