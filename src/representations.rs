mod coefficient;
pub mod default;

use crate::{
    coefficient::Coefficient,
    parser::Token,
    printer::AtomPrinter,
    state::{RecycledAtom, Workspace},
};
use std::{cmp::Ordering, hash::Hash, ops::DerefMut};

pub use self::default::{
    Add, AddView, Fun, ListIterator, ListSlice, Mul, MulView, Num, NumView, Pow, PowView, Var,
    VarView,
};
use self::default::{FunView, RawAtom};

/// A symbol, for example the name of a variable or the name of a function,
/// together with its properties.
/// Should be created using `get_symbol` of `State`.
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol {
    id: u32,
    wildcard_level: u8,
    is_symmetric: bool,
    is_antisymmetric: bool,
    is_linear: bool,
}

impl Symbol {
    /// Create a new variable symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn init_var(id: u32, wildcard_level: u8) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric: false,
            is_antisymmetric: false,
            is_linear: false,
        }
    }

    /// Create a new function symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn init_fn(
        id: u32,
        wildcard_level: u8,
        is_symmetric: bool,
        is_antisymmetric: bool,
        is_linear: bool,
    ) -> Self {
        Symbol {
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

impl std::fmt::Debug for Symbol {
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
}

impl<'a> AsAtomView<'a> for AtomView<'a> {
    fn as_atom_view(self) -> AtomView<'a> {
        self
    }
}

impl<'a, T: AsRef<Atom>> AsAtomView<'a> for &'a T {
    fn as_atom_view(self) -> AtomView<'a> {
        self.as_ref().as_view()
    }
}

impl AsRef<Atom> for Atom {
    fn as_ref(&self) -> &Atom {
        self
    }
}

impl<'a> AtomView<'a> {
    pub fn to_owned(&self) -> Atom {
        let mut a = Atom::default();
        a.set_from_view(self);
        a
    }

    pub fn clone_into(&self, target: &mut Atom) {
        target.set_from_view(self);
    }

    /// Add two atoms and return the buffer that contains the unnormalized result.
    fn add_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs);
        e
    }

    /// Subtract two atoms and return the buffer that contains the unnormalized result.
    fn sub_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs.neg_no_norm(workspace).as_view());
        e
    }

    /// Multiply two atoms and return the buffer that contains the unnormalized result.
    fn mul_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let a = e.to_mul();

        // TODO: check if self or rhs is mul
        a.extend(*self);
        a.extend(rhs);
        e
    }

    /// Construct `self^exp` and return the buffer that contains the unnormalized result.
    fn pow_no_norm(&self, workspace: &Workspace, exp: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        e.to_pow(*self, exp);
        e
    }

    /// Divide `self` by `div` and return the buffer that contains the unnormalized result.
    fn div_no_norm(&self, workspace: &Workspace, div: AtomView<'_>) -> RecycledAtom {
        self.mul_no_norm(
            workspace,
            div.pow_no_norm(workspace, workspace.new_num(-1).as_view())
                .as_view(),
        )
    }

    /// Negate `self` and return the buffer that contains the unnormalized result.
    fn neg_no_norm(&self, workspace: &Workspace) -> RecycledAtom {
        self.mul_no_norm(workspace, workspace.new_num(-1).as_view())
    }

    /// Add `self` and `rhs`, writing the result in `out`.
    pub fn add_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.add_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Subtract `rhs` from `self, writing the result in `out`.
    pub fn sub_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.sub_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Multiply `self` and `rhs`, writing the result in `out`.
    pub fn mul_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.mul_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Construct `self^exp`, writing the result in `out`.
    pub fn pow_with_ws_into(&self, workspace: &Workspace, exp: AtomView<'_>, out: &mut Atom) {
        self.pow_no_norm(workspace, exp)
            .as_view()
            .normalize(workspace, out);
    }

    /// Divide `self` by `div`, writing the result in `out`.
    pub fn div_with_ws_into(&self, workspace: &Workspace, div: AtomView<'_>, out: &mut Atom) {
        self.div_no_norm(workspace, div)
            .as_view()
            .normalize(workspace, out);
    }

    /// Negate `self`, writing the result in `out`.
    pub fn neg_with_ws_into(&self, workspace: &Workspace, out: &mut Atom) {
        self.neg_no_norm(workspace)
            .as_view()
            .normalize(workspace, out);
    }

    /// Get the symbol of a variable or function.
    #[inline(always)]
    pub fn get_symbol(&self) -> Option<Symbol> {
        match self {
            AtomView::Var(v) => Some(v.get_symbol()),
            AtomView::Fun(f) => Some(f.get_symbol()),
            _ => None,
        }
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
    Empty, // for internal use
}

impl Default for Atom {
    /// Create an atom that represents the number 0.
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
    /// Create an atom that represents the number 0.
    pub fn new() -> Atom {
        Atom::default()
    }

    /// Parse and atom from a string.
    pub fn parse(input: &str) -> Result<Atom, String> {
        Workspace::get_local().with(|ws| Token::parse(input)?.to_atom(ws))
    }

    #[inline]
    pub fn new_var(id: Symbol) -> Atom {
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
    pub fn to_var(&mut self, id: Symbol) -> &mut Var {
        let buffer = std::mem::replace(self, Atom::Empty).into_raw();
        *self = Atom::Var(Var::new_into(id, buffer));
        if let Atom::Var(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_fun(&mut self, id: Symbol) -> &mut Fun {
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

    /// Get the symbol of a variable or function.
    #[inline(always)]
    pub fn get_symbol(&self) -> Option<Symbol> {
        match self {
            Atom::Var(v) => Some(v.get_symbol()),
            Atom::Fun(f) => Some(f.get_symbol()),
            _ => None,
        }
    }

    #[inline(always)]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
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

/// A constructor of a function,
///
/// For example:
/// ```
/// # use symbolica::{
/// #     representations::{Atom, AsAtomView, FunctionBuilder},
/// #     state::{FunctionAttribute, State},
/// # };
/// # fn main() {
/// ///
/// let f_id = State::get_or_insert_fn("f", Some(vec![FunctionAttribute::Symmetric])).unwrap();
/// let fb = FunctionBuilder::new(f_id);
/// let a = fb
///     .add_arg(&Atom::new_num(3))
///     .add_arg(&Atom::new_num(2))
///     .add_arg(&Atom::new_num(1))
///     .finish();
///
/// println!("{}", a);
/// # }
/// ```
pub struct FunctionBuilder {
    handle: RecycledAtom,
}

impl FunctionBuilder {
    /// Create a new `FunctionBuilder`.
    pub fn new(name: Symbol) -> FunctionBuilder {
        let mut a = RecycledAtom::new();
        a.to_fun(name);
        FunctionBuilder { handle: a }
    }

    /// Add an argument to the function.
    pub fn add_arg<'b, T: AsAtomView<'b>>(mut self, arg: T) -> FunctionBuilder {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            f.add_arg(arg.as_atom_view());
        }

        self
    }

    /// Finish the function construction and return an `Atom`.
    pub fn finish(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut f = ws.new_atom();
            self.handle.as_view().normalize(ws, &mut f);
            f.into_inner()
        })
    }
}

/// Create a new function by providing its name as the first argument,
/// followed by the list of arguments. This macro uses [`FunctionBuilder`].
#[macro_export]
macro_rules! fun {
    ($name:ident, $($id:expr),*) => {
        {
            let mut f = FunctionBuilder::new($name);
            $(
                f = f.add_arg(&$id);
            )+
            f.finish()
        }
    };
}

impl Atom {
    /// Take the `self` to a numerical power `exp`
    pub fn npow<T: Into<Coefficient>>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(exp);
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, n.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Take the `self` to the power `exp`. Use [`Atom::npow()`] for a numerical power and [`Atom::rpow()`] for the reverse operation.
    pub fn pow<'a, T: AsAtomView<'a>>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, exp.as_atom_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Take `base` to the power `self`.
    pub fn rpow<'a, T: AsAtomView<'a>>(&self, base: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            base.as_atom_view()
                .pow_no_norm(ws, self.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Add<Atom> for &Atom {
    type Output = Atom;

    fn add(self, mut rhs: Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut rhs, &mut t);
        });

        rhs
    }
}

impl std::ops::Sub<Atom> for &Atom {
    type Output = Atom;

    fn sub(self, mut rhs: Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .sub_no_norm(ws, rhs.as_view())
                .as_view()
                .normalize(ws, &mut t);
            std::mem::swap(&mut rhs, &mut t);
        });

        rhs
    }
}

impl std::ops::Mul<Atom> for &Atom {
    type Output = Atom;

    fn mul(self, mut rhs: Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut rhs, &mut t);
        });

        rhs
    }
}

impl std::ops::Div<Atom> for &Atom {
    type Output = Atom;

    fn div(self, mut rhs: Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut rhs, &mut t);
        });

        rhs
    }
}

impl std::ops::Add<&Atom> for &Atom {
    type Output = Atom;

    fn add(self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, rhs.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Sub<&Atom> for &Atom {
    type Output = Atom;

    fn sub(self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .sub_no_norm(ws, rhs.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Mul<&Atom> for &Atom {
    type Output = Atom;

    fn mul(self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, rhs.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Div<&Atom> for &Atom {
    type Output = Atom;

    fn div(self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, rhs.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Neg for &Atom {
    type Output = Atom;

    fn neg(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().neg_with_ws_into(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Add<&Atom> for Atom {
    type Output = Atom;

    fn add(mut self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl std::ops::Sub<&Atom> for Atom {
    type Output = Atom;

    fn sub(mut self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .sub_no_norm(ws, rhs.as_view())
                .as_view()
                .normalize(ws, &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl std::ops::Mul<&Atom> for Atom {
    type Output = Atom;

    fn mul(mut self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl std::ops::Div<&Atom> for Atom {
    type Output = Atom;

    fn div(mut self, rhs: &Atom) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, rhs.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl std::ops::Neg for Atom {
    type Output = Atom;

    fn neg(mut self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().neg_with_ws_into(ws, &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl std::ops::Add<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;

    fn add(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.add_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Sub<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;

    fn sub(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.sub_no_norm(ws, rhs).as_view().normalize(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Mul<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;

    fn mul(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.mul_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Div<AtomView<'_>> for AtomView<'_> {
    type Output = Atom;

    fn div(self, rhs: AtomView<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.div_with_ws_into(ws, rhs, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Neg for AtomView<'_> {
    type Output = Atom;

    fn neg(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.neg_with_ws_into(ws, &mut t);
            t.into_inner()
        })
    }
}

impl<T: Into<Coefficient>> std::ops::Add<T> for &Atom {
    type Output = Atom;

    fn add(self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, n.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl<T: Into<Coefficient>> std::ops::Sub<T> for &Atom {
    type Output = Atom;

    fn sub(self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view()
                .sub_no_norm(ws, n.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }
}

impl<T: Into<Coefficient>> std::ops::Mul<T> for &Atom {
    type Output = Atom;

    fn mul(self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, n.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl<T: Into<Coefficient>> std::ops::Div<T> for &Atom {
    type Output = Atom;

    fn div(self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, n.as_view(), &mut t);
            t.into_inner()
        })
    }
}

impl<T: Into<Coefficient>> std::ops::Add<T> for Atom {
    type Output = Atom;

    fn add(mut self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().add_with_ws_into(ws, n.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl<T: Into<Coefficient>> std::ops::Sub<T> for Atom {
    type Output = Atom;

    fn sub(mut self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view()
                .sub_no_norm(ws, n.as_view())
                .as_view()
                .normalize(ws, &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl<T: Into<Coefficient>> std::ops::Mul<T> for Atom {
    type Output = Atom;

    fn mul(mut self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().mul_with_ws_into(ws, n.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}

impl<T: Into<Coefficient>> std::ops::Div<T> for Atom {
    type Output = Atom;

    fn div(mut self, rhs: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(rhs);
            let mut t = ws.new_atom();
            self.as_view().div_with_ws_into(ws, n.as_view(), &mut t);
            std::mem::swap(&mut self, &mut t);
        });

        self
    }
}
