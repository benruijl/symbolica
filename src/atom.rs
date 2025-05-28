//! Defines the core structures and functions for handling general mathematical expressions.
//!
//! This module provides the core functionality for representing and manipulating mathematical symbols and expressions.
//! It includes definitions for various types of atoms (numbers, variables, functions, etc.), as well as utilities for
//! parsing, printing, and transforming these expressions.
//!
//! # Examples
//!
//! Parse a new expression and expand it:
//!
//! ```
//! use symbolica::{atom::AtomCore, parse};
//!
//! let a = parse!("(x+1)^2");
//! let b = a.expand();
//! let r = parse!("x^2+2x+1");
//! assert_eq!(b, r);
//! ```
//!
//! Create a new symbol and use it in an expression:
//!
//! ```
//! use symbolica::{atom::Atom, parse, symbol};
//!
//! let x = symbol!("x");
//! let expr = Atom::var(x) + 1;
//! let p = parse!("x + 1");
//! assert_eq!(expr, p);
//! ```
//!
//! Define a function with attributes and use it in an expression:
//!
//! ```
//! use symbolica::{function, parse, symbol};
//! use symbolica::atom::{Symbol, FunctionAttribute, Atom, AtomCore};
//!
//! let f = symbol!("f"; Symmetric);
//! let expr = function!(f, 3, 2) + (1, 4);
//! let p = parse!("f(2,3) + 1/4");
//! assert_eq!(expr, p);
//! ```
mod coefficient;
mod core;
pub mod representation;

use colored::Colorize;
use smartstring::{LazyCompact, SmartString};

use crate::{
    coefficient::Coefficient,
    domains::{float::Complex, rational::Rational},
    parser::Token,
    printer::{AtomPrinter, PrintFunction, PrintOptions},
    state::{RecycledAtom, State, Workspace},
    transformer::StatsOptions,
};

use std::{borrow::Cow, cmp::Ordering, hash::Hash, ops::DerefMut};

pub use self::core::AtomCore;
pub use self::representation::{
    Add, AddView, Fun, InlineNum, InlineVar, KeyLookup, ListIterator, ListSlice, Mul, MulView, Num,
    NumView, Pow, PowView, Var, VarView,
};
use self::representation::{FunView, RawAtom};

/// A symbol with a namespace, and optional positional data (file and line) of its definition.
/// Can be created with the [wrap_symbol!](crate::wrap_symbol) macro or by converting from a string that is
/// written as `namespace::symbol`.
#[derive(Clone)]
pub struct NamespacedSymbol {
    pub namespace: Cow<'static, str>,
    pub symbol: Cow<'static, str>,
    pub file: Cow<'static, str>,
    pub line: usize,
}

impl NamespacedSymbol {
    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn parse(s: &str) -> NamespacedSymbol {
        let (namespace, _partial_symbol) = s.rsplit_once("::").unwrap_or_else(|| {
            panic!(
                "Input {} does not contain a symbol in the format `namespace::symbol`.",
                s
            )
        });

        NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: s.to_string().into(),
            file: "".into(),
            line: 0,
        }
    }

    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn try_parse<S: AsRef<str>>(s: S) -> Option<NamespacedSymbol> {
        let (namespace, _partial_symbol) = s.as_ref().rsplit_once("::")?;
        Some(NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: s.as_ref().to_string().into(),
            file: "".into(),
            line: 0,
        })
    }

    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn try_parse_lit(s: &'static str) -> Option<NamespacedSymbol> {
        let (namespace, _partial_symbol) = s.rsplit_once("::")?;
        Some(NamespacedSymbol {
            namespace: namespace.into(),
            symbol: s.into(),
            file: "".into(),
            line: 0,
        })
    }
}

impl TryFrom<&str> for NamespacedSymbol {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(NamespacedSymbol::parse(value))
    }
}

#[macro_export]
macro_rules! wrap_symbol {
    ($e:literal) => {{
        if let Some(mut s) = $crate::atom::NamespacedSymbol::try_parse_lit($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else {


            let ns = if $crate::state::State::BUILTIN_SYMBOL_NAMES.contains(&$e) {
                "symbolica"
            } else {
                $crate::namespace!()
            };
            $crate::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
    ($e:expr) => {{
        if let Some(mut s) = $crate::atom::NamespacedSymbol::try_parse($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else {
            let ns = if $crate::state::State::is_builtin_name(&$e) {
                "symbolica"
            } else {
                $crate::namespace!()
            };
            $crate::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
}

/// A string representation of an expression with a namespace, and optional positional data (file and line).
/// Can be created with the [wrap_input!](crate::wrap_input) macro.
pub struct DefaultNamespace<'a> {
    pub namespace: Cow<'static, str>,
    pub data: &'a str,
    pub file: Cow<'static, str>,
    pub line: usize,
}

impl DefaultNamespace<'_> {
    /// Parse a string into a namespaced string.
    pub fn attach_namespace(&self, s: &str) -> NamespacedSymbol {
        if let Some(mut s) = NamespacedSymbol::try_parse(s) {
            s.file = self.file.clone();
            s.line = self.line;
            s
        } else if State::BUILTIN_SYMBOL_NAMES.contains(&s) {
            NamespacedSymbol {
                symbol: format!("symbolica::{}", s).into(),
                namespace: "symbolica".into(),
                file: "".into(),
                line: 0,
            }
        } else {
            NamespacedSymbol {
                symbol: format!("{}::{}", self.namespace, s).into(),
                namespace: self.namespace.clone(),
                file: self.file.clone(),
                line: self.line,
            }
        }
    }
}

#[macro_export]
macro_rules! wrap_input {
    ($e:expr) => {
        $crate::atom::DefaultNamespace {
            data: $e.as_ref(),
            namespace: $crate::namespace!().into(),
            file: file!().into(),
            line: line!() as usize,
        }
    };
}

#[macro_export]
macro_rules! with_default_namespace {
    ($e:expr, $namespace: expr) => {
        $crate::atom::DefaultNamespace {
            data: $e.as_ref(),
            namespace: $namespace.into(),
            file: file!().into(),
            line: line!() as usize,
        }
    };
}

/// Get the current namespace, based on the location of the macro invocation.
#[macro_export]
macro_rules! namespace {
    () => {{ env!("CARGO_CRATE_NAME") }};
}

/// Hide the current namespace when printing an atom.
#[macro_export]
macro_rules! hide_namespace {
    ($e:expr) => {
        $crate::atom::AtomCore::printer(
            &$e,
            $crate::printer::PrintOptions {
                hide_namespace: Some($crate::namespace!()),
                ..$crate::printer::PrintOptions::new()
            },
        )
    };
}

/// A function that is called after normalization of the arguments.
/// If the input, the first argument, is normalized, the function should return `false`.
/// Otherwise, the function must return `true` and set the second argument to the normalized value.
///
/// # Examples
///
/// ```
/// use symbolica::atom::{Atom, AtomView, NormalizationFunction};
///
/// let normalize_fn: NormalizationFunction = Box::new(|view, atom| {
///     // Example normalization logic
///     if view.is_zero() {
///         *atom = Atom::num(0);
///         true
///     } else {
///         false
///     }
/// });
/// ```
pub type NormalizationFunction = Box<dyn Fn(AtomView, &mut Atom) -> bool + Send + Sync>;

/// Attributes that can be assigned to functions.
#[derive(Clone, Copy, PartialEq)]
pub enum FunctionAttribute {
    /// The function is symmetric.
    Symmetric,
    /// The function is antisymmetric.
    Antisymmetric,
    /// The function is cyclesymmetric.
    Cyclesymmetric,
    /// The function is linear.
    Linear,
}

/// A symbol, for example the name of a variable or the name of a function,
/// together with its properties.
///
/// Every symbol has a namespace, which is either assigned explicitly
/// as `namespace::symbol` or is assigned by the [symbol!](crate::symbol) or
/// [parse!](crate::parse) macros based on the location of the macro invocation.
///
/// # Examples
///
/// ```
/// use symbolica::symbol;
///
/// let x = symbol!("x");
/// let (x, y) = symbol!("x", "y");
/// let f = symbol!("f"; Symmetric);
/// ```
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap),
)]
pub struct Symbol {
    id: u32,
    wildcard_level: u8,
    is_symmetric: bool,
    is_antisymmetric: bool,
    is_cyclesymmetric: bool,
    is_linear: bool,
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

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), f)
    }
}

/// A builder for creating symbols with optional attributes.
pub struct SymbolBuilder {
    symbol: NamespacedSymbol,
    attributes: Option<Cow<'static, [FunctionAttribute]>>,
    normalization_function: Option<NormalizationFunction>,
    print_function: Option<PrintFunction>,
}

impl SymbolBuilder {
    /// Create a new symbol builder with the given name and namespace.
    /// Use the [symbol!](crate::symbol) macro instead to define symbols with the current namespace.
    pub fn new(symbol: NamespacedSymbol) -> Self {
        SymbolBuilder {
            symbol,
            attributes: None,
            normalization_function: None,
            print_function: None,
        }
    }

    /// Get the symbol associated with `name` if it is already registered,
    /// otherwise define it with the given attributes.
    ///
    /// This function will return an error when an existing symbol is redefined
    /// with different attributes.
    ///
    /// Use the [symbol!](crate::symbol) macro instead to define symbols with the current namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::{atom::{Symbol, FunctionAttribute}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("f")).with_attributes(&[FunctionAttribute::Symmetric]).build().unwrap();
    /// ```
    pub fn with_attributes(
        mut self,
        attributes: impl Into<Cow<'static, [FunctionAttribute]>>,
    ) -> Self {
        self.attributes = Some(attributes.into());
        self
    }

    /// ```
    /// use symbolica::{atom::{AtomView, Symbol}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("f")).with_normalization_function(|view, out| {
    ///     // Example normalization logic that sets odd-length function to 0
    ///     if let AtomView::Fun(f) = view {
    ///         if f.get_nargs() % 2 == 1 {
    ///             out.to_num(0.into());
    ///             true // changed
    ///         } else {
    ///             false
    ///         }
    ///     } else {
    ///         unreachable!()
    ///     }
    /// }).build().unwrap();
    /// ```
    pub fn with_normalization_function(
        mut self,
        normalization_function: impl Fn(AtomView, &mut Atom) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.normalization_function = Some(Box::new(normalization_function));
        self
    }

    /// ```
    /// use symbolica::{atom::Symbol, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("mu")).with_print_function(|view, opt| {
    ///     if !opt.mode.is_latex() {
    ///       None
    ///     } else {
    ///        Some("\\mu".to_string())
    ///     }
    /// }).build().unwrap();
    /// ```
    pub fn with_print_function(
        mut self,
        print_function: impl Fn(AtomView, &PrintOptions) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.print_function = Some(Box::new(print_function));
        self
    }

    /// Create a new symbol or return the existing symbol with the same name.
    ///
    /// This function will return an error when an existing symbol is redefined
    /// with different attributes.
    pub fn build(self) -> Result<Symbol, SmartString<LazyCompact>> {
        if self.attributes.is_none()
            && self.normalization_function.is_none()
            && self.print_function.is_none()
        {
            State::get_state_mut().get_symbol(self.symbol)
        } else {
            State::get_state_mut().get_symbol_with_attributes(
                self.symbol,
                self.attributes.as_ref().map(|x| x.as_ref()).unwrap_or(&[]),
                self.normalization_function,
                self.print_function,
            )
        }
    }
}

impl Symbol {
    /// Create a builder for a new symbol with the given name and namespace.
    ///
    /// Use the [symbol!](crate::symbol) macro instead to define symbols in the current namespace.
    pub fn new(name: NamespacedSymbol) -> SymbolBuilder {
        SymbolBuilder::new(name)
    }

    /// Get the name of the symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_name(), "test::x");
    /// ```
    pub fn get_name(&self) -> &str {
        State::get_name(*self)
    }

    /// Get the name of the symbol without the namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_stripped_name(), "x");
    /// ```
    pub fn get_stripped_name(&self) -> &str {
        let d = State::get_symbol_data(*self);
        &d.name[d.namespace.len() + 2..]
    }

    /// Get the internal id of the symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("x");
    /// println!("id = {}", x.get_id());
    /// ```
    pub fn get_id(&self) -> u32 {
        self.id
    }

    /// Get the definition location of the symbol.
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_namespace(), "test");
    /// ```
    pub fn get_namespace(&self) -> &'static str {
        State::get_symbol_namespace(*self)
    }

    /// Get the wildcard level of the symbol. This property
    /// is used for pattern matching.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("x");
    /// let x_ = symbol!("x_");
    /// let x__ = symbol!("x__");
    /// let x___ = symbol!("x___");
    /// assert_eq!(x.get_wildcard_level(), 0);
    /// assert_eq!(x_.get_wildcard_level(), 1);
    /// assert_eq!(x__.get_wildcard_level(), 2);
    /// assert_eq!(x___.get_wildcard_level(), 3);
    /// ```
    pub fn get_wildcard_level(&self) -> u8 {
        self.wildcard_level
    }

    /// Check if the symbol is symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Symmetric);
    /// assert!(f.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    /// Check if the symbol is antisymmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Antisymmetric);
    /// assert!(f.is_antisymmetric());
    /// ```
    pub fn is_antisymmetric(&self) -> bool {
        self.is_antisymmetric
    }

    /// Check if the symbol is cyclesymmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Cyclesymmetric);
    /// assert!(f.is_cyclesymmetric());
    /// ```
    pub fn is_cyclesymmetric(&self) -> bool {
        self.is_cyclesymmetric
    }

    /// Check if the symbol is linear.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Linear);
    /// assert!(f.is_linear());
    /// ```
    pub fn is_linear(&self) -> bool {
        self.is_linear
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub fn is_builtin(id: Symbol) -> bool {
        State::is_builtin(id)
    }

    /// Expert use: create a new variable symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn raw_var(id: u32, wildcard_level: u8) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric: false,
            is_antisymmetric: false,
            is_cyclesymmetric: false,
            is_linear: false,
        }
    }

    /// Expert use: create a new function symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn raw_fn(
        id: u32,
        wildcard_level: u8,
        is_symmetric: bool,
        is_antisymmetric: bool,
        is_cyclesymmetric: bool,
        is_linear: bool,
    ) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric,
            is_antisymmetric,
            is_cyclesymmetric,
            is_linear,
        }
    }

    pub fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        f: &mut W,
    ) -> Result<(), std::fmt::Error> {
        let data = State::get_symbol_data(*self);
        let (namespace, name) = (&data.namespace, &data.name[data.namespace.len() + 2..]);

        if let Some(custom_print) = &data.custom_print {
            if let Some(s) = custom_print(InlineVar::new(*self).as_view(), opts) {
                f.write_str(&s)?;
                return Ok(());
            }
        }

        if opts.mode.is_latex() {
            match *self {
                Atom::E => f.write_char('e'),
                Atom::PI => f.write_str("\\pi"),
                Atom::COS => f.write_str("\\cos"),
                Atom::SIN => f.write_str("\\sin"),
                Atom::EXP => f.write_str("\\exp"),
                Atom::LOG => f.write_str("\\log"),
                _ => {
                    f.write_str(name)?;
                    if !opts.hide_all_namespaces {
                        f.write_fmt(format_args!("_{{\\tiny \text{{{}}}}}", namespace))
                    } else {
                        Ok(())
                    }
                }
            }
        } else {
            if !opts.hide_all_namespaces
                && !State::is_builtin(*self)
                && opts.hide_namespace != Some(namespace)
            {
                if opts.color_namespace {
                    f.write_fmt(format_args!("{}", namespace.dimmed().italic()))?;
                    f.write_fmt(format_args!("{}", "::".dimmed()))?;
                } else {
                    f.write_fmt(format_args!("{}::", namespace))?;
                }
            }

            if opts.color_builtin_symbols && name.ends_with('_') {
                f.write_fmt(format_args!("{}", name.cyan().italic()))
            } else if opts.color_builtin_symbols && State::is_builtin(*self) {
                f.write_fmt(format_args!("{}", name.purple()))
            } else {
                f.write_str(name)
            }
        }
    }
}

/// The type (variant) of an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomType {
    Num,
    Var,
    Add,
    Mul,
    Pow,
    Fun,
}

impl std::fmt::Display for AtomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomType::Num => write!(f, "Num"),
            AtomType::Var => write!(f, "Var"),
            AtomType::Add => write!(f, "Add"),
            AtomType::Mul => write!(f, "Mul"),
            AtomType::Pow => write!(f, "Pow"),
            AtomType::Fun => write!(f, "Fun"),
        }
    }
}

/// The type (variant) of a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    Add,
    Mul,
    Arg,
    One,
    Pow,
    Empty,
}

/// A (immutable) view of an [Atom].
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

impl From<Symbol> for Atom {
    /// Convert a symbol to an atom. This will allocate memory.
    fn from(symbol: Symbol) -> Atom {
        Atom::var(symbol)
    }
}

impl From<AtomView<'_>> for Atom {
    /// Convert an `AtomView` to an atom. This will allocate memory.
    fn from(view: AtomView) -> Atom {
        view.to_owned()
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

/// A copy-on-write structure for `Atom` and `AtomView`.
#[derive(Clone, Debug)]
pub enum AtomOrView<'a> {
    Atom(Atom),
    View(AtomView<'a>),
}

impl std::fmt::Display for AtomOrView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomOrView::Atom(a) => a.fmt(f),
            AtomOrView::View(a) => a.fmt(f),
        }
    }
}

impl PartialEq for AtomOrView<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AtomOrView::Atom(a), AtomOrView::Atom(b)) => a == b,
            (AtomOrView::View(a), AtomOrView::View(b)) => a == b,
            _ => self.as_view() == other.as_view(),
        }
    }
}

impl Eq for AtomOrView<'_> {}

impl PartialOrd for AtomOrView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomOrView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (AtomOrView::Atom(a1), AtomOrView::Atom(a2)) => a1.as_view().cmp(&a2.as_view()),
            (AtomOrView::Atom(a1), AtomOrView::View(a2)) => a1.as_view().cmp(a2),
            (AtomOrView::View(a1), AtomOrView::Atom(a2)) => a1.cmp(&a2.as_view()),
            (AtomOrView::View(a1), AtomOrView::View(a2)) => a1.cmp(a2),
        }
    }
}

impl Hash for AtomOrView<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AtomOrView::Atom(a) => a.as_view().hash(state),
            AtomOrView::View(a) => a.hash(state),
        }
    }
}

impl<'a, T> From<T> for AtomOrView<'a>
where
    T: Into<Coefficient>,
{
    fn from(v: T) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::num(v.into()))
    }
}

impl<'a> From<Symbol> for AtomOrView<'a> {
    fn from(s: Symbol) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::var(s))
    }
}

impl<'a> From<&'a Symbol> for AtomOrView<'a> {
    fn from(s: &'a Symbol) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::var(*s))
    }
}

impl<'a> From<Atom> for AtomOrView<'a> {
    fn from(a: Atom) -> AtomOrView<'a> {
        AtomOrView::Atom(a)
    }
}

impl<'a> From<&'a Atom> for AtomOrView<'a> {
    fn from(a: &'a Atom) -> AtomOrView<'a> {
        AtomOrView::View(a.as_view())
    }
}

impl<'a> From<AtomView<'a>> for AtomOrView<'a> {
    fn from(a: AtomView<'a>) -> AtomOrView<'a> {
        AtomOrView::View(a)
    }
}

impl<'a> From<&AtomView<'a>> for AtomOrView<'a> {
    fn from(a: &AtomView<'a>) -> AtomOrView<'a> {
        AtomOrView::View(*a)
    }
}

impl<'a> AtomOrView<'a> {
    pub fn into_owned(self) -> Atom {
        match self {
            AtomOrView::Atom(a) => a,
            AtomOrView::View(a) => a.to_owned(),
        }
    }

    pub fn as_view(&'a self) -> AtomView<'a> {
        match self {
            AtomOrView::Atom(a) => a.as_view(),
            AtomOrView::View(a) => *a,
        }
    }

    pub fn as_mut(&mut self) -> &mut Atom {
        match self {
            AtomOrView::Atom(a) => a,
            AtomOrView::View(a) => {
                let mut oa = Atom::default();
                oa.set_from_view(a);
                *self = AtomOrView::Atom(oa);
                match self {
                    AtomOrView::Atom(a) => a,
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl AtomView<'_> {
    pub fn to_owned(&self) -> Atom {
        let mut a = Atom::default();
        a.set_from_view(self);
        a
    }

    pub fn clone_into(&self, target: &mut Atom) {
        target.set_from_view(self);
    }

    /// Print the view using the portable [`PrintOptions::file()`] options.
    pub fn to_plain_string(&self) -> String {
        format!("{}", self.printer(PrintOptions::file()))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        if let AtomView::Add(a) = self {
            a.get_nargs()
        } else {
            1
        }
    }

    /// Print statistics about the operation `op`, such as its duration and term growth.
    pub fn with_stats<F: Fn(AtomView) -> Atom>(&self, op: F, o: &StatsOptions) -> Atom {
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0));

        let t = std::time::Instant::now();
        let out = op(*self);
        let dt = t.elapsed();
        o.print(*self, out.as_view(), start_time, dt);
        out
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        if let AtomView::Num(n) = self {
            n.is_zero()
        } else {
            false
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        if let AtomView::Num(n) = self {
            n.is_one()
        } else {
            false
        }
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
        self.add_normalized(rhs, workspace, out);
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

/// A mathematical expression.
///
/// Most operations are implemented in the [AtomCore] trait.
///
/// # Examples
///
/// Parse a new expression and expand it:
///
/// ```
/// use symbolica::{atom::AtomCore, parse};
///
/// let a = parse!("(x+1)^2");
/// let b = a.expand();
/// let r = parse!("x^2+2x+1");
/// assert_eq!(b, r);
/// ```
///
/// Create a new symbol and use it in an expression:
///
/// ```
/// use symbolica::{atom::Atom, parse, symbol};
///
/// let x = symbol!("x");
/// let expr = Atom::var(x) + 1;
/// let p = parse!("x + 1");
/// assert_eq!(expr, p);
/// ```
///
/// Define a function with attributes and use it in an expression:
///
/// ```
/// use symbolica::{function, parse, symbol};
/// use symbolica::atom::{Symbol, FunctionAttribute, Atom, AtomCore};
///
/// let f = symbol!("f"; Symmetric);
/// let expr = function!(f, 3, 2) + (1, 4);
/// let p = parse!("f(2,3) + 1/4");
/// assert_eq!(expr, p);
/// ```
///
/// # Output
///
/// The output can be controlled with
/// ```
/// use symbolica::{hide_namespace, parse};
/// let a = parse!("x^2+cos(x)");
/// println!("{:+}", a); // print with a leading sign
/// println!("{:#}", a); // print the namespace in front of every variable (e.g. `test::x`)
/// println!("{}", hide_namespace!(a)); // hide the current namespace
/// println!("{:>+}", a); // print with a leading sign and print every term on a new line
/// ```
///
/// Advanced output options can be set using [PrintOptions]. Use [PrintOptions::file()]
/// to print an expression in a format that can be parsed again.
///
/// ```
/// use symbolica::{atom::AtomCore, parse, printer::PrintOptions};
/// let a = parse!("x^2+cos(x)");
/// println!("{}", a.printer(PrintOptions::latex()));
/// println!("{}", a.printer(PrintOptions::mathematica()));
/// println!("{}", a.printer(PrintOptions::file()));
/// println!("{}", a.printer(PrintOptions {
///      color_builtin_symbols: true,
///     ..PrintOptions::new()
/// }));
/// ```
#[must_use]
#[derive(Clone)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub enum Atom {
    Num(Num),
    Var(Var),
    Fun(Fun),
    Pow(Pow),
    Mul(Mul),
    Add(Add),
    Zero,
}

impl Atom {
    /// The built-in function represents a list of function arguments.
    pub const ARG: Symbol = State::ARG;
    /// The built-in function that converts a rational polynomial to a coefficient.
    pub const COEFF: Symbol = State::COEFF;
    /// The exponent function.
    pub const EXP: Symbol = State::EXP;
    /// The logarithm function.
    pub const LOG: Symbol = State::LOG;
    /// The sine function.
    pub const SIN: Symbol = State::SIN;
    /// The cosine function.
    pub const COS: Symbol = State::COS;
    /// The square root function.
    pub const SQRT: Symbol = State::SQRT;
    /// The built-in function that represents an abstract derivative.
    pub const DERIVATIVE: Symbol = State::DERIVATIVE;
    /// The constant e, the base of the natural logarithm.
    pub const E: Symbol = State::E;
    /// The mathematical constant `π`.
    pub const PI: Symbol = State::PI;

    /// The number suffix that represents the imaginary unit.
    /// The suffix `i` can also be used for parsing (e.g. `2+3𝑖` or `2+3i`).
    pub const I_STR: &'static str = "𝑖";
    /// The string representation of the constant `π`.
    pub const PI_STR: &'static str = "𝜋";

    /// The imaginary unit.
    pub fn i() -> Atom {
        Atom::num(Complex::<Rational>::new_i())
    }

    /// Exponentiate the atom.
    pub fn exp(&self) -> Atom {
        FunctionBuilder::new(Atom::EXP).add_arg(self).finish()
    }

    /// Take the logarithm of the atom.
    pub fn log(&self) -> Atom {
        FunctionBuilder::new(Atom::LOG).add_arg(self).finish()
    }

    /// Take the sine the atom.
    pub fn sin(&self) -> Atom {
        FunctionBuilder::new(Atom::SIN).add_arg(self).finish()
    }

    ///  Take the cosine the atom.
    pub fn cos(&self) -> Atom {
        FunctionBuilder::new(Atom::COS).add_arg(self).finish()
    }

    ///  Take the square root of the atom.
    pub fn sqrt(&self) -> Atom {
        FunctionBuilder::new(Atom::SQRT).add_arg(self).finish()
    }
}

impl Default for Atom {
    /// Create an atom that represents the number 0.
    #[inline]
    fn default() -> Self {
        Atom::Zero
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

impl PartialEq for Atom {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

impl Eq for Atom {}

impl Hash for Atom {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state)
    }
}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl<T: Into<Coefficient> + Clone> PartialEq<T> for Atom {
    fn eq(&self, other: &T) -> bool {
        *self == Atom::num(other.clone())
    }
}

impl<T: Into<Coefficient> + Clone> PartialOrd<T> for Atom {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Some(self.cmp(&Atom::num(other.clone().into())))
    }
}

impl Atom {
    /// Create an atom that represents the number 0.
    pub fn new() -> Atom {
        Atom::default()
    }

    /// Parse an atom from a namespaced string. Prefer to use [parse!](crate::parse) instead.
    ///
    /// # Examples
    /// ```rust
    /// use symbolica::{wrap_input, with_default_namespace};
    /// use symbolica::atom::Atom;
    /// let x = Atom::parse(wrap_input!("x")).unwrap();
    /// let x_2 = Atom::parse(with_default_namespace!("x_2", "b")).unwrap();
    /// assert!(x != x_2);
    /// ```
    pub fn parse(input: DefaultNamespace) -> Result<Atom, String> {
        Workspace::get_local().with(|ws| Token::parse(input.data, false)?.to_atom(&input, ws))
    }

    /// Create a new atom that represents a variable.
    #[inline]
    pub fn var(id: Symbol) -> Atom {
        Var::new(id).into()
    }

    /// Create a new atom that represents a number.
    #[inline]
    pub fn num<T: Into<Coefficient>>(num: T) -> Atom {
        let c = num.into();
        if c.is_zero() {
            Atom::Zero
        } else {
            Num::new(c).into()
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.as_view().is_zero()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.as_view().is_one()
    }

    /// Get the number of terms in the expression.
    pub fn nterms(&self) -> usize {
        self.as_view().nterms()
    }

    /// Print the atom using the portable [`PrintOptions::file()`] options.
    pub fn to_plain_string(&self) -> String {
        format!("{}", self.printer(PrintOptions::file()))
    }

    /// Print statistics about the operation `op`, such as its duration and term growth.
    pub fn with_stats<F: Fn(AtomView) -> Atom>(&self, op: F, o: &StatsOptions) -> Atom {
        self.as_view().with_stats(op, o)
    }

    /// Repeatedly apply an operation on the atom until the atom no longer changes.
    pub fn repeat_map<F: Fn(AtomView) -> Atom>(&mut self, op: F) {
        let mut res;
        loop {
            res = op(self.as_view());
            if res == *self {
                break;
            }
            std::mem::swap(self, &mut res);
        }
    }

    #[inline]
    pub fn to_num(&mut self, coeff: Coefficient) -> &mut Num {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Num(Num::new_into(coeff, buffer));
        if let Atom::Num(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_var(&mut self, id: Symbol) -> &mut Var {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Var(Var::new_into(id, buffer));
        if let Atom::Var(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_fun(&mut self, id: Symbol) -> &mut Fun {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Fun(Fun::new_into(id, buffer));
        if let Atom::Fun(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_pow(&mut self, base: AtomView, exp: AtomView) -> &mut Pow {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Pow(Pow::new_into(base, exp, buffer));
        if let Atom::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_mul(&mut self) -> &mut Mul {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Mul(Mul::new_into(buffer));
        if let Atom::Mul(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_add(&mut self) -> &mut Add {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
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
            Atom::Zero => RawAtom::new(),
        }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &AtomView) {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
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
            Atom::Zero => AtomView::ZERO,
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
            Atom::Zero => {}
        }
    }
}

/// A constructor of a function. Consider using the [crate::fun!] macro instead.
///
/// For example:
/// ```
/// # use symbolica::symbol;
/// # use symbolica::atom::{Atom, AtomCore, FunctionBuilder};
/// # fn main() {
/// let f_id = symbol!("f"; Symmetric);
/// let fb = FunctionBuilder::new(f_id);
/// let a = fb
///     .add_arg(&Atom::num(3))
///     .add_arg(&Atom::num(2))
///     .add_arg(&Atom::num(1))
///     .finish();
///
/// println!("{}", a);
/// # }
/// ```
#[derive(Clone)]
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
    pub fn add_arg<'a, T: Into<AtomOrView<'a>>>(mut self, arg: T) -> FunctionBuilder {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            f.add_arg(arg.into().as_view());
        }

        self
    }

    /// Add multiple arguments to the function.
    pub fn add_args<'a, T>(mut self, args: &'a [T]) -> FunctionBuilder
    where
        &'a T: Into<AtomOrView<'a>>,
    {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            for a in args {
                f.add_arg(a.into().as_view());
            }
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

/// A trait that allows to add an argument to a function builder.
pub trait FunctionArgument {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder;
}

impl FunctionArgument for Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for &Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for &mut Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for AtomView<'_> {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(*self)
    }
}

impl FunctionArgument for &AtomView<'_> {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(**self)
    }
}

impl FunctionArgument for Symbol {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        let t = InlineVar::new(*self);
        f.add_arg(t.as_view())
    }
}

impl<T: Into<Coefficient> + Clone> FunctionArgument for T {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(Atom::num(self.clone()))
    }
}

/// Create a new function by providing its name as the first argument,
/// followed by the list of arguments. This macro uses [`FunctionBuilder`].
///
/// # Examples
///
/// ```
/// use symbolica::{atom::Atom, atom::Symbol, function, symbol, parse};
/// let f_id = symbol!("f");
/// let f = function!(symbol!("f"), 3, parse!("x"));
/// ```
#[macro_export]
macro_rules! function {
    ($name: expr) => {
        {
            $crate::atom::FunctionBuilder::new($name).finish()
        }
    };
    ($name: expr, $($id: expr),*) => {
        {
            let mut f = $crate::atom::FunctionBuilder::new($name);
            $(
                f = $crate::atom::FunctionArgument::add_arg_to_function_builder(&$id, f);
            )+
            f.finish()
        }
    };
}

/// Create a new symbol or fetch the existing one with the same name.
/// If no namespace is specified, the symbol is created in the
/// current namespace.
///
/// For example:
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x");
/// let (x, y, z) = symbol!("x", "y", "z");
/// let x_remote = symbol!("remote::x");
/// ```
///
/// Since no attributes were specified in the example above, the symbols
/// will inherit the attributes if the symbol already exists or will be
/// created with the default attributes.
///
/// You can specify attributes for the symbol, using `;` as a separator
/// between symbol names and attributes. The options
/// are [Symmetric](FunctionAttribute::Symmetric), [Antisymmetric](FunctionAttribute::Antisymmetric),
/// [Cyclesymmetric](FunctionAttribute::Cyclesymmetric), and [Linear](FunctionAttribute::Linear).
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x"; Symmetric, Linear);
/// let (x, y, z) = symbol!("x", "y", "z"; Symmetric); // define all as symmetric
/// ```
///
/// Explicitly specifying a symbol without attributes:
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x";);
/// ```
/// will panic if the symbol was previously defined with attributes. Use
/// [try_symbol!] for a fallible version.
///
/// You can specify a normalization function for the symbol, following its
/// attributes with a `;`.
/// ```no_run
/// use symbolica::symbol;
/// use symbolica::atom::AtomView;
/// let x = symbol!("f";; |f, out| {
///     if let AtomView::Fun(ff) = f {
///         if ff.get_nargs() % 2 == 1 {
///            out.to_num(0.into());
///            return true;
///         }
///     }
///     false
/// });
/// ```
///
/// You can also define a custom printing function by adding another `;`:
/// ```no_run
/// use symbolica::symbol;
/// use symbolica::atom::{AtomCore, AtomView};
/// use symbolica::printer::PrintState;
/// let _ = symbol!("mu";;;|a, opt| {
///     if !opt.mode.is_latex() {
///         return None; // use default printer
///     }
///
///     let mut fmt = "\\mu".to_string();
///     if let AtomView::Fun(f) = a {
///         fmt.push_str("_{");
///         let n_args = f.get_nargs();
///         for (i, a) in f.iter().enumerate() {
///             a.format(&mut fmt, opt, PrintState::new()).unwrap();
///             if i < n_args - 1 {
///                 fmt.push_str(",");
///             }
///         }
///         fmt.push_str("}");
///     }
///
///     Some(fmt)
/// });
/// ```
/// which renders the symbol/function as `\mu_{...}` in LaTeX.
#[macro_export]
macro_rules! symbol {
    ($id: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build().unwrap()
    };
    ($id: expr; $($attr: ident),*) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).build().unwrap()
    };
    ($id: expr; $($attr: ident),*; $norm: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_normalization_function($norm).build().unwrap()
    };
    ($id: expr; $($attr: ident),*;; $print: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_print_function($print).build().unwrap()
    };
    ($id: expr; $($attr: ident),*; $norm: expr; $print: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_normalization_function($norm).with_print_function($print).build().unwrap()
    };
    ($($id: expr),*) => {
        {
            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::FunctionAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).build().unwrap(),
                )+
            )
        }
    };
}

/// Try to create a new symbol or fetch the existing one with the same name.
/// This is a fallible version of the [symbol!] macro.
#[macro_export]
macro_rules! try_symbol {
    ($id: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build()
    };
    ($id: expr; $($attr: ident),*) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).build()
    };
    ($id: expr; $($attr: ident),*; $norm: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_normalization_function($norm).build()
    };
    ($id: expr; $($attr: ident),*;; $print: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_print_function($print).build()
    };
    ($id: expr; $($attr: ident),*; $norm: expr; $print: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::FunctionAttribute::$attr,)*]).with_normalization_function($norm).with_print_function($print).build()
    };
    ($($id: expr),*) => {
        {
            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::FunctionAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).build(),
                )+
            )
        }
    };
}

/// Parse an atom from a string.
/// Use [parse_lit!](crate::parse_lit) to parse from literal code and
/// [try_parse!](crate::try_parse) for fallible parsing.
///
/// # Examples
/// Parse from a literal string:
/// ```
/// use symbolica::parse;
/// let a = parse!("x^2 + 5 + f(x)");
/// println!("{}", a);
/// ```
///
/// Parse a constructed string:
/// ```
/// use symbolica::parse;
/// let s = format!("x^{}", 2);
/// let a = parse!(s);
/// println!("{}", a);
/// ```
///
/// Parse using another default namespace:
/// ```
/// use symbolica::parse;
/// let a = parse!("test::x + y", "custom");
/// assert_eq!(a, parse!("test::x + custom::y"));
/// ```
///
/// Parse a complex number:
/// ```
/// use symbolica::parse;
/// let a = parse!("(2+3i)*x");
/// println!("{}", a);
/// ```
///
/// Parse a floating-point number in exponential notation with
/// a custom precision of 5 decimal digits:
/// ```
/// use symbolica::parse;
/// let a = parse!("1.23456e-6`5");
/// println!("{}", a);
/// ```
#[macro_export]
macro_rules! parse {
    ($($all_args:tt)*) => {
        $crate::try_parse!($($all_args)*).unwrap()
    };
}

#[macro_export]
macro_rules! try_parse {
    ($s: expr) => {
        $crate::atom::Atom::parse($crate::wrap_input!($s))
    };
    ($s: expr, $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!($s, $ns)) }};
}

/// Parse an atom from literal code. Use [parse!](crate::parse) to parse from a string.
/// Any new symbols are defined in the current namespace. Use [try_parse_lit](crate::try_parse_lit) for fallible parsing.
///
/// # Examples
/// ```
/// use symbolica::parse_lit;
/// let a = parse_lit!(x ^ 2 + 5 + f(x));
/// println!("{}", a);
/// ```
///
/// Parse using another default namespace:
/// ```
/// use symbolica::{parse, parse_lit};
/// let a = parse_lit!(test::x + y, "custom");
/// assert_eq!(a, parse!("test::x + custom::y"));
/// ```
#[macro_export]
macro_rules! parse_lit {
    ($s: expr) => {{ $crate::atom::Atom::parse($crate::wrap_input!(stringify!($s))).unwrap() }};
    ($s: expr, $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!(stringify!($s), $ns)).unwrap() }};
}

/// Try to parse an atom from literal code. Use [parse_lit!](crate::parse_lit) for parsing that panics on an error.
#[macro_export]
macro_rules! try_parse_lit {
    ($s: expr) => {{ $crate::atom::Atom::parse($crate::wrap_input!(stringify!($s))) }};
    ($s: expr, $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!(stringify!($s), $ns)) }};
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
    pub fn pow<T: AtomCore>(&self, exp: T) -> Atom {
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
    pub fn rpow<T: AtomCore>(&self, base: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            base.as_atom_view()
                .pow_no_norm(ws, self.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Add the atoms in `args`.
    pub fn add_many<T: AtomCore>(args: &[T]) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let add = t.to_add();
            for a in args {
                add.extend(a.as_atom_view());
            }

            t.as_view().normalize(ws, &mut out);
        });
        out
    }

    /// Multiply the atoms in `args`.
    pub fn mul_many<T: AtomCore + Copy>(args: &[T]) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let add = t.to_mul();
            for a in args {
                add.extend(a.as_atom_view());
            }

            t.as_view().normalize(ws, &mut out);
        });
        out
    }
}

mod ops;

impl AsRef<Atom> for Atom {
    fn as_ref(&self) -> &Atom {
        self
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore},
        function,
    };

    use super::FunctionBuilder;

    #[test]
    fn parse_macro() {
        assert_eq!(parse_lit!(x ^ 2 + 5 + f(x)), parse!("x ^ 2 + 5 + f(x)"));
    }

    #[test]
    fn debug() {
        let x = parse!("v1+f1(v2)");
        assert_eq!(
            format!("{:?}", x),
            "AddView { data: [5, 17, 2, 13, 2, 1, 11, 3, 5, 0, 0, 0, 1, 41, 2, 1, 12] }"
        );
        assert_eq!(
            x.get_all_symbols(true),
            [symbol!("v1"), symbol!("v2"), symbol!("f1")]
                .into_iter()
                .collect(),
        );
        assert_eq!(x.as_view().get_byte_size(), 17);
    }

    #[test]
    fn composition() {
        let v1 = parse!("v1");
        let v2 = parse!("v2");
        let f1_id = symbol!("f1");

        let f1 = function!(f1_id, v1, v2, Atom::num(2));

        let r = (-(&v2 + &v1 + 2) * &v2 * 6).npow(5) / &v2.pow(&v1) * &f1 / 4;

        let res = parse!("1/4*(v2^v1)^-1*(-6*v2*(v1+v2+2))^5*f1(v1,v2,2)");
        assert_eq!(res, r);
    }

    #[test]
    fn building() {
        let _ = FunctionBuilder::new(symbol!("a"))
            .add_arg(1)
            .add_args(&[1, 2])
            .add_arg(&symbol!("a"))
            .add_args(&[symbol!("b")])
            .add_args(&[parse!("a")])
            .add_arg(&parse!("a"))
            .add_arg(parse!("a"))
            .add_arg(parse!("a").as_view())
            .finish();
    }
}
