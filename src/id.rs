//! Methods related to pattern matching and replacements.
//!
//! The standard use is through [AtomCore] methods such as [replace_all](AtomCore::replace_all)
//! and [pattern_match](AtomCore::pattern_match).
//!
//! # Examples
//!
//! ```
//! use symbolica::{atom::{Atom, AtomCore}, id::Pattern};
//!
//! let expr = Atom::parse("f(1,2,x) + f(1,2,3)").unwrap();
//! let pat = Pattern::parse("f(1,2,y_)").unwrap();
//! let rhs = Pattern::parse("f(1,2,y_+1)").unwrap();
//!
//! let out = expr.replace_all(&pat, &rhs, None, None);
//! assert_eq!(out, Atom::parse("f(1,2,x+1)+f(1,2,4)").unwrap());
//! ```

use std::{ops::DerefMut, str::FromStr};

use ahash::{HashMap, HashSet};
use dyn_clone::DynClone;

use crate::{
    atom::{
        representation::{InlineVar, ListSlice},
        Atom, AtomCore, AtomType, AtomView, Num, SliceType, Symbol,
    },
    state::Workspace,
    transformer::{Transformer, TransformerError},
};

/// A general expression that can contain pattern-matching wildcards
/// and transformers.
///
/// # Examples
/// Patterns can be created from atoms:
/// ```
/// # use symbolica::atom::{Atom, AtomCore};
/// Atom::parse("x_+1").unwrap().to_pattern();
/// ```
///
/// or by directly parsing them:
/// ```
/// # use symbolica::id::Pattern;
/// Pattern::parse("x_+1").unwrap();
/// ```
#[derive(Clone)]
pub enum Pattern {
    Literal(Atom),
    Wildcard(Symbol),
    Fn(Symbol, Vec<Pattern>),
    Pow(Box<[Pattern; 2]>),
    Mul(Vec<Pattern>),
    Add(Vec<Pattern>),
    Transformer(Box<(Option<Pattern>, Vec<Transformer>)>),
}

impl From<Symbol> for Pattern {
    /// Convert the symbol to a pattern.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::{symbol, id::Pattern};
    ///
    /// let p = symbol!("x_").into();
    /// assert!(matches!(p, Pattern::Wildcard(_)));
    /// ```
    fn from(symbol: Symbol) -> Pattern {
        InlineVar::new(symbol).to_pattern()
    }
}

impl From<Atom> for Pattern {
    fn from(atom: Atom) -> Self {
        atom.to_pattern()
    }
}

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(a) = self.to_atom() {
            a.fmt(f)
        } else {
            std::fmt::Debug::fmt(self, f)
        }
    }
}

pub trait MatchMap: Fn(&MatchStack) -> Atom + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(MatchMap);
impl<T: Clone + Send + Sync + Fn(&MatchStack) -> Atom> MatchMap for T {}

/// A pattern or a map from a list of matched wildcards to an atom.
/// The latter can be used for complex replacements that cannot be
/// expressed using atom transformations.
#[derive(Clone)]
pub enum PatternOrMap {
    Pattern(Pattern),
    Map(Box<dyn MatchMap>),
}

impl Into<PatternOrMap> for Pattern {
    fn into(self) -> PatternOrMap {
        PatternOrMap::Pattern(self)
    }
}

impl std::fmt::Debug for PatternOrMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternOrMap::Pattern(p) => write!(f, "{:?}", p),
            PatternOrMap::Map(_) => write!(f, "Map"),
        }
    }
}

/// A pattern or a map from a list of matched wildcards to an atom.
/// The latter can be used for complex replacements that cannot be
/// expressed using atom transformations.
#[derive(Clone, Copy)]
pub enum BorrowedPatternOrMap<'a> {
    Pattern(&'a Pattern),
    Map(&'a Box<dyn MatchMap>),
}

pub trait BorrowPatternOrMap {
    fn borrow(&self) -> BorrowedPatternOrMap;
}

impl BorrowPatternOrMap for &Pattern {
    fn borrow(&self) -> BorrowedPatternOrMap {
        BorrowedPatternOrMap::Pattern(*self)
    }
}

impl BorrowPatternOrMap for Pattern {
    fn borrow(&self) -> BorrowedPatternOrMap {
        BorrowedPatternOrMap::Pattern(self)
    }
}

impl BorrowPatternOrMap for Box<dyn MatchMap> {
    fn borrow(&self) -> BorrowedPatternOrMap {
        BorrowedPatternOrMap::Map(self)
    }
}

impl BorrowPatternOrMap for &Box<dyn MatchMap> {
    fn borrow(&self) -> BorrowedPatternOrMap {
        BorrowedPatternOrMap::Map(*self)
    }
}

impl BorrowPatternOrMap for PatternOrMap {
    fn borrow(&self) -> BorrowedPatternOrMap {
        match self {
            PatternOrMap::Pattern(p) => BorrowedPatternOrMap::Pattern(p),
            PatternOrMap::Map(m) => BorrowedPatternOrMap::Map(m),
        }
    }
}

impl BorrowPatternOrMap for &PatternOrMap {
    fn borrow(&self) -> BorrowedPatternOrMap {
        match self {
            PatternOrMap::Pattern(p) => BorrowedPatternOrMap::Pattern(p),
            PatternOrMap::Map(m) => BorrowedPatternOrMap::Map(m),
        }
    }
}

impl<'a> BorrowPatternOrMap for BorrowedPatternOrMap<'a> {
    fn borrow(&self) -> BorrowedPatternOrMap {
        *self
    }
}

/// A replacement, specified by a pattern and the right-hand side,
/// with optional conditions and settings.
#[derive(Debug, Clone)]
pub struct Replacement {
    pat: Pattern,
    rhs: PatternOrMap,
    conditions: Option<Condition<PatternRestriction>>,
    settings: Option<MatchSettings>,
}

impl std::fmt::Display for Replacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.pat, self.rhs)?;

        if let Some(c) = &self.conditions {
            write!(f, "; {}", c)?;
        }

        Ok(())
    }
}

impl std::fmt::Display for PatternOrMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternOrMap::Pattern(p) => write!(f, "{}", p),
            PatternOrMap::Map(_) => write!(f, "Map"),
        }
    }
}

impl Replacement {
    pub fn new<R: Into<PatternOrMap>>(pat: Pattern, rhs: R) -> Self {
        Replacement {
            pat,
            rhs: rhs.into(),
            conditions: None,
            settings: None,
        }
    }

    pub fn with_conditions(mut self, conditions: Condition<PatternRestriction>) -> Self {
        self.conditions = Some(conditions);
        self
    }

    pub fn with_settings(mut self, settings: MatchSettings) -> Self {
        self.settings = Some(settings);
        self
    }
}

/// A borrowed version of a [Replacement].
#[derive(Clone, Copy)]
pub struct BorrowedReplacement<'a> {
    pub pattern: &'a Pattern,
    pub rhs: BorrowedPatternOrMap<'a>,
    pub conditions: Option<&'a Condition<PatternRestriction>>,
    pub settings: Option<&'a MatchSettings>,
}

pub trait BorrowReplacement {
    fn borrow(&self) -> BorrowedReplacement;
}

impl BorrowReplacement for Replacement {
    fn borrow(&self) -> BorrowedReplacement {
        BorrowedReplacement {
            pattern: &self.pat,
            rhs: self.rhs.borrow(),
            conditions: self.conditions.as_ref(),
            settings: self.settings.as_ref(),
        }
    }
}

impl BorrowReplacement for &Replacement {
    fn borrow(&self) -> BorrowedReplacement {
        BorrowedReplacement {
            pattern: &self.pat,
            rhs: self.rhs.borrow(),
            conditions: self.conditions.as_ref(),
            settings: self.settings.as_ref(),
        }
    }
}

impl<'a> BorrowReplacement for BorrowedReplacement<'a> {
    fn borrow(&self) -> BorrowedReplacement {
        *self
    }
}

/// The context of an atom.
#[derive(Clone, Copy, Debug)]
pub struct Context {
    /// The level of the function in the expression tree.
    pub function_level: usize,
    /// The type of the parent atom.
    pub parent_type: Option<AtomType>,
    /// The index of the atom in the parent.
    pub index: usize,
}

impl<'a> AtomView<'a> {
    pub(crate) fn to_pattern(self) -> Pattern {
        Pattern::from_view(self, true)
    }

    /// Get all symbols in the expression, optionally including function symbols.
    pub(crate) fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        let mut out = HashSet::default();
        self.get_all_symbols_impl(include_function_symbols, &mut out);
        out
    }

    pub(crate) fn get_all_symbols_impl(
        &self,
        include_function_symbols: bool,
        out: &mut HashSet<Symbol>,
    ) {
        match self {
            AtomView::Num(_) => {}
            AtomView::Var(v) => {
                out.insert(v.get_symbol());
            }
            AtomView::Fun(f) => {
                if include_function_symbols {
                    out.insert(f.get_symbol());
                }
                for arg in f {
                    arg.get_all_symbols_impl(include_function_symbols, out);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                base.get_all_symbols_impl(include_function_symbols, out);
                exp.get_all_symbols_impl(include_function_symbols, out);
            }
            AtomView::Mul(m) => {
                for child in m {
                    child.get_all_symbols_impl(include_function_symbols, out);
                }
            }
            AtomView::Add(a) => {
                for child in a {
                    child.get_all_symbols_impl(include_function_symbols, out);
                }
            }
        }
    }

    /// Get all variables and functions in the expression.
    pub(crate) fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView<'a>> {
        let mut out = HashSet::default();
        self.get_all_indeterminates_impl(enter_functions, &mut out);
        out
    }

    fn get_all_indeterminates_impl(&self, enter_functions: bool, out: &mut HashSet<AtomView<'a>>) {
        match self {
            AtomView::Num(_) => {}
            AtomView::Var(_) => {
                out.insert(*self);
            }
            AtomView::Fun(f) => {
                out.insert(*self);

                if enter_functions {
                    for arg in f {
                        arg.get_all_indeterminates_impl(enter_functions, out);
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                base.get_all_indeterminates_impl(enter_functions, out);
                exp.get_all_indeterminates_impl(enter_functions, out);
            }
            AtomView::Mul(m) => {
                for child in m {
                    child.get_all_indeterminates_impl(enter_functions, out);
                }
            }
            AtomView::Add(a) => {
                for child in a {
                    child.get_all_indeterminates_impl(enter_functions, out);
                }
            }
        }
    }

    /// Returns true iff `self` contains `a` literally.
    pub(crate) fn contains<T: AtomCore>(&self, a: T) -> bool {
        let mut stack = Vec::with_capacity(20);
        stack.push(*self);

        while let Some(c) = stack.pop() {
            if a.as_atom_view() == c {
                return true;
            }

            match c {
                AtomView::Num(_) | AtomView::Var(_) => {}
                AtomView::Fun(f) => {
                    for arg in f {
                        stack.push(arg);
                    }
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();
                    stack.push(base);
                    stack.push(exp);
                }
                AtomView::Mul(m) => {
                    for child in m {
                        stack.push(child);
                    }
                }
                AtomView::Add(a) => {
                    for child in a {
                        stack.push(child);
                    }
                }
            }
        }

        false
    }

    /// Returns true iff `self` contains the symbol `s`.
    pub(crate) fn contains_symbol(&self, s: Symbol) -> bool {
        let mut stack = Vec::with_capacity(20);
        stack.push(*self);
        while let Some(c) = stack.pop() {
            match c {
                AtomView::Num(_) => {}
                AtomView::Var(v) => {
                    if v.get_symbol() == s {
                        return true;
                    }
                }
                AtomView::Fun(f) => {
                    if f.get_symbol() == s {
                        return true;
                    }
                    for arg in f {
                        stack.push(arg);
                    }
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();
                    stack.push(base);
                    stack.push(exp);
                }
                AtomView::Mul(m) => {
                    for child in m {
                        stack.push(child);
                    }
                }
                AtomView::Add(a) => {
                    for child in a {
                        stack.push(child);
                    }
                }
            }
        }

        false
    }

    /// Check if the expression can be considered a polynomial in some variables, including
    /// redefinitions. For example `f(x)+y` is considered a polynomial in `f(x)` and `y`, whereas
    /// `f(x)+x` is not a polynomial.
    ///
    /// Rational powers or powers in variables are not rewritten, e.g. `x^(2y)` is not considered
    /// polynomial in `x^y`.
    pub(crate) fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'a>>> {
        let mut vars = HashMap::default();
        let mut symbol_cache = HashSet::default();
        if self.is_polynomial_impl(
            allow_not_expanded,
            allow_negative_powers,
            &mut vars,
            &mut symbol_cache,
        ) {
            symbol_cache.clear();
            for (k, v) in vars {
                if v {
                    symbol_cache.insert(k);
                }
            }

            Some(symbol_cache)
        } else {
            None
        }
    }

    fn is_polynomial_impl(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
        variables: &mut HashMap<AtomView<'a>, bool>,
        symbol_cache: &mut HashSet<AtomView<'a>>,
    ) -> bool {
        if let Some(x) = variables.get(self) {
            return *x;
        }

        macro_rules! block_check {
            ($e: expr) => {
                symbol_cache.clear();
                $e.get_all_indeterminates_impl(true, symbol_cache);
                for x in symbol_cache.drain() {
                    if variables.contains_key(&x) {
                        return false;
                    } else {
                        variables.insert(x, false); // disallow at any level
                    }
                }

                variables.insert(*$e, true); // overwrites block above
            };
        }

        match self {
            AtomView::Num(_) => true,
            AtomView::Var(_) => {
                variables.insert(*self, true);
                true
            }
            AtomView::Fun(_) => {
                block_check!(self);
                true
            }
            AtomView::Pow(pow_view) => {
                // x^y is allowed if x and y do not appear elsewhere
                let (base, exp) = pow_view.get_base_exp();

                if let AtomView::Num(_) = exp {
                    let (positive, integer) = if let Ok(k) = i64::try_from(exp) {
                        (k >= 0, true)
                    } else {
                        (false, false)
                    };

                    if integer && (allow_negative_powers || positive) {
                        if variables.get(&base) == Some(&true) {
                            return true;
                        }

                        if allow_not_expanded && positive {
                            // do not consider (x+y)^-2 a polynomial in x and y
                            return base.is_polynomial_impl(
                                allow_not_expanded,
                                allow_negative_powers,
                                variables,
                                symbol_cache,
                            );
                        }

                        // turn the base into a variable
                        block_check!(&base);
                        return true;
                    }
                }

                block_check!(self);
                true
            }
            AtomView::Mul(mul_view) => {
                for child in mul_view {
                    if !allow_not_expanded {
                        if let AtomView::Add(_) = child {
                            if variables.get(&child) == Some(&true) {
                                continue;
                            }

                            block_check!(&child);
                            continue;
                        }
                    }

                    if !child.is_polynomial_impl(
                        allow_not_expanded,
                        allow_negative_powers,
                        variables,
                        symbol_cache,
                    ) {
                        return false;
                    }
                }
                true
            }
            AtomView::Add(add_view) => {
                for child in add_view {
                    if !child.is_polynomial_impl(
                        allow_not_expanded,
                        allow_negative_powers,
                        variables,
                        symbol_cache,
                    ) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Replace part of an expression by calling the map `m` on each subexpression.
    /// The function `m`  must return `true` if the expression was replaced and must write the new expression to `out`.
    /// A [Context] object is passed to the function, which contains information about the current position in the expression.
    pub(crate) fn replace_map<F: Fn(AtomView, &Context, &mut Atom) -> bool>(&self, m: &F) -> Atom {
        let mut out = Atom::new();
        self.replace_map_into(m, &mut out);
        out
    }

    /// Replace part of an expression by calling the map `m` on each subexpression.
    /// The function `m`  must return `true` if the expression was replaced and must write the new expression to `out`.
    /// A [Context] object is passed to the function, which contains information about the current position in the expression.
    pub(crate) fn replace_map_into<F: Fn(AtomView, &Context, &mut Atom) -> bool>(
        &self,
        m: &F,
        out: &mut Atom,
    ) {
        let context = Context {
            function_level: 0,
            parent_type: None,
            index: 0,
        };
        Workspace::get_local().with(|ws| {
            self.replace_map_impl(ws, m, context, out);
        });
    }

    fn replace_map_impl<F: Fn(AtomView, &Context, &mut Atom) -> bool>(
        &self,
        ws: &Workspace,
        m: &F,
        mut context: Context,
        out: &mut Atom,
    ) -> bool {
        if m(*self, &context, out) {
            return true;
        }

        let mut changed = false;
        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                out.set_from_view(self);
            }
            AtomView::Fun(f) => {
                let mut fun = ws.new_atom();
                let fun = fun.to_fun(f.get_symbol());

                context.parent_type = Some(AtomType::Fun);
                context.function_level += 1;

                for (i, arg) in f.iter().enumerate() {
                    context.index = i;

                    let mut arg_h = ws.new_atom();
                    changed |= arg.replace_map_impl(ws, m, context, &mut arg_h);
                    fun.add_arg(arg_h.as_view());
                }

                if changed {
                    fun.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                context.parent_type = Some(AtomType::Pow);
                context.index = 0;

                let mut base_h = ws.new_atom();
                changed |= base.replace_map_impl(ws, m, context, &mut base_h);

                context.index = 1;
                let mut exp_h = ws.new_atom();
                changed |= exp.replace_map_impl(ws, m, context, &mut exp_h);

                if changed {
                    let mut pow_h = ws.new_atom();
                    pow_h.to_pow(base_h.as_view(), exp_h.as_view());
                    pow_h.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Mul(mm) => {
                let mut mul_h = ws.new_atom();
                let mul = mul_h.to_mul();

                context.parent_type = Some(AtomType::Mul);

                for (i, child) in mm.iter().enumerate() {
                    context.index = i;
                    let mut child_h = ws.new_atom();
                    changed |= child.replace_map_impl(ws, m, context, &mut child_h);
                    mul.extend(child_h.as_view());
                }

                if changed {
                    mul_h.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Add(a) => {
                let mut add_h = ws.new_atom();
                let add = add_h.to_add();

                context.parent_type = Some(AtomType::Add);

                for (i, child) in a.iter().enumerate() {
                    context.index = i;
                    let mut child_h = ws.new_atom();
                    changed |= child.replace_map_impl(ws, m, context, &mut child_h);
                    add.extend(child_h.as_view());
                }

                if changed {
                    add_h.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
        }

        changed
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    pub(crate) fn replace_all<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Atom {
        let mut out = Atom::new();
        self.replace_all_into(pattern, rhs, conditions, settings, &mut out);
        out
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    pub(crate) fn replace_all_into<R: BorrowPatternOrMap>(
        &self,
        pattern: &Pattern,
        rhs: R,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
        out: &mut Atom,
    ) -> bool {
        Workspace::get_local().with(|ws| {
            self.replace_all_with_ws_into(pattern, rhs.borrow(), ws, conditions, settings, out)
        })
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    pub(crate) fn replace_all_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Atom {
        let mut out = Atom::new();
        self.replace_all_multiple_into(replacements, &mut out);
        out
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// Returns `true` iff a match was found.
    pub(crate) fn replace_all_multiple_into<T: BorrowReplacement>(
        &self,
        replacements: &[T],
        out: &mut Atom,
    ) -> bool {
        Workspace::get_local().with(|ws| {
            let mut rhs_cache = HashMap::default();
            let matched = self.replace_all_no_norm(replacements, ws, 0, 0, &mut rhs_cache, out);

            if matched {
                let mut norm = ws.new_atom();
                out.as_view().normalize(ws, &mut norm);
                std::mem::swap(out, &mut norm);
            }

            matched
        })
    }

    /// Replace all occurrences of the patterns in the target, without normalizing the output.
    fn replace_all_no_norm<T: BorrowReplacement>(
        &self,
        replacements: &[T],
        workspace: &Workspace,
        tree_level: usize,
        fn_level: usize,
        rhs_cache: &mut HashMap<(usize, Vec<(Symbol, Match<'a>)>), Atom>,
        out: &mut Atom,
    ) -> bool {
        let mut beyond_max_level = true;
        for (rep_id, r) in replacements.iter().enumerate() {
            let r = r.borrow();

            let def_c = Condition::default();
            let def_s = MatchSettings::default();
            let conditions = r.conditions.unwrap_or(&def_c);
            let settings = r.settings.unwrap_or(&def_s);

            if let Some(max_level) = settings.level_range.1 {
                if settings.level_is_tree_depth && tree_level > max_level
                    || !settings.level_is_tree_depth && fn_level > max_level
                {
                    continue;
                }
            }

            beyond_max_level = false;

            if settings.level_is_tree_depth && tree_level < settings.level_range.0
                || !settings.level_is_tree_depth && fn_level < settings.level_range.0
            {
                continue;
            }

            if r.pattern.could_match(*self) {
                let mut match_stack = WrappedMatchStack::new(conditions, settings);

                let mut it = AtomMatchIterator::new(&r.pattern, *self);
                if let Some((_, used_flags)) = it.next(&mut match_stack) {
                    let mut rhs_subs = workspace.new_atom();

                    let key = (rep_id, std::mem::take(&mut match_stack.stack.stack));

                    if let Some(rhs) = rhs_cache.get(&key) {
                        match_stack.stack.stack = key.1;
                        rhs_subs.set_from_view(&rhs.as_view());
                    } else {
                        match_stack.stack.stack = key.1;

                        match &r.rhs.borrow() {
                            BorrowedPatternOrMap::Pattern(rhs) => {
                                rhs.replace_wildcards_with_matches_impl(
                                    workspace,
                                    &mut rhs_subs,
                                    &match_stack.stack,
                                    settings.allow_new_wildcards_on_rhs,
                                    None,
                                )
                                .unwrap(); // TODO: escalate?
                            }
                            BorrowedPatternOrMap::Map(f) => {
                                let mut rhs = f(&match_stack.stack);
                                std::mem::swap(rhs_subs.deref_mut(), &mut rhs);
                            }
                        }

                        if rhs_cache.len() < settings.rhs_cache_size
                            && !matches!(r.rhs, BorrowedPatternOrMap::Pattern(Pattern::Literal(_)))
                        {
                            rhs_cache.insert(
                                (rep_id, match_stack.stack.stack.clone()),
                                rhs_subs.deref_mut().clone(),
                            );
                        }
                    }

                    if used_flags.iter().all(|x| *x) {
                        // all used, return rhs
                        out.set_from_view(&rhs_subs.as_view());
                        return true;
                    }

                    match self {
                        AtomView::Mul(m) => {
                            let out = out.to_mul();

                            for (child, used) in m.iter().zip(used_flags) {
                                if !used {
                                    out.extend(child);
                                }
                            }

                            out.extend(rhs_subs.as_view());
                        }
                        AtomView::Add(a) => {
                            let out = out.to_add();

                            for (child, used) in a.iter().zip(used_flags) {
                                if !used {
                                    out.extend(child);
                                }
                            }

                            out.extend(rhs_subs.as_view());
                        }
                        _ => {
                            out.set_from_view(&rhs_subs.as_view());
                        }
                    }

                    return true;
                }
            }
        }

        if beyond_max_level {
            out.set_from_view(self);
            return false;
        }

        // no match found at this level, so check the children
        let submatch = match self {
            AtomView::Fun(f) => {
                let out = out.to_fun(f.get_symbol());

                let mut submatch = false;

                let mut child_buf = workspace.new_atom();
                for child in f {
                    submatch |= child.replace_all_no_norm(
                        replacements,
                        workspace,
                        tree_level + 1,
                        fn_level + 1,
                        rhs_cache,
                        &mut child_buf,
                    );

                    out.add_arg(child_buf.as_view());
                }

                out.set_normalized(!submatch && f.is_normalized());
                submatch
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_out = workspace.new_atom();
                let mut submatch = base.replace_all_no_norm(
                    replacements,
                    workspace,
                    tree_level + 1,
                    fn_level,
                    rhs_cache,
                    &mut base_out,
                );

                let mut exp_out = workspace.new_atom();
                submatch |= exp.replace_all_no_norm(
                    replacements,
                    workspace,
                    tree_level + 1,
                    fn_level,
                    rhs_cache,
                    &mut exp_out,
                );

                let out = out.to_pow(base_out.as_view(), exp_out.as_view());
                out.set_normalized(!submatch && p.is_normalized());
                submatch
            }
            AtomView::Mul(m) => {
                let mul = out.to_mul();

                let mut submatch = false;
                let mut child_buf = workspace.new_atom();
                for child in m {
                    submatch |= child.replace_all_no_norm(
                        replacements,
                        workspace,
                        tree_level + 1,
                        fn_level,
                        rhs_cache,
                        &mut child_buf,
                    );

                    mul.extend(child_buf.as_view());
                }

                mul.set_has_coefficient(m.has_coefficient());
                mul.set_normalized(!submatch && m.is_normalized());
                submatch
            }
            AtomView::Add(a) => {
                let out = out.to_add();
                let mut submatch = false;
                let mut child_buf = workspace.new_atom();
                for child in a {
                    submatch |= child.replace_all_no_norm(
                        replacements,
                        workspace,
                        tree_level + 1,
                        fn_level,
                        rhs_cache,
                        &mut child_buf,
                    );

                    out.extend(child_buf.as_view());
                }
                out.set_normalized(!submatch && a.is_normalized());
                submatch
            }
            _ => {
                out.set_from_view(self); // no children
                false
            }
        };

        submatch
    }

    /// Replace all occurrences of the pattern in the target, returning `true` iff a match was found.
    /// For every matched atom, the first canonical match is used and then the atom is skipped.
    pub(crate) fn replace_all_with_ws_into(
        &self,
        pattern: &Pattern,
        rhs: BorrowedPatternOrMap,
        workspace: &Workspace,
        conditions: Option<&Condition<PatternRestriction>>,
        settings: Option<&MatchSettings>,
        out: &mut Atom,
    ) -> bool {
        let rep = BorrowedReplacement {
            pattern,
            rhs,
            conditions,
            settings,
        };

        let mut rhs_cache = HashMap::default();
        let matched = self.replace_all_no_norm(
            std::slice::from_ref(&rep),
            workspace,
            0,
            0,
            &mut rhs_cache,
            out,
        );

        if matched {
            let mut norm = workspace.new_atom();
            out.as_view().normalize(workspace, &mut norm);
            std::mem::swap(out, &mut norm);
        }

        matched
    }
}

impl FromStr for Pattern {
    type Err = String;

    /// Parse a pattern from a string.
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Pattern::parse(input)
    }
}

impl Pattern {
    pub fn parse(input: &str) -> Result<Pattern, String> {
        // TODO: use workspace instead of owned atom
        Ok(Atom::parse(input)?.to_pattern())
    }

    /// Convert the pattern to an atom, if there are not transformers present.
    pub fn to_atom(&self) -> Result<Atom, &'static str> {
        Workspace::get_local().with(|ws| {
            let mut out = Atom::new();
            self.to_atom_impl(ws, &mut out)?;
            Ok(out)
        })
    }

    fn to_atom_impl(&self, ws: &Workspace, out: &mut Atom) -> Result<(), &'static str> {
        match self {
            Pattern::Literal(a) => {
                out.set_from_view(&a.as_view());
            }
            Pattern::Wildcard(s) => {
                out.to_var(*s);
            }
            Pattern::Fn(s, a) => {
                let mut f = ws.new_atom();
                let fun = f.to_fun(*s);

                for arg in a {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    fun.add_arg(arg_h.as_view());
                }

                f.as_view().normalize(ws, out);
            }
            Pattern::Pow(p) => {
                let mut base = ws.new_atom();
                p[0].to_atom_impl(ws, &mut base)?;

                let mut exp = ws.new_atom();
                p[1].to_atom_impl(ws, &mut exp)?;

                let mut pow_h = ws.new_atom();
                pow_h.to_pow(base.as_view(), exp.as_view());
                pow_h.as_view().normalize(ws, out);
            }
            Pattern::Mul(m) => {
                let mut mul_h = ws.new_atom();
                let mul = mul_h.to_mul();

                for arg in m {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    mul.extend(arg_h.as_view());
                }

                mul_h.as_view().normalize(ws, out);
            }
            Pattern::Add(a) => {
                let mut add_h = ws.new_atom();
                let add = add_h.to_add();

                for arg in a {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    add.extend(arg_h.as_view());
                }

                add_h.as_view().normalize(ws, out);
            }
            Pattern::Transformer(_) => Err("Cannot convert transformer to atom")?,
        }

        Ok(())
    }

    pub fn add(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                // create new literal
                let mut e = workspace.new_atom();
                let a = e.to_add();

                a.extend(l1.as_view());
                a.extend(l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        let mut new_args = vec![];
        if let Pattern::Add(l1) = self {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(self.clone());
        }
        if let Pattern::Add(l1) = rhs {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(rhs.clone());
        }

        // TODO: fuse literal parts
        Pattern::Add(new_args)
    }

    pub fn mul(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                let mut e = workspace.new_atom();
                let a = e.to_mul();

                a.extend(l1.as_view());
                a.extend(l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        let mut new_args = vec![];
        if let Pattern::Mul(l1) = self {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(self.clone());
        }
        if let Pattern::Mul(l1) = rhs {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(rhs.clone());
        }

        // TODO: fuse literal parts
        Pattern::Mul(new_args)
    }

    pub fn div(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l2) = rhs {
            let mut pow = workspace.new_atom();
            pow.to_num((-1).into());

            let mut e = workspace.new_atom();
            e.to_pow(l2.as_view(), pow.as_view());

            let mut b = Atom::default();
            e.as_view().normalize(workspace, &mut b);

            match self {
                Pattern::Mul(m) => {
                    let mut new_args = m.clone();
                    new_args.push(Pattern::Literal(b));
                    Pattern::Mul(new_args)
                }
                Pattern::Literal(l1) => {
                    let mut m = workspace.new_atom();
                    let md = m.to_mul();

                    md.extend(l1.as_view());
                    md.extend(b.as_view());

                    let mut b = Atom::default();
                    m.as_view().normalize(workspace, &mut b);
                    Pattern::Literal(b)
                }
                _ => Pattern::Mul(vec![self.clone(), Pattern::Literal(b)]),
            }
        } else {
            let exp = Num::new((-1).into()).into();

            let rhs = Pattern::Mul(vec![
                self.clone(),
                Pattern::Pow(Box::new([rhs.clone(), Pattern::Literal(exp)])),
            ]);

            match self {
                Pattern::Mul(m) => {
                    let mut new_args = m.clone();
                    new_args.push(rhs);
                    Pattern::Mul(new_args)
                }
                _ => Pattern::Mul(vec![self.clone(), rhs]),
            }
        }
    }

    pub fn pow(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                let mut e = workspace.new_atom();
                e.to_pow(l1.as_view(), l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        Pattern::Pow(Box::new([self.clone(), rhs.clone()]))
    }

    pub fn neg(&self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            let mut e = workspace.new_atom();
            let a = e.to_mul();

            let mut sign = workspace.new_atom();
            sign.to_num((-1).into());

            a.extend(l1.as_view());
            a.extend(sign.as_view());

            let mut b = Atom::default();
            e.as_view().normalize(workspace, &mut b);

            Pattern::Literal(b)
        } else {
            let sign = Num::new((-1).into()).into();

            // TODO: simplify if a literal is already present
            Pattern::Mul(vec![self.clone(), Pattern::Literal(sign)])
        }
    }
}

impl Pattern {
    /// A quick check to see if a pattern can match.
    #[inline]
    fn could_match(&self, target: AtomView) -> bool {
        match (self, target) {
            (Pattern::Fn(f1, _), AtomView::Fun(f2)) => {
                f1.get_wildcard_level() > 0 || *f1 == f2.get_symbol()
            }
            (Pattern::Mul(_), AtomView::Mul(_)) => true,
            (Pattern::Add(_), AtomView::Add(_)) => true,
            (Pattern::Wildcard(_), _) => true,
            (Pattern::Pow(_), AtomView::Pow(_)) => true,
            (Pattern::Literal(p), _) => p.as_view() == target,
            (Pattern::Transformer(_), _) => unreachable!(),
            (_, _) => false,
        }
    }

    /// Check if the expression `atom` contains a wildcard.
    fn has_wildcard(atom: AtomView<'_>) -> bool {
        match atom {
            AtomView::Num(_) => false,
            AtomView::Var(v) => v.get_wildcard_level() > 0,
            AtomView::Fun(f) => {
                if f.get_symbol().get_wildcard_level() > 0 {
                    return true;
                }

                for arg in f {
                    if Self::has_wildcard(arg) {
                        return true;
                    }
                }
                false
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                Self::has_wildcard(base) || Self::has_wildcard(exp)
            }
            AtomView::Mul(m) => {
                for child in m {
                    if Self::has_wildcard(child) {
                        return true;
                    }
                }
                false
            }
            AtomView::Add(a) => {
                for child in a {
                    if Self::has_wildcard(child) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Create a pattern from an atom view.
    pub(crate) fn from_view(atom: AtomView<'_>, is_top_layer: bool) -> Pattern {
        // split up Add and Mul for literal patterns as well so that x+y can match to x+y+z
        if Self::has_wildcard(atom)
            || is_top_layer && matches!(atom, AtomView::Mul(_) | AtomView::Add(_))
        {
            match atom {
                AtomView::Var(v) => Pattern::Wildcard(v.get_symbol()),
                AtomView::Fun(f) => {
                    let name = f.get_symbol();

                    let mut args = Vec::with_capacity(f.get_nargs());
                    for arg in f {
                        args.push(Self::from_view(arg, false));
                    }

                    Pattern::Fn(name, args)
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    Pattern::Pow(Box::new([
                        Self::from_view(base, false),
                        Self::from_view(exp, false),
                    ]))
                }
                AtomView::Mul(m) => {
                    let mut args = Vec::with_capacity(m.get_nargs());

                    for child in m {
                        args.push(Self::from_view(child, false));
                    }

                    Pattern::Mul(args)
                }
                AtomView::Add(a) => {
                    let mut args = Vec::with_capacity(a.get_nargs());
                    for child in a {
                        args.push(Self::from_view(child, false));
                    }

                    Pattern::Add(args)
                }
                AtomView::Num(_) => unreachable!("Number cannot have wildcard"),
            }
        } else {
            let mut oa = Atom::default();
            oa.set_from_view(&atom);
            Pattern::Literal(oa)
        }
    }

    /// Substitute the wildcards in the pattern.
    pub fn replace_wildcards(&self, matches: &HashMap<Symbol, Atom>) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            self.replace_wildcards_impl(matches, ws, &mut out);
        });
        out
    }

    fn replace_wildcards_impl(
        &self,
        matches: &HashMap<Symbol, Atom>,
        ws: &Workspace,
        out: &mut Atom,
    ) {
        match self {
            Pattern::Literal(atom) => out.set_from_view(&atom.as_view()),
            Pattern::Wildcard(symbol) => {
                if let Some(a) = matches.get(&symbol) {
                    out.set_from_view(&a.as_view());
                } else {
                    out.to_var(*symbol);
                }
            }
            Pattern::Fn(symbol, args) => {
                let symbol = if let Some(a) = matches.get(symbol) {
                    a.as_view().get_symbol().expect("Function name expected")
                } else {
                    *symbol
                };

                let mut fun = ws.new_atom();
                let f = fun.to_fun(symbol);

                let mut arg = ws.new_atom();
                for a in args {
                    a.replace_wildcards_impl(matches, ws, &mut arg);
                    f.add_arg(arg.as_view());
                }

                fun.as_view().normalize(ws, out);
            }
            Pattern::Pow(args) => {
                let mut pow = ws.new_atom();

                let mut base = ws.new_atom();
                args[0].replace_wildcards_impl(matches, ws, &mut base);
                let mut exp = ws.new_atom();
                args[1].replace_wildcards_impl(matches, ws, &mut exp);
                pow.to_pow(base.as_view(), exp.as_view());

                pow.as_view().normalize(ws, out);
            }
            Pattern::Mul(args) => {
                let mut mul = ws.new_atom();
                let m = mul.to_mul();

                let mut arg = ws.new_atom();
                for a in args {
                    a.replace_wildcards_impl(matches, ws, &mut arg);
                    m.extend(arg.as_view());
                }

                mul.as_view().normalize(ws, out);
            }
            Pattern::Add(args) => {
                let mut add = ws.new_atom();
                let aa = add.to_add();

                let mut arg = ws.new_atom();
                for a in args {
                    a.replace_wildcards_impl(matches, ws, &mut arg);
                    aa.extend(arg.as_view());
                }

                add.as_view().normalize(ws, out);
            }
            Pattern::Transformer(_) => {
                panic!("Encountered transformer during substitution of wildcards from a map")
            }
        }
    }

    /// Substitute the wildcards in the pattern with the values in the match stack.
    pub fn replace_wildcards_with_matches(&self, match_stack: &MatchStack<'_>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = Atom::new();
            self.replace_wildcards_with_matches_impl(ws, &mut out, match_stack, true, None)
                .unwrap();
            out
        })
    }

    /// Substitute the wildcards in the pattern with the values in the match stack.
    pub fn replace_wildcards_with_matches_impl(
        &self,
        workspace: &Workspace,
        out: &mut Atom,
        match_stack: &MatchStack<'_>,
        allow_new_wildcards_on_rhs: bool,
        transformer_input: Option<&Pattern>,
    ) -> Result<(), TransformerError> {
        match self {
            Pattern::Wildcard(name) => {
                if let Some(w) = match_stack.get(*name) {
                    w.to_atom_into(out);
                } else if allow_new_wildcards_on_rhs {
                    out.to_var(*name);
                } else {
                    Err(TransformerError::ValueError(format!(
                        "Unsubstituted wildcard {}",
                        name
                    )))?;
                }
            }
            Pattern::Fn(mut name, args) => {
                if name.get_wildcard_level() > 0 {
                    if let Some(w) = match_stack.get(name) {
                        if let Match::FunctionName(fname) = w {
                            name = *fname;
                        } else if let Match::Single(a) = w {
                            if let AtomView::Var(v) = a {
                                name = v.get_symbol();
                            } else {
                                Err(TransformerError::ValueError(format!(
                                    "Wildcard must be a function name instead of {}",
                                    w.to_atom()
                                )))?;
                            }
                        } else {
                            Err(TransformerError::ValueError(format!(
                                "Wildcard must be a function name instead of {}",
                                w.to_atom()
                            )))?;
                        }
                    } else if !allow_new_wildcards_on_rhs {
                        Err(TransformerError::ValueError(format!(
                            "Unsubstituted wildcard {}",
                            name
                        )))?;
                    }
                }

                let mut func_h = workspace.new_atom();
                let func = func_h.to_fun(name);

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => func.add_arg(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Arg | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            func.add_arg(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom_into(&mut handle);
                                        func.add_arg(handle.as_view())
                                    }
                                },
                                Match::FunctionName(s) => {
                                    func.add_arg(InlineVar::new(*s).as_view())
                                }
                            }
                        } else if allow_new_wildcards_on_rhs {
                            func.add_arg(workspace.new_var(*w).as_view())
                        } else {
                            Err(TransformerError::ValueError(format!(
                                "Unsubstituted wildcard {}",
                                w
                            )))?;
                        }

                        continue;
                    }

                    let mut handle = workspace.new_atom();
                    arg.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut handle,
                        match_stack,
                        allow_new_wildcards_on_rhs,
                        transformer_input,
                    )?;
                    func.add_arg(handle.as_view());
                }

                func_h.as_view().normalize(workspace, out);
            }
            Pattern::Pow(base_and_exp) => {
                let mut base = workspace.new_atom();
                let mut exp = workspace.new_atom();
                let mut oas = [&mut base, &mut exp];

                for (out, arg) in oas.iter_mut().zip(base_and_exp.iter()) {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => out.set_from_view(s),
                                Match::Multiple(_, _) => {
                                    let mut handle = workspace.new_atom();
                                    w.to_atom_into(&mut handle);
                                    out.set_from_view(&handle.as_view())
                                }
                                Match::FunctionName(s) => {
                                    out.set_from_view(&InlineVar::new(*s).as_view())
                                }
                            }
                        } else if allow_new_wildcards_on_rhs {
                            out.set_from_view(&workspace.new_var(*w).as_view());
                        } else {
                            Err(TransformerError::ValueError(format!(
                                "Unsubstituted wildcard {}",
                                w
                            )))?;
                        }

                        continue;
                    }

                    let mut handle = workspace.new_atom();
                    arg.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut handle,
                        match_stack,
                        allow_new_wildcards_on_rhs,
                        transformer_input,
                    )?;
                    out.set_from_view(&handle.as_view());
                }

                let mut pow_h = workspace.new_atom();
                pow_h.to_pow(oas[0].as_view(), oas[1].as_view());
                pow_h.as_view().normalize(workspace, out);
            }
            Pattern::Mul(args) => {
                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => mul.extend(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Mul | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            mul.extend(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom_into(&mut handle);
                                        mul.extend(handle.as_view())
                                    }
                                },
                                Match::FunctionName(s) => mul.extend(InlineVar::new(*s).as_view()),
                            }
                        } else if allow_new_wildcards_on_rhs {
                            mul.extend(workspace.new_var(*w).as_view());
                        } else {
                            Err(TransformerError::ValueError(format!(
                                "Unsubstituted wildcard {}",
                                w
                            )))?;
                        }

                        continue;
                    }

                    let mut handle = workspace.new_atom();
                    arg.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut handle,
                        match_stack,
                        allow_new_wildcards_on_rhs,
                        transformer_input,
                    )?;
                    mul.extend(handle.as_view());
                }
                mul_h.as_view().normalize(workspace, out);
            }
            Pattern::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => add.extend(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Add | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            add.extend(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom_into(&mut handle);
                                        add.extend(handle.as_view())
                                    }
                                },
                                Match::FunctionName(s) => add.extend(InlineVar::new(*s).as_view()),
                            }
                        } else if allow_new_wildcards_on_rhs {
                            add.extend(workspace.new_var(*w).as_view());
                        } else {
                            Err(TransformerError::ValueError(format!(
                                "Unsubstituted wildcard {}",
                                w
                            )))?;
                        }

                        continue;
                    }

                    let mut handle = workspace.new_atom();
                    arg.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut handle,
                        match_stack,
                        allow_new_wildcards_on_rhs,
                        transformer_input,
                    )?;
                    add.extend(handle.as_view());
                }
                add_h.as_view().normalize(workspace, out);
            }
            Pattern::Literal(oa) => {
                out.set_from_view(&oa.as_view());
            }
            Pattern::Transformer(p) => {
                let (pat, ts) = &**p;

                let pat = if let Some(p) = pat.as_ref() {
                    p
                } else if let Some(input_p) = transformer_input {
                    input_p
                } else {
                    Err(TransformerError::ValueError(
                        "Transformer is missing an expression to act on.".to_owned(),
                    ))?
                };

                let mut handle = workspace.new_atom();
                pat.replace_wildcards_with_matches_impl(
                    workspace,
                    &mut handle,
                    match_stack,
                    allow_new_wildcards_on_rhs,
                    transformer_input,
                )?;

                Transformer::execute_chain(handle.as_view(), ts, workspace, out)?;
            }
        }

        Ok(())
    }
}

impl std::fmt::Debug for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wildcard(arg0) => f.debug_tuple("Wildcard").field(arg0).finish(),
            Self::Fn(arg0, arg1) => f.debug_tuple("Fn").field(arg0).field(arg1).finish(),
            Self::Pow(arg0) => f.debug_tuple("Pow").field(arg0).finish(),
            Self::Mul(arg0) => f.debug_tuple("Mul").field(arg0).finish(),
            Self::Add(arg0) => f.debug_tuple("Add").field(arg0).finish(),
            Self::Literal(arg0) => f.debug_tuple("Literal").field(arg0).finish(),
            Self::Transformer(arg0) => f.debug_tuple("Transformer").field(arg0).finish(),
        }
    }
}

pub trait FilterFn: Fn(&Match) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(FilterFn);
impl<T: Clone + Send + Sync + Fn(&Match) -> bool> FilterFn for T {}

pub trait CmpFn: Fn(&Match, &Match) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(CmpFn);
impl<T: Clone + Send + Sync + Fn(&Match, &Match) -> bool> CmpFn for T {}

pub trait MatchStackFn: Fn(&MatchStack) -> ConditionResult + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(MatchStackFn);
impl<T: Clone + Send + Sync + Fn(&MatchStack) -> ConditionResult> MatchStackFn for T {}

/// Restrictions for a wildcard. Note that a length restriction
/// applies at any level and therefore
/// `x_*f(x_) : length(x) == 2`
/// does not match to `x*y*f(x*y)`, since the pattern `x_` has length
/// 1 inside the function argument.
pub enum WildcardRestriction {
    Length(usize, Option<usize>), // min-max range
    IsAtomType(AtomType),
    IsLiteralWildcard(Symbol),
    Filter(Box<dyn FilterFn>),
    Cmp(Symbol, Box<dyn CmpFn>),
    NotGreedy,
}

impl std::fmt::Display for WildcardRestriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WildcardRestriction::Length(min, Some(max)) => write!(f, "length={}-{}", min, max),
            WildcardRestriction::Length(min, None) => write!(f, "length > {}", min),
            WildcardRestriction::IsAtomType(t) => write!(f, "type = {}", t),
            WildcardRestriction::IsLiteralWildcard(s) => write!(f, "= {}", s),
            WildcardRestriction::Filter(_) => write!(f, "filter"),
            WildcardRestriction::Cmp(s, _) => write!(f, "cmp with {}", s),
            WildcardRestriction::NotGreedy => write!(f, "not greedy"),
        }
    }
}

pub type WildcardAndRestriction = (Symbol, WildcardRestriction);

/// A restriction on a wildcard or wildcards.
pub enum PatternRestriction {
    /// A restriction for a wildcard.
    Wildcard(WildcardAndRestriction),
    /// A function that checks if the restriction is met based on the currently matched wildcards.
    /// If more information is needed to test the restriction, the function should return `Inconclusive`.
    MatchStack(Box<dyn MatchStackFn>),
}

impl std::fmt::Display for PatternRestriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternRestriction::Wildcard((s, r)) => write!(f, "{}: {}", s, r),
            PatternRestriction::MatchStack(_) => write!(f, "match_function"),
        }
    }
}

impl Clone for PatternRestriction {
    fn clone(&self) -> Self {
        match self {
            PatternRestriction::Wildcard(w) => PatternRestriction::Wildcard(w.clone()),
            PatternRestriction::MatchStack(f) => {
                PatternRestriction::MatchStack(dyn_clone::clone_box(f))
            }
        }
    }
}

impl From<WildcardAndRestriction> for PatternRestriction {
    fn from(value: WildcardAndRestriction) -> Self {
        PatternRestriction::Wildcard(value)
    }
}

impl From<WildcardAndRestriction> for Condition<PatternRestriction> {
    fn from(value: WildcardAndRestriction) -> Self {
        PatternRestriction::Wildcard(value).into()
    }
}

static DEFAULT_PATTERN_CONDITION: Condition<PatternRestriction> = Condition::True;

/// A logical expression.
#[derive(Clone, Debug, Default)]
pub enum Condition<T> {
    And(Box<(Condition<T>, Condition<T>)>),
    Or(Box<(Condition<T>, Condition<T>)>),
    Not(Box<Condition<T>>),
    Yield(T),
    #[default]
    True,
    False,
}

impl<T: std::fmt::Display> std::fmt::Display for Condition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Condition::And(a) => write!(f, "({}) & ({})", a.0, a.1),
            Condition::Or(o) => write!(f, "{} | {}", o.0, o.1),
            Condition::Not(n) => write!(f, "!({})", n),
            Condition::True => write!(f, "True"),
            Condition::False => write!(f, "False"),
            Condition::Yield(t) => write!(f, "{}", t),
        }
    }
}

pub trait Evaluate {
    type State<'a>;

    /// Evaluate a condition.
    fn evaluate<'a>(&self, state: &Self::State<'a>) -> Result<ConditionResult, String>;
}

impl<T: Evaluate> Evaluate for Condition<T> {
    type State<'a> = T::State<'a>;

    fn evaluate(&self, state: &T::State<'_>) -> Result<ConditionResult, String> {
        Ok(match self {
            Condition::And(a) => a.0.evaluate(state)? & a.1.evaluate(state)?,
            Condition::Or(o) => o.0.evaluate(state)? | o.1.evaluate(state)?,
            Condition::Not(n) => !n.evaluate(state)?,
            Condition::True => ConditionResult::True,
            Condition::False => ConditionResult::False,
            Condition::Yield(t) => t.evaluate(state)?,
        })
    }
}

impl<T> From<T> for Condition<T> {
    fn from(value: T) -> Self {
        Condition::Yield(value)
    }
}

impl<T, R: Into<Condition<T>>> std::ops::BitOr<R> for Condition<T> {
    type Output = Condition<T>;

    fn bitor(self, rhs: R) -> Self::Output {
        Condition::Or(Box::new((self, rhs.into())))
    }
}

impl<T, R: Into<Condition<T>>> std::ops::BitAnd<R> for Condition<T> {
    type Output = Condition<T>;

    fn bitand(self, rhs: R) -> Self::Output {
        Condition::And(Box::new((self, rhs.into())))
    }
}

impl<T> std::ops::Not for Condition<T> {
    type Output = Condition<T>;

    fn not(self) -> Self::Output {
        Condition::Not(Box::new(self))
    }
}

/// The result of the evaluation of a condition, which can be
/// true, false, or inconclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionResult {
    True,
    False,
    Inconclusive,
}

impl std::ops::BitOr<ConditionResult> for ConditionResult {
    type Output = ConditionResult;

    fn bitor(self, rhs: ConditionResult) -> Self::Output {
        match (self, rhs) {
            (ConditionResult::True, _) => ConditionResult::True,
            (_, ConditionResult::True) => ConditionResult::True,
            (ConditionResult::False, ConditionResult::False) => ConditionResult::False,
            _ => ConditionResult::Inconclusive,
        }
    }
}

impl std::ops::BitAnd<ConditionResult> for ConditionResult {
    type Output = ConditionResult;

    fn bitand(self, rhs: ConditionResult) -> Self::Output {
        match (self, rhs) {
            (ConditionResult::False, _) => ConditionResult::False,
            (_, ConditionResult::False) => ConditionResult::False,
            (ConditionResult::True, ConditionResult::True) => ConditionResult::True,
            _ => ConditionResult::Inconclusive,
        }
    }
}

impl std::ops::Not for ConditionResult {
    type Output = ConditionResult;

    fn not(self) -> Self::Output {
        match self {
            ConditionResult::True => ConditionResult::False,
            ConditionResult::False => ConditionResult::True,
            ConditionResult::Inconclusive => ConditionResult::Inconclusive,
        }
    }
}

impl From<bool> for ConditionResult {
    fn from(value: bool) -> Self {
        if value {
            ConditionResult::True
        } else {
            ConditionResult::False
        }
    }
}

impl ConditionResult {
    pub fn is_true(&self) -> bool {
        matches!(self, ConditionResult::True)
    }

    pub fn is_false(&self) -> bool {
        matches!(self, ConditionResult::False)
    }

    pub fn is_inconclusive(&self) -> bool {
        matches!(self, ConditionResult::Inconclusive)
    }
}

/// A test on one or more patterns that should yield
/// a [ConditionResult] when evaluated.
#[derive(Clone, Debug)]
pub enum Relation {
    Eq(Pattern, Pattern),
    Ne(Pattern, Pattern),
    Gt(Pattern, Pattern),
    Ge(Pattern, Pattern),
    Lt(Pattern, Pattern),
    Le(Pattern, Pattern),
    Contains(Pattern, Pattern),
    IsType(Pattern, AtomType),
    Matches(
        Pattern,
        Pattern,
        Condition<PatternRestriction>,
        MatchSettings,
    ),
}

impl std::fmt::Display for Relation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Relation::Eq(a, b) => write!(f, "{} == {}", a, b),
            Relation::Ne(a, b) => write!(f, "{} != {}", a, b),
            Relation::Gt(a, b) => write!(f, "{} > {}", a, b),
            Relation::Ge(a, b) => write!(f, "{} >= {}", a, b),
            Relation::Lt(a, b) => write!(f, "{} < {}", a, b),
            Relation::Le(a, b) => write!(f, "{} <= {}", a, b),
            Relation::Contains(a, b) => write!(f, "{} contains {}", a, b),
            Relation::IsType(a, b) => write!(f, "{} is type {:?}", a, b),
            Relation::Matches(a, b, _, _) => write!(f, "{} matches {}", a, b),
        }
    }
}

impl Evaluate for Relation {
    type State<'a> = Option<AtomView<'a>>;

    fn evaluate(&self, state: &Option<AtomView>) -> Result<ConditionResult, String> {
        Workspace::get_local().with(|ws| {
            let mut out1 = ws.new_atom();
            let mut out2 = ws.new_atom();
            let m = MatchStack::new();

            let pat = state.map(|x| x.to_pattern());

            Ok(match self {
                Relation::Eq(a, b)
                | Relation::Ne(a, b)
                | Relation::Gt(a, b)
                | Relation::Ge(a, b)
                | Relation::Lt(a, b)
                | Relation::Le(a, b)
                | Relation::Contains(a, b) => {
                    a.replace_wildcards_with_matches_impl(ws, &mut out1, &m, true, pat.as_ref())
                        .map_err(|e| match e {
                            TransformerError::Interrupt => "Interrupted by user".into(),
                            TransformerError::ValueError(v) => v,
                        })?;
                    b.replace_wildcards_with_matches_impl(ws, &mut out2, &m, true, pat.as_ref())
                        .map_err(|e| match e {
                            TransformerError::Interrupt => "Interrupted by user".into(),
                            TransformerError::ValueError(v) => v,
                        })?;

                    match self {
                        Relation::Eq(_, _) => out1 == out2,
                        Relation::Ne(_, _) => out1 != out2,
                        Relation::Gt(_, _) => out1.as_view() > out2.as_view(),
                        Relation::Ge(_, _) => out1.as_view() >= out2.as_view(),
                        Relation::Lt(_, _) => out1.as_view() < out2.as_view(),
                        Relation::Le(_, _) => out1.as_view() <= out2.as_view(),
                        Relation::Contains(_, _) => out1.contains(out2.as_view()),
                        _ => unreachable!(),
                    }
                }
                Relation::Matches(a, pattern, cond, settings) => {
                    a.replace_wildcards_with_matches_impl(ws, &mut out1, &m, true, pat.as_ref())
                        .map_err(|e| match e {
                            TransformerError::Interrupt => "Interrupted by user".into(),
                            TransformerError::ValueError(v) => v,
                        })?;

                    out1.pattern_match(pattern, Some(cond), Some(settings))
                        .next()
                        .is_some()
                }
                Relation::IsType(a, b) => {
                    a.replace_wildcards_with_matches_impl(ws, &mut out1, &m, true, pat.as_ref())
                        .map_err(|e| match e {
                            TransformerError::Interrupt => "Interrupted by user".into(),
                            TransformerError::ValueError(v) => v,
                        })?;

                    match out1.as_ref() {
                        Atom::Var(_) => (*b == AtomType::Var).into(),
                        Atom::Fun(_) => (*b == AtomType::Fun).into(),
                        Atom::Num(_) => (*b == AtomType::Num).into(),
                        Atom::Add(_) => (*b == AtomType::Add).into(),
                        Atom::Mul(_) => (*b == AtomType::Mul).into(),
                        Atom::Pow(_) => (*b == AtomType::Pow).into(),
                        Atom::Zero => (*b == AtomType::Num).into(),
                    }
                }
            }
            .into())
        })
    }
}

impl Evaluate for Condition<PatternRestriction> {
    type State<'a> = MatchStack<'a>;

    fn evaluate(&self, state: &MatchStack) -> Result<ConditionResult, String> {
        Ok(match self {
            Condition::And(a) => a.0.evaluate(state)? & a.1.evaluate(state)?,
            Condition::Or(o) => o.0.evaluate(state)? | o.1.evaluate(state)?,
            Condition::Not(n) => !n.evaluate(state)?,
            Condition::True => ConditionResult::True,
            Condition::False => ConditionResult::False,
            Condition::Yield(t) => match t {
                PatternRestriction::Wildcard((v, r)) => {
                    if let Some((_, value)) = state.stack.iter().find(|(k, _)| k == v) {
                        match r {
                            WildcardRestriction::IsAtomType(t) => match value {
                                Match::Single(AtomView::Num(_)) => *t == AtomType::Num,
                                Match::Single(AtomView::Var(_)) => *t == AtomType::Var,
                                Match::Single(AtomView::Add(_)) => *t == AtomType::Add,
                                Match::Single(AtomView::Mul(_)) => *t == AtomType::Mul,
                                Match::Single(AtomView::Pow(_)) => *t == AtomType::Pow,
                                Match::Single(AtomView::Fun(_)) => *t == AtomType::Fun,
                                _ => false,
                            },
                            WildcardRestriction::IsLiteralWildcard(wc) => match value {
                                Match::Single(AtomView::Var(v)) => wc == &v.get_symbol(),
                                _ => false,
                            },
                            WildcardRestriction::Length(min, max) => match value {
                                Match::Single(_) | Match::FunctionName(_) => {
                                    *min <= 1 && max.map(|m| m >= 1).unwrap_or(true)
                                }
                                Match::Multiple(_, slice) => {
                                    *min <= slice.len()
                                        && max.map(|m| m >= slice.len()).unwrap_or(true)
                                }
                            },
                            WildcardRestriction::Filter(f) => f(value),
                            WildcardRestriction::Cmp(v2, f) => {
                                if let Some((_, value2)) = state.stack.iter().find(|(k, _)| k == v2)
                                {
                                    f(value, value2)
                                } else {
                                    return Ok(ConditionResult::Inconclusive);
                                }
                            }
                            WildcardRestriction::NotGreedy => true,
                        }
                        .into()
                    } else {
                        ConditionResult::Inconclusive
                    }
                }
                PatternRestriction::MatchStack(mf) => mf(state),
            },
        })
    }
}

impl Condition<PatternRestriction> {
    /// Check if the conditions on `var` are met
    fn check_possible(&self, var: Symbol, value: &Match, stack: &MatchStack) -> ConditionResult {
        match self {
            Condition::And(a) => {
                a.0.check_possible(var, value, stack) & a.1.check_possible(var, value, stack)
            }
            Condition::Or(o) => {
                o.0.check_possible(var, value, stack) | o.1.check_possible(var, value, stack)
            }
            Condition::Not(n) => !n.check_possible(var, value, stack),
            Condition::True => ConditionResult::True,
            Condition::False => ConditionResult::False,
            Condition::Yield(restriction) => {
                let (v, r) = match restriction {
                    PatternRestriction::Wildcard((v, r)) => (v, r),
                    PatternRestriction::MatchStack(mf) => {
                        return mf(&stack);
                    }
                };

                if *v != var {
                    match r {
                        WildcardRestriction::Cmp(v, _) if *v == var => {}
                        _ => {
                            // TODO: we can actually return True if the v is in the match stack
                            // same for cmp if both are in the stack
                            return ConditionResult::Inconclusive;
                        }
                    }
                }

                match r {
                    WildcardRestriction::IsAtomType(t) => {
                        let is_type = match t {
                            AtomType::Num => matches!(value, Match::Single(AtomView::Num(_))),
                            AtomType::Var => matches!(value, Match::Single(AtomView::Var(_))),
                            AtomType::Add => matches!(
                                value,
                                Match::Single(AtomView::Add(_))
                                    | Match::Multiple(SliceType::Add, _)
                            ),
                            AtomType::Mul => matches!(
                                value,
                                Match::Single(AtomView::Mul(_))
                                    | Match::Multiple(SliceType::Mul, _)
                            ),
                            AtomType::Pow => matches!(
                                value,
                                Match::Single(AtomView::Pow(_))
                                    | Match::Multiple(SliceType::Pow, _)
                            ),
                            AtomType::Fun => matches!(value, Match::Single(AtomView::Fun(_))),
                        };

                        (is_type == matches!(r, WildcardRestriction::IsAtomType(_))).into()
                    }
                    WildcardRestriction::IsLiteralWildcard(wc) => {
                        if let Match::Single(AtomView::Var(v)) = value {
                            (wc == &v.get_symbol()).into()
                        } else {
                            false.into()
                        }
                    }
                    WildcardRestriction::Length(min, max) => match &value {
                        Match::Single(_) | Match::FunctionName(_) => {
                            (*min <= 1 && max.map(|m| m >= 1).unwrap_or(true)).into()
                        }
                        Match::Multiple(_, slice) => (*min <= slice.len()
                            && max.map(|m| m >= slice.len()).unwrap_or(true))
                        .into(),
                    },
                    WildcardRestriction::Filter(f) => f(value).into(),
                    WildcardRestriction::Cmp(v2, f) => {
                        if *v == var {
                            if let Some((_, value2)) = stack.stack.iter().find(|(k, _)| k == v2) {
                                f(value, value2).into()
                            } else {
                                ConditionResult::Inconclusive
                            }
                        } else if let Some((_, value2)) = stack.stack.iter().find(|(k, _)| k == v) {
                            // var == v2 at this point
                            f(value2, value).into()
                        } else {
                            ConditionResult::Inconclusive
                        }
                    }
                    WildcardRestriction::NotGreedy => true.into(),
                }
            }
        }
    }

    fn get_range_hint(&self, var: Symbol) -> (Option<usize>, Option<usize>) {
        match self {
            Condition::And(a) => {
                let (min1, max1) = a.0.get_range_hint(var);
                let (min2, max2) = a.1.get_range_hint(var);

                (
                    match (min1, min2) {
                        (None, None) => None,
                        (None, Some(m)) => Some(m),
                        (Some(m), None) => Some(m),
                        (Some(m1), Some(m2)) => Some(m1.max(m2)),
                    },
                    match (max1, max2) {
                        (None, None) => None,
                        (None, Some(m)) => Some(m),
                        (Some(m), None) => Some(m),
                        (Some(m1), Some(m2)) => Some(m1.min(m2)),
                    },
                )
            }
            Condition::Or(o) => {
                // take the extremes of the min and max
                let (min1, max1) = o.0.get_range_hint(var);
                let (min2, max2) = o.1.get_range_hint(var);

                (
                    if let (Some(m1), Some(m2)) = (min1, min2) {
                        Some(m1.min(m2))
                    } else {
                        None
                    },
                    if let (Some(m1), Some(m2)) = (max1, max2) {
                        Some(m1.max(m2))
                    } else {
                        None
                    },
                )
            }
            Condition::Not(_) => {
                // the range is disconnected and therefore cannot be described
                // using our range conditions
                (None, None)
            }
            Condition::True | Condition::False => (None, None),
            Condition::Yield(restriction) => {
                let (v, r) = match restriction {
                    PatternRestriction::Wildcard((v, r)) => (v, r),
                    PatternRestriction::MatchStack(_) => {
                        return (None, None);
                    }
                };

                if *v != var {
                    return (None, None);
                }

                match r {
                    WildcardRestriction::Length(min, max) => (Some(*min), *max),
                    WildcardRestriction::IsAtomType(
                        AtomType::Var | AtomType::Num | AtomType::Fun,
                    )
                    | WildcardRestriction::IsLiteralWildcard(_) => (Some(1), Some(1)),
                    _ => (None, None),
                }
            }
        }
    }
}

impl Clone for WildcardRestriction {
    fn clone(&self) -> Self {
        match self {
            Self::Length(min, max) => Self::Length(*min, *max),
            Self::IsAtomType(t) => Self::IsAtomType(*t),
            Self::IsLiteralWildcard(w) => Self::IsLiteralWildcard(*w),
            Self::Filter(f) => Self::Filter(dyn_clone::clone_box(f)),
            Self::Cmp(i, f) => Self::Cmp(*i, dyn_clone::clone_box(f)),
            Self::NotGreedy => Self::NotGreedy,
        }
    }
}

impl std::fmt::Debug for WildcardRestriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(arg0, arg1) => f.debug_tuple("Length").field(arg0).field(arg1).finish(),
            Self::IsAtomType(t) => write!(f, "Is{:?}", t),
            Self::IsLiteralWildcard(arg0) => {
                f.debug_tuple("IsLiteralWildcard").field(arg0).finish()
            }
            Self::Filter(_) => f.debug_tuple("Filter").finish(),
            Self::Cmp(arg0, _) => f.debug_tuple("Cmp").field(arg0).finish(),
            Self::NotGreedy => write!(f, "NotGreedy"),
        }
    }
}

impl std::fmt::Debug for PatternRestriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PatternRestriction::Wildcard(arg0) => f.debug_tuple("Wildcard").field(arg0).finish(),
            PatternRestriction::MatchStack(_) => f.debug_tuple("Match").finish(),
        }
    }
}

/// A part of an expression that was matched to a wildcard.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Match<'a> {
    /// A matched single atom.
    Single(AtomView<'a>),
    /// A matched subexpression of atoms of the same type.
    Multiple(SliceType, Vec<AtomView<'a>>),
    /// A matched function name.
    FunctionName(Symbol),
}

impl<'a> std::fmt::Display for Match<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(a) => a.fmt(f),
            Self::Multiple(t, list) => match t {
                SliceType::Add | SliceType::Mul | SliceType::Arg | SliceType::Pow => {
                    f.write_str("(")?;
                    for (i, a) in list.iter().enumerate() {
                        if i > 0 {
                            match t {
                                SliceType::Add => {
                                    f.write_str("+")?;
                                }
                                SliceType::Mul => {
                                    f.write_str("*")?;
                                }
                                SliceType::Arg => {
                                    f.write_str(",")?;
                                }
                                SliceType::Pow => {
                                    f.write_str("^")?;
                                }
                                _ => unreachable!(),
                            }
                        }
                        a.fmt(f)?;
                    }
                    f.write_str(")")
                }
                SliceType::One => list[0].fmt(f),
                SliceType::Empty => f.write_str("()"),
            },
            Self::FunctionName(name) => name.fmt(f),
        }
    }
}

impl<'a> std::fmt::Debug for Match<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(a) => f.debug_tuple("").field(a).finish(),
            Self::Multiple(t, list) => f.debug_tuple("").field(t).field(list).finish(),
            Self::FunctionName(name) => f.debug_tuple("Fn").field(name).finish(),
        }
    }
}

impl<'a> Match<'a> {
    /// Create a new atom from a matched subexpression.
    /// Arguments lists are wrapped in the function `arg`.
    pub fn to_atom(&self) -> Atom {
        let mut out = Atom::default();
        self.to_atom_into(&mut out);
        out
    }

    /// Create a new atom from a matched subexpression.
    /// Arguments lists are wrapped in the function `arg`.
    pub fn to_atom_into(&self, out: &mut Atom) {
        match self {
            Self::Single(v) => {
                out.set_from_view(v);
            }
            Self::Multiple(t, wargs) => match t {
                SliceType::Add => {
                    let add = out.to_add();
                    for arg in wargs {
                        add.extend(*arg);
                    }

                    add.set_normalized(true);
                }
                SliceType::Mul => {
                    let mul = out.to_mul();
                    for arg in wargs {
                        mul.extend(*arg);
                    }

                    // normalization may be needed, for example
                    // to update the coefficient flag
                }
                SliceType::Arg => {
                    let fun = out.to_fun(Atom::ARG);
                    for arg in wargs {
                        fun.add_arg(*arg);
                    }

                    fun.set_normalized(true);
                }
                SliceType::Pow => {
                    let p = out.to_pow(wargs[0], wargs[1]);
                    p.set_normalized(true);
                }
                SliceType::One => {
                    out.set_from_view(&wargs[0]);
                }
                SliceType::Empty => {
                    let f = out.to_fun(Atom::ARG);
                    f.set_normalized(true);
                }
            },
            Self::FunctionName(n) => {
                out.to_var(*n);
            }
        }
    }
}

/// Settings related to pattern matching.
#[derive(Debug, Clone)]
pub struct MatchSettings {
    /// Specifies wildcards that try to match as little as possible.
    pub non_greedy_wildcards: Vec<Symbol>,
    /// Specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when entering a function, or going one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    pub level_range: (usize, Option<usize>),
    /// Determine whether a level reflects the expression tree depth or the function depth.
    pub level_is_tree_depth: bool,
    /// Allow wildcards on the right-hand side that do not appear in the pattern.
    pub allow_new_wildcards_on_rhs: bool,
    /// The maximum size of the cache for the right-hand side of a replacement.
    /// This can be used to prevent expensive recomputations.
    pub rhs_cache_size: usize,
}

static DEFAULT_MATCH_SETTINGS: MatchSettings = MatchSettings::new();

impl MatchSettings {
    pub const fn new() -> Self {
        Self {
            non_greedy_wildcards: Vec::new(),
            level_range: (0, None),
            level_is_tree_depth: false,
            allow_new_wildcards_on_rhs: false,
            rhs_cache_size: 0,
        }
    }

    /// Create default match settings, but enable caching of the rhs.
    pub fn cached() -> Self {
        Self {
            non_greedy_wildcards: Vec::new(),
            level_range: (0, None),
            level_is_tree_depth: false,
            allow_new_wildcards_on_rhs: false,
            rhs_cache_size: 100,
        }
    }
}

impl Default for MatchSettings {
    /// Create default match settings. Use [`MatchSettings::cached`] to enable caching.
    fn default() -> Self {
        MatchSettings::new()
    }
}

/// An insertion-ordered map of wildcard identifiers to subexpressions.
#[derive(Debug, Clone)]
pub struct MatchStack<'a> {
    stack: Vec<(Symbol, Match<'a>)>,
}

impl<'a> From<Vec<(Symbol, Match<'a>)>> for MatchStack<'a> {
    fn from(value: Vec<(Symbol, Match<'a>)>) -> Self {
        MatchStack { stack: value }
    }
}

impl<'a> MatchStack<'a> {
    pub fn new() -> Self {
        MatchStack { stack: Vec::new() }
    }

    /// Get a match.
    pub fn get(&self, key: Symbol) -> Option<&Match<'a>> {
        for (rk, rv) in self.stack.iter() {
            if rk == &key {
                return Some(rv);
            }
        }
        None
    }

    /// Get a reference to all matches.
    pub fn get_matches(&self) -> &[(Symbol, Match<'a>)] {
        &self.stack
    }

    /// Get the underlying matches `Vec`.
    pub fn into_matches(self) -> Vec<(Symbol, Match<'a>)> {
        self.stack
    }
}

/// An insertion-ordered map of wildcard identifiers to subexpressions.
/// It keeps track of all conditions on wildcards and will check them
/// before inserting.
pub struct WrappedMatchStack<'a, 'b> {
    stack: MatchStack<'a>,
    conditions: &'b Condition<PatternRestriction>,
    settings: &'b MatchSettings,
}

impl<'a> std::fmt::Display for MatchStack<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, (k, v)) in self.stack.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_fmt(format_args!("{}: {}", k, v))?;
        }

        f.write_str("]")
    }
}

impl<'a, 'b> IntoIterator for &'b MatchStack<'a> {
    type Item = &'b (Symbol, Match<'a>);
    type IntoIter = std::slice::Iter<'b, (Symbol, Match<'a>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.stack.iter()
    }
}

impl<'a, 'b> std::fmt::Display for WrappedMatchStack<'a, 'b> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stack.fmt(f)
    }
}

impl<'a, 'b> std::fmt::Debug for WrappedMatchStack<'a, 'b> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchStack")
            .field("stack", &self.stack)
            .finish()
    }
}

impl<'a, 'b> WrappedMatchStack<'a, 'b> {
    /// Create a new match stack wrapped with the conditions and settings.
    pub fn new(
        conditions: &'b Condition<PatternRestriction>,
        settings: &'b MatchSettings,
    ) -> WrappedMatchStack<'a, 'b> {
        WrappedMatchStack {
            stack: MatchStack::new(),
            conditions,
            settings,
        }
    }

    /// Add a new map of identifier `key` to value `value` to the stack and return the size the stack had before inserting this new entry.
    /// If the entry `(key, value)` already exists, it is not inserted again and therefore the returned size is the actual size.
    /// If the `key` exists in the map, but the `value` is different, the insertion is ignored and `None` is returned.
    pub fn insert(&mut self, key: Symbol, value: Match<'a>) -> Option<usize> {
        for (rk, rv) in self.stack.stack.iter() {
            if rk == &key {
                if rv == &value {
                    return Some(self.stack.stack.len());
                } else {
                    return None;
                }
            }
        }

        // test whether the current value passes all conditions
        // or returns an inconclusive result
        self.stack.stack.push((key, value));
        if self
            .conditions
            .check_possible(key, &self.stack.stack.last().unwrap().1, &self.stack)
            == ConditionResult::False
        {
            self.stack.stack.pop();
            None
        } else {
            Some(self.stack.stack.len() - 1)
        }
    }

    /// Get the match stack.
    pub fn get_match_stack(&self) -> &MatchStack<'a> {
        &self.stack
    }

    /// Get a reference to all matches.
    pub fn get_matches(&self) -> &[(Symbol, Match<'a>)] {
        &self.stack.stack
    }

    /// Return the length of the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.stack.len()
    }

    /// Truncate the stack to `len`.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.stack.stack.truncate(len)
    }

    /// Get the range of an identifier based on previous matches and based
    /// on conditions.
    pub fn get_range(&self, identifier: Symbol) -> (usize, Option<usize>) {
        if identifier.get_wildcard_level() == 0 {
            return (1, Some(1));
        }

        for (rk, rv) in self.stack.stack.iter() {
            if rk == &identifier {
                return match rv {
                    Match::Single(_) => (1, Some(1)),
                    Match::Multiple(slice_type, slice) => {
                        match slice_type {
                            SliceType::Empty => (0, Some(0)),
                            SliceType::Arg => (slice.len(), Some(slice.len())),
                            _ => {
                                // the length needs to include 1 since for example x*y is only
                                // length one in f(x*y)
                                // TODO: the length can only be 1 or slice.len() and no values in between
                                // so we could optimize this
                                (1, Some(slice.len()))
                            }
                        }
                    }
                    Match::FunctionName(_) => (1, Some(1)),
                };
            }
        }

        let (minimal, maximal) = self.conditions.get_range_hint(identifier);

        match identifier.get_wildcard_level() {
            1 => (minimal.unwrap_or(1), Some(maximal.unwrap_or(1))), // x_
            2 => (minimal.unwrap_or(1), maximal),                    // x__
            _ => (minimal.unwrap_or(0), maximal),                    // x___
        }
    }
}

struct WildcardIter {
    initialized: bool,
    name: Symbol,
    indices: Vec<u32>,
    size_target: u32,
    min_size: u32,
    max_size: u32,
    greedy: bool,
}

enum PatternIter<'a, 'b> {
    Literal(Option<usize>, AtomView<'b>),
    Wildcard(WildcardIter),
    Fn(
        Option<usize>,
        Symbol,
        &'b [Pattern],
        Box<Option<SubSliceIterator<'a, 'b>>>,
    ), // index first
    Sequence(
        Option<usize>,
        SliceType,
        &'b [Pattern],
        Box<Option<SubSliceIterator<'a, 'b>>>,
    ),
}

/// An iterator that tries to match an entire atom or
/// a subslice to a pattern.
pub struct AtomMatchIterator<'a, 'b> {
    try_match_atom: bool,
    sl_it: Option<SubSliceIterator<'a, 'b>>,
    pattern: &'b Pattern,
    target: AtomView<'a>,
    old_match_stack_len: Option<usize>,
}

impl<'a, 'b> AtomMatchIterator<'a, 'b> {
    pub fn new(pattern: &'b Pattern, target: AtomView<'a>) -> AtomMatchIterator<'a, 'b> {
        let try_match_atom = matches!(pattern, Pattern::Wildcard(_) | Pattern::Literal(_));

        AtomMatchIterator {
            try_match_atom,
            sl_it: None,
            pattern,
            target,
            old_match_stack_len: None,
        }
    }

    pub fn next(
        &mut self,
        match_stack: &mut WrappedMatchStack<'a, 'b>,
    ) -> Option<(usize, &[bool])> {
        if self.try_match_atom {
            self.try_match_atom = false;

            if let Pattern::Wildcard(w) = self.pattern {
                let range = match_stack.get_range(*w);
                if range.0 <= 1 && range.1.map(|w| w >= 1).unwrap_or(true) {
                    // TODO: any problems with matching Single vs a list?
                    if let Some(new_stack_len) = match_stack.insert(*w, Match::Single(self.target))
                    {
                        self.old_match_stack_len = Some(new_stack_len);
                        return Some((new_stack_len, &[]));
                    }
                }
            } else if let Pattern::Literal(w) = self.pattern {
                if w.as_view() == self.target {
                    return Some((match_stack.len(), &[]));
                }
            }
            // TODO: also do type matches, Fn Fn, etc?
        }

        if let Some(oml) = self.old_match_stack_len {
            match_stack.truncate(oml);
            self.old_match_stack_len = None;
        }

        if matches!(self.pattern, Pattern::Literal(_)) {
            // TODO: also catch Pattern:Add(_) and Pattern:Mul(_) without any sub-wildcards
            return None;
        }

        if self.sl_it.is_none() {
            self.sl_it = Some(SubSliceIterator::new(
                self.pattern,
                self.target,
                match_stack,
                true,
                matches!(self.pattern, Pattern::Wildcard(_) | Pattern::Literal(_)),
            ));
        }

        self.sl_it.as_mut().unwrap().next(match_stack)
    }
}

/// An iterator that matches a slice of patterns to a slice of atoms.
/// Use the [`SubSliceIterator::next`] to get the next match, if any.
///  
/// The flag `complete` determines whether the pattern should match the entire
/// slice `target`. The flag `ordered_gapless` determines whether the the patterns
/// may match the slice of atoms in any order. For a non-symmetric function, this
/// flag should likely be set.
pub struct SubSliceIterator<'a, 'b> {
    pattern: &'b [Pattern], // input term
    target: ListSlice<'a>,
    iterators: Vec<PatternIter<'a, 'b>>,
    used_flag: Vec<bool>,
    initialized: bool,
    matches: Vec<usize>,   // track match stack length
    complete: bool,        // match needs to consume entire target
    ordered_gapless: bool, // pattern should appear ordered and have no gaps
    cyclic: bool,          // pattern is cyclic
    do_not_match_to_single_atom_in_list: bool,
    do_not_match_entire_slice: bool,
}

impl<'a, 'b> SubSliceIterator<'a, 'b> {
    /// Create an iterator over a pattern applied to a target.
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        match_stack: &WrappedMatchStack<'a, 'b>,
        do_not_match_to_single_atom_in_list: bool,
        do_not_match_entire_slice: bool,
    ) -> SubSliceIterator<'a, 'b> {
        let mut shortcut_done = false;

        // a pattern and target can either be a single atom or a list
        // for (list, list)  create a subslice iterator on the lists that is not complete
        // for (single, list), upgrade single to a slice with one element

        let (pat_list, target_list) = match (pattern, target) {
            (Pattern::Mul(m1), AtomView::Mul(m2)) => (m1.as_slice(), m2.to_slice()),
            (Pattern::Add(a1), AtomView::Add(a2)) => (a1.as_slice(), a2.to_slice()),
            (Pattern::Mul(arg) | Pattern::Add(arg), _) => {
                shortcut_done = true; // cannot match
                (arg.as_slice(), ListSlice::from_one(target))
            }
            (Pattern::Wildcard(_), AtomView::Mul(m2)) => {
                (std::slice::from_ref(pattern), m2.to_slice())
            }
            (Pattern::Wildcard(_), AtomView::Add(a2)) => {
                (std::slice::from_ref(pattern), a2.to_slice())
            }
            (_, AtomView::Mul(m2)) => {
                if do_not_match_to_single_atom_in_list {
                    shortcut_done = true; // cannot match
                }
                (std::slice::from_ref(pattern), m2.to_slice())
            }
            (_, AtomView::Add(a2)) => {
                if do_not_match_to_single_atom_in_list {
                    shortcut_done = true; // cannot match
                }
                (std::slice::from_ref(pattern), a2.to_slice())
            }
            (_, _) => (std::slice::from_ref(pattern), ListSlice::from_one(target)),
        };

        // shortcut if the number of arguments is wrong
        let min_length: usize = pat_list
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        let mut target_len = target_list.len();
        if do_not_match_entire_slice {
            target_len -= 1;
        }

        if min_length > target_len {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern: pat_list,
            iterators: if shortcut_done {
                Vec::new()
            } else {
                Vec::with_capacity(pat_list.len())
            },
            matches: if shortcut_done {
                Vec::new()
            } else {
                Vec::with_capacity(pat_list.len())
            },
            used_flag: if shortcut_done {
                vec![]
            } else {
                vec![false; target_list.len()]
            },
            target: target_list,

            initialized: shortcut_done,
            complete: false,
            ordered_gapless: false,
            cyclic: false,
            do_not_match_to_single_atom_in_list,
            do_not_match_entire_slice,
        }
    }

    /// Create a new sub-slice iterator.
    pub fn from_list(
        pattern: &'b [Pattern],
        target: ListSlice<'a>,
        match_stack: &WrappedMatchStack<'a, 'b>,
        complete: bool,
        ordered: bool,
        cyclic: bool,
    ) -> SubSliceIterator<'a, 'b> {
        let mut shortcut_done = false;

        // shortcut if the number of arguments is wrong
        let min_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        if min_length > target.len() {
            shortcut_done = true;
        };

        let max_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).1.unwrap_or(target.len()),
                _ => 1,
            })
            .sum();

        if complete && max_length < target.len() {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern,
            iterators: Vec::with_capacity(pattern.len()),
            matches: Vec::with_capacity(pattern.len()),
            used_flag: vec![false; target.len()],
            target,
            initialized: shortcut_done,
            complete,
            ordered_gapless: ordered,
            cyclic,
            do_not_match_to_single_atom_in_list: false,
            do_not_match_entire_slice: false,
        }
    }

    /// Get the next matches, where the map of matches is written into `match_stack`.
    /// The function returns the length of the match stack before the last subiterator
    /// matched. This value can be ignored by the end-user. If `None` is returned,
    /// all potential matches will have been generated and the iterator will generate
    /// `None` if called again.
    pub fn next(
        &mut self,
        match_stack: &mut WrappedMatchStack<'a, 'b>,
    ) -> Option<(usize, &[bool])> {
        let mut forward_pass = !self.initialized;
        self.initialized = true;

        'next_match: loop {
            if !forward_pass && self.iterators.is_empty() {
                return None; // done as all options have been exhausted
            }

            if forward_pass && self.iterators.len() == self.pattern.len() {
                // check the proposed solution for extra conditions
                if self.complete && self.used_flag.iter().any(|x| !*x)
                    || self.do_not_match_to_single_atom_in_list // TODO: a function may have more used_flags? does that clash?
                        && self.used_flag.len() > 1
                        && self.used_flag.iter().map(|x| *x as usize).sum::<usize>() == 1
                {
                    // not done as the entire target is not used
                    forward_pass = false;
                } else {
                    // yield the current match
                    return Some((*self.matches.last().unwrap(), &self.used_flag));
                }
            }

            if forward_pass {
                // add new iterator
                let it = match &self.pattern[self.iterators.len()] {
                    Pattern::Wildcard(name) => {
                        let mut size_left = self.used_flag.iter().filter(|x| !*x).count();
                        let range = match_stack.get_range(*name);

                        if self.do_not_match_entire_slice {
                            size_left -= 1;

                            if size_left < range.0 {
                                forward_pass = false;
                                continue 'next_match;
                            }
                        }

                        let mut range = (
                            range.0,
                            range.1.map(|m| m.min(size_left)).unwrap_or(size_left),
                        );

                        // bound the wildcard length based on the bounds of upcoming patterns
                        if self.complete {
                            let mut new_min = size_left;
                            let mut new_max = size_left;
                            for p in &self.pattern[self.iterators.len() + 1..] {
                                let p_range = if let Pattern::Wildcard(name) = p {
                                    match_stack.get_range(*name)
                                } else {
                                    (1, Some(1))
                                };

                                if new_min > 0 {
                                    if let Some(m) = p_range.1 {
                                        new_min -= m.min(new_min);
                                    } else {
                                        new_min = 0;
                                    }
                                }

                                if new_max < p_range.0 {
                                    forward_pass = false;
                                    continue 'next_match;
                                }

                                new_max -= p_range.0;
                            }

                            range.0 = range.0.max(new_min);
                            range.1 = range.1.min(new_max);

                            if range.0 > range.1 {
                                forward_pass = false;
                                continue 'next_match;
                            }
                        }

                        let greedy = !match_stack.settings.non_greedy_wildcards.contains(name);

                        PatternIter::Wildcard(WildcardIter {
                            initialized: false,
                            name: *name,
                            indices: Vec::new(),
                            size_target: if greedy {
                                range.1 as u32
                            } else {
                                range.0 as u32
                            },
                            min_size: range.0 as u32,
                            max_size: range.1 as u32,
                            greedy,
                        })
                    }
                    Pattern::Fn(name, args) => PatternIter::Fn(None, *name, args, Box::new(None)),
                    Pattern::Pow(base_exp) => PatternIter::Sequence(
                        None,
                        SliceType::Pow,
                        base_exp.as_slice(),
                        Box::new(None),
                    ),
                    Pattern::Mul(pat) => {
                        PatternIter::Sequence(None, SliceType::Mul, pat, Box::new(None))
                    }
                    Pattern::Add(pat) => {
                        PatternIter::Sequence(None, SliceType::Add, pat, Box::new(None))
                    }
                    Pattern::Literal(atom) => PatternIter::Literal(None, atom.as_view()),
                    Pattern::Transformer(_) => panic!("Transformer is not allowed on lhs"),
                };

                self.iterators.push(it);
            } else {
                // update an existing iterator, so pop the latest matches (this implies every iter pushes to the match)
                match_stack.truncate(self.matches.pop().unwrap());
            }

            // assume we are in forward pass mode
            // if the iterator does not match this variable is set to false
            forward_pass = true;

            match self.iterators.last_mut().unwrap() {
                PatternIter::Wildcard(w) => {
                    let mut wildcard_forward_pass = !w.initialized;
                    w.initialized = true;

                    'next_wildcard_match: loop {
                        // a wildcard collects indices in increasing order
                        // find the starting point where the last index can be moved to
                        let start_index =
                            w.indices
                                .last()
                                .map(|x| *x as usize + 1)
                                .unwrap_or_else(|| {
                                    if self.cyclic {
                                        let mut pos =
                                            self.used_flag.iter().position(|x| *x).unwrap_or(0);
                                        while self.used_flag[pos] {
                                            pos = (pos + 1) % self.used_flag.len();
                                        }
                                        pos
                                    } else {
                                        0
                                    }
                                });

                        if !wildcard_forward_pass {
                            let last_iterator_empty = w.indices.is_empty();
                            if let Some(last_index) = w.indices.pop() {
                                self.used_flag[last_index as usize] = false;
                            }

                            if last_iterator_empty {
                                // the wildcard iterator is exhausted for this target size
                                if w.greedy {
                                    if w.size_target > w.min_size {
                                        w.size_target -= 1;
                                    } else {
                                        break;
                                    }
                                } else if w.size_target < w.max_size {
                                    w.size_target += 1;
                                } else {
                                    break;
                                }
                            } else if self.ordered_gapless {
                                // early terminate if a gap would be made
                                // do not early terminate if the first placement
                                // in a cyclic structure has not been fixed
                                if !self.cyclic || self.used_flag.iter().any(|x| *x) {
                                    // drain the entire constructed range and start from scratch
                                    continue 'next_wildcard_match;
                                }
                            }
                        }

                        // check for an empty slice match
                        if w.size_target == 0 && w.indices.is_empty() {
                            if let Some(new_stack_len) = match_stack
                                .insert(w.name, Match::Multiple(SliceType::Empty, Vec::new()))
                            {
                                self.matches.push(new_stack_len);
                                continue 'next_match;
                            } else {
                                wildcard_forward_pass = false;
                                continue 'next_wildcard_match;
                            }
                        }

                        let mut tried_first_option = false;
                        let mut k = start_index;
                        loop {
                            if k == self.target.len() {
                                if self.cyclic && w.indices.len() > 0 {
                                    // allow the wildcard to wrap around
                                    k = 0;
                                } else {
                                    break;
                                }
                            }

                            if self.ordered_gapless && tried_first_option {
                                break;
                            }

                            if self.used_flag[k] {
                                if self.cyclic {
                                    break;
                                }

                                k += 1;
                                continue;
                            }

                            self.used_flag[k] = true;
                            w.indices.push(k as u32);

                            if w.indices.len() == w.size_target as usize {
                                tried_first_option = true;

                                // simplify case of 1 argument, this is important for matching to work, since mul(x) = add(x) = arg(x) for any x
                                let matched = if w.indices.len() == 1 {
                                    match self.target.get(w.indices[0] as usize) {
                                        AtomView::Mul(m) => Match::Multiple(SliceType::Mul, {
                                            let mut v = Vec::new();
                                            for x in m {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        AtomView::Add(a) => Match::Multiple(SliceType::Add, {
                                            let mut v = Vec::new();
                                            for x in a {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        x => Match::Single(x),
                                    }
                                } else {
                                    let mut atoms = Vec::with_capacity(w.indices.len());
                                    for i in &w.indices {
                                        atoms.push(self.target.get(*i as usize));
                                    }

                                    Match::Multiple(self.target.get_type(), atoms)
                                };

                                // add the match to the stack if it is compatible
                                if let Some(new_stack_len) = match_stack.insert(w.name, matched) {
                                    self.matches.push(new_stack_len);
                                    continue 'next_match;
                                } else {
                                    // no match
                                    w.indices.pop();
                                    self.used_flag[k] = false;
                                }
                            }

                            k += 1;
                        }

                        // no match found, try to increase the index of the current last element
                        wildcard_forward_pass = false;
                    }
                }
                PatternIter::Fn(index, name, args, s) => {
                    let mut tried_first_option = false;

                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some((x, _)) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                if name.get_wildcard_level() > 0 {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }

                                self.used_flag[*jj] = false;
                                tried_first_option = true;
                                **s = None;
                                *jj + 1
                            }
                        }
                        None => {
                            if self.cyclic && !self.used_flag.iter().all(|u| *u) {
                                // start after the last used index
                                let mut pos = self.used_flag.iter().position(|x| *x).unwrap_or(0);
                                while self.used_flag[pos] {
                                    pos = (pos + 1) % self.used_flag.len();
                                }
                                pos
                            } else {
                                0
                            }
                        }
                    };

                    // find a new match and create a new iterator
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            if self.cyclic {
                                break;
                            }

                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            // cyclic sequences can start at any position
                            if !self.cyclic || self.used_flag.iter().any(|x| *x) {
                                break;
                            }
                        }

                        tried_first_option = true;

                        if let AtomView::Fun(f) = self.target.get(ii) {
                            let target_name = f.get_symbol();
                            let name_match = if name.get_wildcard_level() > 0 {
                                if let Some(new_stack_len) =
                                    match_stack.insert(*name, Match::FunctionName(target_name))
                                {
                                    self.matches.push(new_stack_len);
                                    true
                                } else {
                                    ii += 1;
                                    continue;
                                }
                            } else {
                                f.get_symbol() == *name
                            };

                            if name_match {
                                let mut it = SubSliceIterator::from_list(
                                    args,
                                    f.to_slice(),
                                    match_stack,
                                    true,
                                    !name.is_antisymmetric() && !name.is_symmetric(),
                                    name.is_cyclesymmetric(),
                                );

                                if let Some((x, _)) = it.next(match_stack) {
                                    *index = Some(ii);
                                    **s = Some(it);
                                    self.matches.push(x);
                                    self.used_flag[ii] = true;

                                    continue 'next_match;
                                }

                                if name.get_wildcard_level() > 0 {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }
                            }
                        }

                        ii += 1;
                    }
                }
                PatternIter::Literal(index, atom) => {
                    let mut tried_first_option = false;
                    let mut ii = match index {
                        Some(jj) => {
                            self.used_flag[*jj] = false;
                            tried_first_option = true;
                            *jj + 1
                        }
                        None => {
                            if self.cyclic && !self.used_flag.iter().all(|u| *u) {
                                // start after the last used index
                                let mut pos = self.used_flag.iter().position(|x| *x).unwrap_or(0);
                                while self.used_flag[pos] {
                                    pos = (pos + 1) % self.used_flag.len();
                                }
                                pos
                            } else {
                                0
                            }
                        }
                    };

                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            if self.cyclic {
                                break;
                            }

                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            // cyclic sequences can start at any position
                            if !self.cyclic || self.used_flag.iter().any(|x| *x) {
                                break;
                            }
                        }

                        tried_first_option = true;

                        if self.target.get(ii) == *atom {
                            *index = Some(ii);
                            self.matches.push(match_stack.len());
                            self.used_flag[ii] = true;
                            continue 'next_match;
                        }
                        ii += 1;
                    }
                }
                PatternIter::Sequence(index, slice_type, pattern, s) => {
                    let mut tried_first_option = false;

                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some((x, _)) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                self.used_flag[*jj] = false;
                                tried_first_option = true;
                                *jj + 1
                            }
                        }
                        None => {
                            if self.cyclic && !self.used_flag.iter().all(|u| *u) {
                                // start after the last used index
                                let mut pos = self.used_flag.iter().position(|x| *x).unwrap_or(0);
                                while self.used_flag[pos] {
                                    pos = (pos + 1) % self.used_flag.len();
                                }
                                pos
                            } else {
                                0
                            }
                        }
                    };

                    // find a new match and create a new iterator
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            if self.cyclic {
                                break;
                            }

                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            // cyclic sequences can start at any position
                            if !self.cyclic || self.used_flag.iter().any(|x| *x) {
                                break;
                            }
                        }

                        tried_first_option = true;

                        let slice = match (self.target.get(ii), &slice_type) {
                            (AtomView::Mul(m), SliceType::Mul) => m.to_slice(),
                            (AtomView::Add(a), SliceType::Add) => a.to_slice(),
                            (AtomView::Pow(a), SliceType::Pow) => a.to_slice(),
                            _ => {
                                ii += 1;
                                continue;
                            }
                        };

                        let ordered = match slice_type {
                            SliceType::Add | SliceType::Mul => false,
                            SliceType::Pow => true, // make sure pattern (base,exp) is not exchanged
                            _ => unreachable!(),
                        };

                        let mut it = SubSliceIterator::from_list(
                            pattern,
                            slice,
                            match_stack,
                            true,
                            ordered,
                            false,
                        );

                        if let Some((x, _)) = it.next(match_stack) {
                            *index = Some(ii);
                            **s = Some(it);
                            self.matches.push(x);
                            self.used_flag[ii] = true;

                            continue 'next_match;
                        }

                        ii += 1;
                    }
                }
            }

            // no match, so fall back one level
            forward_pass = false;
            self.iterators.pop();
        }
    }
}

/// Iterator over the atoms of an expression tree.
pub struct AtomTreeIterator<'a> {
    stack: Vec<(Option<usize>, usize, AtomView<'a>)>,
    settings: MatchSettings,
}

impl<'a> AtomTreeIterator<'a> {
    pub fn new(target: AtomView<'a>, settings: MatchSettings) -> AtomTreeIterator<'a> {
        AtomTreeIterator {
            stack: vec![(None, 0, target)],
            settings,
        }
    }
}

impl<'a> Iterator for AtomTreeIterator<'a> {
    type Item = (Vec<usize>, AtomView<'a>);

    /// Return the next position and atom in the tree.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((ind, level, atom)) = self.stack.pop() {
            if let Some(max_level) = self.settings.level_range.1 {
                if level > max_level {
                    continue;
                }
            }

            if let Some(ind) = ind {
                let slice = match atom {
                    AtomView::Fun(f) => f.to_slice(),
                    AtomView::Pow(p) => p.to_slice(),
                    AtomView::Mul(m) => m.to_slice(),
                    AtomView::Add(a) => a.to_slice(),
                    _ => {
                        continue; // not iterable
                    }
                };

                if ind < slice.len() {
                    let new_atom = slice.get(ind);

                    self.stack.push((Some(ind + 1), level, atom));
                    self.stack.push((None, level, new_atom)); // push the new element on the stack
                }
            } else {
                // return full match and set the position to the first sub element
                let location = self
                    .stack
                    .iter()
                    .map(|(ind, _, _)| ind.unwrap() - 1)
                    .collect::<Vec<_>>();

                let new_level = if let AtomView::Fun(_) = atom {
                    level + 1
                } else if self.settings.level_is_tree_depth {
                    level + 1
                } else {
                    level
                };

                self.stack.push((Some(0), new_level, atom));

                if level >= self.settings.level_range.0 {
                    return Some((location, atom));
                }
            }
        }

        None
    }
}

/// Match a pattern to any subexpression of a target expression.
pub struct PatternAtomTreeIterator<'a, 'b> {
    pattern: &'b Pattern,
    atom_tree_iterator: AtomTreeIterator<'a>,
    current_target: Option<AtomView<'a>>,
    pattern_iter: Option<AtomMatchIterator<'a, 'b>>,
    match_stack: WrappedMatchStack<'a, 'b>,
    tree_pos: Vec<usize>,
    first_match: bool,
}

/// A part of an expression with its position that yields a match.
pub struct PatternMatch<'a, 'b> {
    /// The position (branch) of the match in the tree.
    pub position: &'b [usize],
    /// Flags which subexpressions are matched in case of matching a range.
    pub used_flags: Vec<bool>,
    /// The matched target.
    pub target: AtomView<'a>,
    /// The list of identifications of matched wildcards.
    pub match_stack: &'b MatchStack<'a>,
}

impl<'a: 'b, 'b> PatternAtomTreeIterator<'a, 'b> {
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        conditions: Option<&'b Condition<PatternRestriction>>,
        settings: Option<&'b MatchSettings>,
    ) -> PatternAtomTreeIterator<'a, 'b> {
        PatternAtomTreeIterator {
            pattern,
            atom_tree_iterator: AtomTreeIterator::new(
                target,
                settings.unwrap_or(&DEFAULT_MATCH_SETTINGS).clone(),
            ),
            current_target: None,
            pattern_iter: None,
            match_stack: WrappedMatchStack::new(
                conditions.unwrap_or(&DEFAULT_PATTERN_CONDITION),
                settings.unwrap_or(&DEFAULT_MATCH_SETTINGS),
            ),
            tree_pos: Vec::new(),
            first_match: false,
        }
    }

    /// Generate the next match if it exists, with detailed information about the
    /// matched position. Use the iterator [Self::next] to a map of wildcard matches.
    pub fn next_detailed(&mut self) -> Option<PatternMatch<'a, '_>> {
        loop {
            if let Some(ct) = self.current_target {
                if let Some(it) = self.pattern_iter.as_mut() {
                    if let Some((_, used_flags)) = it.next(&mut self.match_stack) {
                        let a = used_flags.to_vec();

                        self.first_match = true;
                        return Some(PatternMatch {
                            position: &self.tree_pos,
                            used_flags: a,
                            target: ct,
                            match_stack: &self.match_stack.stack,
                        });
                    } else {
                        // no match: bail
                        self.current_target = None;
                        self.pattern_iter = None;
                        continue;
                    }
                } else {
                    // prevent duplicate matches by not matching to single atoms in a list as they will
                    // be tested at a later stage in the atom tree iterator, as we want to store the position
                    self.pattern_iter = Some(AtomMatchIterator::new(self.pattern, ct));
                }
            } else {
                let res = self.atom_tree_iterator.next();

                if let Some((tree_pos, cur_target)) = res {
                    self.tree_pos = tree_pos;
                    self.current_target = Some(cur_target);
                } else {
                    return None;
                }
            }
        }
    }
}

impl<'a: 'b, 'b> Iterator for PatternAtomTreeIterator<'a, 'b> {
    type Item = HashMap<Symbol, Atom>;

    /// Get the match map. Use [PatternAtomTreeIterator::next_detailed] to get more information.
    fn next(&mut self) -> Option<HashMap<Symbol, Atom>> {
        if let Some(_) = self.next_detailed() {
            Some(
                self.match_stack
                    .get_matches()
                    .iter()
                    .map(|(key, m)| (*key, m.to_atom()))
                    .collect(),
            )
        } else {
            None
        }
    }
}

/// Replace a pattern in the target once. Every  call to `next`,
/// will return a new match and replacement until the options are exhausted.
pub struct ReplaceIterator<'a, 'b> {
    rhs: BorrowedPatternOrMap<'b>,
    pattern_tree_iterator: PatternAtomTreeIterator<'a, 'b>,
    target: AtomView<'a>,
}

impl<'a: 'b, 'b> ReplaceIterator<'a, 'b> {
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        rhs: BorrowedPatternOrMap<'b>,
        conditions: Option<&'a Condition<PatternRestriction>>,
        settings: Option<&'a MatchSettings>,
    ) -> ReplaceIterator<'a, 'b> {
        ReplaceIterator {
            pattern_tree_iterator: PatternAtomTreeIterator::new(
                pattern, target, conditions, settings,
            ),
            rhs,
            target,
        }
    }

    fn copy_and_replace(
        out: &mut Atom,
        position: &[usize],
        used_flags: &[bool],
        target: AtomView<'a>,
        rhs: AtomView<'_>,
        workspace: &Workspace,
    ) {
        if let Some((first, rest)) = position.split_first() {
            match target {
                AtomView::Fun(f) => {
                    let slice = f.to_slice();

                    let out = out.to_fun(f.get_symbol());

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);
                            out.add_arg(oa.as_view());
                        } else {
                            out.add_arg(arg);
                        }
                    }
                }
                AtomView::Pow(p) => {
                    let slice = p.to_slice();

                    if *first == 0 {
                        let mut oa = workspace.new_atom();
                        Self::copy_and_replace(
                            &mut oa,
                            rest,
                            used_flags,
                            slice.get(0),
                            rhs,
                            workspace,
                        );
                        out.to_pow(oa.as_view(), slice.get(1));
                    } else {
                        let mut oa = workspace.new_atom();
                        Self::copy_and_replace(
                            &mut oa,
                            rest,
                            used_flags,
                            slice.get(1),
                            rhs,
                            workspace,
                        );
                        out.to_pow(slice.get(0), oa.as_view());
                    }
                }
                AtomView::Mul(m) => {
                    let slice = m.to_slice();

                    let out = out.to_mul();

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);

                            // TODO: do type check or just extend? could be that we get x*y*z -> x*(w*u)*z
                            out.extend(oa.as_view());
                        } else {
                            out.extend(arg);
                        }
                    }
                }
                AtomView::Add(a) => {
                    let slice = a.to_slice();

                    let out = out.to_add();

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);

                            out.extend(oa.as_view());
                        } else {
                            out.extend(arg);
                        }
                    }
                }
                _ => unreachable!("Atom does not have children"),
            }
        } else {
            match target {
                AtomView::Mul(m) => {
                    let out = out.to_mul();

                    for (child, used) in m.iter().zip(used_flags) {
                        if !used {
                            out.extend(child);
                        }
                    }

                    out.extend(rhs);
                }
                AtomView::Add(a) => {
                    let out = out.to_add();

                    for (child, used) in a.iter().zip(used_flags) {
                        if !used {
                            out.extend(child);
                        }
                    }

                    out.extend(rhs);
                }
                _ => {
                    out.set_from_view(&rhs);
                }
            }
        }
    }

    /// Return the next replacement.
    pub fn next_into(&mut self, out: &mut Atom) -> Option<()> {
        let allow = self
            .pattern_tree_iterator
            .atom_tree_iterator
            .settings
            .allow_new_wildcards_on_rhs;
        if let Some(pattern_match) = self.pattern_tree_iterator.next_detailed() {
            Workspace::get_local().with(|ws| {
                let mut new_rhs = ws.new_atom();

                match self.rhs {
                    BorrowedPatternOrMap::Pattern(p) => {
                        p.replace_wildcards_with_matches_impl(
                            ws,
                            &mut new_rhs,
                            pattern_match.match_stack,
                            allow,
                            None,
                        )
                        .unwrap(); // TODO: escalate?
                    }
                    BorrowedPatternOrMap::Map(f) => {
                        let mut new_atom = f(&pattern_match.match_stack);
                        std::mem::swap(&mut new_atom, &mut new_rhs);
                    }
                }

                let mut h = ws.new_atom();
                ReplaceIterator::copy_and_replace(
                    &mut h,
                    pattern_match.position,
                    &pattern_match.used_flags,
                    self.target,
                    new_rhs.as_view(),
                    ws,
                );
                h.as_view().normalize(ws, out);
            });

            Some(())
        } else {
            None
        }
    }
}

impl<'a: 'b, 'b> Iterator for ReplaceIterator<'a, 'b> {
    type Item = Atom;

    fn next(&mut self) -> Option<Self::Item> {
        let mut out = Atom::new();
        self.next_into(&mut out).map(|_| out)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore, Symbol},
        id::{
            ConditionResult, Match, MatchSettings, PatternOrMap, PatternRestriction, Replacement,
            WildcardRestriction,
        },
    };

    use super::Pattern;

    #[test]
    fn replace_wildcards_with_map() {
        let a = Atom::parse("f1(v1__, 5) + v1*v2_ + v3^v3_")
            .unwrap()
            .to_pattern();
        let r = a.replace_wildcards(
            &[
                (Symbol::new("v1__"), Atom::parse("arg(v4, v5)").unwrap()),
                (Symbol::new("v2_"), Atom::new_num(4)),
                (Symbol::new("v3_"), Atom::new_num(5)),
            ]
            .into_iter()
            .collect(),
        );

        let res = Atom::parse("f1(v4, v5, 5) + v1*4 + v3^5").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn replace_wildcards() {
        let a = Atom::parse("f1(v1__, 5) + v1*v2_ + v3^v3_")
            .unwrap()
            .to_pattern();

        let r11 = Atom::new_var(Symbol::new("v4"));
        let r12 = Atom::new_var(Symbol::new("v5"));
        let r2 = Atom::new_num(4);
        let r3 = Atom::new_num(5);

        let r = a.replace_wildcards_with_matches(
            &vec![
                (
                    Symbol::new("v1__"),
                    Match::Multiple(
                        crate::atom::SliceType::Arg,
                        vec![r11.as_view(), r12.as_view()],
                    ),
                ),
                (Symbol::new("v2_"), Match::Single(r2.as_view())),
                (Symbol::new("v3_"), Match::Single(r3.as_view())),
            ]
            .into(),
        );

        let res = Atom::parse("f1(v4, v5, 5) + v1*4 + v3^5").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn replace_map() {
        let a = Atom::parse("v1 + f1(1,2, f1((1+v1)^2), (v1+v2)^2)").unwrap();

        let r = a.replace_map(&|arg, context, out| {
            if context.function_level > 0 {
                arg.expand_into(None, out)
            } else {
                false
            }
        });

        let res = Atom::parse("v1+f1(1,2,f1(2*v1+v1^2+1),v1^2+v2^2+2*v1*v2)").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn overlap() {
        let a = Atom::parse("(v1*(v2+v2^2+1)+v2^2 + v2)").unwrap();
        let p = Pattern::parse("v2+v2^v1_").unwrap();
        let rhs = Pattern::parse("v2*(1+v2^(v1_-1))").unwrap();

        let r = a.replace_all(&p, &rhs, None, None);
        let res = Atom::parse("v1*(v2+v2^2+1)+v2*(v2+1)").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn level_restriction() {
        let a = Atom::parse("v1*f1(v1,f1(v1))").unwrap();
        let p = Pattern::parse("v1").unwrap();
        let rhs = Pattern::parse("1").unwrap();

        let r = a.replace_all(
            &p,
            &rhs,
            None,
            Some(&MatchSettings {
                level_range: (1, Some(1)),
                ..Default::default()
            }),
        );
        let res = Atom::parse("v1*f1(1,f1(v1))").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn multiple() {
        let a = Atom::parse("f(v1,v2)").unwrap();

        let r = a.replace_all_multiple(&[
            Replacement::new(Pattern::parse("v1").unwrap(), Pattern::parse("v2").unwrap()),
            Replacement::new(Pattern::parse("v2").unwrap(), Pattern::parse("v1").unwrap()),
        ]);

        let res = Atom::parse("f(v2,v1)").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn map_rhs() {
        let v1 = Symbol::new("v1_");
        let v2 = Symbol::new("v2_");
        let v4 = Symbol::new("v4_");
        let v5 = Symbol::new("v5_");
        let a = Atom::parse("v1(2,1)*v2(3,1)").unwrap();
        let p = Pattern::parse("v1_(v2_,v3_)*v4_(v5_,v3_)").unwrap();
        let rhs = PatternOrMap::Map(Box::new(move |m| {
            Atom::parse(&format!(
                "{}(mu{})*{}(mu{})",
                m.get(v1).unwrap(),
                m.get(v2).unwrap(),
                m.get(v4).unwrap(),
                m.get(v5).unwrap()
            ))
            .unwrap()
        }));

        let r = a.replace_all(&p, &rhs, None, None);
        let res = Atom::parse("v1(mu2)*v2(mu3)").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn repeat_replace() {
        let mut a = Atom::parse("f(10)").unwrap();
        let p1 = Pattern::parse("f(v1_)").unwrap();
        let rhs1 = Pattern::parse("f(v1_ - 1)").unwrap();

        let rest = (
            Symbol::new("v1_"),
            WildcardRestriction::Filter(Box::new(|x| {
                let n: Result<i64, _> = x.to_atom().try_into();
                if let Ok(y) = n {
                    y > 0i64
                } else {
                    false
                }
            })),
        )
            .into();

        a.repeat_map(|e| e.replace_all(&p1, &rhs1, Some(&rest), None));

        let res = Atom::parse("f(0)").unwrap();
        assert_eq!(a, res);
    }

    #[test]
    fn match_stack_filter() {
        let a = Atom::parse("f(1,2,3,4)").unwrap();
        let p1 = Pattern::parse("f(v1_,v2_,v3_,v4_)").unwrap();
        let rhs1 = Pattern::parse("f(v4_,v3_,v2_,v1_)").unwrap();

        let rest = PatternRestriction::MatchStack(Box::new(|m| {
            for x in m.stack.windows(2) {
                if x[0].1.to_atom() >= x[1].1.to_atom() {
                    return false.into();
                }
            }

            if m.stack.len() == 4 {
                true.into()
            } else {
                ConditionResult::Inconclusive
            }
        }))
        .into();

        let r = a.replace_all(&p1, &rhs1, Some(&rest), None);
        let res = Atom::parse("f(4,3,2,1)").unwrap();
        assert_eq!(r, res);

        let b = Atom::parse("f(1,2,4,3)").unwrap();
        let r = b.replace_all(&p1, &rhs1, Some(&rest), None);
        assert_eq!(r, b);
    }

    #[test]
    fn match_cache() {
        let expr = Atom::parse("f1(1)*f1(2)+f1(1)*f1(2)*f2").unwrap();
        let pat = Pattern::parse("v1_(id1_)*v2_(id2_)").unwrap();

        let expr = expr.replace_all(&pat, &Pattern::parse("f1(id1_)").unwrap(), None, None);

        let res = Atom::parse("f1(1)+f2*f1(1)").unwrap();
        assert_eq!(expr, res);
    }

    #[test]
    fn match_cyclic() {
        let rhs = Pattern::parse("1").unwrap();

        // literal wrap
        let expr = Atom::parse("fc1(1,2,3)").unwrap();
        let p = Pattern::parse("fc1(v1__,v1_,1)").unwrap();
        let expr = expr.replace_all(&p, &rhs, None, None);
        assert_eq!(expr, Atom::new_num(1));

        // multiple wildcard wrap
        let expr = Atom::parse("fc1(1,2,3)").unwrap();
        let p = Pattern::parse("fc1(v1__,2)").unwrap();
        let expr = expr.replace_all(&p, &rhs, None, None);
        assert_eq!(expr, Atom::new_num(1));

        // wildcard wrap
        let expr = Atom::parse("fc1(1,2,3)").unwrap();
        let p = Pattern::parse("fc1(v1__,v1_,2)").unwrap();
        let expr = expr.replace_all(&p, &rhs, None, None);
        assert_eq!(expr, Atom::new_num(1));

        let expr = Atom::parse("fc1(v1,4,3,5,4)").unwrap();
        let p = Pattern::parse("fc1(v1__,v1_,v2_,v1_)").unwrap();
        let expr = expr.replace_all(&p, &rhs, None, None);
        assert_eq!(expr, Atom::new_num(1));

        // function shift
        let expr = Atom::parse("fc1(f1(1),f1(2),f1(3))").unwrap();
        let p = Pattern::parse("fc1(f1(v1_),f1(2),f1(3))").unwrap();
        let expr = expr.replace_all(&p, &rhs, None, None);
        assert_eq!(expr, Atom::new_num(1));
    }

    #[test]
    fn is_polynomial() {
        let e = Atom::parse("v1^2 + (1+v5)^3 / v1 + (1+v3)*(1+v4)^v7 + v1^2 + (v1+v2)^3").unwrap();
        let vars = e.as_view().is_polynomial(true, true).unwrap();
        assert_eq!(vars.len(), 5);

        let e = Atom::parse("(1+v5)^(3/2) / v6 + (1+v3)*(1+v4)^v7 + (v1+v2)^3").unwrap();
        let vars = e.as_view().is_polynomial(false, false).unwrap();
        assert_eq!(vars.len(), 5);
    }
}
