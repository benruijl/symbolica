//! Manage global state and thread-local workspaces.

use byteorder::{ReadBytesExt, WriteBytesExt};
use std::borrow::Cow;
use std::hash::Hash;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::thread::LocalKey;
use std::{
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
};

use ahash::{HashMap, HashMapExt};
use append_only_vec::AppendOnlyVec;
use byteorder::LittleEndian;
use once_cell::sync::Lazy;
use smartstring::alias::String;

use crate::atom::{FunctionAttribute, NamespacedSymbol, NormalizationFunction};
use crate::domains::finite_field::Zp64;
use crate::poly::Variable;
use crate::wrap_symbol;
use crate::{
    atom::{Atom, Symbol},
    coefficient::Coefficient,
    domains::finite_field::FiniteFieldCore,
    LicenseManager,
};

pub(crate) const SYMBOLICA_MAGIC: u32 = 0x37871367;
pub(crate) const EXPORT_FORMAT_VERSION: u16 = 1;

/// An id for a given finite field in a registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteFieldIndex(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VariableListIndex(pub(crate) usize);

/// A mapping from one state to the other. Used during importing
/// for merging a state on file with the current state.
pub struct StateMap {
    pub(crate) symbols: HashMap<u32, Symbol>,
    pub(crate) finite_fields: HashMap<FiniteFieldIndex, FiniteFieldIndex>,
    pub(crate) variables_lists: HashMap<u64, Arc<Vec<Variable>>>,
}

impl StateMap {
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty() && self.finite_fields.is_empty() && self.variables_lists.is_empty()
    }
}

pub(crate) struct SymbolData {
    pub(crate) name: String,
    pub(crate) namespace: Cow<'static, str>,
    pub(crate) file: Cow<'static, str>,
    pub(crate) line: usize,
    pub(crate) function: Option<NormalizationFunction>,
}

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
static ID_TO_STR: AppendOnlyVec<(Symbol, SymbolData)> = AppendOnlyVec::new();
static FINITE_FIELDS: AppendOnlyVec<Zp64> = AppendOnlyVec::new();
static VARIABLE_LISTS: AppendOnlyVec<Arc<Vec<Variable>>> = AppendOnlyVec::new();
static SYMBOL_OFFSET: AtomicUsize = AtomicUsize::new(0);

thread_local!(
    /// A thread-local workspace, that stores recyclable atoms.
    static WORKSPACE: Workspace = const { Workspace::new() }
);

/// A global state, that stores mappings from variable and function names to ids.
pub struct State {
    str_to_id: HashMap<String, Symbol>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub(crate) const ARG: Symbol = Symbol::raw_fn(0, 0, false, false, false, false);
    pub(crate) const COEFF: Symbol = Symbol::raw_fn(1, 0, false, false, false, false);
    pub(crate) const EXP: Symbol = Symbol::raw_fn(2, 0, false, false, false, false);
    pub(crate) const LOG: Symbol = Symbol::raw_fn(3, 0, false, false, false, false);
    pub(crate) const SIN: Symbol = Symbol::raw_fn(4, 0, false, false, false, false);
    pub(crate) const COS: Symbol = Symbol::raw_fn(5, 0, false, false, false, false);
    pub(crate) const SQRT: Symbol = Symbol::raw_fn(6, 0, false, false, false, false);
    pub(crate) const DERIVATIVE: Symbol = Symbol::raw_fn(7, 0, false, false, false, false);
    pub(crate) const E: Symbol = Symbol::raw_var(8, 0);
    pub(crate) const I: Symbol = Symbol::raw_var(9, 0);
    pub(crate) const PI: Symbol = Symbol::raw_var(10, 0);

    /// The list of built-in symbols.
    pub const BUILTIN_SYMBOL_NAMES: [&'static str; 11] = [
        "arg", "coeff", "exp", "log", "sin", "cos", "sqrt", "der", "𝑒", "𝑖", "𝜋",
    ];

    pub fn is_builtin_name<S: AsRef<str>>(str: S) -> bool {
        Self::BUILTIN_SYMBOL_NAMES.contains(&str.as_ref())
    }

    fn new() -> State {
        LicenseManager::check();

        let mut state = State {
            str_to_id: HashMap::new(),
        };

        for x in Self::BUILTIN_SYMBOL_NAMES {
            state.get_symbol_impl(wrap_symbol!(x));
        }

        #[cfg(test)]
        {
            state.initialize_test();
        }

        state
    }

    /// Get the global state.
    #[inline]
    pub(crate) fn get_global_state() -> &'static RwLock<State> {
        &STATE
    }

    /// Initialize the global state for testing purposes by allocating
    /// variables and functions with the names v0, ..., v29, f0, ..., f29,
    /// that can be used in concurrently run unit tests without interference.
    #[cfg(test)]
    fn initialize_test(&mut self) {
        use crate::atom::FunctionAttribute;

        for i in 0..30 {
            let _ = self.get_symbol_impl(wrap_symbol!(format!("v{}", i)));
        }
        for i in 0..30 {
            let _ = self.get_symbol_impl(wrap_symbol!(format!("f{}", i)));
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes_impl(
                wrap_symbol!(format!("fs{}", i)),
                &[FunctionAttribute::Symmetric],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes_impl(
                wrap_symbol!(format!("fc{}", i)),
                &[FunctionAttribute::Cyclesymmetric],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes_impl(
                wrap_symbol!(format!("fa{}", i)),
                &[FunctionAttribute::Antisymmetric],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes_impl(
                wrap_symbol!(format!("fl{}", i)),
                &[FunctionAttribute::Linear],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes_impl(
                wrap_symbol!(format!("fsl{}", i)),
                &[FunctionAttribute::Symmetric, FunctionAttribute::Linear],
                None,
            );
        }
    }

    /// Remove all user-defined symbols from the state. This will invalidate all
    /// currently existing atoms, and hence this function is unsafe.
    ///
    /// Example:
    /// ```
    /// # use symbolica::symbol;
    /// # use symbolica::state::State;
    /// symbol!("f"; Symmetric).unwrap();
    /// unsafe { State::reset(); }
    /// symbol!("f"; Antisymmetric).unwrap();
    /// ```
    pub unsafe fn reset() {
        let mut state = STATE.write().unwrap();

        state.str_to_id.clear();
        SYMBOL_OFFSET.store(ID_TO_STR.len(), Ordering::Relaxed);

        for x in Self::BUILTIN_SYMBOL_NAMES {
            state.get_symbol_impl(wrap_symbol!(x));
        }

        #[cfg(test)]
        {
            state.initialize_test();
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) unsafe fn symbol_from_id(id: u32) -> Symbol {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id as usize].0
    }

    /// Iterate over all defined symbols.
    pub fn symbol_iter() -> impl Iterator<Item = (Symbol, &'static str)> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR
            .iter()
            .skip(SYMBOL_OFFSET.load(Ordering::Relaxed))
            .map(|s| (s.0, s.1.name.as_str()))
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub(crate) fn is_builtin(id: Symbol) -> bool {
        id.get_id() < Self::BUILTIN_SYMBOL_NAMES.len() as u32
    }

    /// Get the symbol for a certain name if the name is already registered,
    /// else register it and return a new symbol without attributes.
    ///
    /// To register a symbol with attributes, use [`symbol!_with_attributes`].
    pub(crate) fn get_symbol(name: NamespacedSymbol) -> Symbol {
        STATE.write().unwrap().get_symbol_impl(name)
    }

    pub(crate) fn get_symbol_impl(&mut self, name: NamespacedSymbol) -> Symbol {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0;
                for x in v.key().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let id = ID_TO_STR.len() - offset;
                let new_symbol = Symbol::raw_var(id as u32, wildcard_level);
                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        function: None,
                    },
                )) - offset;
                assert_eq!(id, id_ret);

                v.insert(new_symbol);
                new_symbol
            }
        }
    }

    /// Get the symbol for a certain name if the name is already registered,
    /// else register it and return a new symbol with the given attributes.
    ///
    /// This function will return an error when an existing symbol is redefined
    /// with different attributes.
    pub(crate) fn get_symbol_with_attributes(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
    ) -> Result<Symbol, String> {
        STATE
            .write()
            .unwrap()
            .get_symbol_with_attributes_impl(name, attributes, None)
    }

    /// Register a new symbol with the given attributes and a specific function
    /// that is called after normalization of the arguments. This function cannot
    /// be exported, and therefore before importing a state, symbols with special
    /// normalization functions must be registered explicitly.
    ///
    /// If the symbol already exists, an error is returned.
    pub(crate) fn get_symbol_with_attributes_and_function(
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
        f: NormalizationFunction,
    ) -> Result<Symbol, String> {
        STATE
            .write()
            .unwrap()
            .get_symbol_with_attributes_impl(name, attributes, Some(f))
    }

    pub(crate) fn get_symbol_with_attributes_impl(
        &mut self,
        name: NamespacedSymbol,
        attributes: &[FunctionAttribute],
        normalization_function: Option<NormalizationFunction>,
    ) -> Result<Symbol, String> {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => {
                let r = *o.get();

                let new_id = Symbol::raw_fn(
                    r.get_id(),
                    r.get_wildcard_level(),
                    attributes.contains(&FunctionAttribute::Symmetric),
                    attributes.contains(&FunctionAttribute::Antisymmetric),
                    attributes.contains(&FunctionAttribute::Cyclesymmetric),
                    attributes.contains(&FunctionAttribute::Linear),
                );

                if r == new_id && normalization_function.is_none() {
                    Ok(r)
                } else {
                    let data = &ID_TO_STR[r.get_id() as usize].1;
                    if data.file.is_empty() {
                        return Err(
                            format!("Symbol {} redefined with new attributes.", data.name).into(),
                        );
                    } else {
                        Err(format!("Symbol {} redefined with new attributes. The first definition occurred here: {}:{}.", data.name, data.file, data.line).into())
                    }
                }
            }
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let id = ID_TO_STR.len() - offset;

                let mut wildcard_level = 0;
                for x in v.key().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let new_symbol = Symbol::raw_fn(
                    id as u32,
                    wildcard_level,
                    attributes.contains(&FunctionAttribute::Symmetric),
                    attributes.contains(&FunctionAttribute::Antisymmetric),
                    attributes.contains(&FunctionAttribute::Cyclesymmetric),
                    attributes.contains(&FunctionAttribute::Linear),
                );

                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        function: normalization_function,
                    },
                )) - offset;
                assert_eq!(id, id_ret);

                v.insert(new_symbol);

                Ok(new_symbol)
            }
        }
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_name(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .name
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_symbol_namespace(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .namespace
            .as_ref()
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_symbol_data(id: Symbol) -> &'static SymbolData {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)].1
    }

    /// Get the user-specified normalization function for the symbol.
    #[inline]
    pub(crate) fn get_normalization_function(id: Symbol) -> Option<&'static NormalizationFunction> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .function
            .as_ref()
    }

    pub(crate) fn get_finite_field(fi: FiniteFieldIndex) -> &'static Zp64 {
        &FINITE_FIELDS[fi.0]
    }

    pub(crate) fn get_or_insert_finite_field(f: Zp64) -> FiniteFieldIndex {
        STATE.write().unwrap().get_or_insert_finite_field_impl(f)
    }

    pub(crate) fn get_or_insert_finite_field_impl(&mut self, f: Zp64) -> FiniteFieldIndex {
        for (i, f2) in FINITE_FIELDS.iter().enumerate() {
            if f.get_prime() == f2.get_prime() {
                return FiniteFieldIndex(i);
            }
        }

        let index = FINITE_FIELDS.push(f);
        FiniteFieldIndex(index)
    }

    pub(crate) fn get_variable_list(fi: VariableListIndex) -> Arc<Vec<Variable>> {
        VARIABLE_LISTS[fi.0].clone()
    }

    pub(crate) fn get_or_insert_variable_list(f: Arc<Vec<Variable>>) -> VariableListIndex {
        STATE.write().unwrap().get_or_insert_variable_list_impl(f)
    }

    pub(crate) fn get_or_insert_variable_list_impl(
        &mut self,
        f: Arc<Vec<Variable>>,
    ) -> VariableListIndex {
        for (i, f2) in VARIABLE_LISTS.iter().enumerate() {
            if f2 == &f {
                return VariableListIndex(i);
            }
        }

        let index = VARIABLE_LISTS.push(f);
        VariableListIndex(index)
    }

    /// Write the state to a binary stream.
    #[inline(always)]
    pub fn export<W: Write>(dest: &mut W) -> Result<(), std::io::Error> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        dest.write_u32::<LittleEndian>(SYMBOLICA_MAGIC)?;
        dest.write_u16::<LittleEndian>(EXPORT_FORMAT_VERSION)?;

        dest.write_u64::<LittleEndian>(
            ID_TO_STR.len() as u64 - SYMBOL_OFFSET.load(Ordering::Relaxed) as u64,
        )?;

        for (s, n) in State::symbol_iter() {
            dest.write_u32::<LittleEndian>(n.as_bytes().len() as u32)?;
            dest.write_all(n.as_bytes())?;

            let namespace = s.get_namespace();
            dest.write_u32::<LittleEndian>(namespace.as_bytes().len() as u32)?;
            dest.write_all(namespace.as_bytes())?;

            dest.write_u8(s.get_wildcard_level())?;
            dest.write_u8(s.is_symmetric() as u8)?;
            dest.write_u8(s.is_antisymmetric() as u8)?;
            dest.write_u8(s.is_cyclesymmetric() as u8)?;
            dest.write_u8(s.is_linear() as u8)?;
        }

        dest.write_u64::<LittleEndian>(FINITE_FIELDS.len() as u64)?;
        for x in FINITE_FIELDS.iter() {
            dest.write_u64::<LittleEndian>(x.get_prime())?;
        }

        dest.write_u64::<LittleEndian>(VARIABLE_LISTS.len() as u64)?;
        for x in VARIABLE_LISTS.iter() {
            dest.write_u64::<LittleEndian>(x.len() as u64)?;
            for y in x.iter() {
                match y {
                    Variable::Symbol(s) => {
                        dest.write_u8(0)?;
                        dest.write_u32::<LittleEndian>(s.get_id())?;
                    }
                    Variable::Temporary(u) => {
                        dest.write_u8(1)?;
                        dest.write_u64::<LittleEndian>(*u as u64)?;
                    }
                    Variable::Function(v, t) => {
                        dest.write_u8(2)?;
                        dest.write_u32::<LittleEndian>(v.get_id())?;
                        t.as_view().write(dest.by_ref())?;
                    }
                    Variable::Other(t) => {
                        dest.write_u8(3)?;
                        t.as_view().write(dest.by_ref())?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Import a state, merging it with the current state.
    /// Upon a conflict, i.e. when a symbol with the same name but different attributes is
    /// encountered, `conflict_fn` is called with the conflicting name as argument which
    /// should yield a new name for the symbol.
    #[inline(always)]
    pub fn import<R: Read>(
        source: &mut R,
        conflict_fn: Option<Box<dyn Fn(&str) -> String>>,
    ) -> Result<StateMap, std::io::Error> {
        let magic = source.read_u32::<LittleEndian>()?;

        if magic != SYMBOLICA_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number: the file is not exported from Symbolica",
            ));
        }

        let version = source.read_u16::<LittleEndian>()?;
        if version != EXPORT_FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid export format version",
            ));
        }

        let mut state_map = StateMap {
            symbols: HashMap::default(),
            finite_fields: HashMap::default(),
            variables_lists: HashMap::default(),
        };

        let n_symbols = source.read_u64::<LittleEndian>()?;
        let mut attributes = vec![];
        for x in 0..n_symbols {
            let l = source.read_u32::<LittleEndian>()?;
            let mut v = vec![0; l as usize];
            source.read_exact(&mut v)?;

            let mut str: String = std::string::String::from_utf8(v).unwrap().into();

            let l = source.read_u32::<LittleEndian>()?;
            let mut v = vec![0; l as usize];
            source.read_exact(&mut v)?;

            let namespace: String = std::string::String::from_utf8(v).unwrap().into();

            let wildcard_level = source.read_u8()?;
            let is_symmetric = source.read_u8()? != 0;
            let is_antisymmetric = source.read_u8()? != 0;
            let is_cyclesymmetric = source.read_u8()? != 0;
            let is_linear = source.read_u8()? != 0;

            attributes.clear();
            if is_antisymmetric {
                attributes.push(FunctionAttribute::Antisymmetric);
            }
            if is_symmetric {
                attributes.push(FunctionAttribute::Symmetric);
            }
            if is_cyclesymmetric {
                attributes.push(FunctionAttribute::Cyclesymmetric);
            }
            if is_linear {
                attributes.push(FunctionAttribute::Linear);
            }

            loop {
                match Symbol::new_with_attributes(
                    NamespacedSymbol {
                        symbol: str.to_string().into(),
                        namespace: namespace.to_string().into(),
                        file: "".into(),
                        line: 0,
                    },
                    &attributes,
                ) {
                    Ok(id) => {
                        if x as u32 != id.get_id() {
                            state_map.symbols.insert(x as u32, id);
                        }
                        break;
                    }
                    Err(_) => {
                        if let Some(f) = &conflict_fn {
                            let new_name = f(&str);

                            let mut new_wildcard_level = 0;
                            for x in new_name.chars().rev() {
                                if x != '_' {
                                    break;
                                }
                                new_wildcard_level += 1;
                            }

                            if wildcard_level == new_wildcard_level {
                                str = new_name;
                            }
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Symbol conflict for {}::{}", namespace, str),
                            ));
                        }
                    }
                }
            }
        }

        let n_finite_fields = source.read_u64::<LittleEndian>()?;
        for x in 0..n_finite_fields {
            let prime = source.read_u64::<LittleEndian>()?;
            let id = State::get_or_insert_finite_field(Zp64::new(prime));
            if x != id.0 as u64 {
                state_map
                    .finite_fields
                    .insert(FiniteFieldIndex(x as usize), id);
            }
        }

        let n_variable_lists = source.read_u64::<LittleEndian>()?;
        for x in 0..n_variable_lists {
            let n_vars = source.read_u64::<LittleEndian>()?;
            let mut variables = vec![];
            for _ in 0..n_vars {
                match source.read_u8()? {
                    0 => {
                        let id = source.read_u32::<LittleEndian>()?;
                        if let Some(new_id) = state_map.symbols.get(&id) {
                            variables.push(Variable::Symbol(*new_id));
                        } else {
                            variables.push(Variable::Symbol(ID_TO_STR[id as usize].0))
                        }
                    }
                    1 => {
                        let u = source.read_u64::<LittleEndian>()?;
                        variables.push(Variable::Temporary(u as usize))
                    }
                    2 => {
                        let id = source.read_u32::<LittleEndian>()?;
                        let symb = if let Some(new_id) = state_map.symbols.get(&id) {
                            *new_id
                        } else {
                            ID_TO_STR[id as usize].0
                        };

                        let mut f = Atom::new();
                        f.read(&mut *source)?;

                        let f_r = f.as_view().rename(&state_map);
                        variables.push(Variable::Function(symb, Arc::new(f_r)));
                    }
                    3 => {
                        let mut f = Atom::new();
                        f.read(&mut *source)?;

                        let f_r = f.as_view().rename(&state_map);
                        variables.push(Variable::Other(Arc::new(f_r)));
                    }
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid variable type",
                        ));
                    }
                }
            }

            // see if variables are seen before
            let vars = Arc::new(variables);
            let new_id = State::get_or_insert_variable_list(vars.clone());
            if x != new_id.0 as u64 {
                state_map.variables_lists.insert(x, vars);
            }
        }

        Ok(state_map)
    }
}

/// A workspace that stores recyclable atoms. Upon dropping, the atoms automatically returned to a
/// thread-local workspace (which may be a different one than the one it was created by).
pub struct Workspace {
    atom_buffer: RefCell<Vec<Atom>>,
}

impl Workspace {
    const ATOM_BUFFER_MAX: usize = 30;
    const ATOM_CACHE_SIZE_MAX: usize = 20_000_000;

    /// Create a new workspace.
    const fn new() -> Self {
        Workspace {
            atom_buffer: RefCell::new(Vec::new()),
        }
    }

    /// Get a thread-local workspace.
    #[inline]
    pub fn get_local() -> &'static LocalKey<Workspace> {
        LicenseManager::check();

        &WORKSPACE
    }

    /// Return a recycled atom from this workspace. The atom may have the same value as before.
    #[inline]
    pub fn new_atom(&self) -> RecycledAtom {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            if let Some(b) = a.pop() {
                b.into()
            } else {
                Atom::default().into()
            }
        } else {
            Atom::default().into() // very rare
        }
    }

    /// Create a new variable from a recycled atom from this workspace.
    #[inline]
    pub fn new_var(&self, id: Symbol) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_var(id);
        owned
    }

    /// Create a new number from a recycled atom from this workspace.
    #[inline]
    pub fn new_num<T: Into<Coefficient>>(&self, num: T) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_num(num.into());
        owned
    }

    pub fn return_atom(&self, atom: Atom) {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            a.push(atom);
        }
    }
}

/// A wrapper around [Atom] that stores the underlying buffer
/// in a thread-local storage cache when dropped.
#[derive(PartialEq, Eq, Debug, Hash, Clone)]
pub struct RecycledAtom(Atom);

impl From<Atom> for RecycledAtom {
    fn from(a: Atom) -> Self {
        RecycledAtom(a)
    }
}

impl std::fmt::Display for RecycledAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Default for RecycledAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl RecycledAtom {
    /// Get a recycled atom from a thread-local workspace.
    #[inline]
    pub fn new() -> RecycledAtom {
        Workspace::get_local().with(|ws| ws.new_atom())
    }

    /// Wrap an atom so that it gets recycled upon dropping.
    pub fn wrap(atom: Atom) -> RecycledAtom {
        RecycledAtom(atom)
    }

    #[inline]
    pub fn new_var(id: Symbol) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_var(id);
        owned
    }

    /// Create a new number from a recycled atom from this workspace.
    #[inline]
    pub fn new_num<T: Into<Coefficient>>(num: T) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_num(num.into());
        owned
    }

    /// Yield the atom, which will now no longer be recycled upon dropping.
    pub fn into_inner(mut self) -> Atom {
        std::mem::replace(&mut self.0, Atom::Zero)
    }
}

impl Deref for RecycledAtom {
    type Target = Atom;

    fn deref(&self) -> &Atom {
        &self.0
    }
}

impl DerefMut for RecycledAtom {
    fn deref_mut(&mut self) -> &mut Atom {
        &mut self.0
    }
}

impl AsRef<Atom> for RecycledAtom {
    fn as_ref(&self) -> &Atom {
        self.deref()
    }
}

impl Drop for RecycledAtom {
    #[inline]
    fn drop(&mut self) {
        if let Atom::Zero = self.0 {
            return;
        }

        if self.0.get_capacity() > Workspace::ATOM_CACHE_SIZE_MAX {
            return;
        }

        let _ = WORKSPACE.try_with(
            #[inline(always)]
            |ws| {
                if let Ok(mut a) = ws.atom_buffer.try_borrow_mut() {
                    if a.len() < Workspace::ATOM_BUFFER_MAX {
                        a.push(std::mem::replace(&mut self.0, Atom::Zero));
                    }
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::{
        atom::{Atom, AtomView},
        parse, symbol,
    };

    use super::State;

    #[test]
    fn state_export_import() {
        let mut export = vec![];
        State::export(&mut export).unwrap();

        let i = State::import(&mut Cursor::new(&export), None).unwrap();
        assert!(i.is_empty());
    }

    #[test]
    fn custom_normalization() {
        let _real_log = symbol!(
            "custom_normalization_real_log";;
            |input, out| {
                if let AtomView::Fun(f) = input {
                    if f.get_nargs() == 1 {
                        let arg = f.iter().next().unwrap();
                        if let AtomView::Fun(f2) = arg {
                            if f2.get_symbol() == Atom::EXP {
                                if f2.get_nargs() == 1 {
                                    out.set_from_view(&f2.iter().next().unwrap());
                                    return true;
                                }
                            }
                        }
                    }
                }

                false
            }
        )
        .unwrap();

        let e = parse!("custom_normalization_real_log(exp(x))").unwrap();
        assert_eq!(e, parse!("x").unwrap());
    }
}
