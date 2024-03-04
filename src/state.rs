use std::hash::Hash;
use std::sync::RwLock;
use std::{
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
};

use ahash::{HashMap, HashMapExt};
use append_only_vec::AppendOnlyVec;
use once_cell::sync::Lazy;
use smartstring::alias::String;

use crate::{
    coefficient::Coefficient,
    domains::finite_field::{FiniteField, FiniteFieldCore},
    representations::{AsAtomView, Atom, AtomView, Identifier},
    LicenseManager, LICENSE_MANAGER,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteFieldIndex(pub(crate) usize);

#[derive(Clone, Copy, PartialEq)]
pub enum FunctionAttribute {
    Symmetric,
    Antisymmetric,
    Linear,
}

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
static ID_TO_STR: AppendOnlyVec<String> = AppendOnlyVec::<String>::new();
static FINITE_FIELDS: AppendOnlyVec<FiniteField<u64>> = AppendOnlyVec::<FiniteField<u64>>::new();

/// A global state, that stores mappings from variable and function names to ids.
#[derive(Clone)]
pub struct State {
    str_to_id: HashMap<String, Identifier>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub const ARG: Identifier = Identifier::init_fn(0, 0, false, false, false);
    pub const COEFF: Identifier = Identifier::init_fn(1, 0, false, false, false);
    pub const EXP: Identifier = Identifier::init_fn(2, 0, false, false, false);
    pub const LOG: Identifier = Identifier::init_fn(3, 0, false, false, false);
    pub const SIN: Identifier = Identifier::init_fn(4, 0, false, false, false);
    pub const COS: Identifier = Identifier::init_fn(5, 0, false, false, false);
    pub const SQRT: Identifier = Identifier::init_fn(6, 0, false, false, false);
    pub const DERIVATIVE: Identifier = Identifier::init_fn(7, 0, false, false, false);
    pub const E: Identifier = Identifier::init_var(8, 0);
    pub const I: Identifier = Identifier::init_var(9, 0);
    pub const PI: Identifier = Identifier::init_var(10, 0);

    pub const BUILTIN_VAR_LIST: [&'static str; 11] = [
        "arg", "coeff", "exp", "log", "sin", "cos", "sqrt", "der", "𝑒", "𝑖", "𝜋",
    ];

    // TODO: make private
    pub fn new() -> State {
        LICENSE_MANAGER.get_or_init(LicenseManager::new).check();

        let mut state = State {
            str_to_id: HashMap::new(),
        };

        for x in Self::BUILTIN_VAR_LIST {
            state.get_or_insert_var(x);
        }

        state
    }

    pub fn get_global_state() -> &'static RwLock<State> {
        &STATE
    }

    /// Iterate over all defined symbols.
    pub fn symbol_iter(&self) -> impl Iterator<Item = &str> {
        ID_TO_STR.iter().map(|s| s.as_str())
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub fn is_builtin(id: Identifier) -> bool {
        id.get_id() < Self::BUILTIN_VAR_LIST.len() as u32
    }

    /// Get the id for a certain name if the name is already registered,
    /// else register it and return a new id.
    pub fn get_or_insert_var<S: AsRef<str>>(&mut self, name: S) -> Identifier {
        match self.str_to_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                if ID_TO_STR.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0;
                for x in name.as_ref().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let new_index = ID_TO_STR.push(name.as_ref().into());

                let new_id = Identifier::init_var(new_index as u32, wildcard_level);
                v.insert(new_id);
                new_id
            }
        }
    }

    /// Get the id of a certain function name if the name is already registered,
    /// else register it and return a new id.
    ///
    /// Providing an attribute `None` means that the attributes will be fetched from
    /// the state if the function exists, or the attribute list will be empty if not.
    pub fn get_or_insert_fn<S: AsRef<str>>(
        &mut self,
        name: S,
        attributes: Option<Vec<FunctionAttribute>>,
    ) -> Result<Identifier, String> {
        match self.str_to_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => {
                let r = *o.get();

                if let Some(attributes) = attributes {
                    let new_id = Identifier::init_fn(
                        r.get_id(),
                        r.get_wildcard_level(),
                        attributes.contains(&FunctionAttribute::Symmetric),
                        attributes.contains(&FunctionAttribute::Antisymmetric),
                        attributes.contains(&FunctionAttribute::Linear),
                    );

                    if r == new_id {
                        Ok(r)
                    } else {
                        Err(
                            format!("Function {} redefined with new attributes", name.as_ref())
                                .into(),
                        )
                    }
                } else {
                    Ok(r)
                }
            }
            Entry::Vacant(v) => {
                if ID_TO_STR.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let new_index = ID_TO_STR.push(name.as_ref().into());

                let mut wildcard_level = 0;
                for x in name.as_ref().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let new_id = if let Some(attributes) = attributes {
                    Identifier::init_fn(
                        new_index as u32,
                        wildcard_level,
                        attributes.contains(&FunctionAttribute::Symmetric),
                        attributes.contains(&FunctionAttribute::Antisymmetric),
                        attributes.contains(&FunctionAttribute::Linear),
                    )
                } else {
                    Identifier::init_fn(new_index as u32, wildcard_level, false, false, false)
                };

                v.insert(new_id);

                Ok(new_id)
            }
        }
    }

    /// Get the name for a given id.
    pub fn get_name<'a>(id: Identifier) -> &'a String {
        &ID_TO_STR[id.get_id() as usize]
    }

    pub fn get_finite_field<'a>(fi: FiniteFieldIndex) -> &'a FiniteField<u64> {
        &FINITE_FIELDS[fi.0]
    }

    pub fn get_or_insert_finite_field(&mut self, f: FiniteField<u64>) -> FiniteFieldIndex {
        for (i, f2) in FINITE_FIELDS.iter().enumerate() {
            if f.get_prime() == f2.get_prime() {
                return FiniteFieldIndex(i);
            }
        }

        let index = FINITE_FIELDS.push(f);
        FiniteFieldIndex(index)
    }
}

/// A workspace that stores reusable buffers.
pub struct Workspace {
    atom_stack: Stack<Atom>,
}

impl Workspace {
    pub fn new() -> Self {
        LICENSE_MANAGER.get_or_init(LicenseManager::new).check();

        Workspace {
            atom_stack: Stack::new(),
        }
    }

    #[inline]
    pub fn new_atom(&self) -> BufferHandle<Atom> {
        self.atom_stack.get_buf_ref()
    }

    #[inline]
    pub fn new_var(&self, id: Identifier) -> BufferHandle<Atom> {
        let mut owned = self.new_atom();
        owned.to_var(id);
        owned
    }

    #[inline]
    pub fn new_num<T: Into<Coefficient>>(&self, num: T) -> BufferHandle<Atom> {
        let mut owned = self.new_atom();
        owned.to_num(num.into());
        owned
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            atom_stack: Stack::new(),
        }
    }
}

/// A buffer that can be reset to its initial state.
/// The `new` function may allocate, but the `reset` function must not.
pub trait ResettableBuffer: Sized {
    /// Create a new resettable buffer. May allocate.
    fn new() -> Self;
    /// Reset the buffer to its initial state. Must not allocate.
    fn reset(&mut self);
}

/// A stack of resettable buffers. Any buffer lend from this stack
/// will be returned to it when it is dropped. If a buffer is requested
/// on an empty stack, a new buffer will be created. Use a stack to prevent
/// allocations by recycling used buffers first before creating new ones.
pub struct Stack<T: ResettableBuffer> {
    buffers: RefCell<Vec<T>>,
}

impl<T: ResettableBuffer> Stack<T> {
    /// Create a new stack.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffers: RefCell::new(vec![]),
        }
    }

    /// Get a buffer from the stack if the stack is not empty,
    /// else create a new one.
    #[inline]
    pub fn get_buf_ref(&self) -> BufferHandle<T> {
        let b = if let Ok(mut a) = self.buffers.try_borrow_mut() {
            if let Some(b) = a.pop() {
                b
            } else {
                T::new()
            }
        } else {
            T::new() // should never happen
        };

        BufferHandle {
            buf: Some(b),
            parent: self,
        }
    }

    /// Return a buffer to the stack.
    #[inline]
    fn return_arg(&self, mut b: T) {
        if let Ok(mut a) = self.buffers.try_borrow_mut() {
            b.reset();
            a.push(b);
        }
    }
}

/// A handle to an underlying resettable buffer. When this handle is dropped,
/// the buffer is returned to the stack it was created by.
pub struct BufferHandle<'a, T: ResettableBuffer> {
    buf: Option<T>,
    parent: &'a Stack<T>,
}

impl<'a, T: ResettableBuffer + Eq> Eq for BufferHandle<'a, T> {}

impl<'a, T: ResettableBuffer + PartialEq> PartialEq for BufferHandle<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        self.buf == other.buf
    }
}

impl<'a, T: ResettableBuffer + Hash> Hash for BufferHandle<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.buf.hash(state);
    }
}

impl<'a, T: ResettableBuffer> Deref for BufferHandle<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<'a, T: ResettableBuffer> DerefMut for BufferHandle<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

impl<'a, T: ResettableBuffer> BufferHandle<'a, T> {
    /// Get an immutable reference to the underlying buffer.
    #[inline]
    pub fn get(&self) -> &T {
        self.buf.as_ref().unwrap()
    }

    /// Get a mutable reference to the underlying buffer.
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.buf.as_mut().unwrap()
    }
}

impl<'a, T: ResettableBuffer> Drop for BufferHandle<'a, T> {
    /// Upon dropping the handle, the buffer is returned to the stack it was created by.
    #[inline]
    fn drop(&mut self) {
        self.parent
            .return_arg(std::mem::take(&mut self.buf).unwrap())
    }
}

impl<'a, 'b> AsAtomView<'b> for &'b BufferHandle<'a, Atom> {
    fn as_atom_view(self) -> AtomView<'b> {
        self.as_view()
    }
}
