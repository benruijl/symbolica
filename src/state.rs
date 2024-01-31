use std::hash::Hash;
use std::{
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
};

use ahash::{HashMap, HashMapExt};
use smartstring::alias::String;

use crate::{
    domains::finite_field::{FiniteField, FiniteFieldCore},
    representations::{
        default::Linear, number::Number, AsAtomView, Atom, AtomSet, AtomView, Identifier, OwnedNum,
        OwnedVar,
    },
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

/// A global state, that stores mappings from variable and function names to ids.
#[derive(Clone)]
pub struct State {
    // get variable maps from here
    str_to_var_id: HashMap<String, Identifier>,
    function_attributes: HashMap<Identifier, Vec<FunctionAttribute>>,
    var_info: Vec<(String, usize)>,
    finite_fields: Vec<FiniteField<u64>>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub const ARG: Identifier = Identifier::init(0);
    pub const NUM: Identifier = Identifier::init(1);
    pub const EXP: Identifier = Identifier::init(2);
    pub const LOG: Identifier = Identifier::init(3);
    pub const SIN: Identifier = Identifier::init(4);
    pub const COS: Identifier = Identifier::init(5);
    pub const SQRT: Identifier = Identifier::init(6);
    pub const DERIVATIVE: Identifier = Identifier::init(7);
    pub const E: Identifier = Identifier::init(8);
    pub const I: Identifier = Identifier::init(9);
    pub const PI: Identifier = Identifier::init(10);

    pub(crate) const BUILTIN_VAR_LIST: [&'static str; 11] = [
        "arg", "num", "exp", "log", "sin", "cos", "sqrt", "der", "𝑒", "𝑖", "𝜋",
    ];

    pub fn new() -> State {
        LICENSE_MANAGER.get_or_init(LicenseManager::new).check();

        let mut state = State {
            str_to_var_id: HashMap::new(),
            function_attributes: HashMap::new(),
            var_info: vec![],
            finite_fields: vec![],
        };

        for x in Self::BUILTIN_VAR_LIST {
            state.get_or_insert_var(x);
        }

        state
    }

    /// Iterate over all defined symbols.
    pub fn symbol_iter(&self) -> impl Iterator<Item = &str> {
        self.var_info.iter().map(|(s, _)| s.as_str())
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub fn is_builtin(id: Identifier) -> bool {
        id.to_u32() < Self::BUILTIN_VAR_LIST.len() as u32
    }

    // note: could be made immutable by using frozen collections
    /// Get the id for a certain name if the name is already registered,
    /// else register it and return a new id.
    pub fn get_or_insert_var<S: AsRef<str>>(&mut self, name: S) -> Identifier {
        match self.str_to_var_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                if self.var_info.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0;
                for x in name.as_ref().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let new_id = Identifier::from(self.var_info.len() as u32);
                v.insert(new_id);
                self.var_info.push((name.as_ref().into(), wildcard_level));
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
        match self.str_to_var_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => {
                let r = *o.get();
                let old_attrib = self.function_attributes.get(&r);

                if attributes.is_none() || attributes.as_ref() == old_attrib {
                    Ok(r)
                } else {
                    Err(format!("Function {} redefined with new attributes", name.as_ref()).into())
                }
            }
            Entry::Vacant(v) => {
                if self.var_info.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let mut wildcard_level = 0;
                for x in name.as_ref().chars().rev() {
                    if x != '_' {
                        break;
                    }
                    wildcard_level += 1;
                }

                let new_id = Identifier::from(self.var_info.len() as u32);
                v.insert(new_id);
                self.var_info.push((name.as_ref().into(), wildcard_level));

                self.function_attributes
                    .insert(new_id, attributes.unwrap_or_default());

                Ok(new_id)
            }
        }
    }

    pub fn get_function_attributes(&self, id: Identifier) -> &[FunctionAttribute] {
        self.function_attributes
            .get(&id)
            .map(|x| x.as_ref())
            .unwrap_or(&[])
    }

    /// Get the name for a given id.
    pub fn get_name(&self, id: Identifier) -> &String {
        &self.var_info[id.to_u32() as usize].0
    }

    pub fn get_wildcard_level(&self, id: Identifier) -> usize {
        self.var_info[id.to_u32() as usize].1
    }

    pub fn get_finite_field(&self, fi: FiniteFieldIndex) -> &FiniteField<u64> {
        &self.finite_fields[fi.0]
    }

    pub fn get_or_insert_finite_field(&mut self, f: FiniteField<u64>) -> FiniteFieldIndex {
        for (i, f2) in self.finite_fields.iter().enumerate() {
            if f.get_prime() == f2.get_prime() {
                return FiniteFieldIndex(i);
            }
        }

        self.finite_fields.push(f);
        FiniteFieldIndex(self.finite_fields.len() - 1)
    }
}

/// A workspace that stores reusable buffers.
pub struct Workspace<P: AtomSet = Linear> {
    atom_stack: Stack<Atom<P>>,
}

impl<P: AtomSet> Workspace<P> {
    pub fn new() -> Self {
        LICENSE_MANAGER.get_or_init(LicenseManager::new).check();

        Workspace {
            atom_stack: Stack::new(),
        }
    }

    pub fn new_atom(&self) -> BufferHandle<Atom<P>> {
        self.atom_stack.get_buf_ref()
    }

    pub fn new_var(&self, id: Identifier) -> BufferHandle<Atom<P>> {
        let mut owned = self.new_atom();
        owned.to_var().set_from_id(id);
        owned
    }

    pub fn new_num<T: Into<Number>>(&self, num: T) -> BufferHandle<Atom<P>> {
        let mut owned = self.new_atom();
        owned.to_num().set_from_number(num.into());
        owned
    }
}

impl Default for Workspace<Linear> {
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
        let b = self
            .buffers
            .borrow_mut()
            .pop()
            .map(|mut b| {
                b.reset();
                b
            })
            .unwrap_or_else(|| T::new());

        BufferHandle {
            buf: Some(b),
            parent: self,
        }
    }

    /// Return a buffer to the stack.
    #[inline]
    fn return_arg(&self, b: T) {
        self.buffers.borrow_mut().push(b);
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

impl<'a, 'b, P: AtomSet> AsAtomView<'b, P> for &'b BufferHandle<'a, Atom<P>> {
    fn as_atom_view(self) -> AtomView<'b, P> {
        self.as_view()
    }
}
