use std::{
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
};

use ahash::{HashMap, HashMapExt};
use smartstring::alias::String;

use crate::{
    representations::{
        default::Linear, number::Number, AsAtomView, Atom, AtomSet, AtomView, Identifier, OwnedNum,
        OwnedVar,
    },
    rings::finite_field::{FiniteField, FiniteFieldCore},
    LicenseManager, LICENSE_MANAGER,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniteFieldIndex(pub(crate) usize);

pub(crate) const INPUT_ID: Identifier = Identifier::init(u32::MAX);

#[derive(Clone, Copy, PartialEq)]
pub enum FunctionAttribute {
    Symmetric,
}

pub const ARG: Identifier = Identifier::init(0);
pub const EXP: Identifier = Identifier::init(1);
pub const LOG: Identifier = Identifier::init(2);
pub const SIN: Identifier = Identifier::init(3);
pub const COS: Identifier = Identifier::init(4);
pub const DERIVATIVE: Identifier = Identifier::init(5);
pub(crate) const BUILTIN_VAR_LIST: [&str; 6] = ["arg", "exp", "log", "sin", "cos", "der"];

/// A global state, that stores mappings from variable and function names to ids.
#[derive(Clone)]
pub struct State {
    // get variable maps from here
    str_to_var_id: HashMap<String, Identifier>,
    function_attributes: HashMap<Identifier, Vec<FunctionAttribute>>,
    var_to_str_map: Vec<String>,
    finite_fields: Vec<FiniteField<u64>>,
}

impl State {
    pub fn new() -> State {
        LICENSE_MANAGER.get_or_init(LicenseManager::new).check();

        let mut state = State {
            str_to_var_id: HashMap::new(),
            function_attributes: HashMap::new(),
            var_to_str_map: vec![],
            finite_fields: vec![],
        };

        for x in BUILTIN_VAR_LIST {
            state.get_or_insert_var(x);
        }

        state
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub fn is_builtin(id: Identifier) -> bool {
        id.to_u32() < BUILTIN_VAR_LIST.len() as u32
    }

    // note: could be made immutable by using frozen collections
    /// Get the id for a certain name if the name is already registered,
    /// else register it and return a new id.
    pub fn get_or_insert_var<S: AsRef<str>>(&mut self, name: S) -> Identifier {
        match self.str_to_var_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => *o.get(),
            Entry::Vacant(v) => {
                if self.var_to_str_map.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let new_id = Identifier::from(self.var_to_str_map.len() as u32);
                v.insert(new_id);
                self.var_to_str_map.push(name.as_ref().into());
                new_id
            }
        }
    }

    /// Get the id for a certain name if the name is already registered,
    /// else register it and return a new id.
    pub fn get_or_insert_fn<S: AsRef<str>>(
        &mut self,
        name: S,
        attributes: Option<Vec<FunctionAttribute>>,
    ) -> Identifier {
        match self.str_to_var_id.entry(name.as_ref().into()) {
            Entry::Occupied(o) => {
                let r = *o.get();
                let old_attrib = self.function_attributes.get(&r);

                if &attributes.unwrap_or(vec![]) == old_attrib.unwrap_or(&vec![]) {
                    r
                } else {
                    panic!("Function redefined with new attributes");
                }
            }
            Entry::Vacant(v) => {
                if self.var_to_str_map.len() == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let new_id = Identifier::from(self.var_to_str_map.len() as u32);
                v.insert(new_id);
                self.var_to_str_map.push(name.as_ref().into());

                self.function_attributes
                    .insert(new_id, attributes.unwrap_or(vec![]));

                new_id
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
    pub fn get_name(&self, id: Identifier) -> Option<&String> {
        self.var_to_str_map.get(id.to_u32() as usize)
    }

    pub fn is_wildcard(&self, id: Identifier) -> Option<bool> {
        self.get_name(id).map(|n| n.ends_with('_'))
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
    pub fn new() -> Self {
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
