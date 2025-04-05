//! Utility traits and structures.

use std::ops::Deref;

/// An enum that contains either an owned value of type `T` or a reference to a value of type `T`.
///
/// Use `Into<BorrowedOrOwned<'b, T>>>` to a accept both owned and borrowed values as arguments.
#[derive(Debug, Clone)]
pub enum BorrowedOrOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> From<&'a T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a T) -> Self {
        Self::Borrowed(v)
    }
}

impl<'a, T> From<&'a mut T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a mut T) -> Self {
        Self::Borrowed(v)
    }
}

impl<T> From<T> for BorrowedOrOwned<'_, T> {
    fn from(v: T) -> Self {
        Self::Owned(v)
    }
}

impl<'a, T> BorrowedOrOwned<'a, T> {
    /// Get a reference to the value.
    pub fn borrow(&self) -> &T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t,
        }
    }
}

impl<'a, T: Clone> BorrowedOrOwned<'a, T> {
    /// Yield an owned value.
    pub fn yield_owned(self) -> T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t.clone(),
        }
    }

    /// Create an owned version of `Self`.
    pub fn into_owned(self) -> BorrowedOrOwned<'static, T> {
        match self {
            BorrowedOrOwned::Owned(t) => BorrowedOrOwned::Owned(t),
            BorrowedOrOwned::Borrowed(t) => BorrowedOrOwned::Owned(t.clone()),
        }
    }
}

impl<T> Deref for BorrowedOrOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}
