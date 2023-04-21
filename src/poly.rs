pub mod monomial;
pub mod polynomial;
pub mod gcd;

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add, Sub};

pub const INLINED_EXPONENTS: usize = 5;

pub trait Exponent:
    Hash + Debug + Display + Ord + Sub<Output = Self> + Add<Output = Self> + Clone + Copy
{
    fn zero() -> Self;
    /// Convert the exponent to `u32`. This is always possible, as `u32` is the largest supported exponent type.
    fn to_u32(&self) -> u32;
    /// Convert from `u32`. This function may panic if the exponent is too large.
    fn from_u32(n: u32) -> Self;
    fn is_zero(&self) -> bool;
    fn checked_add(&self, other: &Self) -> Option<Self>;
}

impl Exponent for u32 {
    fn zero() -> Self {
        0
    }

    fn to_u32(&self) -> u32 {
        *self
    }

    fn from_u32(n: u32) -> Self {
        n
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn checked_add(&self, other: &Self) -> Option<Self> {
        u32::checked_add(*self, *other)
    }
}

/// An exponent limited to 255 for efficiency
impl Exponent for u8 {
    fn zero() -> Self {
        0
    }

    fn to_u32(&self) -> u32 {
        *self as u32
    }

    fn from_u32(n: u32) -> Self {
        if n < u8::MAX as u32 {
            n as u8
        } else {
            panic!("Exponent {} too large for u8", n);
        }
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn checked_add(&self, other: &Self) -> Option<Self> {
        u8::checked_add(*self, *other)
    }
}
