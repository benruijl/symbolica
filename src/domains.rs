pub mod algebraic_number;
pub mod atom;
pub mod dual;
pub mod factorized_rational_polynomial;
pub mod finite_field;
pub mod float;
pub mod integer;
pub mod rational;
pub mod rational_polynomial;

use std::fmt::{Debug, Display, Error, Formatter};
use std::hash::Hash;

use integer::Integer;

use crate::poly::Variable;
use crate::printer::{PrintOptions, PrintState};

pub trait InternalOrdering {
    /// Compare two elements using an internal ordering.
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

macro_rules! impl_internal_ordering {
    ($($t:ty),*) => {
        $(
            impl InternalOrdering for $t {
                fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

impl_internal_ordering!(u8);
impl_internal_ordering!(u64);

macro_rules! impl_internal_ordering_range {
    ($($t:ty),*) => {
        $(
            impl<T: InternalOrdering> InternalOrdering for $t {
                fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
                    match self.len().cmp(&other.len()) {
                        std::cmp::Ordering::Equal => (),
                        ord => return ord,
                    }

                    for (i, j) in self.iter().zip(other) {
                        match i.internal_cmp(&j) {
                            std::cmp::Ordering::Equal => {}
                            ord => return ord,
                        }
                    }

                    std::cmp::Ordering::Equal
                }
            }
        )*
    };
}

impl_internal_ordering_range!([T]);
impl_internal_ordering_range!(Vec<T>);

/// A ring that supports a derivative.
pub trait Derivable: Ring {
    /// Take the derivative of `e` in `x`.
    fn derivative(&self, e: &<Self as Ring>::Element, x: &Variable) -> <Self as Ring>::Element;
}

pub trait Ring: Clone + PartialEq + Eq + Hash + Debug + Display {
    type Element: Clone + PartialEq + Eq + Hash + InternalOrdering + Debug;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element);
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element);
    fn neg(&self, a: &Self::Element) -> Self::Element;
    fn zero(&self) -> Self::Element;
    fn one(&self) -> Self::Element;
    /// Return the nth element by computing `n * 1`.
    fn nth(&self, n: u64) -> Self::Element;
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element;
    fn is_zero(a: &Self::Element) -> bool;
    fn is_one(&self, a: &Self::Element) -> bool;
    /// Should return `true` iff `gcd(1,x)` returns `1` for any `x`.
    fn one_is_gcd_unit() -> bool;
    fn characteristic(&self) -> Integer;
    /// The number of elements in the ring. 0 is used for infinite rings.
    fn size(&self) -> Integer;

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element;
    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<(), Error>;

    fn printer<'a>(&'a self, element: &'a Self::Element) -> RingPrinter<'a, Self> {
        RingPrinter::new(self, element)
    }
}

pub trait EuclideanDomain: Ring {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element);
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

pub trait Field: EuclideanDomain {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn inv(&self, a: &Self::Element) -> Self::Element;
}

pub struct RingPrinter<'a, R: Ring> {
    pub ring: &'a R,
    pub element: &'a R::Element,
    pub opts: PrintOptions,
    pub state: PrintState,
}

impl<'a, R: Ring> RingPrinter<'a, R> {
    pub fn new(ring: &'a R, element: &'a R::Element) -> RingPrinter<'a, R> {
        RingPrinter {
            ring,
            element,
            opts: PrintOptions::default(),
            state: PrintState::default(),
        }
    }
}

impl<'a, R: Ring> Display for RingPrinter<'a, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring.format(
            self.element,
            &self.opts.clone().update_with_fmt(f),
            self.state.clone().update_with_fmt(f),
            f,
        )
    }
}
