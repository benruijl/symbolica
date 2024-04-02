pub mod algebraic_number;
pub mod factorized_rational_polynomial;
pub mod finite_field;
pub mod float;
pub mod integer;
pub mod rational;
pub mod rational_polynomial;

use std::fmt::{Debug, Display, Error, Formatter};
use std::hash::Hash;

use crate::printer::PrintOptions;

pub trait Ring: Clone + PartialEq + Eq + Hash + Debug + Display {
    type Element: Clone + PartialEq + Eq + Hash + PartialOrd + Debug;

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
    fn is_characteristic_zero(&self) -> bool;

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element;
    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        in_product: bool, // can be used to add parentheses
        f: &mut Formatter<'_>,
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
    pub in_product: bool,
}

impl<'a, R: Ring> RingPrinter<'a, R> {
    pub fn new(ring: &'a R, element: &'a R::Element) -> RingPrinter<'a, R> {
        RingPrinter {
            ring,
            element,
            opts: PrintOptions::default(),
            in_product: false,
        }
    }
}

impl<'a, R: Ring> Display for RingPrinter<'a, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring
            .fmt_display(self.element, &self.opts, self.in_product, f)
    }
}
