pub mod finite_field;
pub mod linear_system;
pub mod integer;
pub mod rational;

use std::fmt::{Debug, Display};

pub trait Ring: Clone + Copy + PartialEq + Debug + Display {
    type Element: Clone + PartialEq + Debug + Display;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn neg(&self, a: &Self::Element) -> Self::Element;
    fn zero() -> Self::Element;
    fn one(&self) -> Self::Element;
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element;
    fn is_zero(a: &Self::Element) -> bool;
    fn is_one(&self, a: &Self::Element) -> bool;

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element;
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
