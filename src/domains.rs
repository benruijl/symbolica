//! Defines core algebraic traits and data structures.
//!
//! The core trait is [Ring], which has two binary operations, addition and multiplication.
//! Each ring has an associated element type, that should not be confused with the ring type itself.
//! For example:
//! - The ring of integers [Z](type@integer::Z) has elements of type [Integer].
//! - The ring of rational numbers [Q](type@rational::Q) has elements of type [Rational](rational::Rational).
//! - The ring of finite fields [FiniteField](finite_field::FiniteField) has elements of type [FiniteField](finite_field::FiniteFieldElement).
//! - The ring of polynomials [PolynomialRing](super::poly::polynomial::PolynomialRing) has elements of type [MultivariatePolynomial](super::poly::polynomial::MultivariatePolynomial).
//!
//! In general, the ring elements do not implement operations such as addition or multiplication,
//! but rather the ring itself does. Most Symbolica structures are generic over the ring type.
//!
//! An extension of the ring trait is the [`EuclideanDomain`] trait, which adds the ability to compute remainders, quotients, and gcds.
//! Another extension is the [`Field`] trait, which adds the ability to divide and invert elements.
pub mod algebraic_number;
pub mod atom;
pub mod dual;
pub mod factorized_rational_polynomial;
pub mod finite_field;
pub mod float;
pub mod integer;
pub mod rational;
pub mod rational_polynomial;

use std::borrow::Borrow;
use std::fmt::{Debug, Display, Error, Formatter};
use std::hash::Hash;
use std::ops::{Add, Mul, Sub};

use integer::Integer;

use crate::poly::Variable;
use crate::printer::{PrintOptions, PrintState};

/// The internal ordering trait is used to compare elements of a ring.
/// This ordering is defined even for rings that do not have a total ordering, such
/// as complex numbers.
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

/// Rings whose elements contain all the knowledge of the ring itself,
/// for example integers. A counterexample would be finite field elements,
/// as they do not store the prime.
pub trait SelfRing: Clone + PartialEq + Eq + Hash + InternalOrdering + Debug + Display {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error>;

    fn format_string(&self, opts: &PrintOptions, state: PrintState) -> String {
        let mut s = String::new();
        self.format(opts, state, &mut s)
            .expect("Could not write to string");
        s
    }
}

/// A ring is a set with two binary operations, addition and multiplication.
/// Examples of rings include the integers, rational numbers, and polynomials.
///
/// Each ring has an element type, that should not be confused with the ring type itself.
/// For example:
/// - The ring of integers [Z](type@integer::Z) has elements of type [Integer].
/// - The ring of rational numbers [Q](type@rational::Q) has elements of type [Rational](rational::Rational).
/// - The ring of finite fields [FiniteField](finite_field::FiniteField) has elements of type [FiniteField](finite_field::FiniteFieldElement).
/// - The ring of polynomials [PolynomialRing](super::poly::polynomial::PolynomialRing) has elements of type [MultivariatePolynomial](super::poly::polynomial::MultivariatePolynomial).
///
/// In general, the ring elements do not implement operations such as addition or multiplication,
/// but rather the ring itself does. Most Symbolica structures are generic over the ring type.
///
/// An extension of the ring trait is the [`EuclideanDomain`] trait, which adds the ability to compute remainders, quotients, and gcds.
/// Another extension is the [`Field`] trait, which adds the ability to divide and invert elements.
pub trait Ring: Clone + PartialEq + Eq + Hash + Debug + Display {
    /// The element of a ring. For example, the elements of the ring of integers [Z](type@integer::Z), `Z::Element`, are [Integer].
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
    fn nth(&self, n: Integer) -> Self::Element;
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element;
    fn is_zero(&self, a: &Self::Element) -> bool;
    fn is_one(&self, a: &Self::Element) -> bool;
    /// Should return `true` iff `gcd(1,x)` returns `1` for any `x`.
    fn one_is_gcd_unit() -> bool;
    fn characteristic(&self) -> Integer;
    /// The number of elements in the ring. 0 is used for infinite rings.
    fn size(&self) -> Integer;

    /// Return the result of dividing `a` by `b`, if possible and if the result is unique.
    /// For example, in [Z](type@integer::Z), `4/2` is possible but `3/2` is not.
    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element>;

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element;
    /// Format a ring element with custom [PrintOptions] and [PrintState].
    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error>;

    /// Create a new printer for the given ring element that
    /// can be used in a [format!] macro.
    fn printer<'a>(&'a self, element: &'a Self::Element) -> RingPrinter<'a, Self> {
        RingPrinter::new(self, element)
    }
}

/// A Euclidean domain is a ring that supports division with remainder, quotients, and gcds.
pub trait EuclideanDomain: Ring {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element);
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

/// A field is a ring that supports division and inversion.
pub trait Field: EuclideanDomain {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn inv(&self, a: &Self::Element) -> Self::Element;

    /// Find the shortest linear recurrence relation for a given series `s`,
    /// using the Berlekamp-Massey algorithm.
    ///
    /// Yields a vector `c` such that `s[i] = sum(j, 0, m, c[j] * s[i-j-1])` for `i > m`.
    ///
    /// # Example
    /// ```rust
    /// use symbolica::domains::{Field, rational::Q};
    /// let res = Q.find_linear_recurrence_relation(&[0.into(), 1.into(), 1.into(), 3.into(),
    ///                                               5.into(), 11.into(), 21.into()]);
    /// assert_eq!(res, [1.into(), 2.into()]); // s[i] = 1 * s[i-1] + 2 * s[i-2]
    /// ```
    fn find_linear_recurrence_relation(&self, series: &[Self::Element]) -> Vec<Self::Element> {
        let mut c = vec![self.one()];
        let mut c_old = vec![self.one()];
        let mut tmp = vec![];
        let mut seq_len = 0;
        let mut m = 1;
        let mut b_inv = self.one();
        for (n, s) in series.iter().enumerate() {
            let mut error = s.clone();
            for i in 1..=seq_len {
                self.add_mul_assign(&mut error, &c[i], &series[n - i]);
            }

            if self.is_zero(&error) {
                m += 1;
            } else if 2 * seq_len <= n {
                tmp.clone_from(&c);

                let factor = self.mul(&error, &b_inv);

                if c.len() < m + c_old.len() {
                    c.resize(m + c_old.len(), self.zero());
                }

                for (j, c_j) in c_old.iter().enumerate() {
                    self.sub_mul_assign(&mut c[j + m], c_j, &factor);
                }
                seq_len = n + 1 - seq_len;
                std::mem::swap(&mut c_old, &mut tmp);
                b_inv = self.inv(&error);
                m = 1;
            } else {
                let factor = self.mul(&error, &b_inv);

                if c.len() < m + c_old.len() {
                    c.resize(m + c_old.len(), self.zero());
                }

                for (j, c_j) in c_old.iter().enumerate() {
                    self.sub_mul_assign(&mut c[j + m], c_j, &factor);
                }
                m += 1;
            }
        }

        c.drain(0..c.len() - seq_len);
        for x in &mut c {
            *x = self.neg(x);
        }
        c
    }
}

/// Rings that can be upgraded to fields, such as `IntegerRing` and `PolynomialRing`.
/// The most common upgrade is by creating a fraction field, such as `Q[x]`.
pub trait UpgradeToField: Ring {
    type Upgraded: Field;

    /// Upgrade the ring to a field.
    fn upgrade(self) -> Self::Upgraded;

    /// Upgrade an element of the ring to an element of the upgraded field.
    fn upgrade_element(
        &self,
        element: <Self as Ring>::Element,
    ) -> <Self::Upgraded as Ring>::Element;
}

impl<T: Field> UpgradeToField for T {
    type Upgraded = Self;

    fn upgrade(self) -> Self::Upgraded {
        self
    }

    fn upgrade_element(
        &self,
        element: <Self as Ring>::Element,
    ) -> <Self::Upgraded as Ring>::Element {
        element
    }
}

/// Provides an interface for printing elements of a ring with optional customization,
/// suitable as an argument to [format!]. Internally, it will call [Ring::format].
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

impl<R: Ring> Display for RingPrinter<'_, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring
            .format(
                self.element,
                &self.opts.update_with_fmt(f),
                self.state.update_with_fmt(f),
                f,
            )
            .map(|_| ())
    }
}

/// A ring element wrapped together with its ring.
#[derive(Clone)]
pub struct WrappedRingElement<R: Ring, C: Clone + Borrow<R>> {
    pub ring: C,
    pub element: R::Element,
}

impl<R: Ring, C: Clone + Borrow<R>> WrappedRingElement<R, C> {
    pub fn new(ring: C, element: R::Element) -> Self {
        WrappedRingElement { ring, element }
    }

    pub fn ring(&self) -> &R {
        self.ring.borrow()
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Ring for WrappedRingElement<R, C> {
    type Element = WrappedRingElement<R, C>;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().add(&a.element, &b.element),
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().sub(&a.element, &b.element),
        }
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().mul(&a.element, &b.element),
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.ring().add_assign(&mut a.element, &b.element);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.ring().sub_assign(&mut a.element, &b.element);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.ring().mul_assign(&mut a.element, &b.element);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.ring()
            .add_mul_assign(&mut a.element, &b.element, &c.element);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.ring()
            .sub_mul_assign(&mut a.element, &b.element, &c.element);
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().neg(&a.element),
        }
    }

    fn zero(&self) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().zero(),
        }
    }

    fn one(&self) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().one(),
        }
    }

    fn nth(&self, n: Integer) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().nth(n),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().pow(&b.element, e),
        }
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        self.ring().is_zero(&a.element)
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        self.ring().is_one(&a.element)
    }

    fn one_is_gcd_unit() -> bool {
        R::one_is_gcd_unit()
    }

    fn characteristic(&self) -> Integer {
        self.ring().characteristic()
    }

    fn size(&self) -> Integer {
        self.ring().size()
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().sample(rng, range),
        }
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        self.ring().format(&element.element, opts, state, f)
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        Some(WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().try_div(&a.element, &b.element)?,
        })
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Debug for WrappedRingElement<R, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring()
            .format(
                &self.element,
                &PrintOptions::default(),
                PrintState::default(),
                f,
            )
            .map(|_| ())
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Display for WrappedRingElement<R, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring()
            .format(
                &self.element,
                &PrintOptions::default(),
                PrintState::default(),
                f,
            )
            .map(|_| ())
    }
}

impl<R: Ring, C: Clone + Borrow<R>> PartialEq for WrappedRingElement<R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.element == other.element
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Eq for WrappedRingElement<R, C> {}

impl<R: Ring, C: Clone + Borrow<R>> Hash for WrappedRingElement<R, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.element.hash(state)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> InternalOrdering for WrappedRingElement<R, C> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.element.internal_cmp(&other.element)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> SelfRing for WrappedRingElement<R, C> {
    fn is_zero(&self) -> bool {
        self.ring().is_zero(&self.element)
    }

    fn is_one(&self) -> bool {
        self.ring().is_one(&self.element)
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        self.ring().format(&self.element, opts, state, f)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Add for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn add(self, rhs: Self) -> Self::Output {
        WrappedRingElement {
            element: self.ring().add(&self.element, &rhs.element),
            ring: self.ring,
        }
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Sub for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn sub(self, rhs: Self) -> Self::Output {
        WrappedRingElement {
            element: self.ring().sub(&self.element, &rhs.element),
            ring: self.ring,
        }
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Mul for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn mul(self, rhs: Self) -> Self::Output {
        WrappedRingElement {
            element: self.ring().mul(&self.element, &rhs.element),
            ring: self.ring,
        }
    }
}

impl<R: EuclideanDomain, C: Clone + Borrow<R>> EuclideanDomain for WrappedRingElement<R, C> {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().rem(&a.element, &b.element),
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        let (quot, rem) = self.ring().quot_rem(&a.element, &b.element);
        (
            WrappedRingElement {
                ring: self.ring.clone(),
                element: quot,
            },
            WrappedRingElement {
                ring: self.ring.clone(),
                element: rem,
            },
        )
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().gcd(&a.element, &b.element),
        }
    }
}

impl<R: Field, C: Clone + Borrow<R>> Field for WrappedRingElement<R, C> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().div(&a.element, &b.element),
        }
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.ring().div_assign(&mut a.element, &b.element);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        WrappedRingElement {
            ring: self.ring.clone(),
            element: self.ring().inv(&a.element),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::domains::{Field, Ring, rational::Q};

    #[test]
    fn linear_recurrence() {
        let series = [
            2.into(),
            2.into(),
            1.into(),
            2.into(),
            1.into(),
            191.into(),
            393.into(),
            132.into(),
        ];

        let res = Q.find_linear_recurrence_relation(&series);

        for (i, x) in series.iter().enumerate().skip(res.len()) {
            let mut c = Q.zero();
            for j in 0..res.len() {
                c += &res[j] * &series[i - j - 1];
            }

            assert_eq!(&c, x);
        }
    }
}
