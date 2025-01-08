//! Finite fields and modular rings.

use rand::Rng;
use std::fmt::{Display, Error, Formatter};
use std::hash::Hash;
use std::ops::{Deref, Neg};

use crate::domains::integer::Integer;
use crate::poly::gcd::PolynomialGCD;
use crate::poly::Variable::Temporary;
use crate::printer::{PrintOptions, PrintState};

use super::algebraic_number::AlgebraicExtension;
use super::integer::Z;
use super::{EuclideanDomain, Field, InternalOrdering, Ring};

const HENSEL_LIFTING_MASK: [u8; 128] = [
    255, 85, 51, 73, 199, 93, 59, 17, 15, 229, 195, 89, 215, 237, 203, 33, 31, 117, 83, 105, 231,
    125, 91, 49, 47, 5, 227, 121, 247, 13, 235, 65, 63, 149, 115, 137, 7, 157, 123, 81, 79, 37, 3,
    153, 23, 45, 11, 97, 95, 181, 147, 169, 39, 189, 155, 113, 111, 69, 35, 185, 55, 77, 43, 129,
    127, 213, 179, 201, 71, 221, 187, 145, 143, 101, 67, 217, 87, 109, 75, 161, 159, 245, 211, 233,
    103, 253, 219, 177, 175, 133, 99, 249, 119, 141, 107, 193, 191, 21, 243, 9, 135, 29, 251, 209,
    207, 165, 131, 25, 151, 173, 139, 225, 223, 53, 19, 41, 167, 61, 27, 241, 239, 197, 163, 57,
    183, 205, 171, 1,
];

/// A 32-bit integer finite field.
pub type Zp = FiniteField<u32>;
/// A 64-bit integer finite field.
pub type Zp64 = FiniteField<u64>;

pub trait ToFiniteField<UField: FiniteFieldWorkspace>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    fn to_finite_field(
        &self,
        field: &FiniteField<UField>,
    ) -> <FiniteField<UField> as Ring>::Element;
}

impl ToFiniteField<u32> for u32 {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        field.to_element(*self)
    }
}

impl ToFiniteField<u64> for u64 {
    fn to_finite_field(&self, field: &FiniteField<u64>) -> <FiniteField<u64> as Ring>::Element {
        field.to_element(*self)
    }
}

impl ToFiniteField<Two> for u32 {
    fn to_finite_field(&self, field: &FiniteField<Two>) -> <FiniteField<Two> as Ring>::Element {
        field.to_element(Two((*self % 2) as u8))
    }
}

impl ToFiniteField<Two> for u64 {
    fn to_finite_field(&self, field: &FiniteField<Two>) -> <FiniteField<Two> as Ring>::Element {
        field.to_element(Two((*self % 2) as u8))
    }
}

/// A Galois field `GF(p,n)` is a finite field with `p^n` elements.
/// It provides methods to upgrade and downgrade to Galois fields with the
/// same prime but with a different power.
pub trait GaloisField: Field {
    type Base: Field;

    fn get_extension_degree(&self) -> u64;
    // Convert a number from the finite field to standard form `[0,p)`.
    fn to_integer(&self, a: &Self::Element) -> Integer;
    /// Convert a number from the finite field to symmetric form `[-p/2,p/2]`.
    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer;

    /// Upgrade the field to `GF(p,new_pow)`.
    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Ring>::Element: Copy;

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Ring>::Element;

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Ring>::Element,
    ) -> Self::Element;
}

impl<UField: FiniteFieldWorkspace> GaloisField for FiniteField<UField>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField>,
{
    type Base = Self;

    fn get_extension_degree(&self) -> u64 {
        1
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).to_integer()
    }

    #[inline(always)]
    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer {
        let i = self.from_element(a).to_integer();
        let p = self.get_prime().to_integer();

        if &i * &2.into() > p {
            &i - &p
        } else {
            i
        }
    }

    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<FiniteField<UField>>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Ring>::Element: Copy,
    {
        AlgebraicExtension::galois_field(self.clone(), new_pow, Temporary(0))
    }

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Ring>::Element {
        larger_field.constant(e.clone())
    }

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Ring>::Element,
    ) -> Self::Element {
        e.poly.get_constant()
    }
}

/// A number in a finite field.
#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq)]
pub struct FiniteFieldElement<UField>(pub(crate) UField);

impl<UField: PartialOrd> InternalOrdering for FiniteFieldElement<UField> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait FiniteFieldWorkspace: Clone + Display + Eq + Hash {
    /// Convert to u64.
    fn to_u64(&self) -> u64;
    fn to_integer(&self) -> Integer {
        self.to_u64().into()
    }
}

pub trait FiniteFieldCore<UField: FiniteFieldWorkspace>: Field {
    fn new(p: UField) -> Self;
    fn get_prime(&self) -> UField;
    /// Convert a number to a representative in a prime field.
    fn to_element(&self, a: UField) -> Self::Element;
    /// Convert a number from the finite field to standard form `[0,p)`.
    fn from_element(&self, a: &Self::Element) -> UField;
}

/// The modular ring `Z / mZ`, where `m` can be any positive integer. In most cases,
/// `m` will be a prime, and the domain will be a field.
///
/// [Zp] ([`FiniteField<u32>`]) and [Zp64] ([`FiniteField<u64>`]) use Montgomery modular arithmetic
/// to increase the performance of the multiplication operator. For the prime `2`, use [type@Z2] instead.
///
/// For `m` larger than `2^64`, use [`FiniteField<Integer>`].
///
/// The special field [`FiniteField<Mersenne64>`] can be used to have even faster arithmetic
/// for a field with Mersenne prime `2^61-1`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FiniteField<UField> {
    p: UField,
    m: UField,
    one: FiniteFieldElement<UField>,
}

impl Zp {
    /// Create a new finite field. `n` must be a prime larger than 2.
    pub fn new(p: u32) -> Zp {
        if p % 2 == 0 {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_32(p),
            one: FiniteFieldElement(Self::get_one(p)),
        }
    }

    /// Returns the unit element in Montgomory form, ie.e 1 + 2^32 mod a.
    fn get_one(a: u32) -> u32 {
        if a as u64 <= 1u64 << 31 {
            let res = (((1u64 << 31) % a as u64) << 1) as u32;

            if res < a {
                res
            } else {
                res - a
            }
        } else {
            a.wrapping_neg()
        }
    }

    /// Returns -a^-1 mod 2^32.
    fn inv_2_32(a: u32) -> u32 {
        let mut ret: u32 = HENSEL_LIFTING_MASK[((a >> 1) & 127) as usize] as u32;
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret
    }
}

impl FiniteFieldWorkspace for u32 {
    fn to_u64(&self) -> u64 {
        *self as u64
    }
}

impl FiniteFieldCore<u32> for Zp {
    /// Create a new finite field. `n` must be a prime larger than 2.
    fn new(p: u32) -> Zp {
        Self::new(p)
    }

    fn get_prime(&self) -> u32 {
        self.p
    }

    /// Convert a number in a prime field a % n to Montgomory form.
    #[inline(always)]
    fn to_element(&self, a: u32) -> FiniteFieldElement<u32> {
        // TODO: slow, faster alternatives may need assembly
        FiniteFieldElement((((a as u64) << 32) % self.p as u64) as u32)
    }

    /// Convert a number from Montgomory form to standard form.
    #[inline(always)]
    fn from_element(&self, a: &FiniteFieldElement<u32>) -> u32 {
        self.mul(a, &FiniteFieldElement(1)).0
    }
}

impl Ring for Zp {
    type Element = FiniteFieldElement<u32>;

    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut t = a.0 as u64 + b.0 as u64;

        if t >= self.p as u64 {
            t -= self.p as u64;
        }

        FiniteFieldElement(t as u32)
    }

    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let t = a.0 as u64 * b.0 as u64;
        let m = (t as u32).wrapping_mul(self.m);
        let u = ((t.wrapping_add(m as u64 * self.p as u64)) >> 32) as u32;

        // correct for overflow
        if u < (t >> 32) as u32 {
            return FiniteFieldElement(u.wrapping_sub(self.p));
        }

        if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if a.0 == 0 {
            *a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        FiniteFieldElement(0)
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        self.one
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.to_finite_field(self)
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if e >= self.get_prime() as u64 - 1 {
            e = e % (self.get_prime() as u64 - 1);
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        a.0 == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a == &self.one
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime().into()
    }

    fn size(&self) -> Integer {
        self.get_prime().into()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if Self::is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0.max(0)..range.1.min(self.p as i64));
        FiniteFieldElement(r as u32)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for Zp {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        FiniteFieldElement(0)
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), FiniteFieldElement(0))
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.one()
    }
}

impl Field for Zp {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(a.0 != 0, "0 is not invertible");

        // apply multiplication with 1 twice to get the correct scaling of R=2^32
        // see the paper [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf).
        let x_mont = self
            .mul(&self.mul(a, &FiniteFieldElement(1)), &FiniteFieldElement(1))
            .0;

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u32 = 1;
        let mut u3 = x_mont;
        let mut v1: u32 = 0;
        let mut v3 = self.p;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        debug_assert!(u3 == 1);
        if even_iter {
            FiniteFieldElement(u1)
        } else {
            FiniteFieldElement(self.p - u1)
        }
    }
}

impl FiniteFieldWorkspace for u64 {
    #[inline]
    fn to_u64(&self) -> u64 {
        *self
    }
}

impl Zp64 {
    /// Create a new finite field. `n` must be a prime larger than 2.
    fn new(p: u64) -> Zp64 {
        if p % 2 == 0 {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_64(p),
            one: FiniteFieldElement(Self::get_one(p)),
        }
    }

    /// Returns the unit element in Montgomory form, ie.e 1 + 2^64 mod a.
    fn get_one(a: u64) -> u64 {
        if a as u128 <= 1u128 << 63 {
            let res = (((1u128 << 63) % a as u128) << 1) as u64;

            if res < a {
                res
            } else {
                res - a
            }
        } else {
            a.wrapping_neg()
        }
    }

    /// Returns -a^-1 mod 2^64.
    fn inv_2_64(a: u64) -> u64 {
        let mut ret: u64 = HENSEL_LIFTING_MASK[((a >> 1) & 127) as usize] as u64;
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret
    }
}

impl FiniteFieldCore<u64> for Zp64 {
    /// Create a new finite field. `n` must be a prime larger than 2.
    fn new(p: u64) -> Zp64 {
        Self::new(p)
    }

    fn get_prime(&self) -> u64 {
        self.p
    }

    /// Convert a number in a prime field a % n to Montgomory form.
    #[inline(always)]
    fn to_element(&self, a: u64) -> FiniteFieldElement<u64> {
        // TODO: slow, faster alternatives may need assembly
        FiniteFieldElement((((a as u128) << 64) % self.p as u128) as u64)
    }

    /// Convert a number from Montgomory form to standard form.
    #[inline(always)]
    fn from_element(&self, a: &FiniteFieldElement<u64>) -> u64 {
        self.mul(a, &FiniteFieldElement(1)).0
    }
}

impl<UField: Display> Display for FiniteField<UField> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, " % {}", self.p)
    }
}

impl Ring for Zp64 {
    type Element = FiniteFieldElement<u64>;
    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // avoid f128 arithmetic
        let (r, overflow) = a.0.overflowing_add(b.0);
        if overflow || r >= self.p {
            FiniteFieldElement(r.wrapping_sub(self.p))
        } else {
            FiniteFieldElement(r)
        }
    }
    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let t = a.0 as u128 * b.0 as u128;
        let m = (t as u64).wrapping_mul(self.m);
        let u = ((t.wrapping_add(m as u128 * self.p as u128)) >> 64) as u64;

        // correct for overflow
        if u < (t >> 64) as u64 {
            return FiniteFieldElement(u.wrapping_sub(self.p));
        }

        if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if a.0 == 0 {
            *a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        FiniteFieldElement(0)
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        self.one
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.to_finite_field(self)
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if e >= self.get_prime() as u64 - 1 {
            e = e % (self.get_prime() as u64 - 1);
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        a.0 == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a == &self.one
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime().into()
    }

    fn size(&self) -> Integer {
        self.get_prime().into()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if Self::is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0.max(0)..range.1.min(self.p.min(i64::MAX as u64) as i64));
        FiniteFieldElement(r as u64)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for Zp64 {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        FiniteFieldElement(0)
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), FiniteFieldElement(0))
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.one()
    }
}

impl Field for Zp64 {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(a.0 != 0, "0 is not invertible");

        // apply multiplication with 1 twice to get the correct scaling of R=2^64
        // see the paper [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf).
        let x_mont = self
            .mul(&self.mul(a, &FiniteFieldElement(1)), &FiniteFieldElement(1))
            .0;

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u64 = 1;
        let mut u3 = x_mont;
        let mut v1: u64 = 0;
        let mut v3 = self.p;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        debug_assert!(u3 == 1);
        if even_iter {
            FiniteFieldElement(u1)
        } else {
            FiniteFieldElement(self.p - u1)
        }
    }
}

/// The finite field with 0 and 1 as elements.
pub type Z2 = FiniteField<Two>;
/// The finite field with 0 and 1 as elements.
pub const Z2: FiniteField<Two> = Z2::new();

/// A finite field element of the prime 2.
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Two(pub(crate) u8);

impl Default for Z2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Z2 {
    /// Create a new finite field with prime 2.
    pub const fn new() -> Z2 {
        FiniteField {
            p: Two(2),
            m: Two(2),
            one: FiniteFieldElement(Two(1)),
        }
    }

    // Get the prime 2.
    pub fn get_prime() -> Two {
        Two(2)
    }
}

impl Two {
    pub const fn new() -> Two {
        Two(2)
    }
}

impl Default for Two {
    fn default() -> Self {
        Two(2)
    }
}

impl Deref for Two {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for Two {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl Display for Two {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FiniteFieldWorkspace for Two {
    fn to_u64(&self) -> u64 {
        self.0 as u64
    }
}

impl FiniteFieldCore<Two> for FiniteField<Two> {
    fn new(p: Two) -> Self {
        FiniteField {
            p,
            m: p,
            one: FiniteFieldElement(Two(1)),
        }
    }

    fn get_prime(&self) -> Two {
        Two(2)
    }

    fn to_element(&self, a: Two) -> Self::Element {
        a.0 % 2
    }

    fn from_element(&self, a: &Self::Element) -> Two {
        Two(*a)
    }
}

impl Ring for FiniteField<Two> {
    type Element = u8;

    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        *a * *b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        *a
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        (n % 2i32).to_i64().unwrap() as u8
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e == 0 {
            1
        } else {
            *b
        }
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        *a == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == 1
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        2.into()
    }

    fn size(&self) -> Integer {
        2.into()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if *b == 0 {
            None
        } else {
            Some(*a)
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        rng.gen_range(0..2)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).0.into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Two> {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        0
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), 0)
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        1
    }
}

impl Field for FiniteField<Two> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(*a != 0, "0 is not invertible");
        1
    }
}

/// The 64-bit Mersenne prime 2^61 -1.
///
/// Can be used for faster finite field arithmetic
/// w.r.t using Montgomery numbers.
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Mersenne64(u64);

impl Default for Mersenne64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Mersenne64 {
    pub fn new() -> Self {
        Mersenne64(Self::PRIME)
    }

    const SHIFT: u8 = 61;
    pub const PRIME: u64 = (1 << Mersenne64::SHIFT) - 1;
}

impl std::fmt::Debug for Mersenne64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl Display for Mersenne64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FiniteFieldWorkspace for Mersenne64 {
    fn to_u64(&self) -> u64 {
        Self::PRIME
    }
}

impl FiniteFieldCore<Mersenne64> for FiniteField<Mersenne64> {
    fn new(p: Mersenne64) -> Self {
        FiniteField {
            p,
            m: p,
            one: FiniteFieldElement(Mersenne64(1)),
        }
    }

    fn get_prime(&self) -> Mersenne64 {
        Mersenne64(Mersenne64::PRIME)
    }

    fn to_element(&self, a: Mersenne64) -> Self::Element {
        if a.0 >= Mersenne64::PRIME {
            a.0 - Mersenne64::PRIME
        } else {
            a.0
        }
    }

    fn from_element(&self, a: &Self::Element) -> Mersenne64 {
        Mersenne64(*a)
    }
}

impl Ring for FiniteField<Mersenne64> {
    type Element = u64;

    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut sum = a + b; // cannot overflow
        if sum >= Mersenne64::PRIME {
            sum -= Mersenne64::PRIME;
        }
        sum
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if *a >= *b {
            *a - *b
        } else {
            *a + (Mersenne64::PRIME - *b)
        }
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let v = *a as u128 * *b as u128;
        let q = (v >> Mersenne64::SHIFT) as u64;
        let r = (v as u64 & Mersenne64::PRIME) + (q & Mersenne64::PRIME);

        if r >= Mersenne64::PRIME {
            r - Mersenne64::PRIME
        } else {
            r
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if *a == 0 {
            *a
        } else {
            Mersenne64::PRIME - a
        }
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        self.to_element(Mersenne64(n.to_finite_field(self)))
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if e >= self.get_prime().0 - 1 {
            e = e % (self.get_prime().0 - 1);
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        *a == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == 1
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        Mersenne64::PRIME.into()
    }

    fn size(&self) -> Integer {
        Mersenne64::PRIME.into()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if Self::is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng
            .gen_range(range.0.max(0)..range.1.min(Mersenne64::PRIME.min(i64::MAX as u64) as i64));
        r as u64
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).0.into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Mersenne64> {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        0
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), 0)
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        1
    }
}

impl Field for FiniteField<Mersenne64> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(*a != 0, "0 is not invertible");

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u64 = 1;
        let mut u3 = *a;
        let mut v1: u64 = 0;
        let mut v3 = Mersenne64::PRIME;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        debug_assert!(u3 == 1);
        if even_iter {
            u1
        } else {
            Mersenne64::PRIME - u1
        }
    }
}

impl FiniteFieldWorkspace for Integer {
    /// Panics when the modulus is larger than 2^64.
    fn to_u64(&self) -> u64 {
        if self <= &u64::MAX {
            match self {
                Integer::Natural(x) => *x as u64,
                Integer::Double(x) => *x as u64,
                Integer::Large(_) => unreachable!(),
            }
        } else {
            panic!("Modulus {} is too large to be converted to u64", self)
        }
    }

    fn to_integer(&self) -> Integer {
        self.clone()
    }
}

/// A finite field with a large prime modulus.
/// We use the symmetric representation, as this is the most efficient.
impl FiniteFieldCore<Integer> for FiniteField<Integer> {
    fn new(m: Integer) -> FiniteField<Integer> {
        FiniteField {
            p: m.clone(),
            m: Integer::one(),
            one: FiniteFieldElement(Integer::one()),
        }
    }

    #[inline]
    fn get_prime(&self) -> Integer {
        self.p.clone()
    }

    fn to_element(&self, a: Integer) -> Integer {
        a.symmetric_mod(&self.p)
    }

    fn from_element(&self, a: &Integer) -> Integer {
        if a.is_negative() {
            a.clone() + &self.p
        } else {
            a.clone()
        }
    }
}

impl FiniteField<Integer> {
    #[inline(always)]
    fn normalize(&self, mut c: Integer) -> Integer {
        self.normalize_mut(&mut c);
        c
    }

    #[inline(always)]
    fn normalize_mut(&self, c: &mut Integer) {
        let two_c = &*c + &*c;

        if two_c.is_negative() {
            if -two_c >= self.p {
                *c += &self.p;
            }
        } else {
            if two_c >= self.p {
                *c -= &self.p;
            }
        }
    }

    /// Compute the inverse when `a` and the modulus are coprime,
    /// otherwise panic.
    pub fn try_inv(&self, a: &Integer) -> Option<Integer> {
        if a.is_zero() {
            return None;
        }

        let mut u1 = Integer::one();
        let mut u3 = a.clone();
        let mut v1 = Integer::zero();
        let mut v3 = self.get_prime();
        let mut even_iter: bool = true;

        while !v3.is_zero() {
            let (q, t3) = Z.quot_rem(&u3, &v3);
            let t1 = &u1 + &(&q * &v1);
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        if !u3.is_one() {
            return None;
        }

        if even_iter {
            Some(u1)
        } else {
            Some(&self.p - &u1)
        }
    }
}

impl Ring for FiniteField<Integer> {
    type Element = Integer;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a + b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a - b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a * b).symmetric_mod(&self.p)
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
        self.normalize_mut(a);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
        self.normalize_mut(a);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.neg()
    }

    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.symmetric_mod(&self.p)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        // FIXME: intermediate mods
        b.pow(e).symmetric_mod(&self.p)
    }

    fn is_zero(a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime()
    }

    fn size(&self) -> Integer {
        self.get_prime()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if let Some(r) = self.try_inv(b) {
            Some(self.mul(a, &r))
        } else {
            None
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        Z.sample(rng, range).symmetric_mod(&self.p)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Integer> {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), Integer::zero())
    }

    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::one()
    }
}

impl Field for FiniteField<Integer> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, &self.inv(b));
    }

    /// Compute the inverse when `a` and the modulus are coprime,
    /// otherwise panic.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        if let Some(r) = self.try_inv(a) {
            r
        } else {
            panic!("{} is not invertible mod {}", a, self.p);
        }
    }
}

/// Do a deterministic Miller test to check if `n` is a prime.
/// Since `n` is a `u64`, a basis of only 7 witnesses has to be tested.
///
/// Based on [Wojciech Izykowski's implementation](https://github.com/wizykowski/miller-rabin).
pub fn is_prime_u64(n: u64) -> bool {
    // shortest SPRP basis from Jim Sinclair for testing primality of u64
    let witnesses: [u64; 7] = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];

    if n < 2 {
        return false;
    }

    if n % 2 == 0 {
        return n == 2;
    }

    let mut s = 0;
    let mut d = n - 1;
    while d % 2 == 0 {
        d /= 2;
        s += 1;
    }

    let f = Zp64::new(n);
    let neg_one = FiniteFieldElement(n.wrapping_sub(f.one().0));

    'test: for a in witnesses {
        let a = f.to_element(a);

        if a.0 == 0 {
            continue;
        }

        let mut x = f.pow(&a, d);

        if x == f.one() || x == neg_one {
            continue;
        }

        for _ in 0..s {
            x = f.mul(&x, &x);

            if x == f.one() {
                return false;
            }
            if x == neg_one {
                continue 'test;
            }
        }

        return false;
    }

    true
}

/// An iterator over consecutive 64-bit primes.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct PrimeIteratorU64 {
    current_number: u64,
}

impl PrimeIteratorU64 {
    /// Create a new prime iterator that is larger than `start`.
    pub fn new(start: u64) -> PrimeIteratorU64 {
        PrimeIteratorU64 {
            current_number: start.max(1),
        }
    }
}

impl Iterator for PrimeIteratorU64 {
    type Item = u64;

    /// Yield the next prime or `None` if `u64::MAX` has been reached.
    fn next(&mut self) -> Option<u64> {
        while self.current_number < u64::MAX {
            self.current_number += 1;

            if is_prime_u64(self.current_number) {
                return Some(self.current_number);
            }
        }

        None
    }
}

#[cfg(test)]
mod test {
    use super::{FiniteFieldCore, Zp};
    use crate::domains::Ring;

    #[test]
    fn pow() {
        let field = Zp::new(31);

        let mut q = field.one();
        let x = field.to_element(3);
        for i in 0..100 {
            let r = field.pow(&x, i);
            assert_eq!(r, q);
            q = field.mul(&q, &x);
        }
    }
}
