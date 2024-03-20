use rand::Rng;
use std::fmt::{Display, Error, Formatter};
use std::hash::Hash;
use std::ops::Neg;

use crate::domains::integer::Integer;
use crate::printer::PrintOptions;

use super::integer::Z;
use super::{EuclideanDomain, Field, Ring};

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

/// A number in a finite field.
#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq)]
pub struct FiniteFieldElement<UField>(pub(crate) UField);

pub trait FiniteFieldWorkspace: Clone + Display + Eq + Hash {
    /// Convert to u64.
    fn to_u64(&self) -> u64;
}

pub trait FiniteFieldCore<UField: FiniteFieldWorkspace>: Field {
    fn new(p: UField) -> Self;
    fn get_prime(&self) -> UField;
    /// Convert a number to a representative in a prime field.
    fn to_element(&self, a: UField) -> Self::Element;
    /// Convert a number from the finite field to standard form `[0,p)`.
    fn from_element(&self, a: &Self::Element) -> UField;
    /// Convert a number from the finite field to symmetric form `[-p/2,p/2]`.
    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer;
}

/// The modular ring `Z / mZ`, where `m` can be any positive integer. In most cases,
/// `m` will be a prime, and the domain will be a field.
///
/// `Zp` and `Zp64` use Montgomery modular arithmetic
/// to increase the performance of the multiplication operator.
///
/// For `m` larger than `2^64`, use `FiniteField<Integer>`.
///
/// The special field `FiniteField<Mersenne64>` can be used to have even faster arithmetic
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
        assert!(p % 2 != 0);

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

    /// Convert a number from Montgomory form to symmetric form.
    #[inline(always)]
    fn to_symmetric_integer(&self, a: &FiniteFieldElement<u32>) -> Integer {
        let i = self.from_element(a) as u64;

        if i * 2 > self.get_prime() as u64 {
            &Integer::from(i) - &Integer::from(self.get_prime() as u64)
        } else {
            i.into()
        }
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
    fn nth(&self, n: u64) -> Self::Element {
        if n > u32::MAX as u64 {
            self.to_element((n % self.p as u64) as u32)
        } else {
            self.to_element(n as u32 % self.p)
        }
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        let mut b = *b;
        let mut x = self.one();
        while e != 0 {
            if e & 1 != 0 {
                x = self.mul(&x, &b);
            }
            b = self.mul(&b, &b);
            e /= 2;
        }

        x
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

    fn is_characteristic_zero(&self) -> bool {
        false
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0.max(0)..range.1.min(self.p as i64));
        FiniteFieldElement(r as u32)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if opts.symmetric_representation_for_finite_field {
            self.to_symmetric_integer(element).fmt(f)
        } else {
            self.from_element(element).fmt(f)
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
        assert!(p % 2 != 0);

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

    /// Convert a number from Montgomory form to symmetric form.
    #[inline(always)]
    fn to_symmetric_integer(&self, a: &FiniteFieldElement<u64>) -> Integer {
        let i = self.from_element(a);

        if i > self.get_prime() / 2 {
            &Integer::from(i) - &Integer::from(self.get_prime())
        } else {
            i.into()
        }
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
    fn nth(&self, n: u64) -> Self::Element {
        self.to_element(n)
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        let mut b = *b;
        let mut x = self.one();
        while e != 0 {
            if e & 1 != 0 {
                x = self.mul(&x, &b);
            }
            b = self.mul(&b, &b);
            e /= 2;
        }

        x
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

    fn is_characteristic_zero(&self) -> bool {
        false
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0.max(0)..range.1.min(self.p.min(i64::MAX as u64) as i64));
        FiniteFieldElement(r as u64)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if opts.symmetric_representation_for_finite_field {
            self.to_symmetric_integer(element).fmt(f)
        } else {
            self.from_element(element).fmt(f)
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
        f.write_fmt(format_args!("{}", Self::PRIME))
    }
}

impl Display for Mersenne64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}", Self::PRIME))
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

    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer {
        if *a * 2 > Mersenne64::PRIME {
            &Integer::from(*a) - &Integer::from(Mersenne64::PRIME)
        } else {
            (*a).into()
        }
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
    fn nth(&self, n: u64) -> Self::Element {
        self.to_element(Mersenne64(n))
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        let mut b = *b;
        let mut x = self.one();
        while e != 0 {
            if e & 1 != 0 {
                x = self.mul(&x, &b);
            }
            b = self.mul(&b, &b);
            e /= 2;
        }

        x
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

    fn is_characteristic_zero(&self) -> bool {
        false
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng
            .gen_range(range.0.max(0)..range.1.min(Mersenne64::PRIME.min(i64::MAX as u64) as i64));
        r as u64
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if opts.symmetric_representation_for_finite_field {
            self.to_symmetric_integer(element).fmt(f)
        } else {
            self.from_element(element).fmt(f)
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
        if self <= &u64::MAX.into() {
            match self {
                Integer::Natural(x) => *x as u64,
                Integer::Double(x) => *x as u64,
                Integer::Large(_) => unreachable!(),
            }
        } else {
            panic!("Modulus {} is too large to be converted to u64", self)
        }
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
        a.clone()
    }

    fn to_symmetric_integer(&self, a: &Integer) -> Integer {
        a.clone()
    }
}

impl Ring for FiniteField<Integer> {
    type Element = Integer;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a + b).symmetric_mod(&self.p)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a - b).symmetric_mod(&self.p)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a * b).symmetric_mod(&self.p)
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        // TODO: optimize
        *a = self.add(a, b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &(b * c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &(b * c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.neg().symmetric_mod(&self.p)
    }

    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        Integer::from(n).symmetric_mod(&self.p)
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

    fn is_characteristic_zero(&self) -> bool {
        false
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        Z.sample(rng, range).symmetric_mod(&self.p)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if opts.symmetric_representation_for_finite_field {
            self.to_symmetric_integer(element).fmt(f)
        } else {
            self.from_element(element).fmt(f)
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
        assert!(!a.is_zero(), "0 is not invertible");

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

        assert!(u3.is_one(), "{} is not invertible mod {}", a, self.p);
        if even_iter {
            u1
        } else {
            &self.p - &u1
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

/// An interator over consecutive 64-bit primes.
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
