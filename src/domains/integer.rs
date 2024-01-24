use std::{
    cmp::Ordering,
    fmt::{Display, Error, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
};

use rand::Rng;
use rug::{
    integer::IntegerExt64,
    ops::{Pow, RemRounding},
    Complete, Integer as ArbitraryPrecisionInteger,
};

use crate::{printer::PrintOptions, state::State, utils};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, Mersenne64, ToFiniteField},
    rational::Rational,
    EuclideanDomain, Ring,
};

pub const SMALL_PRIMES: [i64; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IntegerRing;

impl IntegerRing {
    pub fn new() -> IntegerRing {
        IntegerRing
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Integer {
    Natural(i64),
    Double(i128),
    Large(ArbitraryPrecisionInteger),
}

impl From<i64> for Integer {
    #[inline]
    fn from(value: i64) -> Self {
        Integer::Natural(value)
    }
}

impl From<u64> for Integer {
    #[inline]
    fn from(value: u64) -> Self {
        if value <= i64::MAX as u64 {
            Integer::Natural(value as i64)
        } else {
            Integer::Large(value.into())
        }
    }
}

impl std::fmt::Debug for Integer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Natural(n) => std::fmt::Display::fmt(n, f),
            Self::Double(n) => std::fmt::Display::fmt(n, f),
            Self::Large(n) => std::fmt::Display::fmt(n, f),
        }
    }
}

impl ToFiniteField<u32> for Integer {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        match self {
            &Integer::Natural(n) => field.to_element(n.rem_euclid(field.get_prime() as i64) as u32),
            &Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u32),
            Integer::Large(r) => field.to_element(r.mod_u(field.get_prime())),
        }
    }
}

impl ToFiniteField<u64> for Integer {
    fn to_finite_field(&self, field: &FiniteField<u64>) -> <FiniteField<u64> as Ring>::Element {
        match self {
            &Integer::Natural(n) => {
                if field.get_prime() > i64::MAX as u64 {
                    field.to_element((n as i128).rem_euclid(field.get_prime() as i128) as u64)
                } else {
                    field.to_element(n.rem_euclid(field.get_prime() as i64) as u64)
                }
            }
            &Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u64),
            Integer::Large(r) => field.to_element(r.mod_u64(field.get_prime())),
        }
    }
}

impl ToFiniteField<Integer> for Integer {
    fn to_finite_field(
        &self,
        field: &FiniteField<Integer>,
    ) -> <FiniteField<Integer> as Ring>::Element {
        field.to_element(self.clone())
    }
}

impl ToFiniteField<Mersenne64> for Integer {
    fn to_finite_field(
        &self,
        _field: &FiniteField<Mersenne64>,
    ) -> <FiniteField<Mersenne64> as Ring>::Element {
        match self {
            &Integer::Natural(n) => n.rem_euclid(Mersenne64::PRIME as i64) as u64,
            &Integer::Double(n) => n.rem_euclid(Mersenne64::PRIME as i128) as u64,
            Integer::Large(r) => r.mod_u64(Mersenne64::PRIME),
        }
    }
}

pub trait FromFiniteField<UField: FiniteFieldWorkspace>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    fn from_finite_field(
        field: &FiniteField<UField>,
        element: <FiniteField<UField> as Ring>::Element,
    ) -> Self;
    fn from_prime(field: &FiniteField<UField>) -> Self;
}

impl FromFiniteField<u32> for Integer {
    fn from_finite_field(
        field: &FiniteField<u32>,
        element: <FiniteField<u32> as Ring>::Element,
    ) -> Self {
        Integer::Natural(field.from_element(&element) as i64)
    }

    fn from_prime(field: &FiniteField<u32>) -> Self {
        Integer::Natural(field.get_prime() as i64)
    }
}

impl FromFiniteField<u64> for Integer {
    fn from_finite_field(
        field: &FiniteField<u64>,
        element: <FiniteField<u64> as Ring>::Element,
    ) -> Self {
        let r = field.from_element(&element);
        if r <= i64::MAX as u64 {
            Integer::Natural(r as i64)
        } else {
            Integer::Large(ArbitraryPrecisionInteger::from(r))
        }
    }

    fn from_prime(field: &FiniteField<u64>) -> Self {
        let r = field.get_prime();
        if r <= i64::MAX as u64 {
            Integer::Natural(r as i64)
        } else {
            Integer::Large(ArbitraryPrecisionInteger::from(r))
        }
    }
}

impl FromFiniteField<Mersenne64> for Integer {
    fn from_finite_field(_field: &FiniteField<Mersenne64>, element: u64) -> Self {
        Integer::Natural(element as i64)
    }

    fn from_prime(_field: &FiniteField<Mersenne64>) -> Self {
        Integer::Natural(Mersenne64::PRIME as i64)
    }
}

impl Integer {
    pub fn new(num: i64) -> Integer {
        Integer::Natural(num)
    }

    #[inline]
    fn simplify(&mut self) {
        match self {
            Integer::Double(n) => {
                *self = Integer::from_double(*n);
            }
            Integer::Large(l) => {
                if let Some(n) = l.to_i64() {
                    *self = Integer::Natural(n);
                } else if let Some(n) = l.to_i128() {
                    *self = Integer::Double(n);
                }
            }
            _ => {}
        }
    }

    #[inline]
    pub fn from_large(n: ArbitraryPrecisionInteger) -> Integer {
        if let Some(n) = n.to_i64() {
            Integer::Natural(n)
        } else if let Some(n) = n.to_i128() {
            Integer::Double(n)
        } else {
            Integer::Large(n)
        }
    }

    #[inline]
    pub fn from_double(n: i128) -> Integer {
        if n >= i64::MIN as i128 && n <= i64::MAX as i128 {
            Integer::Natural(n as i64)
        } else {
            Integer::Double(n)
        }
    }

    #[inline]
    pub fn from_f64(f: f64) -> Integer {
        Self::from_large(ArbitraryPrecisionInteger::from_f64(f).unwrap())
    }

    pub fn to_rational(&self) -> Rational {
        match self {
            Integer::Natural(n) => Rational::Natural(*n, 1),
            &Integer::Double(n) => Rational::Large(n.into()),
            Integer::Large(r) => Rational::Large(r.into()),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            Integer::Natural(n) => *n == 0,
            _ => false,
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        match self {
            Integer::Natural(n) => *n == 1,
            _ => false,
        }
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        match self {
            Integer::Natural(n) => *n < 0,
            Integer::Double(n) => *n < 0,
            Integer::Large(r) => ArbitraryPrecisionInteger::from(r.signum_ref()) == -1,
        }
    }

    #[inline]
    pub fn zero() -> Integer {
        Integer::Natural(0)
    }

    #[inline]
    pub fn one() -> Integer {
        Integer::Natural(1)
    }

    pub fn abs(&self) -> Integer {
        match self {
            Integer::Natural(n) => {
                if *n == i64::MIN {
                    Integer::Double((*n as i128).abs())
                } else {
                    Integer::Natural(n.abs())
                }
            }
            Integer::Double(n) => {
                if *n == i128::MIN {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n).abs())
                } else {
                    Integer::Double(n.abs())
                }
            }
            Integer::Large(n) => Integer::Large(n.clone().abs()),
        }
    }

    pub fn abs_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Integer::Large(n1), Integer::Large(n2)) => n1.as_abs().cmp(&n2.as_abs()),
            (Integer::Natural(n1), Integer::Large(n2)) => {
                if *n1 == i64::MIN {
                    ArbitraryPrecisionInteger::from(*n1).as_abs().cmp(n2)
                } else {
                    n2.as_abs()
                        .partial_cmp(&n1.abs())
                        .unwrap_or(Ordering::Equal)
                        .reverse()
                }
            }
            (Integer::Large(n1), Integer::Natural(n2)) => {
                if *n1 == i64::MIN {
                    n1.as_abs()
                        .cmp(&ArbitraryPrecisionInteger::from(*n2).as_abs())
                } else {
                    n1.as_abs()
                        .partial_cmp(&n2.abs())
                        .unwrap_or(Ordering::Equal)
                }
            }
            (_, _) => self.abs().cmp(&other.abs()),
        }
    }

    /// Compute `n` factorial (`n!`).
    pub fn factorial(n: u32) -> Integer {
        if n <= 20 {
            let mut f: i64 = 1;
            for x in 2..=n as i64 {
                f *= x;
            }
            Integer::Natural(f)
        } else {
            Integer::Large(rug::Integer::factorial(n).complete())
        }
    }

    /// Compute the binomial coefficient `(n k) = n!/(k!(n-k)!)`.
    ///
    /// The implementation does not to overflow.
    pub fn binom(n: i64, mut k: i64) -> Integer {
        if n < 0 || k < 0 || k > n {
            return Integer::zero();
        }
        if k > n / 2 {
            k = n - k
        }
        let mut res = Integer::one();
        for i in 1..=k {
            res *= n - k + i;
            res /= i;
        }
        res
    }

    /// Compute the multinomial coefficient `(k_1+...+k_n)!/(k_1!*...*k_n!)`
    ///
    /// The implementation does not to overflow.
    pub fn multinom(k: &[u32]) -> Integer {
        let mut mcr = Integer::one();
        let mut accum = 0i64;
        for v in k {
            if let Some(res) = accum.checked_add(*v as i64) {
                accum = res;
            } else {
                panic!("Sum of occurrences exceeds i64: {:?}", k);
            }

            mcr *= &Self::binom(accum, *v as i64);
        }
        mcr
    }

    pub fn pow(&self, e: u64) -> Integer {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        if e == 0 {
            return Integer::one();
        }

        match self {
            Integer::Natural(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Natural(pn)
                } else if let Some(pn) = (*n1 as i128).checked_pow(e) {
                    Integer::Double(pn)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Double(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Double(pn)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Large(r) => Integer::Large(r.pow(e).into()),
        }
    }

    /// Use Garner's algorithm for the Chinese remainder theorem
    /// to reconstruct an `x` that satisfies `n1 = x % p1` and `n2 = x % p2`.
    /// The `x` will be in the range `[-p1*p2/2,p1*p2/2]`.
    pub fn chinese_remainder(n1: Integer, n2: Integer, p1: Integer, p2: Integer) -> Integer {
        // make sure n1 < n2
        if match (&n1, &n2) {
            (Integer::Natural(n1), Integer::Natural(n2)) => n1 > n2,
            (Integer::Natural(_), Integer::Large(_)) => false,
            (Integer::Natural(_), Integer::Double(_)) => false,
            (Integer::Double(_), Integer::Natural(_)) => true,
            (Integer::Double(_), Integer::Large(_)) => false,
            (Integer::Large(_), Integer::Natural(_)) => true,
            (Integer::Large(_), Integer::Double(_)) => true,
            (Integer::Double(n1), Integer::Double(n2)) => n1 > n2,
            (Integer::Large(r1), Integer::Large(r2)) => r1 > r2,
        } {
            return Self::chinese_remainder(n2, n1, p2, p1);
        }

        let p1 = match p1 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Double(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let p2 = match p2 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Double(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        let n1 = match n1 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Double(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let n2 = match n2 {
            Integer::Natural(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Double(n) => ArbitraryPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        // convert to mixed-radix notation
        let gamma1 = (p1.clone() % p2.clone())
            .invert(&p2)
            .unwrap_or_else(|_| panic!("Could not invert {} in {}", p1, p2));

        let v1 = ((n2 - n1.clone()) * gamma1) % p2.clone();

        // convert to standard representation
        let r = v1 * p1.clone() + n1;

        let res = if r.clone() * 2 > p1.clone() * p2.clone() {
            r - p1 * p2
        } else {
            r
        };

        Integer::from_large(res)
    }

    /// Perform the symmetric mod `p` on `self`.
    #[inline]
    pub fn symmetric_mod(&self, p: &Integer) -> Integer {
        let c = self % p;

        if &c * &2u64.into() > *p {
            &c - p
        } else {
            c
        }
    }

    /// Compute the modular inverse of `self` in the ring with size `n`.
    /// `self` and `n` must be coprime.
    pub fn mod_inverse(&self, n: &Integer) -> Integer {
        let mut t0 = Integer::zero();
        let mut t1 = Integer::one();
        let mut r0 = n.clone();
        let mut r1 = self.clone();

        while !r1.is_zero() {
            let (q, r) = IntegerRing::new().quot_rem(&r0, &r1);
            (t1, t0) = (&t0 - &(&q * &t1), t1);
            (r1, r0) = (r, r1);
        }

        if r0 > Integer::one() {
            panic!("{} is not invertible in ring {}", self, n);
        }
        if t0.is_negative() {
            t0 += &n;
        }

        t0
    }
}

impl Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Integer::Natural(n) => n.fmt(f),
            Integer::Double(n) => n.fmt(f),
            Integer::Large(r) => r.fmt(f),
        }
    }
}

impl Display for IntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Integer::Natural(n1), Integer::Natural(n2)) => n1.partial_cmp(n2),
            (Integer::Natural(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Natural(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Natural(n1), Integer::Double(n2)) => (*n1 as i128).partial_cmp(n2),
            (Integer::Double(n1), Integer::Natural(n2)) => n1.partial_cmp(&(*n2 as i128)),
            (Integer::Double(n1), Integer::Double(n2)) => n1.partial_cmp(n2),
            (Integer::Double(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Double(n2)) => n1.partial_cmp(n2),
        }
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Ring for IntegerRing {
    type Element = Integer;

    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        // TODO: recycle a gmp integer coming from b * c if a is not a gmp integer
        a.add_assign(&(b * c))
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        a.sub_assign(&(b * c))
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    #[inline]
    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        if n <= i64::MAX as u64 {
            Integer::Natural(n as i64)
        } else {
            Integer::Large(ArbitraryPrecisionInteger::from(n))
        }
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e)
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        match a {
            Integer::Natural(r) => *r == 0,
            Integer::Double(_) => false,
            Integer::Large(_) => false,
        }
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        match a {
            Integer::Natural(r) => *r == 1,
            Integer::Double(_) => false,
            Integer::Large(_) => false,
        }
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn is_characteristic_zero(&self) -> bool {
        true
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Integer::Natural(r)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        _state: Option<&State>,
        _opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl EuclideanDomain for IntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a % b
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        if b.is_zero() {
            panic!("Cannot divide by zero");
        }

        match (a, b) {
            (Integer::Natural(aa), Integer::Natural(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb) {
                    (Integer::Natural(q), a - &(b * &Integer::Natural(q)))
                } else {
                    (Integer::Double(-(i64::MIN as i128)), Integer::Natural(0))
                }
            }
            (Integer::Natural(a), Integer::Double(b)) => {
                // we always have |a| <= |b|
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Natural(-1), Integer::from_double(*a as i128 + *b))
                    } else {
                        (Integer::Natural(1), Integer::from_double(*a as i128 - *b))
                    }
                } else {
                    (Integer::Natural(0), Integer::Natural(*a))
                }
            }
            (Integer::Double(a), Integer::Natural(b)) => {
                let r = ArbitraryPrecisionInteger::from(*a)
                    .div_rem_euc(ArbitraryPrecisionInteger::from(*b));
                (Integer::from_large(r.0), Integer::from_large(r.1))
            }
            (Integer::Double(aa), Integer::Double(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb) {
                    (Integer::Double(q), a - &(b * &Integer::Double(q)))
                } else {
                    (
                        Integer::Large(ArbitraryPrecisionInteger::from(i128::MIN).neg()),
                        Integer::Natural(0),
                    )
                }
            }
            (Integer::Natural(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Natural(-1), Integer::from_large((a + b).into()))
                    } else {
                        (Integer::Natural(1), Integer::from_large((a - b).into()))
                    }
                } else {
                    (Integer::Natural(0), Integer::Natural(*a))
                }
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                let r = a.clone().div_rem_euc(ArbitraryPrecisionInteger::from(*b));
                (Integer::from_large(r.0), Integer::from_large(r.1))
            }
            (Integer::Large(a), Integer::Large(b)) => {
                let r = a.clone().div_rem_euc(b.clone());
                (Integer::from_large(r.0), Integer::from_large(r.1))
            }

            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Natural(-1), Integer::from_large((a + b).into()))
                    } else {
                        (Integer::Natural(1), Integer::from_large((a - b).into()))
                    }
                } else {
                    (Integer::Natural(0), Integer::Double(*a))
                }
            }
            (Integer::Large(a), Integer::Double(b)) => {
                let r = a.clone().div_rem_euc(ArbitraryPrecisionInteger::from(*b));
                (Integer::from_large(r.0), Integer::from_large(r.1))
            }
        }
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                let gcd = utils::gcd_signed(*n1, *n2);
                if gcd == i64::MAX as u64 + 1 {
                    // n1 == n2 == u64::MIN
                    Integer::Double(gcd as i128)
                } else {
                    Integer::Natural(gcd as i64)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => {
                let r1 = ArbitraryPrecisionInteger::from(*n1);
                Integer::from_large(r1.gcd(r2))
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from_large(r1.clone().gcd(r2)),
            (Integer::Natural(r1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Natural(r1)) => {
                Integer::from_double(utils::gcd_signed_i128(*r1 as i128, *r2) as i128)
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                let gcd = utils::gcd_signed_i128(*r1, *r2);
                if gcd == i128::MAX as u128 + 1 {
                    Integer::Large(ArbitraryPrecisionInteger::from(gcd))
                } else {
                    Integer::from_double(gcd as i128)
                }
            }
            (Integer::Double(r1), Integer::Large(r2)) => {
                Integer::from_large(ArbitraryPrecisionInteger::from(*r1).clone().gcd(r2))
            }
            (Integer::Large(r1), Integer::Double(r2)) => {
                Integer::from_large(r1.clone().gcd(&ArbitraryPrecisionInteger::from(*r2)))
            }
        }
    }
}

impl<'a, 'b> Add<&'b Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_add(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 + *n2 as i128)
                }
            }
            (Integer::Natural(n1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Natural(n1)) => {
                if let Some(num) = (*n1 as i128).checked_add(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r2) + *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_add(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) + *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => Integer::from_large((*n1 + r2).into()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from_large((*n1 + r2).into()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from_large((r1 + r2).into()),
        }
    }
}

impl<'a, 'b> Sub<&'b Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_sub(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 - *n2 as i128)
                }
            }
            (Integer::Natural(n1), Integer::Double(r2)) => {
                if let Some(num) = (*n1 as i128).checked_sub(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Natural(r2)) => {
                if let Some(num) = r1.checked_sub(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_sub(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2)) => Integer::from_large((*n1 - r2).into()),
            (Integer::Large(r1), Integer::Natural(n2)) => Integer::from_large((r1 - *n2).into()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from_large((*n1 - r2).into()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from_large((r1 - *n2).into()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from_large((r1 - r2).into()),
        }
    }
}

impl<'a, 'b> Mul<&'b Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_mul(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 * *n2 as i128)
                }
            }
            (Integer::Natural(n1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Natural(n1)) => {
                if let Some(num) = (*n1 as i128).checked_mul(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r2) * *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_mul(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) * *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => Integer::from_large((n1 * r2).into()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from_large((n1 * r2).into()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from_large((r1 * r2).into()),
        }
    }
}

impl<'a, 'b> Div<&'b Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Natural(n1), Integer::Natural(n2)) => {
                if let Some(num) = n1.checked_div(*n2) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 / *n2 as i128)
                }
            }
            (Integer::Natural(n1), Integer::Double(r2)) => {
                if let Some(num) = (*n1 as i128).checked_div(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Natural(r2)) => {
                if let Some(num) = r1.checked_div(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_div(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2)) => Integer::from_large((*n1 / r2).into()),
            (Integer::Large(r1), Integer::Natural(n2)) => Integer::from_large((r1 / *n2).into()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from_large((*n1 / r2).into()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from_large((r1 / *n2).into()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from_large((r1 / r2).into()),
        }
    }
}

impl<'a> Add<i64> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_add(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 + rhs as i128)
                }
            }
            Integer::Double(n1) => {
                if let Some(num) = n1.checked_add(rhs as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) + rhs)
                }
            }
            Integer::Large(n1) => Integer::from_large((n1 + rhs).into()),
        }
    }
}

impl<'a> Sub<i64> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_sub(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 - rhs as i128)
                }
            }
            Integer::Double(n1) => {
                if let Some(num) = n1.checked_sub(rhs as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) - rhs)
                }
            }
            Integer::Large(n1) => Integer::from_large((n1 - rhs).into()),
        }
    }
}

impl<'a> Mul<i64> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_mul(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 * rhs as i128)
                }
            }
            Integer::Double(n1) => {
                if let Some(num) = n1.checked_mul(rhs as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) * rhs)
                }
            }
            Integer::Large(n1) => Integer::from_large((n1 * rhs).into()),
        }
    }
}

impl<'a> Div<i64> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: i64) -> Integer {
        match self {
            Integer::Natural(n1) => {
                if let Some(num) = n1.checked_div(rhs) {
                    Integer::Natural(num)
                } else {
                    Integer::Double(*n1 as i128 / rhs as i128)
                }
            }
            Integer::Double(n1) => {
                if let Some(num) = n1.checked_div(rhs as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n1) / rhs)
                }
            }
            Integer::Large(n1) => Integer::from_large((n1 / rhs).into()),
        }
    }
}

impl<'a> AddAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.add_assign(*r),
                Integer::Double(r) => l.add_assign(*r),
                Integer::Large(r) => l.add_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self + rhs;
        }
    }
}

impl<'a> SubAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.sub_assign(*r),
                Integer::Double(r) => l.sub_assign(*r),
                Integer::Large(r) => l.sub_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self - rhs;
        }
    }
}

impl<'a> MulAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.mul_assign(*r),
                Integer::Double(r) => l.mul_assign(*r),
                Integer::Large(r) => l.mul_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self * rhs;
        }
    }
}

impl<'a> DivAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn div_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.div_assign(*r),
                Integer::Double(r) => l.div_assign(*r),
                Integer::Large(r) => l.div_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self / rhs;
        }
    }
}

impl MulAssign<i64> for Integer {
    #[inline]
    fn mul_assign(&mut self, rhs: i64) {
        *self = (&*self) * rhs;
    }
}

impl DivAssign<i64> for Integer {
    #[inline]
    fn div_assign(&mut self, rhs: i64) {
        *self = (&*self) / rhs;
    }
}

impl<'a> Neg for &'a Integer {
    type Output = Integer;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Integer::Natural(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::Natural(neg)
                } else {
                    Integer::Double((*n as i128).neg())
                }
            }
            Integer::Double(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::from_double(neg)
                } else {
                    Integer::Large(ArbitraryPrecisionInteger::from(*n).neg())
                }
            }
            Integer::Large(r) => Integer::from_large(r.neg().into()),
        }
    }
}

impl<'a> Rem for &'a Integer {
    type Output = Integer;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            panic!("Cannot divide by zero");
        }

        match (self, rhs) {
            (Integer::Natural(a), Integer::Natural(b)) => {
                if let Some(q) = a.checked_rem_euclid(*b) {
                    Integer::Natural(q)
                } else {
                    Integer::Natural(0)
                }
            }
            (Integer::Natural(a), Integer::Double(b)) => {
                // b must be larger than a, so division is never needed
                if *a < 0 {
                    if *b > 0 {
                        Integer::from_double(*a as i128 + *b)
                    } else {
                        Integer::from_double(*a as i128 - *b)
                    }
                } else {
                    Integer::Natural(*a)
                }
            }
            (Integer::Natural(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        Integer::from_large((a + b).into())
                    } else {
                        Integer::from_large((a - b).into())
                    }
                } else {
                    Integer::Natural(*a)
                }
            }
            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        Integer::from_large((a + b).into())
                    } else {
                        Integer::from_large((a - b).into())
                    }
                } else {
                    Integer::Double(*a)
                }
            }
            (Integer::Double(a), Integer::Natural(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b as i128) {
                    Integer::from_double(r)
                } else {
                    Integer::Natural(0)
                }
            }
            (Integer::Double(a), Integer::Double(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b as i128) {
                    Integer::from_double(r)
                } else {
                    Integer::Natural(0)
                }
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                Integer::from_large(a.rem_euc(ArbitraryPrecisionInteger::from(*b)))
            }
            (Integer::Large(a), Integer::Double(b)) => {
                Integer::from_large(a.rem_euc(ArbitraryPrecisionInteger::from(*b)))
            }
            (Integer::Large(a), Integer::Large(b)) => Integer::from_large(a.rem_euc(b).into()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ArbitraryPrecisionIntegerRing;

impl Display for ArbitraryPrecisionIntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArbitraryPrecisionIntegerRing {
    pub fn new() -> ArbitraryPrecisionIntegerRing {
        ArbitraryPrecisionIntegerRing
    }
}

impl Ring for ArbitraryPrecisionIntegerRing {
    type Element = ArbitraryPrecisionInteger;

    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        a.add_assign(b * c)
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        a.sub_assign(b * c)
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }

    #[inline]
    fn zero(&self) -> Self::Element {
        ArbitraryPrecisionInteger::new()
    }

    #[inline]
    fn one(&self) -> Self::Element {
        ArbitraryPrecisionInteger::from(1)
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        ArbitraryPrecisionInteger::from(n)
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        b.clone().pow(e as u32)
    }

    #[inline]
    fn is_zero(a: &Self::Element) -> bool {
        a.is_zero()
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == self.one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn is_characteristic_zero(&self) -> bool {
        true
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        ArbitraryPrecisionInteger::from(r)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        _state: Option<&State>,
        _opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl EuclideanDomain for ArbitraryPrecisionIntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() % b
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.clone().div_rem_euc(b.clone())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone().gcd(b)
    }
}
