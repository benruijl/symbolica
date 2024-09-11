use std::{
    cmp::Ordering,
    fmt::{Display, Error, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    str::FromStr,
};

use rand::Rng;
use rug::{
    ops::{Pow, RemRounding},
    Complete, Integer as MultiPrecisionInteger,
};

use crate::{printer::PrintOptions, utils};

use super::{
    finite_field::{
        FiniteField, FiniteFieldCore, FiniteFieldWorkspace, Mersenne64, ToFiniteField, Two, Zp,
        Zp64, Z2,
    },
    rational::Rational,
    EuclideanDomain, InternalOrdering, Ring,
};

pub const SMALL_PRIMES: [i64; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// The integer ring.
pub type Z = IntegerRing;
/// The integer ring.
pub const Z: IntegerRing = IntegerRing::new();

/// The integer ring.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntegerRing;

impl Default for IntegerRing {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegerRing {
    pub const fn new() -> IntegerRing {
        IntegerRing
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Integer {
    Natural(i64),
    Double(i128),
    Large(MultiPrecisionInteger),
}

impl InternalOrdering for Integer {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(self, other)
    }
}

macro_rules! from_with_cast {
    ($base: ty) => {
        impl From<$base> for Integer {
            #[inline]
            fn from(value: $base) -> Self {
                Integer::Natural(value as i64)
            }
        }

        impl PartialEq<$base> for Integer {
            #[inline]
            fn eq(&self, other: &$base) -> bool {
                match self {
                    Integer::Natural(n) => *n == *other as i64,
                    _ => false,
                }
            }
        }

        impl PartialEq<Integer> for $base {
            #[inline]
            fn eq(&self, other: &Integer) -> bool {
                other == self
            }
        }

        impl PartialOrd<$base> for Integer {
            #[inline]
            fn partial_cmp(&self, other: &$base) -> Option<Ordering> {
                match self {
                    Integer::Natural(n) => n.partial_cmp(&(*other as i64)),
                    x => {
                        if x.is_negative() {
                            Some(Ordering::Less)
                        } else {
                            Some(Ordering::Greater)
                        }
                    }
                }
            }
        }

        impl PartialOrd<Integer> for $base {
            #[inline]
            fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
                other.partial_cmp(self).map(|x| x.reverse())
            }
        }
    };
}

from_with_cast!(i8);
from_with_cast!(i16);
from_with_cast!(i32);
from_with_cast!(i64);
from_with_cast!(u8);
from_with_cast!(u16);
from_with_cast!(u32);
from_with_cast!(usize);

macro_rules! cmp_with_conv {
    ($base: ty) => {
        impl PartialEq<$base> for Integer {
            #[inline]
            fn eq(&self, other: &$base) -> bool {
                self == &Integer::from(*other)
            }
        }

        impl PartialOrd<$base> for Integer {
            #[inline]
            fn partial_cmp(&self, other: &$base) -> Option<Ordering> {
                self.partial_cmp(&Integer::from(*other))
            }
        }

        impl PartialEq<Integer> for $base {
            #[inline]
            fn eq(&self, other: &Integer) -> bool {
                other == self
            }
        }

        impl PartialOrd<Integer> for $base {
            #[inline]
            fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
                other.partial_cmp(self).map(|x| x.reverse())
            }
        }
    };
}

impl From<i128> for Integer {
    #[inline]
    fn from(value: i128) -> Self {
        Integer::from_double(value)
    }
}

impl From<u64> for Integer {
    #[inline]
    fn from(value: u64) -> Self {
        if value <= i64::MAX as u64 {
            Integer::Natural(value as i64)
        } else {
            Integer::Double(value as i128)
        }
    }
}

impl From<u128> for Integer {
    #[inline]
    fn from(value: u128) -> Self {
        if value <= i128::MAX as u128 {
            Integer::from_double(value as i128)
        } else {
            Integer::Large(value.into())
        }
    }
}

impl From<MultiPrecisionInteger> for Integer {
    /// Convert from a multi-precision integer to an Integer, potentially
    /// downcasting the number.
    #[inline]
    fn from(n: MultiPrecisionInteger) -> Self {
        if let Some(n) = n.to_i64() {
            Integer::Natural(n)
        } else if let Some(n) = n.to_i128() {
            Integer::Double(n)
        } else {
            Integer::Large(n)
        }
    }
}

cmp_with_conv!(u64);
cmp_with_conv!(u128);
cmp_with_conv!(i128);

impl FromStr for Integer {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 20 {
            if let Ok(n) = s.parse::<i64>() {
                return Ok(Integer::Natural(n));
            }
        }

        if s.len() <= 40 {
            if let Ok(n) = s.parse::<i128>() {
                return Ok(Integer::Double(n));
            }
        }

        if let Ok(n) = s.parse::<MultiPrecisionInteger>() {
            Ok(Integer::Large(n))
        } else {
            Err("Could not parse integer")
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
    fn to_finite_field(&self, field: &Zp) -> <Zp as Ring>::Element {
        match self {
            &Integer::Natural(n) => field.to_element(n.rem_euclid(field.get_prime() as i64) as u32),
            &Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u32),
            Integer::Large(r) => field.to_element(r.mod_u(field.get_prime())),
        }
    }
}

impl ToFiniteField<u64> for Integer {
    fn to_finite_field(&self, field: &Zp64) -> <Zp64 as Ring>::Element {
        match self {
            &Integer::Natural(n) => {
                if field.get_prime() > i64::MAX as u64 {
                    field.to_element((n as i128).rem_euclid(field.get_prime() as i128) as u64)
                } else {
                    field.to_element(n.rem_euclid(field.get_prime() as i64) as u64)
                }
            }
            &Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u64),
            Integer::Large(r) => {
                field.to_element(r.rem_euc(field.get_prime()).complete().to_u64().unwrap())
            }
        }
    }
}

impl ToFiniteField<Two> for Integer {
    fn to_finite_field(&self, field: &Z2) -> <Z2 as Ring>::Element {
        match self {
            &Integer::Natural(n) => field.to_element(Two(n.rem_euclid(2) as u8)),
            &Integer::Double(n) => field.to_element(Two(n.rem_euclid(2) as u8)),
            Integer::Large(r) => field.to_element(Two(r.mod_u(2) as u8)),
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
            Integer::Large(r) => r.rem_euc(Mersenne64::PRIME).complete().to_u64().unwrap(),
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
    fn from_finite_field(field: &Zp, element: <Zp as Ring>::Element) -> Self {
        Integer::Natural(field.from_element(&element) as i64)
    }

    fn from_prime(field: &Zp) -> Self {
        Integer::Natural(field.get_prime() as i64)
    }
}

impl FromFiniteField<u64> for Integer {
    fn from_finite_field(field: &Zp64, element: <Zp64 as Ring>::Element) -> Self {
        let r = field.from_element(&element);
        if r <= i64::MAX as u64 {
            Integer::Natural(r as i64)
        } else {
            Integer::Double(r as i128)
        }
    }

    fn from_prime(field: &Zp64) -> Self {
        let r = field.get_prime();
        if r <= i64::MAX as u64 {
            Integer::Natural(r as i64)
        } else {
            Integer::Double(r as i128)
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
    fn simplify(&mut self) -> &mut Self {
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
        self
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
        Self::from(MultiPrecisionInteger::from_f64(f).unwrap())
    }

    pub fn to_rational(&self) -> Rational {
        self.into()
    }

    pub fn to_multi_prec(self) -> MultiPrecisionInteger {
        match self {
            Integer::Natural(n) => n.into(),
            Integer::Double(d) => d.into(),
            Integer::Large(l) => l,
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
            Integer::Large(r) => MultiPrecisionInteger::from(r.signum_ref()) == -1,
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

    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Integer::Natural(n) => Some(*n),
            _ => None,
        }
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
                    Integer::Large(MultiPrecisionInteger::from(*n).abs())
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
            (Integer::Natural(n1), Integer::Large(n2)) => n2
                .as_abs()
                .partial_cmp(&n1.unsigned_abs())
                .unwrap_or(Ordering::Equal)
                .reverse(),
            (Integer::Double(n1), Integer::Large(n2)) => n2
                .as_abs()
                .partial_cmp(&n1.unsigned_abs())
                .unwrap_or(Ordering::Equal)
                .reverse(),
            (Integer::Large(n1), Integer::Natural(n2)) => n1
                .as_abs()
                .partial_cmp(&n2.unsigned_abs())
                .unwrap_or(Ordering::Equal),
            (Integer::Large(n1), Integer::Double(n2)) => n1
                .as_abs()
                .partial_cmp(&n2.unsigned_abs())
                .unwrap_or(Ordering::Equal),
            (_, _) => Ord::cmp(&self.abs(), &other.abs()),
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
                    Integer::Large(MultiPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Double(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Double(pn)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Large(r) => Integer::Large(r.pow(e).into()),
        }
    }

    pub fn quot_rem(&self, b: &Integer) -> (Integer, Integer) {
        if b.is_zero() {
            panic!("Cannot divide by zero");
        }

        match (self, b) {
            (Integer::Natural(aa), Integer::Natural(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb) {
                    (Integer::Natural(q), self - &(b * &Integer::Natural(q)))
                } else {
                    (Integer::Double(-(i64::MIN as i128)), Integer::zero())
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
                    (Integer::zero(), Integer::Natural(*a))
                }
            }
            (Integer::Double(aa), Integer::Natural(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb as i128) {
                    let q = Integer::from_double(q);
                    (q.clone(), self - &(b * &q))
                } else {
                    (
                        Integer::Large(MultiPrecisionInteger::from(i128::MIN).neg()),
                        Integer::zero(),
                    )
                }
            }
            (Integer::Double(aa), Integer::Double(bb)) => {
                let q = Integer::from_double(aa.div_euclid(*bb)); // b != -1
                (q.clone(), self - &(b * &q))
            }
            (Integer::Natural(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Natural(-1), Integer::from((a + b).complete()))
                    } else {
                        (Integer::Natural(1), Integer::from((a - b).complete()))
                    }
                } else {
                    (Integer::zero(), Integer::Natural(*a))
                }
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                let r = a.clone().div_rem_euc(MultiPrecisionInteger::from(*b));
                (Integer::from(r.0), Integer::from(r.1))
            }
            (Integer::Large(a), Integer::Large(b)) => {
                let r = a.clone().div_rem_euc(b.clone());
                (Integer::from(r.0), Integer::from(r.1))
            }

            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Natural(-1), Integer::from((a + b).complete()))
                    } else {
                        (Integer::Natural(1), Integer::from((a - b).complete()))
                    }
                } else {
                    (Integer::zero(), Integer::Double(*a))
                }
            }
            (Integer::Large(a), Integer::Double(b)) => {
                let r = a.clone().div_rem_euc(MultiPrecisionInteger::from(*b));
                (Integer::from(r.0), Integer::from(r.1))
            }
        }
    }

    pub fn gcd(&self, b: &Integer) -> Integer {
        match (self, b) {
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
                let r1 = MultiPrecisionInteger::from(*n1);
                Integer::from(r1.gcd(r2))
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from(r1.clone().gcd(r2)),
            (Integer::Natural(r1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Natural(r1)) => {
                Integer::from_double(utils::gcd_signed_i128(*r1 as i128, *r2) as i128)
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                let gcd = utils::gcd_signed_i128(*r1, *r2);
                if gcd == i128::MAX as u128 + 1 {
                    Integer::Large(MultiPrecisionInteger::from(gcd))
                } else {
                    Integer::from_double(gcd as i128)
                }
            }
            (Integer::Double(r1), Integer::Large(r2)) => {
                Integer::from(MultiPrecisionInteger::from(*r1).clone().gcd(r2))
            }
            (Integer::Large(r1), Integer::Double(r2)) => {
                Integer::from(r1.clone().gcd(&MultiPrecisionInteger::from(*r2)))
            }
        }
    }

    /// Compute the least common multiple of two integers.
    pub fn lcm(&self, b: &Integer) -> Integer {
        let g = self.gcd(b);
        if g.is_zero() {
            Integer::zero()
        } else {
            (self / &g) * b
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
            Integer::Natural(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let p2 = match p2 {
            Integer::Natural(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        let n1 = match n1 {
            Integer::Natural(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let n2 = match n2 {
            Integer::Natural(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
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

        Integer::from(res)
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
            let (q, r) = Z.quot_rem(&r0, &r1);
            (t1, t0) = (&t0 - &(&q * &t1), t1);
            (r1, r0) = (r, r1);
        }

        if r0 > Integer::one() {
            panic!("{} is not invertible in ring {}", self, n);
        }
        if t0.is_negative() {
            t0 += n;
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
        if let Integer::Large(l) = a {
            // prevent the creation of a GMP integer b * c
            match (b, c) {
                (Integer::Natural(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                (Integer::Double(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Natural(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Double(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                _ => {
                    return *a += b * c;
                }
            }

            a.simplify();
            return;
        }

        *a += b * c;
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        if let Integer::Large(l) = a {
            match (b, c) {
                (Integer::Natural(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                (Integer::Double(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Natural(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Double(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                _ => {
                    return *a -= b * c;
                }
            }

            a.simplify();
            return;
        }

        *a -= b * c;
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
            Integer::Double(n as i128)
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

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn size(&self) -> Integer {
        0.into()
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Integer::Natural(r)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        if opts.explicit_rational_polynomial {
            match element {
                Integer::Natural(n) => n.fmt(f),
                Integer::Double(n) => n.fmt(f),
                Integer::Large(r) => {
                    // write the GMP number in hexadecimal representation,
                    // since the conversion is much faster than for the decimal representation
                    if r.is_negative() {
                        write!(f, "-#{:X}", r.as_abs())
                    } else if f.sign_plus() {
                        write!(f, "+#{:X}", r)
                    } else {
                        write!(f, "#{:X}", r)
                    }
                }
            }
        } else {
            element.fmt(f)
        }
    }
}

impl EuclideanDomain for IntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a % b
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b)
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

impl<'b> Add<&'b Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: &'b Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Natural(n) => Integer::from(*n + r),
                Integer::Double(n) => Integer::from(*n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else {
            &self + rhs
        }
    }
}

impl Add<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Natural(n) => Integer::from(n + r),
                Integer::Double(n) => Integer::from(n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(n) => Integer::from(n + r),
                Integer::Double(n) => Integer::from(n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else {
            self + &rhs
        }
    }
}

impl<'a> Add<Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: Integer) -> Integer {
        rhs + self
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
                    Integer::Large(MultiPrecisionInteger::from(*r2) + *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_add(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) + *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => Integer::from((*n1 + r2).complete()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from((*n1 + r2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 + r2).complete()),
        }
    }
}

impl<'a> Sub<&'a Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: &Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Natural(r) => Integer::from(s - r),
                Integer::Double(r) => Integer::from(s - r),
                Integer::Large(r) => Integer::from(s - r),
            }
        } else {
            &self - rhs
        }
    }
}

impl<'a> Sub<Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(s) => Integer::from(*s - r),
                Integer::Double(s) => Integer::from(*s - r),
                Integer::Large(s) => Integer::from(s - r),
            }
        } else {
            self - &rhs
        }
    }
}

impl Sub<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Natural(r) => Integer::from(s - r),
                Integer::Double(r) => Integer::from(s - r),
                Integer::Large(r) => Integer::from(s - r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(s) => Integer::from(s - r),
                Integer::Double(s) => Integer::from(s - r),
                Integer::Large(s) => Integer::from(s - r),
            }
        } else {
            self - &rhs
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
                    Integer::Large(MultiPrecisionInteger::from(*n1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Natural(r2)) => {
                if let Some(num) = r1.checked_sub(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_sub(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2)) => Integer::from((*n1 - r2).complete()),
            (Integer::Large(r1), Integer::Natural(n2)) => Integer::from((r1 - *n2).complete()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from((*n1 - r2).complete()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from((r1 - *n2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 - r2).complete()),
        }
    }
}

impl<'a> Mul<&'a Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: &'a Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Natural(n) => Integer::from(*n * r),
                Integer::Double(n) => Integer::from(*n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else {
            &self * rhs
        }
    }
}

impl<'a> Mul<Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: Integer) -> Integer {
        rhs * self
    }
}

impl Mul<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Natural(n) => Integer::from(n * r),
                Integer::Double(n) => Integer::from(n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(n) => Integer::from(n * r),
                Integer::Double(n) => Integer::from(n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else {
            self * &rhs
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
                    Integer::Large(MultiPrecisionInteger::from(*r2) * *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_mul(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) * *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Natural(n1)) => Integer::from((n1 * r2).complete()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from((n1 * r2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 * r2).complete()),
        }
    }
}

impl<'a> Div<&'a Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: &Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Natural(r) => Integer::from(s / r),
                Integer::Double(r) => Integer::from(s / r),
                Integer::Large(r) => Integer::from(s / r),
            }
        } else {
            &self / rhs
        }
    }
}

impl<'a> Div<Integer> for &'a Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(s) => Integer::from(*s / r),
                Integer::Double(s) => Integer::from(*s / r),
                Integer::Large(s) => Integer::from(s / r),
            }
        } else {
            self / &rhs
        }
    }
}

impl Div<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Natural(r) => Integer::from(s / r),
                Integer::Double(r) => Integer::from(s / r),
                Integer::Large(r) => Integer::from(s / r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Natural(s) => Integer::from(s / r),
                Integer::Double(s) => Integer::from(s / r),
                Integer::Large(s) => Integer::from(s / r),
            }
        } else {
            self / &rhs
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
                    Integer::Large(MultiPrecisionInteger::from(*n1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Natural(r2)) => {
                if let Some(num) = r1.checked_div(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_div(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Natural(n1), Integer::Large(r2)) => Integer::from((*n1 / r2).complete()),
            (Integer::Large(r1), Integer::Natural(n2)) => Integer::from((r1 / *n2).complete()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from((*n1 / r2).complete()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from((r1 / *n2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 / r2).complete()),
        }
    }
}

macro_rules! bin_op_int {
    ($base: ty) => {
        impl Add<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: $base) -> Integer {
                self + Integer::from(rhs)
            }
        }

        impl<'a> Add<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: $base) -> Integer {
                self + Integer::from(rhs)
            }
        }

        impl<'a> Add<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: Integer) -> Integer {
                rhs + self
            }
        }

        impl<'a> Add<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: &'a Integer) -> Integer {
                rhs + self
            }
        }

        impl Sub<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: $base) -> Integer {
                self - Integer::from(rhs)
            }
        }

        impl<'a> Sub<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: $base) -> Integer {
                self - Integer::from(rhs)
            }
        }

        impl<'a> Sub<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: Integer) -> Integer {
                (-rhs) + self
            }
        }

        impl<'a> Sub<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: &'a Integer) -> Integer {
                -(rhs + -(self as i64))
            }
        }

        impl Mul<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: $base) -> Integer {
                self * Integer::from(rhs)
            }
        }

        impl<'a> Mul<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: $base) -> Integer {
                self * Integer::from(rhs)
            }
        }

        impl<'a> Mul<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: Integer) -> Integer {
                rhs * self
            }
        }

        impl<'a> Mul<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: &'a Integer) -> Integer {
                rhs * self
            }
        }

        impl Div<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn div(self, rhs: $base) -> Integer {
                &self / rhs
            }
        }

        impl<'a> Div<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn div(self, rhs: $base) -> Integer {
                self / Integer::from(rhs)
            }
        }

        impl Rem<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn rem(self, rhs: $base) -> Integer {
                &self % rhs
            }
        }

        impl<'a> Rem<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn rem(self, rhs: $base) -> Integer {
                self % Integer::from(rhs)
            }
        }

        impl AddAssign<$base> for Integer {
            #[inline]
            fn add_assign(&mut self, rhs: $base) {
                *self += Integer::from(rhs);
            }
        }

        impl SubAssign<$base> for Integer {
            #[inline]
            fn sub_assign(&mut self, rhs: $base) {
                *self -= Integer::from(rhs);
            }
        }

        impl MulAssign<$base> for Integer {
            #[inline]
            fn mul_assign(&mut self, rhs: $base) {
                *self *= Integer::from(rhs);
            }
        }

        impl DivAssign<$base> for Integer {
            #[inline]
            fn div_assign(&mut self, rhs: $base) {
                *self /= Integer::from(rhs);
            }
        }
    };
}

bin_op_int!(i8);
bin_op_int!(i16);
bin_op_int!(i32);
bin_op_int!(i64);
bin_op_int!(i128);
bin_op_int!(u8);
bin_op_int!(u16);
bin_op_int!(u32);
bin_op_int!(u64);
bin_op_int!(u128);

impl AddAssign<Integer> for Integer {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.add_assign(r),
                Integer::Double(r) => l.add_assign(r),
                Integer::Large(r) => l.add_assign(r),
            }

            self.simplify();
        } else {
            *self = rhs + &*self;
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

impl SubAssign<Integer> for Integer {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.sub_assign(r),
                Integer::Double(r) => l.sub_assign(r),
                Integer::Large(r) => l.sub_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self - rhs;
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

impl MulAssign<Integer> for Integer {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.mul_assign(r),
                Integer::Double(r) => l.mul_assign(r),
                Integer::Large(r) => l.mul_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self * rhs;
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

impl DivAssign<Integer> for Integer {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Natural(r) => l.div_assign(r),
                Integer::Double(r) => l.div_assign(r),
                Integer::Large(r) => l.div_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self / rhs;
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

impl Neg for Integer {
    type Output = Integer;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Integer::Natural(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::Natural(neg)
                } else {
                    Integer::Double((n as i128).neg())
                }
            }
            Integer::Double(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::from_double(neg)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(n).neg())
                }
            }
            Integer::Large(r) => Integer::from(-r),
        }
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
                    Integer::Large(MultiPrecisionInteger::from(*n).neg())
                }
            }
            Integer::Large(r) => Integer::from(r.clone().neg()),
        }
    }
}

impl Rem<Integer> for Integer {
    type Output = Integer;

    fn rem(self, rhs: Self) -> Self::Output {
        &self % &rhs
    }
}

impl<'a> Rem<&'a Integer> for Integer {
    type Output = Integer;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        &self % rhs
    }
}

impl<'a> Rem<Integer> for &'a Integer {
    type Output = Integer;

    fn rem(self, rhs: Integer) -> Self::Output {
        self % &rhs
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
                if let Some(r) = a.checked_rem_euclid(*b) {
                    Integer::Natural(r)
                } else {
                    Integer::zero()
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
                        Integer::from((a + b).complete())
                    } else {
                        Integer::from((a - b).complete())
                    }
                } else {
                    Integer::Natural(*a)
                }
            }
            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        Integer::from((a + b).complete())
                    } else {
                        Integer::from((a - b).complete())
                    }
                } else {
                    Integer::Double(*a)
                }
            }
            (Integer::Double(a), Integer::Natural(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b as i128) {
                    Integer::from_double(r)
                } else {
                    Integer::zero()
                }
            }
            (Integer::Double(a), Integer::Double(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b) {
                    Integer::from_double(r)
                } else {
                    Integer::zero()
                }
            }
            (Integer::Large(a), Integer::Natural(b)) => {
                Integer::from(a.rem_euc(MultiPrecisionInteger::from(*b)))
            }
            (Integer::Large(a), Integer::Double(b)) => {
                Integer::from(a.rem_euc(MultiPrecisionInteger::from(*b)))
            }
            (Integer::Large(a), Integer::Large(b)) => Integer::from(a.rem_euc(b).complete()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MultiPrecisionIntegerRing;

impl Display for MultiPrecisionIntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Default for MultiPrecisionIntegerRing {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiPrecisionIntegerRing {
    pub fn new() -> MultiPrecisionIntegerRing {
        MultiPrecisionIntegerRing
    }
}

impl InternalOrdering for MultiPrecisionInteger {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(self, other)
    }
}

impl Ring for MultiPrecisionIntegerRing {
    type Element = MultiPrecisionInteger;

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
        MultiPrecisionInteger::new()
    }

    #[inline]
    fn one(&self) -> Self::Element {
        MultiPrecisionInteger::from(1)
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        MultiPrecisionInteger::from(n)
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

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn size(&self) -> Integer {
        0.into()
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        MultiPrecisionInteger::from(r)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        _opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl EuclideanDomain for MultiPrecisionIntegerRing {
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

#[cfg(test)]
mod test {
    use std::ops::{Add, Div, Mul, Rem, Sub};

    use rug::Complete;

    use super::Integer;

    #[test]
    fn binary_ops() {
        let a = Integer::from(5);
        let b: Integer = 7.into();

        assert_eq!(a < 5, false);
        assert_eq!(5 > a, false);
        assert_eq!(a < Integer::from(-891273892173892178922i128), false);

        let c = 2u32;

        assert_eq!(c + b.clone(), 9);
        assert_eq!(c + &b, 9);
        assert_eq!(c * b.clone(), 14);
        assert_eq!(c * &b, 14);
        assert_eq!(b.clone() + c, 9);
        assert_eq!(&b + c, 9);
        assert_eq!(b.clone() * c, 14);
        assert_eq!(&b * c, 14);
        assert_eq!(b.clone() / c, 3);
        assert_eq!(&b / c, 3);
        assert_eq!(&b % c, 1);
        assert_eq!(b.clone() % c, 1);

        macro_rules! try_variants {
            ($a: expr, $b: expr, $res: expr, $op: tt) => {
                assert_eq!($a.clone().$op(&$b), $res);
                assert_eq!($a.clone().$op($b.clone()), $res);
                assert_eq!((&$a).$op($b.clone()), $res);
                assert_eq!((&$a).$op(&$b), $res);
            };
        }

        try_variants!(a, b, 12, add);
        try_variants!(a, b, -2, sub);
        try_variants!(a, b, 35, mul);
        try_variants!(a, b, 0, div);
        try_variants!(a, b, 5, rem);

        let a = Integer::from(5123123132i64).pow(5);
        let b: Integer = Integer::from(-312223132i64).pow(5);

        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("3529178341193418202448766865967598093745792000000")
                    .unwrap()
                    .complete()
            ),
            add
        );
        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("3529184275300451286008027827753913822719081764864")
                    .unwrap()
                    .complete()
            ),
            sub
        );
        try_variants!(a, b,  Integer::from(
            rug::Integer::parse("-10471269811147586074167526453409671971439869211545667023340431153576356451994427981246234624")
                .unwrap()
                .complete()
        ), mul);
        try_variants!(a, b, -1189456, div);
        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("1700675215712075116094879895131557158845440")
                    .unwrap()
                    .complete()
            ),
            rem
        );
    }
}
