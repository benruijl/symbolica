use std::{
    f64::consts::{LOG10_2, LOG2_10},
    fmt::{self, Debug, Display, Formatter, LowerExp, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use serde::{Deserialize, Serialize};
use wide::{f64x2, f64x4};

use crate::domains::integer::Integer;

use super::rational::Rational;
use rug::{
    ops::{CompleteRound, Pow},
    Assign, Float as MultiPrecisionFloat,
};

pub trait NumericalFloatLike:
    PartialEq
    + Clone
    + Debug
    + LowerExp
    + Display
    + std::ops::Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
{
    fn mul_add(&self, a: &Self, b: &Self) -> Self;
    fn neg(&self) -> Self;
    fn zero(&self) -> Self;
    /// Create a zero that should only be used as a temporary value,
    /// as for some types it may have wrong precision information.
    fn new_zero() -> Self;
    fn one(&self) -> Self;
    fn pow(&self, e: u64) -> Self;
    fn inv(&self) -> Self;

    fn from_usize(&self, a: usize) -> Self;
    fn from_i64(&self, a: i64) -> Self;

    /// Get the number of precise binary digits.
    fn get_precision(&self) -> u32;
    fn get_epsilon(&self) -> f64;
    /// Return true iff the precision is fixed, or false
    /// if the precision is changed dynamically.
    fn fixed_precision(&self) -> bool;

    /// Sample a point on the interval [0, 1].
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self;
}

pub trait SingleFloat: NumericalFloatLike {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_finite(&self) -> bool;
}

pub trait RealNumberLike: SingleFloat {
    fn to_usize_clamped(&self) -> usize;
    fn to_f64(&self) -> f64;
}

/// A float that can be constructed without any parameters, such as f64.
pub trait ConstructibleFloat: NumericalFloatLike {
    fn new_one() -> Self;
    fn new_from_usize(a: usize) -> Self;
    fn new_from_i64(a: i64) -> Self;
    /// Sample a point on the interval [0, 1].
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

pub trait Real: NumericalFloatLike {
    fn norm(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn log(&self) -> Self;
    fn exp(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan2(&self, x: &Self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn powf(&self, e: &Self) -> Self;
}

impl NumericalFloatLike for f64 {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        f64::mul_add(*self, *a, *b)
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        0.
    }

    #[inline(always)]
    fn new_zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one(&self) -> Self {
        1.
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        // FIXME: use binary exponentiation
        debug_assert!(e <= i32::MAX as u64);
        self.powi(e as i32)
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        1. / self
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        a as f64
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        53
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        f64::EPSILON / 2.
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        true
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        rng.gen()
    }
}

impl SingleFloat for f64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == 0.
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        *self == 1.
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }
}

impl RealNumberLike for f64 {
    fn to_usize_clamped(&self) -> usize {
        *self as usize
    }

    fn to_f64(&self) -> f64 {
        *self
    }
}

impl ConstructibleFloat for f64 {
    #[inline(always)]
    fn new_one() -> Self {
        1.
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl Real for f64 {
    #[inline(always)]
    fn norm(&self) -> Self {
        f64::abs(*self)
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        (*self).ln()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        (*self).exp()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        (*self).sin()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        (*self).cos()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        (*self).tan()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        (*self).asin()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        (*self).acos()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        (*self).atan2(*x)
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        (*self).sinh()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        (*self).cosh()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        (*self).tanh()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        (*self).asinh()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        (*self).acosh()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        (*self).atanh()
    }

    #[inline]
    fn powf(&self, e: &f64) -> Self {
        (*self).powf(*e)
    }
}

impl From<&Rational> for f64 {
    fn from(value: &Rational) -> Self {
        value.to_f64()
    }
}

impl From<Rational> for f64 {
    fn from(value: Rational) -> Self {
        value.to_f64()
    }
}

/// A multi-precision floating point type. Operations on this type
/// loosely track the precision of the result, but always overestimate.
/// Some operations may improve precision, such as `sqrt` or adding an
/// infinite-precision integer.
///
/// Floating point output with less than five significant binary digits
/// should be considered unreliable.
#[derive(Clone, PartialEq)]
pub struct Float(MultiPrecisionFloat);

impl Debug for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // print only the significant digits
        // the original float value may not be reconstructible
        // from this output
        if f.precision().is_none() {
            f.write_fmt(format_args!(
                "{0:.1$}",
                self.0,
                (self.0.prec() as f64 * LOG10_2).floor() as usize
            ))
        } else {
            Display::fmt(&self.0, f)
        }
    }
}

impl LowerExp for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if f.precision().is_none() {
            f.write_fmt(format_args!(
                "{0:.1$e}",
                self.0,
                (self.0.prec() as f64 * LOG10_2).floor() as usize
            ))
        } else {
            LowerExp::fmt(&self.0, f)
        }
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Neg for Float {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

impl Add<&Float> for Float {
    type Output = Self;

    /// Add two floats, while keeping loose track of the precision.
    /// The precision of the output will be at most 2 binary digits too high.
    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        let mut r = self.0 + &rhs.0;

        if let Some(e) = r.get_exp() {
            if let Some(e1) = e1 {
                if let Some(e2) = rhs.0.get_exp() {
                    // the max is at most 2 binary digits off
                    let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);

                    // set the min precision to 1, from this point on the result is unreliable
                    r.set_prec(1.max(max_prec.min(r.prec() as i32)) as u32);
                }
            }
        }

        r.into()
    }
}

impl Add<Float> for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            rhs + &self
        } else {
            self + &rhs
        }
    }
}

impl Sub<&Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        let mut r = self.0 - &rhs.0;

        if let Some(e) = r.get_exp() {
            if let Some(e1) = e1 {
                if let Some(e2) = rhs.0.get_exp() {
                    let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
                    r.set_prec(1.max(max_prec.min(r.prec() as i32)) as u32);
                }
            }
        }

        r.into()
    }
}

impl Sub<Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            -rhs + &self
        } else {
            self - &rhs
        }
    }
}

impl Mul<&Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 * &rhs.0).into()
    }
}

impl Mul<Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.prec() < self.prec() {
            (rhs.0 * self.0).into()
        } else {
            (self.0 * rhs.0).into()
        }
    }
}

impl Div<&Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: &Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / &rhs.0).into()
    }
}

impl Div<Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: &Float) {
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        self.0.add_assign(&rhs.0);

        if let Some(e) = self.0.get_exp() {
            if let Some(e1) = e1 {
                if let Some(e2) = rhs.0.get_exp() {
                    let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
                    self.set_prec(1.max(max_prec.min(self.prec() as i32)) as u32);
                }
            }
        }
    }
}

impl AddAssign<Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: Float) {
        self.add_assign(&rhs)
    }
}

impl SubAssign<&Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: &Float) {
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        self.0.sub_assign(&rhs.0);

        if let Some(e) = self.0.get_exp() {
            if let Some(e1) = e1 {
                if let Some(e2) = rhs.0.get_exp() {
                    let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
                    self.set_prec(1.max(max_prec.min(self.prec() as i32)) as u32);
                }
            }
        }
    }
}

impl SubAssign<Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: Float) {
        self.sub_assign(&rhs)
    }
}

impl MulAssign<&Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: &Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(&rhs.0);
    }
}

impl MulAssign<Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(rhs.0);
    }
}

impl DivAssign<&Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: &Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(&rhs.0);
    }
}

impl DivAssign<Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(rhs.0);
    }
}

impl Add<i64> for Float {
    type Output = Self;

    /// Add an infinite-precision `i64` to the float.
    #[inline]
    fn add(mut self, rhs: i64) -> Self::Output {
        if rhs == 0 {
            return self;
        }

        let Some(e1) = self.0.get_exp() else {
            return self + rhs;
        };

        let e2 = rhs.unsigned_abs().ilog2() + 1;
        let old_prec = self.prec();

        if e1 < 0 || e1.unsigned_abs() < e2 {
            self.set_prec(self.prec() + e2.checked_add_signed(-e1).unwrap() + 1);
        }

        let mut r = self.0 + rhs;

        if let Some(e) = r.get_exp() {
            r.set_prec((1.max(old_prec as i32 + 1 - (e1 - e))) as u32);
        }

        r.into()
    }
}

impl Add<Float> for i64 {
    type Output = Float;

    /// Add a float to an infinite-precision `i64`.
    #[inline]
    fn add(self, rhs: Float) -> Self::Output {
        rhs + self
    }
}

impl Sub<i64> for Float {
    type Output = Self;

    /// Subtract an infinite-precision `i64` from a float.
    #[inline]
    fn sub(self, rhs: i64) -> Self::Output {
        self + -rhs
    }
}

impl Sub<Float> for i64 {
    type Output = Float;

    /// Subtract a float from an infinite-precision `i64`.
    #[inline]
    fn sub(self, rhs: Float) -> Self::Output {
        -rhs + self
    }
}

impl Mul<i64> for Float {
    type Output = Self;

    /// Multiply an infinite-precision `i64` to a float.
    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        (self.0 * rhs).into()
    }
}

impl Mul<Float> for i64 {
    type Output = Float;

    /// Multiply a float to an infinite-precision `i64`.
    #[inline]
    fn mul(self, rhs: Float) -> Self::Output {
        (self * rhs.0).into()
    }
}

impl Div<i64> for Float {
    type Output = Self;

    /// Divide an infinite-precision `i64` from a float.
    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        (self.0 / rhs).into()
    }
}

impl Div<Float> for i64 {
    type Output = Float;

    /// Divide a float from an infinite-precision `i64`.
    #[inline]
    fn div(self, rhs: Float) -> Self::Output {
        (self / rhs.0).into()
    }
}

impl Add<Rational> for Float {
    type Output = Self;

    /// Add an infinite-precision rational to the float.
    #[inline]
    fn add(mut self, rhs: Rational) -> Self::Output {
        if rhs.is_zero() {
            return self;
        }

        let Some(e1) = self.0.get_exp() else {
            let np = self.prec();
            return (self.0 + rhs.to_multi_prec_float(np).0).into();
        };

        fn get_bits(i: &Integer) -> i32 {
            match i {
                Integer::Natural(n) => n.unsigned_abs().ilog2() as i32 + 1,
                Integer::Double(n) => n.unsigned_abs().ilog2() as i32 + 1,
                Integer::Large(r) => r.significant_bits() as i32,
            }
        }

        // TODO: check off-by-one errors
        let e2 = get_bits(rhs.numerator_ref()) - get_bits(rhs.denominator_ref());

        let old_prec = self.prec();

        if e1 < 0 || e1 < e2 {
            self.set_prec(self.prec() + (e2 - e1) as u32 + 1);
        }

        let np = self.prec();
        let mut r = self.0 + rhs.to_multi_prec_float(np).0;

        if let Some(e) = r.get_exp() {
            r.set_prec((1.max(old_prec as i32 + 1 - (e1 - e))) as u32);
        }

        r.into()
    }
}

impl Sub<Rational> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rational) -> Self::Output {
        self + -rhs
    }
}

impl Mul<Rational> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rational) -> Self::Output {
        (self.0 * rhs.to_multi_prec()).into()
    }
}

impl Div<Rational> for Float {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Rational) -> Self::Output {
        (self.0 / rhs.to_multi_prec()).into()
    }
}

impl Float {
    pub fn new(prec: u32) -> Self {
        Float(MultiPrecisionFloat::new(prec))
    }

    pub fn with_val<T>(prec: u32, val: T) -> Self
    where
        MultiPrecisionFloat: Assign<T>,
    {
        Float(MultiPrecisionFloat::with_val(prec, val))
    }

    pub fn prec(&self) -> u32 {
        self.0.prec()
    }

    pub fn set_prec(&mut self, prec: u32) {
        self.0.set_prec(prec);
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    /// Parse a float from a string.
    /// Precision can be specified by a trailing backtick followed by the precision.
    /// For example: ```1.234`20``` for a precision of 20 decimal digits.
    /// The precision is allowed to be a floating point number.
    ///  If `prec` is `None` and no precision is specified, the precision is derived from the string, with
    /// a minimum of 53 bits (`f64` precision).
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, String> {
        if let Some(prec) = prec {
            Ok(Float(
                MultiPrecisionFloat::parse(s)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
        } else if let Some((f, p)) = s.split_once('`') {
            let prec = (p
                .parse::<f64>()
                .map_err(|e| format!("Invalid precision: {}", e))?
                * LOG2_10)
                .ceil() as u32;
            Ok(Float(
                MultiPrecisionFloat::parse(f)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
        } else {
            // get the number of accurate digits
            let digits = s
                .chars()
                .skip_while(|x| *x == '.' || *x == '0')
                .take_while(|x| x.is_ascii_digit())
                .count();

            let prec = ((digits as f64 * LOG2_10).ceil() as u32).max(53);
            Ok(Float(
                MultiPrecisionFloat::parse(s)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        self.0.to_string_radix(16, None).into_bytes()
    }

    pub fn deserialize(d: &[u8], prec: u32) -> Float {
        MultiPrecisionFloat::parse_radix(d, 16)
            .unwrap()
            .complete(prec)
            .into()
    }

    pub fn to_rational(&self) -> Rational {
        self.0.to_rational().unwrap().into()
    }

    pub fn into_inner(self) -> MultiPrecisionFloat {
        self.0
    }
}

impl From<MultiPrecisionFloat> for Float {
    fn from(value: MultiPrecisionFloat) -> Self {
        Float(value)
    }
}

impl NumericalFloatLike for Float {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        a.clone() * b + self
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0.clone()).into()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        Float::new(self.prec())
    }

    #[inline(always)]
    fn new_zero() -> Self {
        Float::new(1)
    }

    #[inline(always)]
    fn one(&self) -> Self {
        Float::with_val(self.prec(), 1.)
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        rug::ops::Pow::pow(&self.0, e)
            .complete(self.prec() as i64)
            .into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.clone().recip().into()
    }

    /// Convert from a `usize`. This may involve a loss of precision.
    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        Float::with_val(self.prec(), a)
    }

    /// Convert from a `i64`. This may involve a loss of precision.
    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        Float::with_val(self.prec(), a)
    }

    fn get_precision(&self) -> u32 {
        self.prec()
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.prec() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        false
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let f: f64 = rng.gen();
        Float::with_val(self.prec(), f)
    }
}

impl SingleFloat for Float {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0.
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0 == 1.
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
}

impl RealNumberLike for Float {
    fn to_usize_clamped(&self) -> usize {
        self.0
            .to_integer()
            .unwrap()
            .to_usize()
            .unwrap_or(usize::MAX)
    }

    fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }
}

impl Real for Float {
    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.clone().abs().into()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        self.0.sqrt_ref().complete(self.prec() as i64 + 1).into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        // Log grows in precision if the input is less than 1/e and more than e
        let e = self.0.get_exp().unwrap();
        if !(0..2).contains(&e) {
            self.0
                .ln_ref()
                .complete((self.0.prec() + e.unsigned_abs().ilog2() + 1) as i64)
                .into()
        } else {
            self.0.clone().ln().into()
        }
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        if let Some(e) = self.0.get_exp() {
            // Exp grows in precision when e < 0
            self.0
                .exp_ref()
                .complete(1.max(self.0.prec() as i32 - e + 1) as i64)
                .into()
        } else {
            self.0.clone().exp().into()
        }
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        self.0.clone().sin().into()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        self.0.clone().cos().into()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        self.0.clone().tan().into()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        self.0.clone().asin().into()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        self.0.clone().acos().into()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        self.0.clone().atan2(&x.0).into()
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        self.0.clone().sinh().into()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        self.0.clone().cosh().into()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        if let Some(e) = self.0.get_exp() {
            if e > 0 {
                return self
                    .0
                    .tanh_ref()
                    .complete((self.0.prec() + 3 * e.unsigned_abs() + 1) as i64)
                    .into();
            }
        }

        self.0.clone().tanh().into()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        self.0.clone().asinh().into()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        self.0.clone().acosh().into()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        self.0.clone().atanh().into()
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        let mut c = self.0.clone();
        if let Some(exp) = e.0.get_exp() {
            if let Some(eb) = self.0.get_exp() {
                // eb is (over)estimate of ln(self)
                // TODO: prevent taking the wrong branch when self = 1
                if eb == 0 {
                    c.set_prec(1.max((self.0.prec() as i32 - exp + 1) as u32));
                } else {
                    c.set_prec(
                        1.max(
                            (self.0.prec() as i32)
                                .min((e.0.prec() as i32) + eb.unsigned_abs().ilog2() as i32)
                                - exp
                                + 1,
                        ) as u32,
                    );
                }
            }
        }

        c.pow(&e.0).into()
    }
}

impl Rational {
    // Convert the rational number to a multi-precision float with precision `prec`.
    pub fn to_multi_prec_float(&self, prec: u32) -> Float {
        Float::with_val(
            prec,
            rug::Rational::from((
                self.numerator().to_multi_prec(),
                self.denominator().to_multi_prec(),
            )),
        )
    }
}

/// A float that does linear error propagation.
#[derive(Copy, Clone)]
pub struct ErrorPropagatingFloat<T: NumericalFloatLike> {
    value: T,
    prec: f64,
}

impl<T: NumericalFloatLike> Neg for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        ErrorPropagatingFloat {
            value: -self.value,
            prec: self.prec,
        }
    }
}

impl<T: RealNumberLike> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        // TODO: handle r = 0
        let r = self.value.clone() + &rhs.value;
        ErrorPropagatingFloat {
            prec: (self.get_num().to_f64().abs() * self.prec
                + rhs.get_num().to_f64().abs() * rhs.prec)
                / r.clone().to_f64().abs(),
            value: r,
        }
    }
}

impl<T: RealNumberLike> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: RealNumberLike> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

impl<T: RealNumberLike> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: RealNumberLike> Mul<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value.clone() * &rhs.value,
            prec: self.prec + rhs.prec,
        }
    }
}

impl<T: RealNumberLike + Add<Rational, Output = T>> Add<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Rational) -> Self::Output {
        let v = self.value.to_f64();
        let prec = self.prec * v.abs() / (v + rhs.to_f64()).abs();
        let r = self.value + rhs;
        ErrorPropagatingFloat { prec, value: r }.truncate()
    }
}

impl<T: RealNumberLike + Add<Rational, Output = T>> Sub<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rational) -> Self::Output {
        self + -rhs
    }
}

impl<T: RealNumberLike + Mul<Rational, Output = T>> Mul<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rational) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value * rhs,
            prec: self.prec,
        }
    }
}

impl<T: RealNumberLike + Div<Rational, Output = T>> Div<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Rational) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value.clone() / rhs,
            prec: self.prec,
        }
    }
}

impl<T: RealNumberLike> Mul<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: RealNumberLike> Div<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value.clone() / &rhs.value,
            prec: self.prec + rhs.prec,
        }
    }
}

impl<T: RealNumberLike> Div<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<T: RealNumberLike> AddAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() + rhs;
    }
}

impl<T: RealNumberLike> AddAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: RealNumberLike> SubAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() - rhs;
    }
}

impl<T: RealNumberLike> SubAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: RealNumberLike> MulAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() * rhs;
    }
}

impl<T: RealNumberLike> MulAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.mul_assign(&rhs)
    }
}

impl<T: RealNumberLike> DivAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() / rhs;
    }
}

impl<T: RealNumberLike> DivAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.div_assign(&rhs)
    }
}

impl<T: NumericalFloatLike> ErrorPropagatingFloat<T> {
    /// Create a new precision tracking float with a number of precise decimal digits `prec`.
    /// The `prec` must be smaller than the precision of the underlying float.
    pub fn new(value: T, prec: f64) -> Self {
        ErrorPropagatingFloat {
            value,
            prec: 10f64.pow(-prec),
        }
    }

    /// Get the number.
    #[inline(always)]
    pub fn get_num(&self) -> &T {
        &self.value
    }

    /// Get the precision in number of decimal digits.
    #[inline(always)]
    pub fn get_precision(&self) -> f64 {
        -self.prec.log10()
    }

    /// Truncate the precision to the maximal number of stable decimal digits
    /// of the underlying float.
    #[inline(always)]
    pub fn truncate(mut self) -> Self {
        if self.value.fixed_precision() {
            self.prec = self.prec.max(self.value.get_epsilon());
        }
        self
    }
}

impl<T: NumericalFloatLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let p = self.get_precision() as usize;

        if p == 0 {
            f.write_char('0')
        } else {
            f.write_fmt(format_args!(
                "{0:.1$e}",
                self.value,
                self.get_precision() as usize
            ))
        }
    }
}

impl<T: NumericalFloatLike> Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)?;
        f.write_fmt(format_args!("`{}", self.get_precision()))
    }
}

impl<T: NumericalFloatLike> LowerExp for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<T: NumericalFloatLike> PartialEq for ErrorPropagatingFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: ignore precision for partial equality?
        self.value == other.value
    }
}

impl<T: NumericalFloatLike + PartialOrd> PartialOrd for ErrorPropagatingFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: RealNumberLike> NumericalFloatLike for ErrorPropagatingFloat<T> {
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        a.clone() * b + self
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn zero(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.zero(),
            prec: 2f64.pow(-(self.value.get_precision() as f64)),
        }
    }

    fn new_zero() -> Self {
        ErrorPropagatingFloat {
            value: T::new_zero(),
            prec: 2f64.powi(-53),
        }
    }

    fn one(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.one(),
            prec: 2f64.pow(-(self.value.get_precision() as f64)),
        }
    }

    fn pow(&self, e: u64) -> Self {
        ErrorPropagatingFloat {
            value: self.value.pow(e),
            prec: self.prec * e as f64,
        }
    }

    fn inv(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.inv(),
            prec: self.prec,
        }
    }

    fn from_usize(&self, a: usize) -> Self {
        ErrorPropagatingFloat {
            value: self.value.from_usize(a),
            prec: self.prec,
        }
    }

    fn from_i64(&self, a: i64) -> Self {
        ErrorPropagatingFloat {
            value: self.value.from_i64(a),
            prec: self.prec,
        }
    }

    fn get_precision(&self) -> u32 {
        // return the precision of the underlying float instead
        // of the current tracked precision
        self.value.get_precision()
    }

    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.value.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        ErrorPropagatingFloat {
            value: self.value.sample_unit(rng),
            prec: self.prec,
        }
    }
}

impl<T: RealNumberLike> SingleFloat for ErrorPropagatingFloat<T> {
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }
}

impl<T: RealNumberLike> RealNumberLike for ErrorPropagatingFloat<T> {
    fn to_usize_clamped(&self) -> usize {
        self.value.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }
}

impl<T: Real + RealNumberLike> Real for ErrorPropagatingFloat<T> {
    fn norm(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.norm(),
            prec: self.prec,
        }
        .truncate()
    }

    fn sqrt(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.sqrt(),
            prec: self.prec / 2.,
        }
        .truncate()
    }

    fn log(&self) -> Self {
        let r = self.value.log();
        ErrorPropagatingFloat {
            prec: self.prec / r.clone().to_f64().abs(),
            value: r,
        }
        .truncate()
    }

    fn exp(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.value.to_f64().abs() * self.prec,
            value: self.value.exp(),
        }
        .truncate()
    }

    fn sin(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() / self.value.tan().to_f64().abs(),
            value: self.value.sin(),
        }
        .truncate()
    }

    fn cos(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() * self.value.tan().to_f64().abs(),
            value: self.value.cos(),
        }
        .truncate()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        let tt = t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() * (tt.inv() + tt),
            value: t,
        }
        .truncate()
    }

    fn asin(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.asin();
        let tt = (1. - v * v).sqrt() * t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn acos(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.acos();
        let tt = (1. - v * v).sqrt() * t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn atan2(&self, x: &Self) -> Self {
        let t = self.value.atan2(&x.value);
        let r = self.clone() / x;
        let r2 = r.value.to_f64().abs();

        let tt = (1. + r2 * r2) * t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: r.prec * r2 / tt,
            value: t,
        }
        .truncate()
    }

    fn sinh(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() / self.value.tanh().to_f64().abs(),
            value: self.value.sinh(),
        }
        .truncate()
    }

    fn cosh(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() * self.value.tanh().to_f64().abs(),
            value: self.value.cosh(),
        }
        .truncate()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let tt = t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * self.value.to_f64().abs() * (tt.inv() - tt),
            value: t,
        }
        .truncate()
    }

    fn asinh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.asinh();
        let tt = (1. + v * v).sqrt() * t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn acosh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.acosh();
        let tt = (v * v - 1.).sqrt() * t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn atanh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.atanh();
        let tt = (1. - v * v) * t.to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn powf(&self, e: &Self) -> Self {
        let v = self.value.to_f64().abs();
        ErrorPropagatingFloat {
            value: self.value.powf(&e.value),
            prec: (self.prec + e.prec * v.ln().abs()) * e.value.clone().to_f64().abs(),
        }
        .truncate()
    }
}

macro_rules! simd_impl {
    ($t:ty, $p:ident) => {
        impl NumericalFloatLike for $t {
            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                *self * *a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -self
            }

            #[inline(always)]
            fn zero(&self) -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn new_zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn one(&self) -> Self {
                Self::ONE
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // FIXME: use binary exponentiation
                debug_assert!(e <= i32::MAX as u64);
                (*self).powf(e as f64)
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                Self::ONE / *self
            }

            #[inline(always)]
            fn from_usize(&self, a: usize) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn from_i64(&self, a: i64) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn get_precision(&self) -> u32 {
                53
            }

            #[inline(always)]
            fn get_epsilon(&self) -> f64 {
                f64::EPSILON / 2.
            }

            #[inline(always)]
            fn fixed_precision(&self) -> bool {
                true
            }

            fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
                Self::from(rng.gen::<f64>())
            }
        }

        impl Real for $t {
            #[inline(always)]
            fn norm(&self) -> Self {
                (*self).abs()
            }

            #[inline(always)]
            fn sqrt(&self) -> Self {
                (*self).sqrt()
            }

            #[inline(always)]
            fn log(&self) -> Self {
                (*self).ln()
            }

            #[inline(always)]
            fn exp(&self) -> Self {
                (*self).exp()
            }

            #[inline(always)]
            fn sin(&self) -> Self {
                (*self).sin()
            }

            #[inline(always)]
            fn cos(&self) -> Self {
                (*self).cos()
            }

            #[inline(always)]
            fn tan(&self) -> Self {
                (*self).tan()
            }

            #[inline(always)]
            fn asin(&self) -> Self {
                (*self).asin()
            }

            #[inline(always)]
            fn acos(&self) -> Self {
                (*self).acos()
            }

            #[inline(always)]
            fn atan2(&self, x: &Self) -> Self {
                (*self).atan2(*x)
            }

            #[inline(always)]
            fn sinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn cosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn tanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn asinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn acosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn atanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn powf(&self, e: &Self) -> Self {
                (*self).$p(*e)
            }
        }

        impl From<&Rational> for $t {
            fn from(value: &Rational) -> Self {
                value.to_f64().into()
            }
        }
    };
}

simd_impl!(f64x2, pow_f64x2);
simd_impl!(f64x4, pow_f64x4);

impl From<Float> for Rational {
    fn from(value: Float) -> Self {
        value.to_rational()
    }
}

impl LowerExp for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lower-exp is ignored for integers
        f.write_fmt(format_args!("{}", self))
    }
}

impl NumericalFloatLike for Rational {
    fn mul_add(&self, a: &Self, c: &Self) -> Self {
        &(self * a) + c
    }

    fn neg(&self) -> Self {
        self.neg()
    }

    fn zero(&self) -> Self {
        Self::zero()
    }

    fn new_zero() -> Self {
        Self::zero()
    }

    fn one(&self) -> Self {
        Self::one()
    }

    fn pow(&self, e: u64) -> Self {
        self.pow(e)
    }

    fn inv(&self) -> Self {
        self.inv()
    }

    fn from_usize(&self, a: usize) -> Self {
        a.into()
    }

    fn from_i64(&self, a: i64) -> Self {
        a.into()
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        0.
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        true
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let rng1 = rng.gen::<i64>();
        let rng2 = rng.gen::<i64>();

        if rng1 > rng2 {
            (rng2, rng1).into()
        } else {
            (rng1, rng2).into()
        }
    }
}

impl SingleFloat for Rational {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.is_one()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        true
    }
}

impl RealNumberLike for Rational {
    fn to_usize_clamped(&self) -> usize {
        f64::from(self).to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        f64::from(self)
    }
}

#[derive(Copy, Clone, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Complex {
            re: T::default(),
            im: T::default(),
        }
    }
}

impl<T: ConstructibleFloat + Real> ConstructibleFloat for Complex<T> {
    fn new_from_i64(a: i64) -> Self {
        Complex {
            re: T::new_from_i64(a),
            im: T::new_zero(),
        }
    }

    fn new_from_usize(a: usize) -> Self {
        Complex {
            re: T::new_from_usize(a),
            im: T::new_zero(),
        }
    }

    fn new_one() -> Self {
        Complex {
            re: T::new_one(),
            im: T::new_zero(),
        }
    }

    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Complex {
            re: T::new_sample_unit(rng),
            im: T::new_zero(),
        }
    }
}

impl<T: NumericalFloatLike> Complex<T> {
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    #[inline]
    pub fn new_zero() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    #[inline]
    pub fn new_i() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_one(),
        }
    }

    #[inline]
    pub fn one(&self) -> Self {
        Complex {
            re: self.re.one(),
            im: self.im.zero(),
        }
    }

    #[inline]
    pub fn conj(&self) -> Self {
        Complex {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }

    #[inline]
    pub fn zero(&self) -> Self {
        Complex {
            re: self.re.zero(),
            im: self.im.zero(),
        }
    }

    #[inline]
    pub fn i(&self) -> Complex<T> {
        Complex {
            re: self.re.zero(),
            im: self.im.one(),
        }
    }

    #[inline]
    pub fn norm_squared(&self) -> T {
        self.re.clone() * &self.re + self.im.clone() * &self.im
    }
}

impl<T: Real> Complex<T> {
    #[inline]
    pub fn arg(&self) -> T {
        self.im.atan2(&self.re)
    }

    #[inline]
    pub fn to_polar_coordinates(self) -> (T, T) {
        (self.norm_squared().sqrt(), self.arg())
    }

    #[inline]
    pub fn from_polar_coordinates(r: T, phi: T) -> Complex<T> {
        Complex::new(r.clone() * phi.cos(), r.clone() * phi.sin())
    }
}

impl<T: NumericalFloatLike> Add<Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: NumericalFloatLike> Add<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<T: NumericalFloatLike> Add<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re + &rhs.re, self.im + &rhs.im)
    }
}

impl<T: NumericalFloatLike> Add<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Add<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'a, T: NumericalFloatLike> Add<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T: NumericalFloatLike> Add<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T: NumericalFloatLike> Add<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: NumericalFloatLike> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<T: NumericalFloatLike> AddAssign<T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T: NumericalFloatLike> AddAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }
}

impl<T: NumericalFloatLike> AddAssign<&T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        self.re += rhs;
    }
}

impl<T: NumericalFloatLike> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: NumericalFloatLike> Sub<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<T: NumericalFloatLike> Sub<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re - &rhs.re, self.im - &rhs.im)
    }
}

impl<T: NumericalFloatLike> Sub<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Sub<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'a, T: NumericalFloatLike> Sub<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T: NumericalFloatLike> Sub<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T: NumericalFloatLike> Sub<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: NumericalFloatLike> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<T: NumericalFloatLike> SubAssign<T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T: NumericalFloatLike> SubAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }
}

impl<T: NumericalFloatLike> SubAssign<&T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        self.re -= rhs;
    }
}

impl<T: NumericalFloatLike> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<T: NumericalFloatLike> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re * &rhs, self.im * &rhs)
    }
}

impl<T: NumericalFloatLike> Mul<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        Complex::new(
            self.re.clone() * &rhs.re - self.im.clone() * &rhs.im,
            self.re.clone() * &rhs.im + self.im.clone() * &rhs.re,
        )
    }
}

impl<T: NumericalFloatLike> Mul<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Mul<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'a, T: NumericalFloatLike> Mul<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T: NumericalFloatLike> Mul<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T: NumericalFloatLike> Mul<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: NumericalFloatLike> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<&T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> Div for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<T: NumericalFloatLike> Div<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re / &rhs, self.im / &rhs)
    }
}

impl<T: NumericalFloatLike> Div<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.im.clone() * &rhs.re - self.re.clone() * &rhs.im;
        Complex::new(re / &n, im / &n)
    }
}

impl<T: NumericalFloatLike> Div<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Div<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'a, T: NumericalFloatLike> Div<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'b, T: NumericalFloatLike> Div<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'b, T: NumericalFloatLike> Div<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: NumericalFloatLike> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<T> for Complex<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<&Complex<T>> for Complex<T> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<&T> for Complex<T> {
    fn div_assign(&mut self, rhs: &T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> Neg for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: NumericalFloatLike> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.re, f)?;
        f.write_char('+')?;
        Display::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: NumericalFloatLike> Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Debug::fmt(&self.re, f)?;
        f.write_char('+')?;
        Debug::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: NumericalFloatLike> LowerExp for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        LowerExp::fmt(&self.re, f)?;
        f.write_char('+')?;
        LowerExp::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: SingleFloat> SingleFloat for Complex<T> {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        true
    }
}

impl<T: NumericalFloatLike> NumericalFloatLike for Complex<T> {
    #[inline]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() + (a.clone() * b)
    }

    #[inline]
    fn neg(&self) -> Self {
        Complex {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }

    #[inline]
    fn zero(&self) -> Self {
        Complex {
            re: self.re.zero(),
            im: self.im.zero(),
        }
    }

    fn new_zero() -> Self {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    fn one(&self) -> Self {
        Complex {
            re: self.re.one(),
            im: self.im.zero(),
        }
    }

    fn pow(&self, e: u64) -> Self {
        // TODO: use binary exponentiation
        let mut r = self.one();
        for _ in 0..e {
            r *= self;
        }
        r
    }

    fn inv(&self) -> Self {
        let n = self.norm_squared();
        Complex::new(self.re.clone() / &n, -self.im.clone() / &n)
    }

    fn from_usize(&self, a: usize) -> Self {
        Complex {
            re: self.re.from_usize(a),
            im: self.im.zero(),
        }
    }

    fn from_i64(&self, a: i64) -> Self {
        Complex {
            re: self.re.from_i64(a),
            im: self.im.zero(),
        }
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.re.get_precision().min(self.im.get_precision())
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        (2.0f64).powi(-(self.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.re.fixed_precision() || self.im.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        Complex {
            re: self.re.sample_unit(rng),
            im: self.im.zero(),
        }
    }
}

/// Following the same conventions and formulas as num::Complex.
impl<T: Real> Real for Complex<T> {
    #[inline]
    fn norm(&self) -> Self {
        Complex::new(self.norm_squared().sqrt(), self.im.zero())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        let (r, phi) = self.clone().to_polar_coordinates();
        Complex::from_polar_coordinates(r.sqrt(), phi / self.re.from_usize(2))
    }

    #[inline]
    fn log(&self) -> Self {
        Complex::new(self.norm().re.log(), self.arg())
    }

    #[inline]
    fn exp(&self) -> Self {
        let r = self.re.exp();
        Complex::new(r.clone() * self.im.cos(), r * self.im.sin())
    }

    #[inline]
    fn sin(&self) -> Self {
        Complex::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    #[inline]
    fn cos(&self) -> Self {
        Complex::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    #[inline]
    fn tan(&self) -> Self {
        let (r, i) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = r.cos() + i.cosh();
        Self::new(r.sin() / &m, i.sinh() / m)
    }

    #[inline]
    fn asin(&self) -> Self {
        let i = self.i();
        -i.clone() * ((self.one() - self.clone() * self).sqrt() + i * self).log()
    }

    #[inline]
    fn acos(&self) -> Self {
        let i = self.i();
        -i.clone() * (i * (self.one() - self.clone() * self).sqrt() + self).log()
    }

    #[inline]
    fn atan2(&self, x: &Self) -> Self {
        // TODO: pick proper branch
        let r = self.clone() / x;
        let i = self.i();
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + &i * &r).log() - (&one - &i * r).log()) / (two * i)
    }

    #[inline]
    fn sinh(&self) -> Self {
        Complex::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    #[inline]
    fn cosh(&self) -> Self {
        Complex::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    #[inline]
    fn tanh(&self) -> Self {
        let (two_re, two_im) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = two_re.cosh() + two_im.cos();
        Self::new(two_re.sinh() / &m, two_im.sin() / m)
    }

    #[inline]
    fn asinh(&self) -> Self {
        let one = self.one();
        (self.clone() + (one + self.clone() * self).sqrt()).log()
    }

    #[inline]
    fn acosh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        &two * (((self.clone() + &one) / &two).sqrt() + ((self.clone() - one) / &two).sqrt()).log()
    }

    #[inline]
    fn atanh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + self).log() - (one - self).log()) / two
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        if e.re == self.re.zero() && e.im == self.im.zero() {
            self.one()
        } else if e.im == self.im.zero() {
            let (r, phi) = self.clone().to_polar_coordinates();
            Self::from_polar_coordinates(r.powf(&e.re), phi * e.re.clone())
        } else {
            (e * self.log()).exp()
        }
    }
}

impl<'a, T: NumericalFloatLike + From<&'a Rational>> From<&'a Rational> for Complex<T> {
    fn from(value: &'a Rational) -> Self {
        let c: T = value.into();
        let zero = c.zero();
        Complex::new(c, zero)
    }
}

#[cfg(test)]
mod test {
    use rug::Complete;

    use super::*;

    #[test]
    fn double() {
        let a = 5.;
        let b = 7.;

        let r = a.sqrt() + b.log() + b.sin() - a.cos() + b.tan() - 0.3.asin() + 0.5.acos()
            - a.atan2(b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - 0.7.asinh()
            + b.acosh() / 0.4.atanh()
            + b.powf(a);
        assert_eq!(r, 17293.219725825093);
    }

    #[test]
    fn error_propagation() {
        let a = ErrorPropagatingFloat::new(5., 16.);
        let b = ErrorPropagatingFloat::new(7., 16.);
        let c = ErrorPropagatingFloat::new(0.3, 16.);
        let d = ErrorPropagatingFloat::new(0.5, 16.);
        let e = ErrorPropagatingFloat::new(0.7, 16.);
        let f = ErrorPropagatingFloat::new(0.4, 16.);

        let r = a.sqrt() + b.log() + b.sin() - a.cos() + b.tan() - c.asin() + d.acos()
            - a.atan2(&b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - e.asinh()
            + b.acosh() / f.atanh()
            + b.powf(&a);
        assert_eq!(r.value, 17293.219725825093);
        // error is 14.836811363436391 when the f64 could have theoretically grown in between
        assert_eq!(r.get_precision(), 14.836795991431746);
    }

    #[test]
    fn error_truncation() {
        let a = ErrorPropagatingFloat::new(0.0000000123456789, 9.)
            .exp()
            .log();
        assert_eq!(a.get_precision(), 8.046104745509947);
    }

    #[test]
    fn large_cancellation() {
        let a = ErrorPropagatingFloat::new(Float::with_val(200, 1e-50), 60.);
        let r = (a.exp() - a.one()) / a;
        println!("{}", r.value.prec());
        assert_eq!(format!("{}", r), "1.000000000e0");
        assert_eq!(r.get_precision(), 10.205999132807323);
    }

    #[test]
    fn complex() {
        let a = Complex::new(1., 2.);
        let b: Complex<f64> = Complex::new(3., 4.);

        let r = &a.sqrt() + &b.log() - a.exp() + b.sin() - a.cos() + b.tan() - a.asin() + b.acos()
            - a.atan2(&b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - a.asinh()
            + &b.acosh() / a.atanh()
            + b.powf(&a);
        assert_eq!(r, Complex::new(0.1924131450685842, -39.83285329561913));
    }

    #[test]
    fn float_int() {
        let a = Float::with_val(53, 0.123456789123456);
        let b = a / 10i64 * 1300;
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 12345.6789);
        let b = a - 12345;
        assert_eq!(b.get_precision(), 40);
    }

    #[test]
    fn float_rational() {
        let a = Float::with_val(53, 1000);
        let b: Float = a * Rational::from((-3001, 30)) / Rational::from((1, 2));
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 1000);
        let b: Float = a + Rational::from(
            rug::Rational::parse("-3128903712893789123789213781279/30890231478123748912372")
                .unwrap()
                .complete(),
        );
        assert_eq!(b.get_precision(), 71);
    }

    #[test]
    fn float_cancellation() {
        let a = Float::with_val(10, 1000);
        let b = a + 10i64;
        assert_eq!(b.get_precision(), 11);

        let a = Float::with_val(53, -1001);
        let b = a + 1000i64;
        assert_eq!(b.get_precision(), 45); // tight bound is 44 digits

        let a = Float::with_val(53, 1000);
        let b = Float::with_val(100, -1001);
        let c = a + b;
        assert_eq!(c.get_precision(), 45); // tight bound is 44 digits

        let a = Float::with_val(20, 1000);
        let b = Float::with_val(40, 1001);
        let c = a + b;
        assert_eq!(c.get_precision(), 22);

        let a = Float::with_val(4, 18.0);
        let b = Float::with_val(24, -17.9199009);
        let c = a + b;
        assert_eq!(c.get_precision(), 1); // capped at 1

        let a = Float::with_val(24, 18.00000);
        let b = Float::with_val(24, -17.992);
        let c = a + b;
        assert_eq!(c.get_precision(), 14);
    }

    #[test]
    fn float_growth() {
        let a = Float::with_val(53, 0.01);
        let b = a.exp();
        assert_eq!(b.get_precision(), 60);

        let a = Float::with_val(53, 0.8);
        let b = a.exp();
        assert_eq!(b.get_precision(), 54);

        let a = Float::with_val(53, 200);
        let b = a.exp();
        assert_eq!(b.get_precision(), 46);

        let a = Float::with_val(53, 0.8);
        let b = a.log();
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 300.0);
        let b = a.log();
        assert_eq!(b.get_precision(), 57);

        let a = Float::with_val(53, 1.5709);
        let b = a.sin();
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 14.);
        let b = a.tanh();
        assert_eq!(b.get_precision(), 66);

        let a = Float::with_val(53, 1.);
        let b = Float::with_val(53, 0.1);
        let b = a.powf(&b);
        assert_eq!(b.get_precision(), 57);

        let a = Float::with_val(53, 1.);
        let b = Float::with_val(200, 0.1);
        let b = a.powf(&b);
        assert_eq!(b.get_precision(), 57);
    }

    #[test]
    fn powf_prec() {
        let a = Float::with_val(53, 10.);
        let b = Float::with_val(200, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57);

        let a = Float::with_val(200, 2.);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 58);

        let a = Float::with_val(53, 3.);
        let b = Float::with_val(200, 20.);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 49);

        let a = Float::with_val(200, 1.);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57); // a=1 is anomalous

        let a = Float::with_val(200, 0.4);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57);
    }
}
