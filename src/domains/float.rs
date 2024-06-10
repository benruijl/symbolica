use core::fmt;
use std::{
    f64::consts::LOG2_10,
    fmt::{Display, LowerExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use wide::{f64x2, f64x4};

use super::rational::Rational;
use rug::{
    ops::{CompleteRound, Pow},
    Assign, Float as MultiPrecisionFloat, Rational as MultiPrecisionRational,
};

pub trait NumericalFloatLike:
    PartialEq
    + Clone
    + std::fmt::Debug
    + std::fmt::LowerExp
    + std::fmt::Display
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
    fn mul_add(&self, a: &Self, a: &Self) -> Self;
    fn neg(&self) -> Self;
    fn norm(&self) -> Self;
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

    /// Sample a point on the interval [0, 1].
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self;
}

pub trait NumericalFloatComparison: NumericalFloatLike + PartialOrd {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_finite(&self) -> bool;
    fn max(&self, other: &Self) -> Self;

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
    fn powf(&self, e: Self) -> Self;
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
    fn norm(&self) -> Self {
        f64::abs(*self)
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

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        rng.gen()
    }
}

impl NumericalFloatComparison for f64 {
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

    fn max(&self, other: &Self) -> Self {
        (*self).max(*other)
    }

    fn to_usize_clamped(&self) -> usize {
        *self as usize
    }

    fn to_f64(&self) -> f64 {
        *self
    }
}

impl ConstructibleFloat for f64 {
    fn new_one() -> Self {
        1.
    }

    fn new_from_usize(a: usize) -> Self {
        a as f64
    }

    fn new_from_i64(a: i64) -> Self {
        a as f64
    }

    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl Real for f64 {
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
    fn powf(&self, e: f64) -> Self {
        (*self).powf(e)
    }
}

impl From<&Rational> for f64 {
    fn from(value: &Rational) -> Self {
        match value {
            Rational::Natural(n, d) => *n as f64 / *d as f64,
            Rational::Large(l) => l.to_f64(),
        }
    }
}

impl From<Rational> for f64 {
    fn from(value: Rational) -> Self {
        match value {
            Rational::Natural(n, d) => n as f64 / d as f64,
            Rational::Large(l) => l.to_f64(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct Float(MultiPrecisionFloat);

impl fmt::Debug for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self.0))
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}", self.0))
    }
}

impl LowerExp for Float {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{:e}", self.0))
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

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 + &rhs.0).into()
    }
}

impl Add<Float> for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            (rhs.0 + self.0).into()
        } else {
            (self.0 + rhs.0).into()
        }
    }
}

impl Sub<&Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 - &rhs.0).into()
    }
}

impl Sub<Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 - rhs.0).into()
    }
}

impl Mul<&Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 * &rhs.0).into()
    }
}

impl Mul<Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
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
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / &rhs.0).into()
    }
}

impl Div<Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: &Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.add_assign(&rhs.0);
    }
}

impl AddAssign<Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.add_assign(rhs.0);
    }
}

impl SubAssign<&Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: &Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.sub_assign(&rhs.0);
    }
}

impl SubAssign<Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.sub_assign(rhs.0);
    }
}

impl MulAssign<&Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: &Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(&rhs.0);
    }
}

impl MulAssign<Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(rhs.0);
    }
}

impl DivAssign<&Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: &Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(&rhs.0);
    }
}

impl DivAssign<Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: Float) {
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(rhs.0);
    }
}

impl Add<i64> for Float {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: i64) -> Self::Output {
        if self.prec() < 63 {
            self.set_prec(63);
        }

        (self.0 + rhs).into()
    }
}

impl Mul<i64> for Float {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: i64) -> Self::Output {
        if self.prec() < 63 {
            self.set_prec(63);
        }

        (self.0 * rhs).into()
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

    /// Parse a float from a string. If `prec` is `None`, the precision is derived from the string, with
    /// a minimum of 53 bits (f64 precision).
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, rug::float::ParseFloatError> {
        if let Some(prec) = prec {
            return Ok(Float(MultiPrecisionFloat::parse(s)?.complete(prec)));
        } else {
            let prec = ((s.len() as f64 * LOG2_10).ceil() as u32).max(53);
            Ok(Float(MultiPrecisionFloat::parse(s)?.complete(prec)))
        }
    }

    pub fn serialize(&self) -> String {
        self.0.to_string_radix(16, None)
    }

    pub fn to_rational(&self) -> Rational {
        Rational::from_large(self.0.to_rational().unwrap())
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
        self.0
            .mul_add_ref(&a.0, &b.0)
            .complete(self.prec().max(a.prec()).max(b.prec()))
            .into()
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0.clone()).into()
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.clone().abs().into()
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
        rug::ops::Pow::pow(&self.0, e).complete(self.prec()).into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.clone().recip().into()
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        Float::with_val(self.prec(), a)
    }

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

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let f: f64 = rng.gen();
        Float::with_val(self.prec(), f)
    }
}

impl NumericalFloatComparison for Float {
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

    fn max(&self, other: &Self) -> Self {
        if self.0 > other.0 {
            self.clone()
        } else {
            other.clone()
        }
    }

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
    fn sqrt(&self) -> Self {
        self.0.clone().sqrt().into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        self.0.clone().ln().into()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        self.0.clone().exp().into()
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
    fn powf(&self, e: Self) -> Self {
        rug::ops::Pow::pow(&self.0, e.0).into()
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

impl<T: NumericalFloatComparison> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
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

impl<T: NumericalFloatComparison> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: NumericalFloatComparison> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

impl<T: NumericalFloatComparison> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: NumericalFloatComparison> Mul<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value.clone() * &rhs.value,
            prec: self.prec + rhs.prec,
        }
    }
}

impl<T: NumericalFloatComparison> Mul<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: NumericalFloatComparison> Div<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            value: self.value.clone() / &rhs.value,
            prec: self.prec + rhs.prec, // TODO: check
        }
    }
}

impl<T: NumericalFloatComparison> Div<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<T: NumericalFloatComparison> AddAssign<&ErrorPropagatingFloat<T>>
    for ErrorPropagatingFloat<T>
{
    #[inline]
    fn add_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() + rhs;
    }
}

impl<T: NumericalFloatComparison> AddAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: NumericalFloatComparison> SubAssign<&ErrorPropagatingFloat<T>>
    for ErrorPropagatingFloat<T>
{
    #[inline]
    fn sub_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() - rhs;
    }
}

impl<T: NumericalFloatComparison> SubAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: NumericalFloatComparison> MulAssign<&ErrorPropagatingFloat<T>>
    for ErrorPropagatingFloat<T>
{
    #[inline]
    fn mul_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() * rhs;
    }
}

impl<T: NumericalFloatComparison> MulAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.mul_assign(&rhs)
    }
}

impl<T: NumericalFloatComparison> DivAssign<&ErrorPropagatingFloat<T>>
    for ErrorPropagatingFloat<T>
{
    #[inline]
    fn div_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() / rhs;
    }
}

impl<T: NumericalFloatComparison> DivAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
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
        self.prec = self.prec.max(self.value.get_epsilon());
        self
    }
}

impl<T: NumericalFloatLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = self.get_precision() as usize;

        if p == 0 {
            return f.write_fmt(format_args!("0"));
        } else {
            f.write_fmt(format_args!(
                "{0:.1$e}",
                self.value,
                self.get_precision() as usize - 1
            ))
        }
    }
}

impl<T: NumericalFloatLike> fmt::Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("{}`{}", self.value, self.get_precision()))
    }
}

impl<T: NumericalFloatLike> fmt::LowerExp for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as Display>::fmt(&self, f)
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

impl<T: NumericalFloatComparison> NumericalFloatLike for ErrorPropagatingFloat<T> {
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        a.clone() * b + self
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn norm(&self) -> Self {
        todo!()
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

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        ErrorPropagatingFloat {
            value: self.value.sample_unit(rng),
            prec: self.prec,
        }
    }
}

impl<T: NumericalFloatComparison + Into<f64>> NumericalFloatComparison
    for ErrorPropagatingFloat<T>
{
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    fn max(&self, other: &Self) -> Self {
        if self.value > other.value {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn to_usize_clamped(&self) -> usize {
        self.value.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }
}

impl<T: Real + NumericalFloatComparison> Real for ErrorPropagatingFloat<T> {
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
            value: self.value.exp(),
            prec: self.value.clone().to_f64().abs() * self.prec,
        }
        .truncate()
    }

    fn sin(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.clone().to_f64().abs() / self.value.tan().to_f64().abs(),
            value: self.value.sin(),
        }
        .truncate()
    }

    fn cos(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.clone().to_f64().abs() * self.value.tan().to_f64().abs(),
            value: self.value.cos(),
        }
        .truncate()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        let tt = t.clone().to_f64();
        ErrorPropagatingFloat {
            prec: self.prec * self.value.clone().to_f64().abs() * (tt.inv() + tt),
            value: t,
        }
        .truncate()
    }

    fn asin(&self) -> Self {
        let v = self.value.clone().to_f64();
        let t = self.value.asin();
        let tt = (1. - v * v).sqrt() * t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn acos(&self) -> Self {
        let v = self.value.clone().to_f64();
        let t = self.value.acos();
        let tt = (1. - v * v).sqrt() * t.clone().to_f64().abs();
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
            prec: self.prec * self.value.clone().to_f64().abs() / self.value.tanh().to_f64().abs(),
            value: self.value.sinh(),
        }
        .truncate()
    }

    fn cosh(&self) -> Self {
        ErrorPropagatingFloat {
            prec: self.prec * self.value.clone().to_f64().abs() * self.value.tanh().to_f64().abs(),
            value: self.value.cosh(),
        }
        .truncate()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let tt = t.clone().to_f64();
        ErrorPropagatingFloat {
            prec: self.prec * self.value.clone().to_f64().abs() * (tt.inv() - tt),
            value: t,
        }
        .truncate()
    }

    fn asinh(&self) -> Self {
        let v = self.value.clone().to_f64();
        let t = self.value.asinh();
        let tt = (1. + v * v).sqrt() * t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn acosh(&self) -> Self {
        let v = self.value.clone().to_f64();
        let t = self.value.acosh();
        let tt = (v * v - 1.).sqrt() * t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn atanh(&self) -> Self {
        let v = self.value.clone().to_f64();
        let t = self.value.atanh();
        let tt = (1. - v * v) * t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            prec: self.prec * v.abs() / tt,
            value: t,
        }
        .truncate()
    }

    fn powf(&self, e: Self) -> Self {
        let v = self.value.clone().to_f64().abs();
        ErrorPropagatingFloat {
            value: self.value.powf(e.value.clone()),
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
            fn norm(&self) -> Self {
                (*self).abs()
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

            fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
                Self::from(rng.gen::<f64>())
            }
        }

        impl Real for $t {
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
            fn powf(&self, e: Self) -> Self {
                (*self).$p(e)
            }
        }

        impl From<&Rational> for $t {
            fn from(value: &Rational) -> Self {
                match value {
                    Rational::Natural(n, d) => Self::from(*n as f64 / *d as f64),
                    Rational::Large(l) => Self::from(l.to_f64()),
                }
            }
        }
    };
}

simd_impl!(f64x2, pow_f64x2);
simd_impl!(f64x4, pow_f64x4);

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

    fn norm(&self) -> Self {
        self.abs()
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
        if a < i64::MAX as usize {
            Rational::Natural(a as i64, 1)
        } else {
            Rational::Large(MultiPrecisionRational::from(a))
        }
    }

    fn from_i64(&self, a: i64) -> Self {
        Rational::Natural(a, 1)
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        0.
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let rng1 = rng.gen::<i64>();
        let rng2 = rng.gen::<i64>();

        if rng1 > rng2 {
            Rational::Natural(rng2, rng1)
        } else {
            Rational::Natural(rng1, rng2)
        }
    }
}

impl NumericalFloatComparison for Rational {
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

    fn max(&self, other: &Self) -> Self {
        if self > other {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn to_usize_clamped(&self) -> usize {
        f64::from(self).to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        f64::from(self)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub struct Complex<T: Real> {
    pub re: T,
    pub im: T,
}

impl<T: Real> Complex<T> {
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
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

impl<T: Real> Add<Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Real> Add<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re + &rhs.re, self.im + &rhs.im)
    }
}

impl<'a, 'b, T: Real> Add<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T: Real> Add<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: Real> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<T: Real> AddAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }
}

impl<T: Real> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: Real> Sub<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re - &rhs.re, self.im - &rhs.im)
    }
}

impl<'a, 'b, T: Real> Sub<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T: Real> Sub<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: Real> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<T: Real> SubAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }
}

impl<T: Real> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<T: Real> Mul<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        Complex::new(
            self.re.clone() * &rhs.re - self.im.clone() * &rhs.im,
            self.re.clone() * &rhs.im + self.im.clone() * &rhs.re,
        )
    }
}

impl<'a, 'b, T: Real> Mul<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T: Real> Mul<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: Real> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: Real> MulAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: Real> Div for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<T: Real> Div<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.im.clone() * &rhs.re - self.re.clone() * &rhs.im;
        Complex::new(re / &n, im / &n)
    }
}

impl<'a, 'b, T: Real> Div<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'b, T: Real> Div<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: Real> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: Real> DivAssign<&Complex<T>> for Complex<T> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: Real> Neg for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: Real> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({}+{}i)", self.re, self.im))
    }
}

impl<T: Real> std::fmt::Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({:?}+{:?}i)", self.re, self.im))
    }
}

impl<T: Real> LowerExp for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("({:e}+{:e}i)", self.re, self.im))
    }
}

impl<T: Real> NumericalFloatLike for Complex<T> {
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
    fn norm(&self) -> Self {
        Complex::new(self.norm_squared().sqrt(), self.im.zero())
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
    fn powf(&self, e: Self) -> Self {
        if e.re == self.re.zero() && e.im == self.im.zero() {
            return self.one();
        } else if e.im == self.im.zero() {
            let (r, phi) = self.clone().to_polar_coordinates();
            Self::from_polar_coordinates(r.powf(e.re.clone()), phi * e.re)
        } else {
            (e * self.log()).exp()
        }
    }
}

impl<'a, T: Real + From<&'a Rational>> From<&'a Rational> for Complex<T> {
    fn from(value: &'a Rational) -> Self {
        let c: T = value.into();
        let zero = c.zero();
        Complex::new(c, zero)
    }
}

#[cfg(test)]
mod test {
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
            + b.powf(a);
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
        assert_eq!(
            format!("{}", r),
            "9.9999999996329025378948944031934242812314589571096384477391833e-1"
        );
        assert_eq!(r.get_precision(), 9.904969137116314);
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
            + b.powf(a);
        assert_eq!(r, Complex::new(0.1924131450685842, -39.83285329561913));
    }
}
