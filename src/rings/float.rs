use wide::{f64x2, f64x4};

use super::rational::Rational;

// TODO: add more operators as bounds
pub trait NumericalFloatLike:
    PartialEq
    + Clone
    + std::fmt::Display
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::MulAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
{
    fn add(&self, a: &Self) -> Self;
    fn sub(&self, a: &Self) -> Self;
    fn mul(&self, a: &Self) -> Self;
    fn add_assign(&mut self, a: &Self);
    fn sub_assign(&mut self, a: &Self);
    fn mul_assign(&mut self, a: &Self);
    fn mul_add(&self, a: &Self, a: &Self) -> Self;
    fn neg(&self) -> Self;
    fn abs(&self) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn pow(&self, e: u64) -> Self;
    fn inv(&self) -> Self;
}

pub trait NumericalFloatComparison: NumericalFloatLike {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
}

impl NumericalFloatLike for f64 {
    #[inline(always)]
    fn add(&self, a: &Self) -> Self {
        self + a
    }

    #[inline(always)]
    fn sub(&self, a: &Self) -> Self {
        self - a
    }

    #[inline(always)]
    fn mul(&self, a: &Self) -> Self {
        self * a
    }

    #[inline(always)]
    fn add_assign(&mut self, a: &Self) {
        *self += a
    }

    #[inline(always)]
    fn sub_assign(&mut self, a: &Self) {
        *self -= a
    }

    #[inline(always)]
    fn mul_assign(&mut self, a: &Self) {
        *self *= a
    }

    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        f64::mul_add(*self, *a, *b)
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn abs(&self) -> Self {
        f64::abs(*self)
    }

    #[inline(always)]
    fn zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one() -> Self {
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
}

impl From<&Rational> for f64 {
    fn from(value: &Rational) -> Self {
        match value {
            Rational::Natural(n, d) => *n as f64 / *d as f64,
            Rational::Large(l) => l.to_f64(),
        }
    }
}

macro_rules! simd_impl {
    ($t:ty) => {
        impl NumericalFloatLike for $t {
            #[inline(always)]
            fn add(&self, a: &Self) -> Self {
                *self + *a
            }

            #[inline(always)]
            fn sub(&self, a: &Self) -> Self {
                *self - *a
            }

            #[inline(always)]
            fn mul(&self, a: &Self) -> Self {
                *self * a
            }

            #[inline(always)]
            fn add_assign(&mut self, a: &Self) {
                *self += a
            }

            #[inline(always)]
            fn sub_assign(&mut self, a: &Self) {
                *self -= a
            }

            #[inline(always)]
            fn mul_assign(&mut self, a: &Self) {
                *self *= a
            }

            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                *self * *a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -self
            }

            #[inline(always)]
            fn abs(&self) -> Self {
                (*self).abs()
            }

            #[inline(always)]
            fn zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn one() -> Self {
                Self::ONE
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // FIXME: use binary exponentiation
                debug_assert!(e <= i32::MAX as u64);
                self.powf(e as f64)
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                Self::ONE / *self
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

simd_impl!(f64x2);
simd_impl!(f64x4);

impl NumericalFloatLike for Rational {
    fn add(&self, a: &Self) -> Self {
        self + a
    }

    fn sub(&self, a: &Self) -> Self {
        self - a
    }

    fn mul(&self, a: &Self) -> Self {
        self * a
    }

    fn add_assign(&mut self, a: &Self) {
        *self += a
    }

    fn sub_assign(&mut self, a: &Self) {
        *self -= a;
    }

    fn mul_assign(&mut self, a: &Self) {
        *self *= a
    }

    fn mul_add(&self, a: &Self, c: &Self) -> Self {
        &(self * a) + &c
    }

    fn neg(&self) -> Self {
        self.neg()
    }

    fn abs(&self) -> Self {
        self.abs()
    }

    fn zero() -> Self {
        Self::zero()
    }

    fn one() -> Self {
        Self::one()
    }

    fn pow(&self, e: u64) -> Self {
        self.pow(e)
    }

    fn inv(&self) -> Self {
        self.inv()
    }
}

impl NumericalFloatComparison for Rational {
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }
}
