//! A hyperdual number is a number that keep track of (higher order) derivatives in one or multiple
//! variables.
//! ```
//! use symbolica::{
//! create_hyperdual_from_components, create_hyperdual_single_derivative,
//! domains::{float::NumericalFloatLike, rational::Rational},
//! };
//!
//! create_hyperdual_single_derivative!(SingleDual, 3);
//!
//! create_hyperdual_from_components!(
//! Dual,
//! [
//!     [0, 0, 0],
//!     [1, 0, 0],
//!     [0, 1, 0],
//!     [0, 0, 1],
//!     [1, 1, 0],
//!     [1, 0, 1],
//!     [0, 1, 1],
//!     [1, 1, 1],
//!     [2, 0, 0]
//! ]
//! );
//!
//! fn main() {
//! let x = Dual::<Rational>::new_variable(0, (1, 1).into());
//! let y = Dual::new_variable(1, (2, 1).into());
//! let z = Dual::new_variable(2, (3, 1).into());
//!
//! let t3 = x * y * z;
//!
//! println!("{}", t3.inv());
//! }
//! ```

/// Get the size of a dual number with the given maximum depth per variable.
pub const fn get_dual_size<const N: usize>(x: &[usize; N]) -> usize {
    let mut c = 1;
    let mut i = 0;
    while i < x.len() {
        c *= x[i] + 1;
        i += 1;
    }
    c
}

/// Get the dual components for a single derivative in multiple variables.
pub const fn get_single_derivative_dual_components<const N: usize, const N_PLUS_ONE: usize>(
) -> [[usize; N]; N_PLUS_ONE] {
    let mut res = [[0; N]; N_PLUS_ONE];

    let mut i = 1;
    while i < res.len() {
        res[i][i - 1] = 1;
        i += 1;
    }
    res
}

// Get the dual components for a given shape.
pub const fn get_dual_components<const N: usize, const K: usize>(
    x: &[usize; N],
) -> [[usize; N]; K] {
    let mut res = [[0; N]; K];
    let mut cur = [0; N];

    let mut i = 0;
    let mut k;

    let mut max_depth = 0;
    while i < x.len() {
        max_depth += x[i];
        i += 1;
    }
    i = 0;

    let mut d = 0;
    let mut depth_sum = 0;
    while d <= max_depth {
        k = 0;
        'done: loop {
            if depth_sum == d {
                res[i] = cur;
                i += 1;
            }

            while cur[k] + 1 > x[k] {
                depth_sum -= cur[k];
                cur[k] = 0;
                if k + 1 == N {
                    break 'done;
                }

                k += 1;
            }
            cur[k] += 1;
            depth_sum += 1;
            k = 0;
        }

        d += 1;
    }
    res
}

/// Get the index in `r` of the multiplication of `a` and `b` in `r`.
pub const fn get_multiplication_index<const N: usize, const C: usize>(
    r: &[[usize; N]; C],
    a: usize,
    b: usize,
) -> Option<usize> {
    let mut sum = r[a];
    let mut i = 0;
    while i < N {
        sum[i] += r[b][i];
        i += 1;
    }

    i = 0;

    // find the index of the sum of powers
    while i < r.len() {
        let mut j = 0;
        while j < N {
            if r[i][j] != sum[j] {
                break;
            }
            j += 1;
        }
        if j == N {
            break;
        }
        i += 1;
    }

    if i == r.len() {
        None
    } else {
        Some(i)
    }
}

/// Get the size of the multiplication table.
pub const fn get_mult_table_size<const N: usize, const C: usize>(r: &[[usize; N]; C]) -> usize {
    let mut i = 0;
    let mut ri = 0;
    while i < r.len() {
        let mut j = 1; // skip first entry
        while j < r.len() {
            if get_multiplication_index::<N, C>(&r, i, j).is_some() {
                ri += 1;
            }
            j += 1;
        }
        i += 1;
    }

    ri
}

// Construct the multiplication table for a dual.
pub const fn get_mult_table<const N: usize, const C: usize, const T: usize>(
    r: &[[usize; N]; C],
) -> [(usize, usize, usize); T] {
    let mut res = [(0, 0, 0); T];

    let mut ri = 0;
    let mut i = 0;
    while i < r.len() {
        let mut j = 1; // skip first entry
        while j < r.len() {
            if let Some(index) = get_multiplication_index::<N, C>(&r, i, j) {
                res[ri] = (i, j, index);
                ri += 1;
            }
            j += 1;
        }
        i += 1;
    }

    res
}

/// Create a new hyperdual number, with only single derivatives and with `$var` variables.
///
/// For example:
/// ```
/// # use symbolica::{
/// # create_hyperdual_single_derivative,
/// # domains::{float::NumericalFloatLike, rational::Rational},
/// # };
/// create_hyperdual_single_derivative!(Dual, 3);
///
/// fn main() {
///     let x = Dual::<Rational>::new_variable(0, (1, 1).into());
///     let y = Dual::new_variable(1, (2, 1).into());
///     let z = Dual::new_variable(2, (3, 1).into());
///
///     let t3 = x * y * z;
///
///     println!("{}", t3.inv());
/// }
/// ```
#[macro_export]
macro_rules! create_hyperdual_single_derivative {
    ($t: ident, $var: expr) => {
        $crate::create_hyperdual_from_components!(
            $t,
            $crate::domains::dual::get_single_derivative_dual_components::<{ $var }, { $var + 1 }>(
            )
        );
    };
}

/// Create a new hyperdual number, with a given derivative depth per variable.
#[macro_export]
macro_rules! create_hyperdual_from_depths {
    ($t: ident, $var: expr) => {
        $crate::create_hyperdual_from_components!(
            $t,
            $crate::domains::dual::get_dual_components::<
                { $var.len() },
                { $crate::domains::dual::get_dual_size(&$var) },
            >(&$var)
        );
    };
}

/// Create a new hyperdual number, from a specification of the components.
/// The first components must be the real value and the single derivatives of the variables in order.
///
/// For example:
/// ```
/// # use symbolica::{
/// # create_hyperdual_from_components,
/// # domains::{float::NumericalFloatLike, rational::Rational},
/// # };
/// create_hyperdual_from_components!(
///     Dual,
///     [
///         [0, 0, 0],
///         [1, 0, 0],
///         [0, 1, 0],
///         [0, 0, 1],
///         [1, 1, 0],
///         [1, 0, 1],
///         [0, 1, 1],
///         [1, 1, 1],
///         [2, 0, 0]
///     ]
/// );
///
/// fn main() {
///     let x = Dual::<Rational>::new_variable(0, (1, 1).into());
///     let y = Dual::new_variable(1, (2, 1).into());
///     let z = Dual::new_variable(2, (3, 1).into());
///
///     let t3 = x * y * z;
///
///     println!("{}", t3.inv());
/// }
/// ```
#[macro_export]
macro_rules! create_hyperdual_from_components {
    ($t: ident, $var: expr) => {
        /// A hyperdual with a given shape.
        /// The multiplication table is precomputed and (partially) unrolled
        /// for performance.
        #[derive(Clone, Debug)]
        pub struct $t<T> {
            pub values: [T; $var.len()],
        }

        impl<T> $t<T> {
            /// The shape of the hyperdual.
            pub const SHAPE: [[usize; { $var[0].len() }]; { $var.len() }] = $var;
            const MAX_POW: usize = {
                let mut c = 0;
                let mut i = 0;
                let last = $var[$var.len() - 1];
                while i < last.len() {
                    c += last[i];
                    i += 1;
                }
                c
            };
            const MULT_TABLE: [(usize, usize, usize); {
                $crate::domains::dual::get_mult_table_size(&$var)
            }] = $crate::domains::dual::get_mult_table(&$var);
        }

        impl<T: $crate::domains::float::NumericalFloatLike> $t<T> {
            /// Create a new dual variable for the variable `var`, i.e. `value + 1*ε_var`,
            /// inheriting the floating point settings from `self`.
            #[allow(dead_code)]
            pub fn variable(&self, var: usize, value: T) -> Self {
                let mut values = std::array::from_fn(|_| self.values[0].zero());
                values[0] = value;
                values[1 + var] = self.values[0].one();
                $t { values }
            }
        }

        impl<T: $crate::domains::float::ConstructibleFloat> $t<T> {
            /// Create a new dual variable for the variable `var`, i.e. `value + 1*ε_var`.
            #[allow(dead_code)]
            pub fn new_variable(var: usize, value: T) -> Self {
                let mut values = std::array::from_fn(|_| T::new_zero());
                values[0] = value;
                values[1 + var] = T::new_one();
                $t { values }
            }
        }

        impl<T: for<'a> std::ops::MulAssign<&'a T>> std::ops::Mul<&T> for $t<T> {
            type Output = Self;

            fn mul(mut self, other: &T) -> Self::Output {
                for s in self.values.iter_mut() {
                    *s *= other;
                }

                self
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::fmt::Display for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;
                for (i, (v, s)) in self.values.iter().zip(Self::SHAPE).enumerate() {
                    if i > 0 && i < self.values.len() {
                        f.write_char('+')?;
                    }
                    f.write_char('(')?;
                    std::fmt::Display::fmt(v, f)?;
                    f.write_char(')')?;
                    if i > 0 {
                        f.write_char('*')?;
                    }
                    for (i, p) in s.iter().enumerate() {
                        if *p > 0 {
                            f.write_char('ε')?;
                            std::fmt::Display::fmt(&i, f)?;
                            if *p > 1 {
                                f.write_fmt(format_args!("^{}", p))?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::fmt::LowerExp for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;
                for (i, (v, s)) in self.values.iter().zip(Self::SHAPE).enumerate() {
                    if i > 0 && i < self.values.len() {
                        f.write_char('+')?;
                    }
                    f.write_char('(')?;
                    std::fmt::LowerExp::fmt(v, f)?;
                    f.write_char(')')?;
                    if i > 0 {
                        f.write_char('*')?;
                    }
                    for (i, p) in s.iter().enumerate() {
                        if *p > 0 {
                            f.write_char('ε')?;
                            std::fmt::Display::fmt(&i, f)?;
                            if *p > 1 {
                                f.write_fmt(format_args!("^{}", p))?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> PartialEq for $t<T> {
            fn eq(&self, other: &Self) -> bool {
                // only compare the real part
                self.values[0].eq(&other.values[0])
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike + PartialOrd> PartialOrd for $t<T> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.values[0].partial_cmp(&other.values[0])
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Neg for $t<T> {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                let mut res = self.clone();
                for (s, o) in res.values.iter_mut().zip(self.values.into_iter()) {
                    *s = -o;
                }
                res
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Add<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: &Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s += o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Add<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
                    *s += o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Sub<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: &Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s -= o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Sub<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
                    *s -= o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Mul<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: &Self) -> Self::Output {
                let mut res = self.clone();

                for s in &mut res.values {
                    *s *= rhs.values[0].clone();
                }

                // the compiler will (partially) unroll this
                for (si, oi, index) in Self::MULT_TABLE.iter() {
                    unsafe {
                        *res.values.get_unchecked_mut(*index) +=
                            self.values.get_unchecked(*si).clone() * rhs.values.get_unchecked(*oi);
                    }
                }

                res
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Mul<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self * &rhs
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Div<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn div(self, rhs: &Self) -> Self::Output {
                use $crate::domains::float::NumericalFloatLike;
                self * rhs.inv()
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::Div<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                use $crate::domains::float::NumericalFloatLike;
                self * rhs.inv()
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::AddAssign<&$t<T>> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: &$t<T>) {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s += o;
                }
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::AddAssign<$t<T>> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: $t<T>) {
                self.add_assign(&rhs)
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::SubAssign<&$t<T>> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: &$t<T>) {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s -= o;
                }
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::SubAssign<$t<T>> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: $t<T>) {
                self.sub_assign(&rhs)
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::MulAssign<&$t<T>> for $t<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: &$t<T>) {
                *self = self.clone() * rhs.clone();
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::MulAssign<$t<T>> for $t<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: $t<T>) {
                *self = self.clone() * rhs;
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::DivAssign<&$t<T>> for $t<T> {
            #[inline]
            fn div_assign(&mut self, rhs: &$t<T>) {
                use $crate::domains::float::NumericalFloatLike;
                *self = rhs.inv() * &*self;
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike> std::ops::DivAssign<$t<T>> for $t<T> {
            #[inline]
            fn div_assign(&mut self, rhs: $t<T>) {
                use $crate::domains::float::NumericalFloatLike;
                *self = rhs.inv() * &*self;
            }
        }

        impl<T: $crate::domains::float::NumericalFloatLike>
            $crate::domains::float::NumericalFloatLike for $t<T>
        {
            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                self.clone() * a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -self.clone()
            }

            #[inline(always)]
            fn zero(&self) -> Self {
                $t {
                    values: std::array::from_fn(|_| self.values[0].zero()),
                }
            }

            #[inline(always)]
            fn new_zero() -> Self {
                $t {
                    values: std::array::from_fn(|_| T::new_zero()),
                }
            }

            #[inline(always)]
            fn one(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].one();
                res
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // TODO: use binary exponentiation
                let mut res = self.clone();
                for _ in 1..e {
                    res = res.clone() * self;
                }
                res
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                let e = self.values[0].inv();
                let mut r = self.clone() * &e;
                r.values[0] = r.values[0].zero();

                let mut accum = r.one();
                let mut res = r.one();

                for i in 1..Self::MAX_POW + 1 {
                    accum = accum * &r;
                    if i % 2 == 0 {
                        res = res + accum.clone()
                    } else {
                        res = res - accum.clone()
                    }
                }

                res * &e
            }

            #[inline(always)]
            fn from_usize(&self, a: usize) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].from_usize(a);
                res
            }

            #[inline(always)]
            fn from_i64(&self, a: i64) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].from_i64(a);
                res
            }

            #[inline(always)]
            fn get_precision(&self) -> u32 {
                self.values[0].get_precision()
            }

            #[inline(always)]
            fn get_epsilon(&self) -> f64 {
                self.values[0].get_epsilon()
            }

            #[inline(always)]
            fn fixed_precision(&self) -> bool {
                self.values[0].fixed_precision()
            }

            fn sample_unit<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].sample_unit(rng);
                res
            }
        }

        impl<T: $crate::domains::float::SingleFloat> $crate::domains::float::SingleFloat for $t<T> {
            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.values[0].is_zero()
            }

            #[inline(always)]
            fn is_one(&self) -> bool {
                self.values[0].is_one()
            }

            #[inline(always)]
            fn is_finite(&self) -> bool {
                self.values[0].is_finite()
            }

            #[inline(always)]
            fn from_rational(&self, rat: &Rational) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].from_rational(rat);
                res
            }
        }

        impl<T: $crate::domains::float::ConstructibleFloat>
            $crate::domains::float::ConstructibleFloat for $t<T>
        {
            #[inline(always)]
            fn new_one() -> Self {
                let mut res = <Self as $crate::domains::float::NumericalFloatLike>::new_zero();
                res.values[0] = T::new_one();
                res
            }

            #[inline(always)]
            fn new_from_usize(a: usize) -> Self {
                let mut res = <Self as $crate::domains::float::NumericalFloatLike>::new_zero();
                res.values[0] = T::new_from_usize(a);
                res
            }

            #[inline(always)]
            fn new_from_i64(a: i64) -> Self {
                let mut res = <Self as $crate::domains::float::NumericalFloatLike>::new_zero();
                res.values[0] = T::new_from_i64(a);
                res
            }

            #[inline(always)]
            fn new_sample_unit<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
                let mut res = <Self as $crate::domains::float::NumericalFloatLike>::new_zero();
                res.values[0] = T::new_sample_unit(rng);
                res
            }
        }

        impl<T: $crate::domains::float::Real> $crate::domains::float::Real for $t<T> {
            #[inline(always)]
            fn pi(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].pi();
                res
            }

            #[inline(always)]
            fn e(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].e();
                res
            }

            #[inline(always)]
            fn euler(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].euler();
                res
            }

            #[inline(always)]
            fn phi(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].phi();
                res
            }

            #[inline(always)]
            fn i(&self) -> Option<Self> {
                let mut res = self.zero();
                res.values[0] = self.values[0].i()?;
                Some(res)
            }

            #[inline(always)]
            fn norm(&self) -> Self {
                let n = self.values[0].norm();
                if n == self.values[0] {
                    return self.clone();
                } else {
                    let scale = n / &self.values[0];
                    self.clone() * &scale
                }
            }

            #[inline(always)]
            fn sqrt(&self) -> Self {
                use $crate::domains::float::NumericalFloatLike;
                let e = self.values[0].sqrt();

                let norm = self.values[0].inv(); // TODO: check 0
                let mut r = self.clone() * &norm;
                r.values[0] = self.values[0].zero();

                let mut accum = self.one();
                let mut res = self.one();
                let mut num = self.values[0].one();

                let mut scale = 1;
                for p in 1..Self::MAX_POW + 1 {
                    scale *= p;
                    num = num.clone() * (num.from_usize(2).inv() - &num.from_usize(p as usize - 1));
                    accum = accum * &r;
                    res = res + accum.clone() * &num * &num.from_usize(scale).inv();
                }

                res * &e
            }

            #[inline(always)]
            fn log(&self) -> Self {
                let e = self.values[0].log();

                let norm = self.values[0].inv(); // TODO: check 0
                let mut r = self.clone() * &norm;
                r.values[0] = self.values[0].zero();

                let mut accum = r.clone();
                let mut res = self.one();
                res.values[0] = e.clone();

                let mut scale = -1;
                for p in 1..Self::MAX_POW + 1 {
                    scale *= -1;
                    res += accum.clone() * &e.from_i64(p as i64 * scale).inv();
                    accum = accum * &r;
                }

                res
            }

            #[inline(always)]
            fn exp(&self) -> Self {
                let e = self.values[0].exp();
                let mut res = self.one();

                let mut r = self.clone();
                r.values[0] = self.values[0].zero();
                let mut accum = self.one();
                let mut scale = 1;
                for p in 0..Self::MAX_POW {
                    scale *= p + 1;
                    accum = accum * &r; // TODO: many multiplications with 0
                    res = res + accum.clone() * &e.from_usize(scale).inv();
                }

                res * &e
            }

            #[inline(always)]
            fn sin(&self) -> Self {
                let s = self.values[0].sin();
                let c = self.values[0].cos();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = s.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let mut b = if i % 2 == 1 { c.clone() } else { s.clone() };

                    if i % 4 >= 2 {
                        b = b.neg();
                    }

                    let s = sp.clone() * &b * &e.from_usize(scale).inv();

                    sp = sp * &p;

                    e = e + s;
                }

                e
            }

            #[inline(always)]
            fn cos(&self) -> Self {
                let s = self.values[0].sin();
                let c = self.values[0].cos();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = c.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let mut b = if i % 2 == 1 { s.clone() } else { -c.clone() };

                    if i % 4 < 2 {
                        b = b.neg();
                    }

                    let s = sp.clone() * &b * &e.from_usize(scale).inv();

                    sp = sp * &p;

                    e = e + s;
                }

                e
            }

            #[inline(always)]
            fn tan(&self) -> Self {
                // TODO: improve
                self.sin() / self.cos()
            }

            #[inline(always)]
            fn asin(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn acos(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn atan2(&self, _x: &Self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn sinh(&self) -> Self {
                let s = self.values[0].sinh();
                let c = self.values[0].cosh();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = s.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let b = if i % 2 == 1 { &c } else { &s };

                    let s = sp.clone() * b * &e.from_usize(scale).inv();

                    sp = sp * &p;

                    e = e + s;
                }

                e
            }

            #[inline(always)]
            fn cosh(&self) -> Self {
                let s = self.values[0].sinh();
                let c = self.values[0].cosh();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = c.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let b = if i % 2 == 1 { &s } else { &c };

                    let s = sp.clone() * b * &e.from_usize(scale).inv();

                    sp = sp * &p;

                    e = e + s;
                }

                e
            }

            #[inline(always)]
            fn tanh(&self) -> Self {
                self.sinh() / self.cosh()
            }

            #[inline(always)]
            fn asinh(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn acosh(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn atanh(&self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powf(&self, e: &Self) -> Self {
                // TODO: improve
                (self.log() * e).exp()
            }
        }
    };
}

#[cfg(test)]
mod test {
    use crate::domains::{
        float::{NumericalFloatLike, Real},
        rational::Rational,
    };

    create_hyperdual_from_depths!(Dual, [1, 2, 3]);

    #[test]
    fn real_functions() {
        let x = Dual::<f64> {
            values: [
                0.2, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        };

        fn approx_eq(a: Dual<f64>, b: &Dual<f64>) -> bool {
            for (a, b) in a.values.iter().zip(b.values.iter()) {
                if (a - b).abs() / (a + b) > 1e-10
                    && (*a != 0. || *b > 1e-10)
                    && (*b != 0. || *a > 1e-10)
                {
                    return false;
                }
            }

            true
        }

        assert!(approx_eq(x.sqrt() * x.sqrt(), &x));
        assert!(approx_eq(x.log().exp(), &x));
        assert!(approx_eq(x.exp().log(), &x));
        assert!(approx_eq(x.sin() * x.sin() + x.cos() * x.cos(), &x.one()));
        assert!(approx_eq((-x.clone()).norm(), &x.norm()));
        assert!(approx_eq(
            x.cosh() * x.cosh() - x.sinh() * x.sinh(),
            &x.one()
        ));
        assert_eq!(x.cosh() * x.cosh() - x.sinh() * x.sinh(), x.one());
    }

    #[test]
    fn dual_inv() {
        let x = Dual::<Rational>::new_variable(0, (1, 1).into());
        let y = Dual::new_variable(1, (2, 1).into());
        let z = Dual::new_variable(2, (3, 1).into());

        let t3 = x * y * z;
        let one = t3.clone() * t3.inv();

        assert_eq!(one, t3.one());
    }
}
