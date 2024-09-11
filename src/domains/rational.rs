use std::{
    borrow::Cow,
    fmt::{Display, Error, Formatter, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    poly::{gcd::LARGE_U32_PRIMES, polynomial::PolynomialRing, Exponent},
    printer::PrintOptions,
};

use super::{
    finite_field::{
        FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField, Two, Zp, Z2,
    },
    integer::{Integer, IntegerRing, Z},
    EuclideanDomain, Field, InternalOrdering, Ring,
};

/// The field of rational numbers.
pub type Q = FractionField<IntegerRing>;
pub type RationalField = FractionField<IntegerRing>;
/// The field of rational numbers.
pub const Q: FractionField<IntegerRing> = FractionField::new(Z);

/// The fraction field of `R`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FractionField<R: Ring> {
    ring: R,
}

impl<R: Ring> FractionField<R> {
    pub const fn new(ring: R) -> FractionField<R> {
        FractionField { ring }
    }
}

impl<R: EuclideanDomain + FractionNormalization> FractionField<R> {
    pub fn to_element_numerator(&self, numerator: R::Element) -> <Self as Ring>::Element {
        Fraction {
            numerator,
            denominator: self.ring.one(),
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> FractionField<R> {
    pub fn to_element(
        &self,
        mut numerator: R::Element,
        mut denominator: R::Element,
        do_gcd: bool,
    ) -> <Self as Ring>::Element {
        if do_gcd {
            let g = self.ring.gcd(&numerator, &denominator);
            if !self.ring.is_one(&g) {
                numerator = self.ring.quot_rem(&numerator, &g).0;
                denominator = self.ring.quot_rem(&denominator, &g).0;
            }
        }

        let f = self.ring.get_normalization_factor(&denominator);

        if self.ring.is_one(&f) {
            Fraction {
                numerator,
                denominator,
            }
        } else {
            Fraction {
                numerator: self.ring.mul(&numerator, &f),
                denominator: self.ring.mul(&denominator, &f),
            }
        }
    }
}

impl<R: Ring> Display for FractionField<R> {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait FractionNormalization: Ring {
    /// Get the factor that normalizes the element `a`.
    /// - For a field, this is the inverse of `a`.
    /// - For the integers, this is the sign of `a`.
    /// - For a polynomial ring, this is the normalization factor of the leading coefficient.
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element;
}

impl FractionNormalization for Z {
    fn get_normalization_factor(&self, a: &Integer) -> Integer {
        if *a < 0 {
            (-1).into()
        } else {
            1.into()
        }
    }
}

impl<R: Ring + FractionNormalization, E: Exponent> FractionNormalization for PolynomialRing<R, E> {
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element {
        a.constant(a.ring.get_normalization_factor(&a.lcoeff()))
    }
}

impl<T: Field> FractionNormalization for T {
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element {
        self.inv(a)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Fraction<R: Ring> {
    numerator: R::Element,
    denominator: R::Element,
}

impl<R: Ring> Fraction<R> {
    pub fn numerator(&self) -> R::Element {
        self.numerator.clone()
    }

    pub fn denominator(&self) -> R::Element {
        self.denominator.clone()
    }

    pub fn numerator_ref(&self) -> &R::Element {
        &self.numerator
    }

    pub fn denominator_ref(&self) -> &R::Element {
        &self.denominator
    }
}

impl<R: Ring> InternalOrdering for Fraction<R> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.numerator
            .internal_cmp(&other.numerator)
            .then_with(|| self.denominator.internal_cmp(&other.denominator))
    }
}

impl<R: EuclideanDomain> Ring for FractionField<R> {
    type Element = Fraction<R>;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let r = &self.ring;

        if a.denominator == b.denominator {
            let num = r.add(&a.numerator, &b.numerator);
            let g = r.gcd(&num, &a.denominator);
            if !r.is_one(&g) {
                return Fraction {
                    numerator: r.quot_rem(&num, &g).0,
                    denominator: r.quot_rem(&a.denominator, &g).0,
                };
            } else {
                return Fraction {
                    numerator: num,
                    denominator: a.denominator.clone(),
                };
            }
        }

        let denom_gcd = r.gcd(&a.denominator, &b.denominator);

        let mut a_den_red = Cow::Borrowed(&a.denominator);
        let mut b_den_red = Cow::Borrowed(&b.denominator);

        if !r.is_one(&denom_gcd) {
            a_den_red = Cow::Owned(r.quot_rem(&a.denominator, &denom_gcd).0);
            b_den_red = Cow::Owned(r.quot_rem(&b.denominator, &denom_gcd).0);
        }

        let num1 = r.mul(&a.numerator, &b_den_red);
        let num2 = r.mul(&b.numerator, &a_den_red);
        let mut num = r.add(&num1, &num2);

        // TODO: prefer small * large over medium * medium sized operations
        // a_denom_red.as_ref() * &other.denominator may be faster
        // TODO: add size hint trait with default implementation?
        let mut den = r.mul(b_den_red.as_ref(), &a.denominator);

        let g = r.gcd(&num, &denom_gcd);

        if !r.is_one(&g) {
            num = r.quot_rem(&num, &g).0;
            den = r.quot_rem(&den, &g).0;
        }

        Fraction {
            numerator: num,
            denominator: den,
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let r = &self.ring;
        let gcd1 = r.gcd(&a.numerator, &b.denominator);
        let gcd2 = r.gcd(&a.denominator, &b.numerator);

        if r.is_one(&gcd1) {
            if r.is_one(&gcd2) {
                Fraction {
                    numerator: r.mul(&a.numerator, &b.numerator),
                    denominator: r.mul(&a.denominator, &b.denominator),
                }
            } else {
                Fraction {
                    numerator: r.mul(&a.numerator, &r.quot_rem(&b.numerator, &gcd2).0),
                    denominator: r.mul(&r.quot_rem(&a.denominator, &gcd2).0, &b.denominator),
                }
            }
        } else if r.is_one(&gcd2) {
            Fraction {
                numerator: r.mul(&r.quot_rem(&a.numerator, &gcd1).0, &b.numerator),
                denominator: r.mul(&a.denominator, &r.quot_rem(&b.denominator, &gcd1).0),
            }
        } else {
            Fraction {
                numerator: r.mul(
                    &r.quot_rem(&a.numerator, &gcd1).0,
                    &r.quot_rem(&b.numerator, &gcd2).0,
                ),
                denominator: r.mul(
                    &r.quot_rem(&a.denominator, &gcd2).0,
                    &r.quot_rem(&b.denominator, &gcd1).0,
                ),
            }
        }
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
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        Fraction {
            numerator: self.ring.neg(&a.numerator),
            denominator: a.denominator.clone(),
        }
    }

    fn zero(&self) -> Self::Element {
        Fraction {
            numerator: self.ring.zero(),
            denominator: self.ring.one(),
        }
    }

    fn one(&self) -> Self::Element {
        Fraction {
            numerator: self.ring.one(),
            denominator: self.ring.one(),
        }
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        Fraction {
            numerator: self.ring.nth(n),
            denominator: self.ring.one(),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        Fraction {
            numerator: self.ring.pow(&b.numerator, e),
            denominator: self.ring.pow(&b.denominator, e),
        }
    }

    fn is_zero(a: &Self::Element) -> bool {
        R::is_zero(&a.numerator)
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        self.ring.is_one(&a.numerator) && self.ring.is_one(&a.denominator)
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn size(&self) -> Integer {
        // TODO: this is an overestimate
        self.ring.size() * self.ring.size()
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        Fraction {
            numerator: self.ring.sample(rng, range),
            denominator: self.ring.one(),
        }
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        self.ring.fmt_display(&element.numerator, opts, true, f)?;
        if !self.ring.is_one(&element.denominator) {
            f.write_char('/')?;
            self.ring.fmt_display(&element.denominator, opts, true, f)?;
        }

        Ok(())
    }
}

impl<R: EuclideanDomain + FractionNormalization> EuclideanDomain for FractionField<R> {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let gcd_num = self.ring.gcd(&a.numerator, &b.numerator);
        let gcd_den = self.ring.gcd(&a.denominator, &b.denominator);

        let d1 = self.ring.quot_rem(&a.denominator, &gcd_den).0;
        let lcm = self.ring.mul(&d1, &b.denominator);

        Fraction {
            numerator: gcd_num,
            denominator: lcm,
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> Field for FractionField<R> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        let f = self.ring.get_normalization_factor(&a.numerator);

        Fraction {
            numerator: self.ring.mul(&a.denominator, &f),
            denominator: self.ring.mul(&a.numerator, &f),
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization, E: Exponent> PolynomialRing<FractionField<R>, E> {
    pub fn to_rational_polynomial(
        &self,
        e: &<Self as Ring>::Element,
    ) -> Fraction<PolynomialRing<R, E>> {
        let mut lcm = self.ring.ring.one();
        for x in &e.coefficients {
            let g = self.ring.ring.gcd(&lcm, x.denominator_ref());
            lcm = self
                .ring
                .ring
                .mul(&lcm, &self.ring.ring.quot_rem(x.denominator_ref(), &g).0);
        }

        let e2 = e.map_coeff(
            |c| {
                self.ring.ring.mul(
                    &c.numerator,
                    &self.ring.ring.quot_rem(&lcm, &c.denominator).0,
                )
            },
            self.ring.ring.clone(),
        );

        Fraction {
            denominator: e2.constant(lcm),
            numerator: e2,
        }
    }
}

/// A rational number.
pub type Rational = Fraction<IntegerRing>;

impl Default for Rational {
    fn default() -> Self {
        Rational::zero()
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.numerator.fmt(f)?;
        if self.denominator != 1 {
            f.write_char('/')?;
            write!(f, "{}", self.denominator)?;
        }

        Ok(())
    }
}
impl From<f64> for Rational {
    /// Convert a floating point number to its exact rational number equivalent.
    /// Use [`Rational::truncate_denominator`] to get an approximation with a smaller denominator.
    #[inline]
    fn from(f: f64) -> Self {
        assert!(f.is_finite());

        // taken from num-traits
        let bits: u64 = f.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xfffffffffffff) << 1
        } else {
            (bits & 0xfffffffffffff) | 0x10000000000000
        };
        // Exponent bias + mantissa shift
        exponent -= 1023 + 52;

        // superfluous factors of 2 will be divided out in the conversion to rational
        if exponent < 0 {
            (
                (sign as i64 * mantissa as i64).into(),
                Integer::from(2).pow(-exponent as u64),
            )
                .into()
        } else {
            (
                &Integer::from(sign as i64 * mantissa as i64)
                    * &Integer::from(2).pow(exponent as u64),
                1.into(),
            )
                .into()
        }
    }
}

impl<T: Into<Integer>> From<T> for Rational {
    #[inline]
    fn from(value: T) -> Self {
        Rational {
            numerator: value.into(),
            denominator: 1.into(),
        }
    }
}

impl From<&Integer> for Rational {
    fn from(value: &Integer) -> Self {
        Rational {
            numerator: value.clone(),
            denominator: 1.into(),
        }
    }
}

impl<T: Into<Integer>> From<(T, T)> for Rational {
    #[inline]
    fn from((num, den): (T, T)) -> Self {
        Q.to_element(num.into(), den.into(), true)
    }
}

impl From<rug::Rational> for Rational {
    fn from(value: rug::Rational) -> Self {
        let (num, den) = value.into_numer_denom();
        Q.to_element(num.into(), den.into(), false)
    }
}

impl ToFiniteField<u32> for Rational {
    fn to_finite_field(&self, field: &Zp) -> <Zp as Ring>::Element {
        field.div(
            &self.numerator.to_finite_field(field),
            &self.denominator.to_finite_field(field),
        )
    }
}

impl ToFiniteField<Two> for Rational {
    fn to_finite_field(&self, field: &Z2) -> <Z2 as Ring>::Element {
        field.div(
            &self.numerator.to_finite_field(field),
            &self.denominator.to_finite_field(field),
        )
    }
}

impl Rational {
    pub fn from_unchecked<T: Into<Integer>>(num: T, den: T) -> Rational {
        Q.to_element(num.into(), den.into(), false)
    }

    pub fn is_negative(&self) -> bool {
        self.numerator < 0
    }

    pub fn is_integer(&self) -> bool {
        self.denominator.is_one()
    }

    pub fn zero() -> Rational {
        Rational {
            numerator: 0.into(),
            denominator: 1.into(),
        }
    }

    pub fn one() -> Rational {
        Rational {
            numerator: 1.into(),
            denominator: 1.into(),
        }
    }

    pub fn abs(&self) -> Rational {
        if self.is_negative() {
            self.clone().neg()
        } else {
            self.clone()
        }
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.numerator.is_one() && self.denominator.is_one()
    }

    pub fn pow(&self, e: u64) -> Rational {
        Q.pow(self, e)
    }

    pub fn inv(&self) -> Rational {
        Q.inv(self)
    }

    pub fn neg(&self) -> Rational {
        Q.neg(self)
    }

    pub fn to_f64(&self) -> f64 {
        rug::Rational::from((
            self.numerator.clone().to_multi_prec(),
            self.denominator.clone().to_multi_prec(),
        ))
        .to_f64()
    }

    pub fn to_multi_prec(self) -> rug::Rational {
        rug::Rational::from((
            self.numerator.to_multi_prec(),
            self.denominator.to_multi_prec(),
        ))
    }

    /// Return a best approximation of the rational number where the denominator
    /// is less than or equal to `max_denominator`.
    pub fn truncate_denominator(&self, max_denominator: &Integer) -> Rational {
        assert!(!max_denominator.is_zero() && !max_denominator.is_negative());

        if self.denominator_ref() < max_denominator {
            return self.clone();
        }

        let (mut p0, mut q0, mut p1, mut q1) = (
            Integer::zero(),
            Integer::one(),
            Integer::one(),
            Integer::zero(),
        );

        let (mut n, mut d) = (self.numerator_ref().abs(), self.denominator());
        loop {
            let a = &n / &d;
            let q2 = &q0 + &(&a * &q1);
            if &q2 > max_denominator {
                break;
            }
            (p1, p0, q0, q1) = (p0 + &(&a * &p1), p1, q1, q2);
            (d, n) = (&n - &a * &d, d);
        }

        let k = &(max_denominator - &q0) / &q1;
        let bound1: Rational = (p0 + &(&k * &p1), &q0 + &(&k * &q1)).into();
        let bound2: Rational = (p1, q1).into();

        let res = if (&bound2 - self).abs() <= (&bound1 - self).abs() {
            bound2
        } else {
            bound1
        };

        if self.is_negative() {
            res.neg()
        } else {
            res
        }
    }

    /// Round the rational to the one with the smallest numerator or denominator in the interval
    /// `[self * (1-relative_error), self * (1+relative_error)]`, where
    /// `0 < relative_error < 1`.
    pub fn round(&self, relative_error: &Rational) -> Rational {
        if self.is_zero() {
            return Rational::zero();
        }

        if self.is_negative() {
            self.round_in_interval(
                self.clone() * (Rational::one() + relative_error),
                self.clone() * (Rational::one() - relative_error),
            )
        } else {
            self.round_in_interval(
                self.clone() * (Rational::one() - relative_error),
                self.clone() * (Rational::one() + relative_error),
            )
        }
    }

    /// Round the rational to the one with the smallest numerator or denominator in the interval
    /// `[l, u]`, where `l < u`.
    pub fn round_in_interval(&self, mut l: Rational, mut u: Rational) -> Rational {
        assert!(l < u);

        let mut flip = false;
        if l.is_negative() && u.is_negative() {
            flip = true;
            (l, u) = (-u, -l);
        } else if l.is_negative() {
            return Rational::zero();
        }

        // use continued fractions to find the best approximation in an interval
        let (mut ln, mut ld) = (l.numerator(), l.denominator());
        let (mut un, mut ud) = (u.numerator(), u.denominator());

        // h1/k1 accumulates the shared continued fraction terms of l and u
        let (mut h1, mut h0, mut k1, mut k0): (Integer, Integer, Integer, Integer) =
            (1.into(), 0.into(), 0.into(), 1.into());

        loop {
            let a = &(&ln - &1.into()) / &ld; // get next term in continued fraction
            (ld, ud, ln, un) = (&un - &a * &ud, &ln - &a * &ld, ud, ld); // subtract and invert
            (h1, h0) = (&a * &h1 + &h0, h1);
            (k1, k0) = (&a * &k1 + &k0, k1);
            if ln <= ld {
                let res: Rational = (h1 + &h0, k1 + &k0).into();

                if flip {
                    return -res;
                } else {
                    return res;
                }
            }
        }
    }

    /// Reconstruct a rational number `q` from a value `v` in a prime field `p`,
    /// such that `q ≡ v mod p`.
    ///
    /// From "Maximal Quotient Rational Reconstruction: An Almost
    /// Optimal Algorithm for Rational Reconstruction" by Monagan.
    pub fn maximal_quotient_reconstruction(
        v: &Integer,
        p: &Integer,
        acceptance_scale: Option<Integer>,
    ) -> Result<Rational, &'static str> {
        let mut acceptance_scale = match acceptance_scale {
            Some(t) => t.clone(),
            None => {
                // set t to 2^20*ceil(log2(m))
                let ceil_log2 = match &p {
                    Integer::Natural(n) => u64::BITS as u64 - (*n as u64).leading_zeros() as u64,
                    Integer::Double(n) => u128::BITS as u64 - (*n as u128).leading_zeros() as u64,
                    Integer::Large(n) => {
                        let mut pos = 0;
                        while let Some(p) = n.find_one(pos) {
                            if let Some(p2) = pos.checked_add(p) {
                                if p2 == u32::MAX {
                                    return Err("Could not reconstruct, as the log is too large");
                                }

                                pos += 1;
                            } else {
                                return Err("Could not reconstruct, as the log is too large");
                            }
                        }
                        pos as u64
                    }
                };

                &Integer::new(2i64 << 10) * &Integer::new(ceil_log2 as i64)
            }
        };

        if v.is_zero() {
            return if p > &acceptance_scale {
                Ok(Rational::zero())
            } else {
                Err("Could not reconstruct: u=0 and t <= m")
            };
        }

        let mut n = Integer::zero();
        let mut d = Integer::zero();
        let (mut t, mut old_t) = (Integer::one(), Integer::zero());
        let (mut r, mut old_r) = (if v.is_negative() { v + p } else { v.clone() }, p.clone());

        while !r.is_zero() && old_r > acceptance_scale {
            let q = &old_r / &r;
            if q > acceptance_scale {
                n = r.clone();
                d = t.clone();
                acceptance_scale = q.clone();
            }
            (r, old_r) = (&old_r - &(&q * &r), r);
            (t, old_t) = (&old_t - &(&q * &t), t);
        }

        if d.is_zero() || !Z.gcd(&n, &d).is_one() {
            return Err("Reconstruction failed");
        }
        if d < Integer::zero() {
            n = n.neg();
            d = d.neg();
        }

        Ok((n, d).into())
    }

    /// Return the rational number that corresponds to `f` evaluated at sample point `sample`,
    /// i.e. `f(sample)`, if such a number exists and if the evaluations were not unlucky.
    ///
    /// The procedure can be repeated with a different starting prime, by setting `prime_start`
    /// to a non-zero value.
    pub fn rational_reconstruction<
        F: Fn(&Zp, &[<Zp as Ring>::Element]) -> <Zp as Ring>::Element,
        R: Ring,
    >(
        f: F,
        sample: &[R::Element],
        prime_start: Option<usize>,
    ) -> Result<Rational, &'static str>
    where
        Zp: FiniteFieldCore<u32>,
        R::Element: ToFiniteField<u32>,
    {
        let mut cur_result = Integer::one();
        let mut prime_accum = Integer::one();
        let mut prime_sample_point = vec![];
        let mut prime_start = prime_start.unwrap_or(0);

        let mut last_guess = Rational::zero();
        for i in 0..sample.len() {
            if prime_start + i >= LARGE_U32_PRIMES.len() {
                return Err("Ran out of primes for rational reconstruction");
            }

            let p = LARGE_U32_PRIMES[prime_start]; // TODO: support u64
            prime_start += 1;

            let field = FiniteField::<u32>::new(p);
            prime_sample_point.clear();
            for x in sample {
                prime_sample_point.push(x.to_finite_field(&field));
            }

            let eval = f(&field, &prime_sample_point);

            // NOTE: check bounds if upgraded to u64 primes!
            let eval_conv = Integer::Natural(field.from_element(&eval).to_u64() as i64);

            if i == 0 {
                cur_result = eval_conv;
            } else {
                let new_result = Integer::chinese_remainder(
                    eval_conv,
                    cur_result.clone(),
                    Integer::Natural(p as i64),
                    prime_accum.clone(),
                );

                if cur_result == new_result {
                    return Ok(last_guess);
                }
                cur_result = new_result;
            }

            prime_accum *= &Integer::Natural(p as i64);

            if cur_result < Integer::zero() {
                cur_result += &prime_accum;
            }

            if let Ok(q) =
                Rational::maximal_quotient_reconstruction(&cur_result, &prime_accum, None)
            {
                if q == last_guess {
                    return Ok(q);
                } else {
                    last_guess = q;
                }
            }
        }

        Err("Reconstruction failed")
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.denominator == other.denominator {
            return self.numerator.cmp(&other.numerator);
        }

        let a = self.numerator_ref() * other.denominator_ref();
        let b = self.denominator_ref() * other.numerator_ref();

        a.cmp(&b)
    }
}

impl Add<Rational> for Rational {
    type Output = Rational;

    fn add(self, other: Rational) -> Self::Output {
        Q.add(&self, &other)
    }
}

impl Sub<Rational> for Rational {
    type Output = Rational;

    fn sub(self, other: Rational) -> Self::Output {
        self.add(&other.neg())
    }
}

impl Mul<Rational> for Rational {
    type Output = Rational;

    fn mul(self, other: Rational) -> Self::Output {
        Q.mul(&self, &other)
    }
}

impl Div<Rational> for Rational {
    type Output = Rational;

    fn div(self, other: Rational) -> Self::Output {
        Q.div(&self, &other)
    }
}

impl<'a> Add<&'a Rational> for Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        Q.add(&self, other)
    }
}

impl<'a> Sub<&'a Rational> for Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a> Mul<&'a Rational> for Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        Q.mul(&self, other)
    }
}

impl<'a> Div<&'a Rational> for Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        Q.div(&self, other)
    }
}

impl<'a, 'b> Add<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        Q.add(self, other)
    }
}

impl<'a, 'b> Sub<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        Q.sub(self, other)
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Q.neg(&self)
    }
}

impl<'a, 'b> Mul<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        Q.mul(self, other)
    }
}

impl<'a, 'b> Div<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        Q.div(self, other)
    }
}

impl<'a> AddAssign<&'a Rational> for Rational {
    fn add_assign(&mut self, other: &'a Rational) {
        Q.add_assign(self, other)
    }
}

impl<'a> SubAssign<&'a Rational> for Rational {
    fn sub_assign(&mut self, other: &'a Rational) {
        self.add_assign(&other.neg())
    }
}

impl<'a> MulAssign<&'a Rational> for Rational {
    fn mul_assign(&mut self, other: &'a Rational) {
        Q.mul_assign(self, other)
    }
}

impl<'a> DivAssign<&'a Rational> for Rational {
    fn div_assign(&mut self, other: &'a Rational) {
        Q.div_assign(self, other)
    }
}

impl AddAssign<Rational> for Rational {
    fn add_assign(&mut self, other: Rational) {
        Q.add_assign(self, &other)
    }
}

impl SubAssign<Rational> for Rational {
    fn sub_assign(&mut self, other: Rational) {
        self.add_assign(&other.neg())
    }
}

impl MulAssign<Rational> for Rational {
    fn mul_assign(&mut self, other: Rational) {
        Q.mul_assign(self, &other)
    }
}

impl DivAssign<Rational> for Rational {
    fn div_assign(&mut self, other: Rational) {
        Q.div_assign(self, &other)
    }
}

impl<'a> std::iter::Sum<&'a Self> for Rational {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Rational::zero(), |a, b| a + b)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::Atom,
        domains::{
            integer::Z,
            rational::{FractionField, Rational, Q},
            Field, Ring,
        },
        poly::polynomial::PolynomialRing,
    };

    #[test]
    fn rounding() {
        let r: Rational = (11, 10).into();
        let res = r.round_in_interval((1, 1).into(), (12, 10).into());
        assert_eq!(res, (1, 1).into());

        let r: Rational = (11, 10).into();
        let res = r.round_in_interval((2, 1).into(), (3, 1).into());
        assert_eq!(res, (2, 1).into());

        let r: Rational = (503, 1500).into();
        let res = r.round(&(1, 10).into());
        assert_eq!(res, (1, 3).into());

        let r: Rational = (-503, 1500).into();
        let res = r.round(&(1, 10).into());
        assert_eq!(res, (-1, 3).into());

        let r = crate::domains::float::Float::from(rug::Float::with_val(
            1000,
            rug::float::Constant::Pi,
        ))
        .to_rational();
        let res = r.round(&(1, 100000000).into());
        assert_eq!(res, (93343, 29712).into());
    }

    #[test]
    fn fraction_int() {
        let f = FractionField::new(Z);
        let b = f.neg(&f.nth(3));
        let d = f.div(&f.add(&f.nth(100), &b), &b);
        assert_eq!(d, f.to_element((-97).into(), 3.into(), false));
    }

    #[test]
    fn fraction_poly() {
        let poly = Atom::parse("-3/2*x^2+1/5x+4")
            .unwrap()
            .to_polynomial::<_, u8>(&Q, None);

        let f = FractionField::new(Z);
        let poly2 = poly.map_coeff(
            |c| f.to_element(c.numerator(), c.denominator(), false),
            f.clone(),
        );

        let p = PolynomialRing::from_poly(&poly2);
        let rat = p.to_rational_polynomial(&poly2);
        let f = FractionField::new(PolynomialRing::from_poly(&rat.numerator));

        let b = f.neg(&f.nth(3));
        let c = f.add(&rat, &b);
        let d = f.div(&c, &rat);

        let num = Atom::parse("-10-2*x+15*x^2")
            .unwrap()
            .to_polynomial::<_, u8>(&Z, None);
        let den = Atom::parse("-40-2*x+15*x^2")
            .unwrap()
            .to_polynomial::<_, u8>(&Z, None);

        assert_eq!(d, f.to_element(num, den, false));
    }
}
