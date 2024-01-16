use std::{
    fmt::{Display, Error, Formatter, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use rug::{
    integer::IntegerExt64, ops::Pow, Integer as ArbitraryPrecisionInteger,
    Rational as ArbitraryPrecisionRational,
};

use crate::{poly::gcd::LARGE_U32_PRIMES, state::State, utils};

use super::{
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::{Integer, IntegerRing},
    EuclideanDomain, Field, Ring,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RationalField;

impl RationalField {
    pub fn new() -> RationalField {
        RationalField
    }
}

/// A rational number.
///
/// Explicit construction of `Rational::Natural`
/// is only valid if the conventions are followed:
/// `Rational::Natural(n,d)` should have `d > 0` and
/// `gcd(n,d)=1`.
// TODO: convert to Rational(Integer, Integer)?
// TODO: prevent construction of explicit rational
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Rational {
    Natural(i64, i64),
    Large(ArbitraryPrecisionRational),
}

impl ToFiniteField<u32> for Rational {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Ring>::Element {
        match self {
            &Rational::Natural(n, d) => {
                let mut ff = field.to_element(n.rem_euclid(field.get_prime() as i64) as u32);

                if d != 1 {
                    let df = field.to_element(d.rem_euclid(field.get_prime() as i64) as u32);
                    field.div_assign(&mut ff, &df);
                }

                ff
            }
            Rational::Large(r) => field.div(
                &field.to_element(r.numer().mod_u(field.get_prime())),
                &field.to_element(r.denom().mod_u(field.get_prime())),
            ),
        }
    }
}

impl From<&Integer> for Rational {
    fn from(val: &Integer) -> Self {
        match val {
            Integer::Natural(n) => Rational::Natural(*n, 1),
            Integer::Large(l) => Rational::Large(l.into()),
        }
    }
}

impl Rational {
    pub fn new(mut num: i64, mut den: i64) -> Rational {
        if den == 0 {
            panic!("Division by zero");
        }

        let gcd = utils::gcd_signed(num, den);
        if gcd != 1 {
            if gcd == i64::MAX as u64 + 1 {
                // num = den = u64::MIN
                num = 1;
                den = 1;
            } else {
                num /= gcd as i64;
                den /= gcd as i64;
            }
        }

        if den < 0 {
            if let Some(neg) = den.checked_neg() {
                Rational::Natural(num, neg).neg()
            } else {
                Rational::Large(ArbitraryPrecisionRational::from((num, den)))
            }
        } else {
            Rational::Natural(num, den)
        }
    }

    pub fn from_num_den(num: Integer, den: Integer) -> Rational {
        match (num, den) {
            (Integer::Natural(n), Integer::Natural(d)) => Rational::new(n, d),
            (Integer::Natural(n), Integer::Large(d)) => {
                Rational::from_large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Large(n), Integer::Natural(d)) => {
                Rational::from_large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Large(n), Integer::Large(d)) => {
                Rational::from_large(ArbitraryPrecisionRational::from((n, d)))
            }
        }
    }

    pub fn from_large(r: ArbitraryPrecisionRational) -> Rational {
        if let Some(d) = r.denom().to_i64() {
            if let Some(n) = r.numer().to_i64() {
                return Rational::Natural(n, d);
            }
        }

        Rational::Large(r)
    }

    pub fn from_finite_field_u32(
        field: FiniteField<u32>,
        element: &<FiniteField<u32> as Ring>::Element,
    ) -> Rational {
        Rational::Natural(field.from_element(element) as i64, 1)
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Rational::Natural(n, _) => *n < 0,
            Rational::Large(r) => ArbitraryPrecisionInteger::from(r.numer().signum_ref()) == -1,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Rational::Natural(_, d) => *d == 1,
            Rational::Large(r) => r.is_integer(),
        }
    }

    pub fn numerator(&self) -> Integer {
        match self {
            Rational::Natural(n, _) => Integer::Natural(*n),
            Rational::Large(r) => Integer::Large(r.numer().clone()),
        }
    }

    pub fn denominator(&self) -> Integer {
        match self {
            Rational::Natural(_, d) => Integer::Natural(*d),
            Rational::Large(r) => Integer::Large(r.denom().clone()),
        }
    }

    pub fn zero() -> Rational {
        Rational::Natural(0, 1)
    }

    pub fn one() -> Rational {
        Rational::Natural(1, 1)
    }

    pub fn abs(&self) -> Rational {
        if self.is_negative() {
            self.clone().neg()
        } else {
            self.clone()
        }
    }

    pub fn is_zero(&self) -> bool {
        self == &Rational::Natural(0, 1)
    }

    pub fn is_one(&self) -> bool {
        self == &Rational::Natural(1, 1)
    }

    pub fn pow(&self, e: u64) -> Rational {
        if e > u32::MAX as u64 {
            panic!("Power of exponentation is larger than 2^32: {}", e);
        }
        let e = e as u32;

        match self {
            Rational::Natural(n1, d1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    if let Some(pd) = d1.checked_pow(e) {
                        return Rational::Natural(pn, pd);
                    }
                }

                Rational::Large(ArbitraryPrecisionRational::from((*n1, *d1)).pow(e))
            }
            Rational::Large(r) => Rational::Large(r.pow(e).into()),
        }
    }

    pub fn inv(&self) -> Rational {
        match self {
            Rational::Natural(n, d) => {
                if *n < 0 {
                    if let Some(neg) = n.checked_neg() {
                        Rational::Natural(-d, neg)
                    } else {
                        Rational::Large(ArbitraryPrecisionRational::from((*n, *d)).recip())
                    }
                } else {
                    Rational::Natural(*d, *n)
                }
            }
            Rational::Large(r) => Rational::from_large(r.clone().recip()),
        }
    }

    pub fn neg(&self) -> Rational {
        match self {
            Rational::Natural(n, d) => {
                if let Some(neg) = n.checked_neg() {
                    Rational::Natural(neg, *d)
                } else {
                    Rational::Large(ArbitraryPrecisionRational::from((*n, *d)).neg())
                }
            }
            Rational::Large(r) => Rational::from_large(r.neg().into()),
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
                    Integer::Large(n) => {
                        let mut pos = 0;
                        while let Some(p) = n.find_one_64(pos) {
                            if p == i64::MAX as u64 {
                                return Err("Could not reconstruct, as the log is too large");
                            }
                            pos = p + 1;
                        }
                        pos
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
        let (mut r, mut old_r) = (v.clone(), p.clone());

        while !r.is_one() && old_r > acceptance_scale {
            let q = &old_r / &r;
            if q > acceptance_scale {
                n = r.clone();
                d = t.clone();
                acceptance_scale = q.clone();
            }
            (r, old_r) = (&old_r - &(&q * &r), r);
            (t, old_t) = (&old_t - &(&q * &t), t);
        }

        if d.is_zero() || !IntegerRing::new().gcd(&n, &d).is_one() {
            return Err("Reconstruction failed");
        }
        if d < Integer::zero() {
            n = n.neg();
            d = d.neg();
        }

        Ok(Rational::from_num_den(n, d))
    }

    /// Return the rational number that corresponds to `f` evaluated at sample point `sample`,
    /// i.e. `f(sample)`, if such a number exists and if the evaluations were not unlucky.
    ///
    /// The procedure can be repeated with a different starting prime, by setting `prime_start`
    /// to a non-zero value.
    pub fn rational_reconstruction<
        F: Fn(
            &FiniteField<u32>,
            &[<FiniteField<u32> as Ring>::Element],
        ) -> <FiniteField<u32> as Ring>::Element,
        R: Ring,
    >(
        f: F,
        sample: &[R::Element],
        prime_start: Option<usize>,
    ) -> Result<Rational, &'static str>
    where
        FiniteField<u32>: FiniteFieldCore<u32>,
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

            let field = FiniteField::new(p);
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

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rational::Natural(n, d) => {
                if *d == 1 {
                    n.fmt(f)
                } else {
                    n.fmt(f)?;
                    f.write_char('/')?;
                    write!(f, "{}", d)
                }
            }
            Rational::Large(r) => r.fmt(f),
        }
    }
}

impl Display for RationalField {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Ring for RationalField {
    type Element = Rational;

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                if let Some(lcm) = d2.checked_mul(d1 / utils::gcd_signed(*d1, *d2) as i64) {
                    if let Some(num2) = n2.checked_mul(lcm / d2) {
                        if let Some(num1) = n1.checked_mul(lcm / d1) {
                            if let Some(num) = num1.checked_add(num2) {
                                let g = utils::gcd_signed(num, lcm) as i64;
                                return Rational::Natural(num / g, lcm / g);
                            }
                        }
                    }
                }
                Rational::from_large(
                    ArbitraryPrecisionRational::from((*n1, *d1))
                        + ArbitraryPrecisionRational::from((*n2, *d2)),
                )
            }
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::from_large(r1 + r2)
            }
            (Rational::Large(r1), Rational::Large(r2)) => Rational::from_large((r1 + r2).into()),
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                let gcd1 = utils::gcd_signed(*n1, *d2);
                let (n1, d2) = if gcd1 == i64::MAX as u64 + 1 {
                    (-1, -1)
                } else {
                    (n1 / gcd1 as i64, d2 / gcd1 as i64)
                };

                let gcd2 = utils::gcd_signed(*d1, *n2);
                let (d1, n2) = if gcd2 == i64::MAX as u64 + 1 {
                    (-1, -1)
                } else {
                    (d1 / gcd2 as i64, n2 / gcd2 as i64)
                };

                match (n2).checked_mul(n1) {
                    Some(nn) => match (d1).checked_mul(d2) {
                        Some(nd) => Rational::Natural(nn, nd),
                        None => Rational::Large(ArbitraryPrecisionRational::from((
                            nn,
                            ArbitraryPrecisionInteger::from(d1)
                                * ArbitraryPrecisionInteger::from(d2),
                        ))),
                    },
                    None => Rational::Large(ArbitraryPrecisionRational::from((
                        ArbitraryPrecisionInteger::from(n1) * ArbitraryPrecisionInteger::from(n2),
                        ArbitraryPrecisionInteger::from(d1) * ArbitraryPrecisionInteger::from(d2),
                    ))),
                }
            }
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::from_large(r1 * r2)
            }
            (Rational::Large(r1), Rational::Large(r2)) => Rational::from_large((r1 * r2).into()),
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
        self.add_assign(a, &(b * c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &(b * c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.neg()
    }

    fn zero(&self) -> Self::Element {
        Rational::Natural(0, 1)
    }

    fn one(&self) -> Self::Element {
        Rational::Natural(1, 1)
    }

    #[inline]
    fn nth(&self, n: u64) -> Self::Element {
        if n <= i64::MAX as u64 {
            Rational::Natural(n as i64, 1)
        } else {
            Rational::Large(ArbitraryPrecisionRational::from(n))
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e)
    }

    fn is_zero(a: &Self::Element) -> bool {
        match a {
            Rational::Natural(r, _) => *r == 0,
            Rational::Large(_) => false,
        }
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        match a {
            Rational::Natural(r, d) => *r == 1 && *d == 1,
            Rational::Large(_) => false,
        }
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn is_characteristic_zero(&self) -> bool {
        true
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.gen_range(range.0..range.1);
        Rational::Natural(r, 1)
    }

    fn fmt_display(
        &self,
        element: &Self::Element,
        _state: Option<&State>,
        _in_product: bool,
        f: &mut Formatter<'_>,
    ) -> Result<(), Error> {
        element.fmt(f)
    }
}

impl EuclideanDomain for RationalField {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Rational::Natural(0, 0)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), Rational::Natural(0, 0))
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (Rational::Natural(n1, d1), Rational::Natural(n2, d2)) => {
                let gcd_num = utils::gcd_signed(*n1, *n2);
                let gcd_den = utils::gcd_signed(*d1, *d2);

                let d1 = if gcd_den == i64::MAX as u64 + 1 {
                    -1
                } else {
                    *d1 / gcd_den as i64
                };

                let lcm = d2.checked_mul(d1);

                if gcd_num == i64::MAX as u64 + 1 || lcm.is_none() {
                    Rational::Large(ArbitraryPrecisionRational::from((
                        ArbitraryPrecisionInteger::from(gcd_num),
                        ArbitraryPrecisionInteger::from(*d2) * ArbitraryPrecisionInteger::from(d1),
                    )))
                } else {
                    Rational::Natural(gcd_num as i64, lcm.unwrap())
                }
            }
            (Rational::Natural(n1, d1), Rational::Large(r2))
            | (Rational::Large(r2), Rational::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Rational::from_large(ArbitraryPrecisionRational::from((
                    r1.numer().clone().gcd(r2.numer()),
                    r1.denom().clone().lcm(r2.denom()),
                )))
            }
            (Rational::Large(r1), Rational::Large(r2)) => {
                Rational::from_large(ArbitraryPrecisionRational::from((
                    r1.numer().clone().gcd(r2.numer()),
                    r1.denom().clone().lcm(r2.denom()),
                )))
            }
        }
    }
}

impl Field for RationalField {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        a.inv()
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if let (Rational::Large(n1), Rational::Large(n2)) = (self, other) {
            return n1.partial_cmp(n2);
        }

        let a = &self.numerator() * &other.denominator();
        let b = &self.denominator() * &other.numerator();

        a.partial_cmp(&b)
    }
}

impl Add<Rational> for Rational {
    type Output = Rational;

    fn add(self, other: Rational) -> Self::Output {
        RationalField::new().add(&self, &other)
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
        RationalField::new().mul(&self, &other)
    }
}

impl Div<Rational> for Rational {
    type Output = Rational;

    fn div(self, other: Rational) -> Self::Output {
        RationalField::new().div(&self, &other)
    }
}

impl<'a> Add<&'a Rational> for Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        RationalField::new().add(&self, other)
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
        RationalField::new().mul(&self, other)
    }
}

impl<'a> Div<&'a Rational> for Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        RationalField::new().div(&self, other)
    }
}

impl<'a, 'b> Add<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        RationalField::new().add(self, other)
    }
}

impl<'a, 'b> Sub<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        self.add(&other.neg())
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        RationalField::new().neg(&self)
    }
}

impl<'a, 'b> Mul<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        RationalField::new().mul(self, other)
    }
}

impl<'a, 'b> Div<&'a Rational> for &'b Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        RationalField::new().div(self, other)
    }
}

impl<'a> AddAssign<&'a Rational> for Rational {
    fn add_assign(&mut self, other: &'a Rational) {
        RationalField::new().add_assign(self, other)
    }
}

impl<'a> SubAssign<&'a Rational> for Rational {
    fn sub_assign(&mut self, other: &'a Rational) {
        self.add_assign(&other.neg())
    }
}

impl<'a> MulAssign<&'a Rational> for Rational {
    fn mul_assign(&mut self, other: &'a Rational) {
        RationalField::new().mul_assign(self, other)
    }
}

impl<'a> DivAssign<&'a Rational> for Rational {
    fn div_assign(&mut self, other: &'a Rational) {
        RationalField::new().div_assign(self, other)
    }
}

impl AddAssign<Rational> for Rational {
    fn add_assign(&mut self, other: Rational) {
        RationalField::new().add_assign(self, &other)
    }
}

impl SubAssign<Rational> for Rational {
    fn sub_assign(&mut self, other: Rational) {
        self.add_assign(&other.neg())
    }
}

impl MulAssign<Rational> for Rational {
    fn mul_assign(&mut self, other: Rational) {
        RationalField::new().mul_assign(self, &other)
    }
}

impl DivAssign<Rational> for Rational {
    fn div_assign(&mut self, other: Rational) {
        RationalField::new().div_assign(self, &other)
    }
}

impl<'a> std::iter::Sum<&'a Self> for Rational {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Rational::zero(), |a, b| a + b)
    }
}
