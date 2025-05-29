//! Defines and handles coefficients of Symbolica expressions.
//!
//! It includes implementations for different types of coefficients such as rational numbers,
//! floating-point numbers, finite field elements, rational polynomials and the conversions
//! between them in [Coefficient] and [CoefficientView].
//! These connect with Symbolica's standalone data types such as [Integer],
//! which should be used instead of [Coefficient] if possible.
//!
//! Additionally, the module contains functions for normalizing coefficients, converting
//! coefficients to floats with a specified precision, and setting the coefficient ring
//! for multivariate rational polynomials.
use std::{
    cmp::Ordering,
    f64::consts::LOG2_10,
    ops::{Add, Div, Mul, Neg},
    sync::Arc,
};

use ahash::HashMap;
use bytes::Buf;
use rug::{integer::Order, ops::NegAssign};
use smallvec::{SmallVec, smallvec};

use crate::{
    atom::{Atom, AtomView},
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring,
        algebraic_number::AlgebraicExtension,
        atom::AtomField,
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, ToFiniteField,
            Zp64,
        },
        float::{Complex, Float, NumericalFloatLike, Real, SingleFloat},
        integer::{Integer, IntegerRing, Z},
        rational::{Fraction, Q, Rational},
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    poly::{INLINED_EXPONENTS, Variable, polynomial::MultivariatePolynomial},
    state::{FiniteFieldIndex, State, Workspace},
};

pub trait ConvertToRing: Ring {
    /// Convert from an `Integer` to a Ring.
    fn element_from_integer(&self, number: Integer) -> Self::Element;

    /// Convert from a Symbolica `Coefficient` to a Ring.
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element;

    /// Convert from a Symbolica `CoefficientView` to a Ring.
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Self::Element;
}

/// A coefficient that can appear in a Symbolica expression.
/// In most cases, this is a rational number but it can also be a finite field element or
/// a rational polynomial. If no additions of mixed-types such as rational numbers
/// or floats is expected, use the standalone structs such as [Rational] and [Float] instead.
///
/// The borrowed version of this is [CoefficientView].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Coefficient {
    Complex(Complex<Rational>),
    Float(Complex<Float>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(RationalPolynomial<IntegerRing, u16>),
}

impl Coefficient {
    /// Construct a coefficient from a finite field element.
    pub fn from_finite_field(field: Zp64, element: FiniteFieldElement<u64>) -> Self {
        let index = State::get_or_insert_finite_field(field);
        Coefficient::FiniteField(element, index)
    }
}

impl From<i64> for Coefficient {
    fn from(value: i64) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<&i64> for Coefficient {
    fn from(value: &i64) -> Self {
        Coefficient::Complex(Rational::from(*value).into())
    }
}

impl From<i32> for Coefficient {
    fn from(value: i32) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<&i32> for Coefficient {
    fn from(value: &i32) -> Self {
        Coefficient::Complex(Rational::from(*value).into())
    }
}

impl From<f64> for Coefficient {
    fn from(value: f64) -> Self {
        Coefficient::Float(Float::with_val(53, value).into())
    }
}

impl From<&f64> for Coefficient {
    fn from(value: &f64) -> Self {
        Coefficient::Float(Float::with_val(53, value).into())
    }
}

impl From<(i64, i64)> for Coefficient {
    #[inline]
    fn from(r: (i64, i64)) -> Self {
        Coefficient::Complex(Rational::from(r).into())
    }
}

impl From<&(i64, i64)> for Coefficient {
    #[inline]
    fn from(r: &(i64, i64)) -> Self {
        Coefficient::Complex(Rational::from(*r).into())
    }
}

impl From<(i64, i64)> for CoefficientView<'_> {
    #[inline]
    fn from(r: (i64, i64)) -> Self {
        CoefficientView::Natural(r.0, r.1, 0, 1)
    }
}

impl From<(i64, i64, i64, i64)> for CoefficientView<'_> {
    #[inline]
    fn from(r: (i64, i64, i64, i64)) -> Self {
        CoefficientView::Natural(r.0, r.1, r.2, r.3)
    }
}

impl From<Integer> for Coefficient {
    fn from(value: Integer) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<(Integer, Integer)> for Coefficient {
    fn from(value: (Integer, Integer)) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<rug::Integer> for Coefficient {
    fn from(value: rug::Integer) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<rug::Rational> for Coefficient {
    fn from(value: rug::Rational) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<Rational> for Coefficient {
    fn from(value: Rational) -> Self {
        Coefficient::Complex(value.into())
    }
}

/*impl From<ComplexCoefficient> for Coefficient {
    fn from(value: ComplexCoefficient) -> Self {
        Coefficient::Complex(value)
    }
}*/

impl From<Float> for Coefficient {
    fn from(value: Float) -> Self {
        Coefficient::Float(value.into())
    }
}

impl From<Complex<Rational>> for Coefficient {
    fn from(value: Complex<Rational>) -> Self {
        Coefficient::Complex(value)
    }
}

impl From<Complex<Float>> for Coefficient {
    fn from(value: Complex<Float>) -> Self {
        Coefficient::Float(value)
    }
}

impl Default for Coefficient {
    fn default() -> Self {
        Coefficient::zero()
    }
}

impl PartialOrd for Coefficient {
    fn partial_cmp(&self, other: &Coefficient) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Complex<Rational> {
    pub fn gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        let gcd_re = self.re.gcd(&other.re);
        let gcd_im = self.im.gcd(&other.im);

        Complex::new(gcd_re, gcd_im)
    }
}

impl Ord for Coefficient {
    fn cmp(&self, other: &Coefficient) -> Ordering {
        match (self, other) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => r1
                .re
                .partial_cmp(&r2.re)
                .unwrap_or(Ordering::Equal)
                .then_with(|| r1.im.partial_cmp(&r2.im).unwrap_or(Ordering::Equal)),
            (Coefficient::FiniteField(n1, _), Coefficient::FiniteField(n2, _)) => n1.0.cmp(&n2.0),
            (Coefficient::Float(f1), Coefficient::Float(f2)) => f1
                .re
                .partial_cmp(&f2.re)
                .unwrap_or(Ordering::Equal)
                .then_with(|| f1.im.partial_cmp(&f2.im).unwrap_or(Ordering::Equal)),
            (Coefficient::RationalPolynomial(n1), Coefficient::RationalPolynomial(n2)) => {
                n1.internal_cmp(n2)
            }
            (Coefficient::Complex(_), _) => Ordering::Less,
            (_, Coefficient::Complex(_)) => Ordering::Greater,
            (Coefficient::Float(_), _) => Ordering::Less,
            (_, Coefficient::Float(_)) => Ordering::Greater,
            (Coefficient::FiniteField(_, _), _) => Ordering::Less,
            (_, Coefficient::FiniteField(_, _)) => Ordering::Greater,
        }
    }
}

impl Coefficient {
    pub fn new() -> Coefficient {
        Coefficient::zero()
    }

    pub fn zero() -> Coefficient {
        Coefficient::Complex(Complex::new_zero())
    }

    pub fn one() -> Coefficient {
        Coefficient::Complex(Complex::new(Rational::one(), Rational::zero()))
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Coefficient::Complex(r) => {
                r.re.is_negative() && r.im.is_zero() || r.im.is_negative() && r.re.is_zero()
            }
            Coefficient::Float(f) => {
                f.re.is_negative() && f.im.is_zero() || f.im.is_negative() && f.re.is_zero()
            }
            Coefficient::FiniteField(_, _) => false,
            Coefficient::RationalPolynomial(r) => r.numerator.lcoeff().is_negative(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Coefficient::Complex(r) => r.is_zero(),
            Coefficient::Float(f) => f.is_zero(),
            Coefficient::FiniteField(num, _field) => num.0 == 0,
            Coefficient::RationalPolynomial(r) => r.numerator.is_zero(),
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Coefficient::Complex(r) => r.is_one(),
            Coefficient::Float(f) => f.is_one(),
            Coefficient::FiniteField(num, field) => {
                let f = State::get_finite_field(*field);
                f.is_one(num)
            }
            Coefficient::RationalPolynomial(r) => r.numerator.is_one(),
        }
    }

    pub fn gcd(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => {
                Coefficient::Complex(r1.gcd(r2))
            }
            (Coefficient::FiniteField(_n1, i1), Coefficient::FiniteField(_n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(*i1).get_prime(),
                        State::get_finite_field(*i2).get_prime()
                    );
                }
                let f = State::get_finite_field(*i1);
                Coefficient::FiniteField(f.one(), *i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                panic!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(rp))
            | (Coefficient::RationalPolynomial(rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let p = RationalPolynomial::from_num_den(
                        rp.numerator.constant(r.re.numerator()),
                        rp.numerator.constant(r.re.denominator()),
                        &Z,
                        false,
                    );

                    let g = p.gcd(rp);
                    if g.is_constant() {
                        (g.numerator.lcoeff(), g.denominator.lcoeff()).into()
                    } else {
                        unreachable!()
                    }
                } else {
                    panic!("Cannot multiply complex number to rational polynomial")
                }
            }
            (Coefficient::RationalPolynomial(p1), Coefficient::RationalPolynomial(p2)) => {
                let r = if p1.get_variables() != p2.get_variables() {
                    let mut p1 = p1.clone();
                    let mut p2 = p2.clone();
                    p1.unify_variables(&mut p2);
                    p1.gcd(&p2)
                } else {
                    p1.gcd(p2)
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(_), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(_)) => Coefficient::Float(f.one()),
            (Coefficient::Float(f1), Coefficient::Float(_f2)) => Coefficient::Float(f1.one()),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                panic!("Cannot take gcd of floats and other coefficient types");
            }
        }
    }
}

impl Neg for Coefficient {
    type Output = Coefficient;

    fn neg(self) -> Coefficient {
        match self {
            Coefficient::Complex(r) => Coefficient::Complex(-r),
            Coefficient::Float(f) => Coefficient::Float(-f),
            Coefficient::FiniteField(n, i) => {
                let f = State::get_finite_field(i);
                Coefficient::FiniteField(f.neg(&n), i)
            }
            Coefficient::RationalPolynomial(p) => Coefficient::RationalPolynomial(-p),
        }
    }
}

impl Add for Coefficient {
    type Output = Coefficient;

    fn add(self, rhs: Coefficient) -> Coefficient {
        match (self, rhs) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => Coefficient::Complex(r1 + r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(rp))
            | (Coefficient::RationalPolynomial(rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let r2 = RationalPolynomial {
                        numerator: rp.numerator.constant(r.re.numerator()),
                        denominator: rp.denominator.constant(r.re.denominator()),
                    };
                    Coefficient::RationalPolynomial(&rp + &r2)
                } else {
                    panic!(
                        "Cannot add complex number to rational polynomial. Convert other number first?"
                    );
                }
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                };

                let r = &p1 + &p2;

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(r), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(r)) => {
                Coefficient::Float(Complex::new(f.re + r.re, f.im + r.im))
            }
            (Coefficient::Float(f1), Coefficient::Float(f2)) => Coefficient::Float(f1 + f2),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                panic!("Cannot add float to finite-field number or rational polynomial");
            }
        }
    }
}

impl Mul for Coefficient {
    type Output = Coefficient;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        match (self, rhs) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => Coefficient::Complex(r1 * r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.mul(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                panic!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(mut rp))
            | (Coefficient::RationalPolynomial(mut rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let gcd1 = Z.gcd(&r.re.numerator(), &rp.denominator.content());
                    let gcd2 = Z.gcd(&r.re.denominator(), &rp.numerator.content());
                    rp.numerator = rp
                        .numerator
                        .div_coeff(&gcd2)
                        .mul_coeff(r.re.numerator().div(&gcd1));
                    rp.denominator = rp
                        .denominator
                        .div_coeff(&gcd1)
                        .mul_coeff(r.re.denominator().div(&gcd2));
                    Coefficient::RationalPolynomial(rp)
                } else {
                    panic!(
                        "Cannot multiply complex number to non-complex number. Convert other number first?"
                    );
                }
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                };

                let r = &p1 * &p2;

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(r), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(r)) => Coefficient::Float(Complex::new(
                f.re.clone() * r.re.clone() - f.im.clone() * r.im.clone(),
                f.re * r.im + f.im * r.re,
            )),
            (Coefficient::Float(f1), Coefficient::Float(f2)) => Coefficient::Float(f1 * f2),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                panic!("Cannot multiply float to finite-field number or rational polynomial");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedRational<'a> {
    Natural(i64, i64),
    Large(SerializedLargeRational<'a>),
}

impl SerializedRational<'_> {
    pub fn is_zero(&self) -> bool {
        match self {
            SerializedRational::Natural(n, _) => *n == 0,
            SerializedRational::Large(r) => r.is_zero(),
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            SerializedRational::Natural(n, _) => *n < 0,
            SerializedRational::Large(r) => r.is_negative(),
        }
    }

    pub fn to_rat(&self) -> Rational {
        match self {
            SerializedRational::Natural(n, d) => Rational::from_unchecked(*n, *d),
            SerializedRational::Large(r) => r.to_rat(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerializedLargeRational<'a> {
    pub(crate) is_negative: bool,
    pub(crate) num_digits: &'a [u8],
    pub(crate) den_digits: &'a [u8],
}

impl SerializedLargeRational<'_> {
    pub fn is_zero(&self) -> bool {
        self.num_digits.is_empty()
    }

    pub fn is_negative(&self) -> bool {
        self.is_negative
    }

    pub fn to_rat(&self) -> Rational {
        if self.num_digits.is_empty() {
            return Rational::zero();
        }

        let mut num = rug::Integer::from_digits(self.num_digits, Order::Lsf);
        let den = rug::Integer::from_digits(self.den_digits, Order::Lsf);
        if self.is_negative {
            num.neg_assign();
        }

        Rational::from_unchecked(num, den)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SerializedRationalPolynomial<'a>(pub &'a [u8]);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SerializedFloat<'a>(pub &'a [u8]);

impl SerializedFloat<'_> {
    pub fn to_float(&self) -> Float {
        let mut d = self.0;
        let prec = d.get_u32_le();
        Float::deserialize(d, prec)
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
}

/// A view of a coefficient that keeps its complicated variants
/// serialized for efficiency.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CoefficientView<'a> {
    /// A complex number `(n_re, d_re, n_im, d_im)` that represents `n_re/d_re + i * n_im/d_im`.
    Natural(i64, i64, i64, i64),
    /// A floating point number `(re, im)` that represents `re + i * im`.
    Float(SerializedFloat<'a>, SerializedFloat<'a>),
    /// A large complex number `(re, im)` that represents `re + i * im`.
    Large(SerializedRational<'a>, SerializedRational<'a>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(SerializedRationalPolynomial<'a>),
}

impl ConvertToRing for Q {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number.into()
    }

    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        match number {
            Coefficient::Complex(r) => {
                if r.is_real() {
                    r.re
                } else {
                    panic!("Cannot convert complex to rational")
                }
            }
            Coefficient::Float(_) => panic!("Cannot convert float to rational"),
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to rational"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Rational {
        match number {
            CoefficientView::Natural(r, d, cr, _cd) => {
                if cr == 0 {
                    Rational::from_unchecked(r, d)
                } else {
                    panic!("Cannot convert complex number to rational")
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    r.to_rat()
                } else {
                    panic!("Cannot convert complex number to rational")
                }
            }
            CoefficientView::Float(_, _) => {
                panic!("Cannot convert float to rational")
            }
            CoefficientView::FiniteField(_, _) => {
                panic!("Cannot convert finite field to rational")
            }
            CoefficientView::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }
}

impl ConvertToRing for IntegerRing {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number
    }

    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Integer {
        match number {
            Coefficient::Complex(r) => {
                if r.is_real() {
                    if r.re.is_integer() {
                        r.re.numerator()
                    } else {
                        panic!("Cannot convert non-integer real number to integer")
                    }
                } else {
                    panic!("Cannot convert complex number to integer")
                }
            }
            Coefficient::Float(_) => panic!("Cannot convert float to integer"),
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to integer"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Integer {
        match number {
            CoefficientView::Natural(nr, dr, ni, _di) => {
                if dr == 1 && ni == 0 {
                    Integer::Natural(nr)
                } else {
                    panic!("Cannot convert rational or complex number to integer")
                }
            }
            CoefficientView::Large(r, i) => {
                if !i.is_zero() {
                    panic!("Cannot convert complex number to integer")
                }
                let r = r.to_rat();
                if !r.is_integer() {
                    panic!("Cannot convert rational to integer")
                }
                r.numerator()
            }
            CoefficientView::Float(_, _) => {
                panic!("Cannot convert float to integer")
            }
            CoefficientView::FiniteField(_, _) => {
                panic!("Cannot convert finite field to integer")
            }
            CoefficientView::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to integer")
            }
        }
    }
}

impl ConvertToRing for AtomField {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        Atom::num(number)
    }

    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        Atom::num(number)
    }

    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Self::Element {
        Atom::num(number.to_owned())
    }
}

impl<UField: FiniteFieldWorkspace> ConvertToRing for FiniteField<UField>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    Integer: ToFiniteField<UField>,
{
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number.to_finite_field(self)
    }

    #[inline]
    fn element_from_coefficient(
        &self,
        number: Coefficient,
    ) -> <FiniteField<UField> as Ring>::Element {
        match number {
            Coefficient::Complex(r) => {
                if r.is_real() {
                    self.div(
                        &r.re.numerator().to_finite_field(self),
                        &r.re.denominator().to_finite_field(self),
                    )
                } else {
                    panic!("Cannot convert complex number to finite field")
                }
            }
            Coefficient::Float(_) => panic!("Cannot convert float to finite field"),
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to other one"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> <FiniteField<UField> as Ring>::Element {
        match number {
            CoefficientView::Natural(n, d, i, _) => {
                if i == 0 {
                    self.div(
                        &Integer::new(n).to_finite_field(self),
                        &Integer::new(d).to_finite_field(self),
                    )
                } else {
                    panic!("Cannot convert complex number to finite field")
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    let l = r.to_rat();
                    self.div(
                        &l.numerator().to_finite_field(self),
                        &l.denominator().to_finite_field(self),
                    )
                } else {
                    panic!("Cannot convert complex number to finite field")
                }
            }
            CoefficientView::Float(_, _) => {
                panic!("Cannot convert float to finite field")
            }
            CoefficientView::FiniteField(_, _) => {
                panic!("Cannot convert finite field to other one")
            }
            CoefficientView::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }
}

impl ConvertToRing for AlgebraicExtension<Q> {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.constant(number.into())
    }

    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        match number {
            Coefficient::Complex(r) => {
                if r.is_real() {
                    self.constant(r.re.clone())
                } else {
                    if &self.poly().exponents == &[0, 2]
                        && self.poly().get_constant() == Rational::one()
                    {
                        self.to_element(
                            self.poly().monomial(r.im.clone(), vec![1])
                                + self.poly().constant(r.re.clone()),
                        )
                    } else {
                        panic!(
                            "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                        )
                    }
                }
            }
            Coefficient::Float(_) => panic!("Cannot convert float to rational"),
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to extension"),
            Coefficient::RationalPolynomial(_) => {
                // TODO: this may be possible!
                panic!("Cannot convert rational polynomial to extension")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Self::Element {
        match number {
            CoefficientView::Natural(r, d, cr, cd) => {
                if cr == 0 {
                    self.constant(Rational::from_unchecked(r, d))
                } else {
                    if &self.poly().exponents == &[0, 2]
                        && self.poly().get_constant() == Rational::one()
                    {
                        self.to_element(
                            self.poly()
                                .monomial(Rational::from_unchecked(cr, cd), vec![1])
                                + self.poly().constant(Rational::from_unchecked(r, d)),
                        )
                    } else {
                        panic!(
                            "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                        )
                    }
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    self.constant(r.to_rat())
                } else {
                    if &self.poly().exponents == &[0, 2]
                        && self.poly().get_constant() == Rational::one()
                    {
                        self.to_element(
                            self.poly().monomial(i.to_rat(), vec![1])
                                + self.poly().constant(r.to_rat()),
                        )
                    } else {
                        panic!(
                            "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                        )
                    }
                }
            }
            CoefficientView::Float(_, _) => {
                panic!("Cannot convert float to rational")
            }
            CoefficientView::FiniteField(_, _) => {
                panic!("Cannot convert finite field to rational")
            }
            CoefficientView::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }
}

impl CoefficientView<'_> {
    pub fn normalize(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(nr, dr, ni, di) => {
                Coefficient::Complex(Complex::new((*nr, *dr).into(), (*ni, *di).into()))
            }
            CoefficientView::Float(_, _)
            | CoefficientView::Large(_, _)
            | CoefficientView::FiniteField(_, _)
            | CoefficientView::RationalPolynomial(_) => self.to_owned(),
        }
    }

    pub fn to_owned(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(nr, dr, ni, di) => Coefficient::Complex(Complex::new(
                Rational::from_unchecked(*nr, *dr),
                Rational::from_unchecked(*ni, *di),
            )),
            CoefficientView::Large(rr, ri) => {
                Coefficient::Complex(Complex::new(rr.to_rat(), ri.to_rat()))
            }
            CoefficientView::Float(r, i) => {
                Coefficient::Float(Complex::new(r.to_float(), i.to_float()))
            }
            CoefficientView::FiniteField(num, field) => Coefficient::FiniteField(*num, *field),
            CoefficientView::RationalPolynomial(p) => {
                Coefficient::RationalPolynomial(p.deserialize())
            }
        }
    }

    pub fn pow(&self, other: &CoefficientView<'_>) -> (Coefficient, Coefficient, Coefficient) {
        // cannot simplify complex exponent
        if let CoefficientView::Natural(_, _, n, _) = other {
            if *n != 0 {
                return (Coefficient::one(), self.to_owned(), other.to_owned());
            }
        } else if let CoefficientView::Large(_, n) = other {
            if !n.is_zero() {
                return (Coefficient::one(), self.to_owned(), other.to_owned());
            }
        }

        fn rat_pow(
            mut base: Rational,
            mut exp: Rational,
        ) -> (Complex<Rational>, Rational, Rational) {
            if exp < 0.into() {
                base = base.inv();
                exp = -exp;
            }

            if base.is_negative() && !exp.is_integer() {
                let pow = exp.numerator() / exp.denominator();
                let rest = exp.numerator() - &pow * exp.denominator();

                if rest == 1 && exp.denominator_ref() == &2 {
                    (
                        Complex::new(Rational::zero(), base.pow(pow.to_i64().unwrap() as u64)),
                        base.abs().into(),
                        Rational::from_unchecked(rest, exp.denominator()).into(),
                    )
                } else {
                    (
                        base.pow(pow.to_i64().unwrap() as u64).into(),
                        base.into(),
                        Rational::from_unchecked(rest, exp.denominator()).into(),
                    )
                }
            } else {
                base = base.pow(exp.numerator().to_i64().unwrap() as u64);
                (
                    Rational::one().into(),
                    base.into(),
                    Rational::from_unchecked(Integer::one(), exp.denominator()).into(),
                )
            }
        }

        // TODO: normalize 4^1/3 to 2^(2/3)?
        match (self, other) {
            (
                &CoefficientView::Natural(n1, d1, ni1, di1),
                &CoefficientView::Natural(n2, d2, _, _),
            ) => {
                if ni1 == 0 {
                    let (coeff, base, exp) = rat_pow(
                        Rational::from_unchecked(n1, d1),
                        Rational::from_unchecked(n2, d2),
                    );
                    (coeff.into(), base.into(), exp.into())
                } else if d2 == 1 {
                    let r = Complex::new(
                        Rational::from_unchecked(n1, d1),
                        Rational::from_unchecked(ni1, di1),
                    )
                    .pow(n2.unsigned_abs());

                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Rational::one().into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(
                            Rational::from_unchecked(n1, d1),
                            Rational::from_unchecked(ni1, di1),
                        )
                        .into(),
                        Rational::from_unchecked(n2, d2).into(),
                    )
                }
            }
            (&CoefficientView::Natural(n1, d1, ni1, di1), &CoefficientView::Large(n2, _)) => {
                if ni1 == 0 {
                    let (coeff, base, exp) = rat_pow(Rational::from_unchecked(n1, d1), n2.to_rat());
                    (coeff.into(), base.into(), exp.into())
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(
                            Rational::from_unchecked(n1, d1),
                            Rational::from_unchecked(ni1, di1),
                        )
                        .into(),
                        n2.to_rat().into(),
                    )
                }
            }
            (&CoefficientView::RationalPolynomial(r), &CoefficientView::Natural(n2, d2, _, _)) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.deserialize().inv();
                    (
                        Coefficient::one(),
                        Coefficient::RationalPolynomial(r.pow(n2.unsigned_abs())),
                        Rational::from_unchecked(1, d2).into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Coefficient::RationalPolynomial(r.deserialize().pow(n2 as u64)),
                        Rational::from_unchecked(1, d2).into(),
                    )
                }
            }
            (&CoefficientView::Large(r, i), &CoefficientView::Natural(n2, d2, _, _)) => {
                if i.is_zero() {
                    let (coeff, base, exp) = rat_pow(r.to_rat(), Rational::from_unchecked(n2, d2));
                    (coeff.into(), base.into(), exp.into())
                } else if d2 == 1 {
                    let r = Complex::new(r.to_rat(), i.to_rat()).pow(n2.unsigned_abs());

                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Rational::one().into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(r.to_rat(), i.to_rat()).into(),
                        Rational::from_unchecked(n2, d2).into(),
                    )
                }
            }
            (&CoefficientView::Float(fr, fi), &CoefficientView::Natural(n2, d2, ni2, di2)) => {
                if d2 == 1 && ni2 == 0 {
                    let r = Complex::new(fr.to_float(), fi.to_float()).pow(n2.unsigned_abs());
                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Coefficient::one(),
                    )
                } else {
                    // FIXME: what precision should be used?
                    let f = fr.to_float();
                    let p = f.prec();
                    (
                        Coefficient::one(),
                        Complex::new(f, fi.to_float())
                            .powf(&Complex::new(
                                Rational::from_unchecked(n2, d2).to_multi_prec_float(p),
                                Rational::from_unchecked(ni2, di2).to_multi_prec_float(p),
                            ))
                            .into(),
                        Coefficient::one(),
                    )
                }
            }
            (&CoefficientView::Float(fr, fi), &CoefficientView::Large(r, d)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(f, fi.to_float())
                        .powf(&Complex::new(
                            r.to_rat().to_multi_prec_float(p),
                            d.to_rat().to_multi_prec_float(p),
                        ))
                        .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Natural(n2, d2, ni2, di2), &CoefficientView::Float(fr, fi)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(
                        Rational::from_unchecked(n2, d2).to_multi_prec_float(p),
                        Rational::from_unchecked(ni2, di2).to_multi_prec_float(p),
                    )
                    .powf(&Complex::new(f, fi.to_float()))
                    .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Large(r, d), &CoefficientView::Float(fr, fi)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(
                        r.to_rat().to_multi_prec_float(p),
                        d.to_rat().to_multi_prec_float(p),
                    )
                    .powf(&Complex::new(f, fi.to_float()))
                    .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Float(fr1, fi1), &CoefficientView::Float(fr2, fi2)) => {
                let p = if fi1.is_zero() && fi2.is_zero() {
                    fr1.to_float().powf(&fr2.to_float()).into()
                } else {
                    Complex::new(fr1.to_float(), fi1.to_float())
                        .powf(&Complex::new(fr2.to_float(), fi2.to_float()))
                        .into()
                };

                (Coefficient::one(), p, Coefficient::one())
            }
            (&CoefficientView::Large(r, i), &CoefficientView::Large(er, _)) => {
                if i.is_zero() {
                    let (coeff, base, exp) = rat_pow(r.to_rat(), er.to_rat());
                    (coeff.into(), base.into(), exp.into())
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(r.to_rat(), i.to_rat()).into(),
                        er.to_rat().into(),
                    )
                }
            }
            _ => {
                unimplemented!(
                    "Power of configuration {:?}^{:?} is not implemented",
                    self,
                    other
                );
            }
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            CoefficientView::Natural(_, d, i, _) => *i == 0 && *d == 1,
            CoefficientView::Float(_, _) => false,
            CoefficientView::Large(r, d) => d.is_zero() && r.to_rat().is_integer(),
            CoefficientView::FiniteField(_, _) => true,
            CoefficientView::RationalPolynomial(_) => false,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            CoefficientView::Natural(_, _, i, _) => *i == 0,
            CoefficientView::Float(_, i) => i.is_zero(),
            CoefficientView::Large(_, i) => i.is_zero(),
            CoefficientView::FiniteField(_, _) => true,
            CoefficientView::RationalPolynomial(_) => true,
        }
    }
}

impl PartialOrd for CoefficientView<'_> {
    fn partial_cmp(&self, other: &CoefficientView) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoefficientView<'_> {
    fn cmp(&self, other: &CoefficientView) -> Ordering {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Rational::from_unchecked(*n1, *d1)
                .cmp(&Rational::from_unchecked(*n2, *d2))
                .then_with(|| {
                    Rational::from_unchecked(*ni1, *di1).cmp(&Rational::from_unchecked(*ni2, *di2))
                }),
            (CoefficientView::Large(n1, d1), CoefficientView::Large(n2, d2)) => n1
                .to_rat()
                .cmp(&n2.to_rat())
                .then_with(|| d1.to_rat().cmp(&d2.to_rat())),
            (CoefficientView::FiniteField(n1, _), CoefficientView::FiniteField(n2, _)) => {
                n1.0.cmp(&n2.0)
            }
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(n2, d)) => {
                Rational::from_unchecked(*n1, *d1)
                    .cmp(&n2.to_rat())
                    .then_with(|| Rational::from_unchecked(*ni1, *di1).cmp(&d.to_rat()))
            }
            (CoefficientView::Large(n1, d), CoefficientView::Natural(n2, d2, ni2, di2)) => n1
                .to_rat()
                .cmp(&Rational::from_unchecked(*n2, *d2))
                .then_with(|| d.to_rat().cmp(&Rational::from_unchecked(*ni2, *di2))),
            (CoefficientView::Float(fr1, fi1), CoefficientView::Float(fr2, fi2)) => fr1
                .to_float()
                .partial_cmp(&fr2.to_float())
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    fi1.to_float()
                        .partial_cmp(&fi2.to_float())
                        .unwrap_or(Ordering::Equal)
                }),
            (CoefficientView::RationalPolynomial(n1), CoefficientView::RationalPolynomial(n2)) => {
                n1.deserialize().internal_cmp(&n2.deserialize())
            }
            (CoefficientView::Natural(_, _, _, _), _) => Ordering::Less,
            (_, CoefficientView::Natural(_, _, _, _)) => Ordering::Greater,
            (CoefficientView::Large(_, _), _) => Ordering::Less,
            (_, CoefficientView::Large(_, _)) => Ordering::Greater,
            (CoefficientView::Float(_, _), _) => Ordering::Less,
            (_, CoefficientView::Float(_, _)) => Ordering::Greater,
            (CoefficientView::FiniteField(_, _), _) => Ordering::Less,
            (_, CoefficientView::FiniteField(_, _)) => Ordering::Greater,
        }
    }
}

impl Add<CoefficientView<'_>> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Coefficient::Complex(
                Complex::new(
                    Rational::from_unchecked(n1, d1),
                    Rational::from_unchecked(ni1, di1),
                ) + Complex::new(
                    Rational::from_unchecked(n2, d2),
                    Rational::from_unchecked(ni2, di2),
                ),
            ),
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(r2, d))
            | (CoefficientView::Large(r2, d), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                Coefficient::Complex(
                    Complex::new(r2.to_rat(), d.to_rat())
                        + Complex::new(
                            Rational::from_unchecked(n1, d1),
                            Rational::from_unchecked(ni1, di1),
                        ),
                )
            }
            (CoefficientView::Large(r1, d1), CoefficientView::Large(r2, d2)) => {
                Coefficient::Complex(
                    Complex::new(r1.to_rat(), d1.to_rat()) + Complex::new(r2.to_rat(), d2.to_rat()),
                )
            }
            (CoefficientView::FiniteField(n1, i1), CoefficientView::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &n2), i1)
            }
            (CoefficientView::FiniteField(_, _), _) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (CoefficientView::Natural(n, d, ni, _di), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d, ni, _di)) => {
                if ni != 0 {
                    panic!("Cannot add complex number with polynomial");
                }
                let r = p.deserialize();
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(Integer::Natural(n)),
                    denominator: r.denominator.constant(Integer::Natural(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::Large(l, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l, d)) => {
                if !d.is_zero() {
                    panic!("Cannot add complex number with polynomial");
                }
                let r = p.deserialize();
                let l = l.to_rat();
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(l.numerator()),
                    denominator: r.denominator.constant(l.denominator()),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let mut p1 = p1.deserialize();
                let mut p2 = p2.deserialize();

                let r = if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                    &p1 + &p2
                } else {
                    &p1 + &p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (CoefficientView::Natural(n, d, ni, di), CoefficientView::Float(rf, ri))
            | (CoefficientView::Float(rf, ri), CoefficientView::Natural(n, d, ni, di)) => {
                Coefficient::Float(Complex::new(
                    rf.to_float() + Rational::from((n, d)),
                    ri.to_float() + Rational::from((ni, di)),
                ))
            }
            (CoefficientView::Large(r, i), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Large(r, i)) => Coefficient::Float(
                Complex::new(fr.to_float() + r.to_rat(), fi.to_float() + i.to_rat()),
            ),
            (CoefficientView::Float(r1, i1), CoefficientView::Float(r2, i2)) => Coefficient::Float(
                Complex::new(r1.to_float() + r2.to_float(), i1.to_float() + i2.to_float()),
            ),
            (CoefficientView::Float(_, _), CoefficientView::RationalPolynomial(_)) => {
                panic!("Cannot add float to rational polynomial");
            }
            (CoefficientView::RationalPolynomial(_), CoefficientView::Float(_, _)) => {
                panic!("Cannot add float to rational polynomial");
            }
        }
    }
}

impl Mul for CoefficientView<'_> {
    type Output = Coefficient;

    fn mul(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Coefficient::Complex(
                Complex::new(
                    Rational::from_unchecked(n1, d1),
                    Rational::from_unchecked(ni1, di1),
                ) * Complex::new(
                    Rational::from_unchecked(n2, d2),
                    Rational::from_unchecked(ni2, di2),
                ),
            ),
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(r2, d2))
            | (CoefficientView::Large(r2, d2), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                Coefficient::Complex(
                    Complex::new(r2.to_rat(), d2.to_rat())
                        * Complex::new(
                            Rational::from_unchecked(n1, d1),
                            Rational::from_unchecked(ni1, di1),
                        ),
                )
            }
            (CoefficientView::Large(r1, d1), CoefficientView::Large(r2, d2)) => {
                Coefficient::Complex(
                    Complex::new(r1.to_rat(), d1.to_rat()) * Complex::new(r2.to_rat(), d2.to_rat()),
                )
            }
            (CoefficientView::FiniteField(n1, i1), CoefficientView::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.mul(&n1, &n2), i1)
            }
            (CoefficientView::FiniteField(_, _), _) => {
                panic!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                panic!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
            }
            (CoefficientView::Natural(n, d, ni, _di), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d, ni, _di)) => {
                if ni != 0 {
                    panic!("Cannot multiply complex number with polynomial");
                }

                let mut r = p.deserialize();
                let (n, d) = (Integer::Natural(n), Integer::Natural(d));

                let gcd1 = Z.gcd(&n, &r.denominator.content());
                let gcd2 = Z.gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::Large(l, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l, d)) => {
                if !d.is_zero() {
                    panic!("Cannot multiply complex number with polynomial");
                }

                let mut r = p.deserialize();
                let l = l.to_rat();
                let (n, d) = (l.numerator_ref(), l.denominator_ref());

                let gcd1 = Z.gcd(n, &r.denominator.content());
                let gcd2 = Z.gcd(d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let mut p1 = p1.deserialize();
                let mut p2 = p2.deserialize();
                let r = if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                    &p1 * &p2
                } else {
                    &p1 * &p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                let c2: Complex<Fraction<IntegerRing>> =
                    Complex::new((n1, d1).into(), (ni1, di1).into());
                let c1 = Complex::new(fr.to_float(), fi.to_float());
                Complex::new(
                    c1.re.clone() * c2.re.clone() - c1.im.clone() * c2.im.clone(),
                    c1.re * c2.im + c1.im * c2.re,
                )
                .into()
            }
            (CoefficientView::Large(r, d), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Large(r, d)) => {
                let c2 = Complex::new(r.to_rat(), d.to_rat());
                let c1 = Complex::new(fr.to_float(), fi.to_float());

                Complex::new(
                    c1.re.clone() * c2.re.clone() - c1.im.clone() * c2.im.clone(),
                    c1.re * c2.im + c1.im * c2.re,
                )
                .into()
            }
            (CoefficientView::Float(r1, i1), CoefficientView::Float(r2, i2)) => {
                let c1 = Complex::new(r1.to_float(), i1.to_float());
                let c2 = Complex::new(r2.to_float(), i2.to_float());
                (c1 * c2).into()
            }
            (CoefficientView::Float(_, _), CoefficientView::RationalPolynomial(_)) => {
                panic!("Cannot multiply float to rational polynomial");
            }
            (CoefficientView::RationalPolynomial(_), CoefficientView::Float(_, _)) => {
                panic!("Cannot multiply float to rational polynomial");
            }
        }
    }
}

impl Add<i64> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: i64) -> Coefficient {
        match self {
            CoefficientView::Natural(n1, d1, ni1, di1) => Coefficient::Complex(
                Complex::new(
                    Rational::from_unchecked(n1, d1),
                    Rational::from_unchecked(ni1, di1),
                ) + Rational::from(other),
            ),
            CoefficientView::Float(r, i) => {
                Coefficient::Float(Complex::new(r.to_float() + other, i.to_float()))
            }
            CoefficientView::Large(r1, d1) => {
                Coefficient::Complex(Complex::new(r1.to_rat(), d1.to_rat()) + Rational::from(other))
            }
            CoefficientView::FiniteField(n1, i1) => {
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &f.element_from_coefficient(other.into())), i1)
            }
            CoefficientView::RationalPolynomial(p) => {
                let p = p.deserialize();
                let a = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Natural(other)),
                    denominator: p.denominator.constant(Integer::Natural(1)),
                };

                Coefficient::RationalPolynomial(&p + &a)
            }
        }
    }
}

impl TryFrom<Atom> for i64 {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for i64 {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for i64 {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            if let CoefficientView::Natural(n, 1, 0, 1) = n.get_coeff_view() {
                Ok(n)
            } else {
                Err("Not an i64")
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Rational {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Rational {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Rational {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, _di) => {
                    if ni == 0 {
                        Ok(Rational::from_unchecked(n, d))
                    } else {
                        Err("Not a rational")
                    }
                }
                CoefficientView::Large(r, d) => {
                    if d.is_zero() {
                        Ok(r.to_rat())
                    } else {
                        Err("Not a rational")
                    }
                }
                _ => Err("Not a rational"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Float {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Float {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Float {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Float(r, i) => {
                    if i.is_zero() {
                        Ok(r.to_float())
                    } else {
                        Err("Cannot convert complex number to float")
                    }
                }
                _ => Err("Not a float"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Float(f, i) => Ok(Complex::new(f.to_float(), i.to_float())),
                _ => Err("Not a float"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => Ok(Complex::new(
                    Rational::from_unchecked(n, d),
                    Rational::from_unchecked(ni, di),
                )),
                CoefficientView::Large(r, i) => Ok(Complex::new(r.to_rat(), i.to_rat())),
                _ => Err("Not a rational"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl AtomView<'_> {
    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    pub(crate) fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.set_coefficient_ring_with_ws_into(vars, ws, &mut out);
            out.into_inner()
        })
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    pub(crate) fn set_coefficient_ring_with_ws_into(
        &self,
        vars: &Arc<Vec<Variable>>,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        match self {
            AtomView::Num(n) => {
                if let CoefficientView::RationalPolynomial(r) = n.get_coeff_view() {
                    let r = r.deserialize();
                    let old_var_map = r.get_variables();
                    if old_var_map != vars {
                        if old_var_map.iter().all(|x| vars.contains(x)) {
                            let n = r.numerator.rearrange_with_growth(vars).unwrap();
                            let d = r.denominator.rearrange_with_growth(vars).unwrap();
                            let r = RationalPolynomial {
                                numerator: n,
                                denominator: d,
                            };
                            out.to_num(Coefficient::RationalPolynomial(r));
                            true
                        } else {
                            let mut n1 = workspace.new_atom();
                            r.numerator.to_expression_with_map(
                                workspace,
                                &HashMap::default(),
                                &mut n1,
                            );

                            let mut n1_conv = workspace.new_atom();
                            n1.as_view().set_coefficient_ring_with_ws_into(
                                vars,
                                workspace,
                                &mut n1_conv,
                            );

                            let mut n2 = workspace.new_atom();
                            r.denominator.to_expression_with_map(
                                workspace,
                                &HashMap::default(),
                                &mut n2,
                            );

                            let mut n2_conv = workspace.new_atom();
                            n2.as_view().set_coefficient_ring_with_ws_into(
                                vars,
                                workspace,
                                &mut n2_conv,
                            );

                            // create n1/n2
                            let mut n3 = workspace.new_atom();
                            let mut exp = workspace.new_atom();
                            exp.to_num(Coefficient::Complex(Rational::from(-1i64).into()));
                            n3.to_pow(n2_conv.as_view(), exp.as_view());

                            let mut m = workspace.new_atom();
                            let mm = m.to_mul();
                            mm.extend(n1_conv.as_view());
                            mm.extend(n3.as_view());
                            m.as_view().normalize(workspace, out);
                            true
                        }
                    } else {
                        out.set_from_view(self);
                        false
                    }
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Var(v) => {
                let id = v.get_symbol();
                if vars.contains(&id.into()) {
                    // change variable into coefficient
                    let mut poly = MultivariatePolynomial::new(&Z, None, vars.clone());
                    let mut e: SmallVec<[u16; INLINED_EXPONENTS]> = smallvec![0; vars.len()];
                    e[vars.iter().position(|x| *x == id.into()).unwrap()] = 1;
                    poly.append_monomial(Integer::one(), &e);
                    let den = poly.one();

                    out.to_num(Coefficient::RationalPolynomial(RationalPolynomial {
                        numerator: poly,
                        denominator: den,
                    }));
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = workspace.new_atom();
                if base.set_coefficient_ring_with_ws_into(vars, workspace, &mut nb) {
                    let mut o = workspace.new_atom();
                    o.to_pow(nb.as_view(), exp);

                    o.as_view().normalize(workspace, out);
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Mul(m) => {
                let mut o = workspace.new_atom();
                let mul = o.to_mul();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in m {
                    changed |= arg.set_coefficient_ring_with_ws_into(vars, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_normalized(!changed);

                if !changed {
                    mul.set_has_coefficient(m.has_coefficient());
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, out);
                    true
                }
            }
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a {
                    changed |= arg.set_coefficient_ring_with_ws_into(vars, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_normalized(!changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, out);
                    true
                }
            }
            AtomView::Fun(_) => {
                // do not propagate into functions
                out.set_from_view(self);
                false
            }
        }
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec``.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    pub(crate) fn coefficients_to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
        let binary_prec = (decimal_prec as f64 * LOG2_10).ceil() as u32;

        Workspace::get_local().with(|ws| self.to_float_impl(binary_prec, true, false, ws, out))
    }

    fn to_float_impl(
        &self,
        binary_prec: u32,
        enter_function: bool,
        enter_exponent: bool,
        ws: &Workspace,
        out: &mut Atom,
    ) {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => {
                    out.to_num(Coefficient::Float(Complex::new(
                        Float::with_val(binary_prec, n) / Float::with_val(binary_prec, d),
                        Float::with_val(binary_prec, ni) / Float::with_val(binary_prec, di),
                    )));
                }
                CoefficientView::Float(r, i) => {
                    let mut f = r.to_float();
                    let mut g = i.to_float();
                    if f.prec() > binary_prec || g.prec() > binary_prec {
                        f.set_prec(binary_prec);
                        g.set_prec(binary_prec);
                        out.to_num(Coefficient::Float(Complex::new(f, g)));
                    } else {
                        out.set_from_view(self);
                    }
                }
                CoefficientView::Large(r, d) => {
                    out.to_num(Coefficient::Float(Complex::new(
                        r.to_rat().to_multi_prec_float(binary_prec),
                        d.to_rat().to_multi_prec_float(binary_prec),
                    )));
                }
                CoefficientView::FiniteField(_, _) => {
                    panic!("Cannot convert finite field to float");
                }
                CoefficientView::RationalPolynomial(_) => {
                    panic!("Cannot convert rational polynomial to float");
                }
            },
            AtomView::Var(v) => {
                let s = v.get_symbol();

                match s {
                    Atom::PI => {
                        out.to_num(Coefficient::Float(
                            Float::with_val(binary_prec, rug::float::Constant::Pi).into(),
                        ));
                    }
                    Atom::E => {
                        out.to_num(Coefficient::Float(
                            Float::with_val(binary_prec, 1).exp().into(),
                        ));
                    }
                    _ => {
                        out.set_from_view(self);
                    }
                }
            }
            AtomView::Fun(f) => {
                if enter_function {
                    let mut o = ws.new_atom();
                    let ff = o.to_fun(f.get_symbol());

                    let mut na = ws.new_atom();
                    for a in f.iter() {
                        a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                        ff.add_arg(na.as_view());
                    }

                    o.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = ws.new_atom();
                base.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut nb);

                let mut o = ws.new_atom();

                if enter_exponent {
                    let mut ne = ws.new_atom();
                    exp.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut ne);
                    o.to_pow(nb.as_view(), ne.as_view());
                } else {
                    o.to_pow(nb.as_view(), exp);
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Mul(m) => {
                let mut o = ws.new_atom();
                let mm = o.to_mul();

                let mut na = ws.new_atom();
                for a in m.iter() {
                    a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                    mm.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Add(a) => {
                let mut o = ws.new_atom();
                let aa = o.to_add();

                let mut na = ws.new_atom();
                for a in a.iter() {
                    a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                    aa.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
        }
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    pub(crate) fn rationalize_coefficients(&self, relative_error: &Rational) -> Atom {
        let mut a = Atom::new();
        Workspace::get_local().with(|ws| {
            self.map_coefficient_impl(
                |c| match c {
                    CoefficientView::Float(r, i) => Coefficient::Complex(Complex::new(
                        r.to_float().to_rational().round(relative_error),
                        i.to_float().to_rational().round(relative_error),
                    )),
                    CoefficientView::Natural(n, d, ni, di) => {
                        let r = Rational::from_unchecked(n, d);
                        let real = r.round(relative_error).into();

                        let r = Rational::from_unchecked(ni, di);
                        let imag = r.round(relative_error).into();

                        Coefficient::Complex(Complex::new(real, imag))
                    }
                    CoefficientView::Large(r, d) => Coefficient::Complex(Complex::new(
                        r.to_rat().round(relative_error),
                        d.to_rat().round(relative_error),
                    )),
                    _ => c.to_owned(),
                },
                true,
                false,
                ws,
                &mut a,
            )
        });
        a
    }

    /// Map all coefficients using a given function.
    pub(crate) fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
    ) -> Atom {
        let mut a = Atom::new();
        self.map_coefficient_into(f, &mut a);
        a
    }

    /// Map all coefficients using a given function.
    pub(crate) fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        Workspace::get_local().with(|ws| self.map_coefficient_impl(f, true, true, ws, out))
    }

    fn map_coefficient_impl<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        coeff_map: F,
        enter_function: bool,
        enter_exponent: bool,
        ws: &Workspace,
        out: &mut Atom,
    ) {
        match self {
            AtomView::Num(n) => {
                out.to_num(coeff_map(n.get_coeff_view()));
            }
            AtomView::Var(_) => out.set_from_view(self),
            AtomView::Fun(f) => {
                if enter_function {
                    let mut o = ws.new_atom();
                    let ff = o.to_fun(f.get_symbol());

                    let mut na = ws.new_atom();
                    for a in f.iter() {
                        a.map_coefficient_impl(
                            coeff_map,
                            enter_function,
                            enter_exponent,
                            ws,
                            &mut na,
                        );
                        ff.add_arg(na.as_view());
                    }

                    o.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = ws.new_atom();
                base.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut nb);

                let mut o = ws.new_atom();

                if enter_exponent {
                    let mut ne = ws.new_atom();
                    exp.map_coefficient_impl(
                        coeff_map,
                        enter_function,
                        enter_exponent,
                        ws,
                        &mut ne,
                    );
                    o.to_pow(nb.as_view(), ne.as_view());
                } else {
                    o.to_pow(nb.as_view(), exp);
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Mul(m) => {
                let mut o = ws.new_atom();
                let mm = o.to_mul();

                let mut na = ws.new_atom();
                for a in m.iter() {
                    a.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut na);
                    mm.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Add(a) => {
                let mut o = ws.new_atom();
                let aa = o.to_add();

                let mut na = ws.new_atom();
                for a in a.iter() {
                    a.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut na);
                    aa.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{Atom, AtomCore},
        domains::float::Float,
        parse,
        printer::{AtomPrinter, PrintOptions},
        symbol,
    };

    use super::Coefficient;

    #[test]
    fn coeff_conversion() {
        let expr = parse!("v1*coeff(v2+v3/v4)+v1*coeff(v2)");
        let res = parse!("coeff((v3+2*v2*v4)/v4)*v1");
        assert_eq!(expr - &res, Atom::new());
    }

    #[test]
    fn coeff_conversion_large() {
        let expr = parse!(
            "coeff(-8123781237821378123128937128937211238971238*v2-1289378192371289372891378127893)"
        );
        let res = parse!(
            "1289378192371289372891378127893+8123781237821378123128937128937211238971238*coeff(v2)"
        );
        assert_eq!(expr + &res, Atom::new());
    }

    #[test]
    fn coefficient_ring() {
        let expr = parse!("v1*v3+v1*(v2+2)^-1*(v2+v3+1)");

        let v2 = symbol!("v2");
        let expr_yz = expr.set_coefficient_ring(&Arc::new(vec![v2.into(), symbol!("v3").into()]));

        let a = ((&expr_yz + &Atom::num((1, 2))) * &Atom::num((3, 4))).expand();

        let a = (a / &Atom::num((3, 4)) - &Atom::num((1, 2))).expand();

        let a = a.set_coefficient_ring(&Arc::new(vec![]));

        let expr = expr.replace(v2).with(Atom::num(3)).expand();

        let a = a.replace(v2).with(Atom::num(3)).expand();

        assert_eq!(a, expr);
    }

    #[test]
    fn float() {
        let expr = parse!("1/2 x + 5.8912734891723 + sin(1.2334)");
        let c = Coefficient::Float(Float::with_val(200, rug::float::Constant::Pi).into());
        let expr = expr * &Atom::num(c);
        let r = format!(
            "{}",
            AtomPrinter::new_with_options(
                expr.expand().as_view(),
                PrintOptions::file_no_namespace()
            )
        );
        assert_eq!(
            r,
            "1.57079632679489661923132169163975144209858469968755291048747*x+21.4724504210349"
        );
    }

    #[test]
    fn float_convert() {
        let expr = parse!("1/2 x + 238947/128903718927 + sin(3/4)");
        let expr = expr.coefficients_to_float(60);
        let r = format!(
            "{}",
            AtomPrinter::new_with_options(expr.as_view(), PrintOptions::file_no_namespace())
        );
        assert_eq!(
            r,
            "5.00000000000000000000000000000000000000000000000000000000000e-1*x+6.81640613709185816359170669566511987261485697756222332885129e-1"
        );
    }

    #[test]
    fn float_to_rat() {
        let expr = parse!("1/2 x + 238947/128903718927 + sin(3/4)");
        let expr = expr.coefficients_to_float(60);
        let expr = expr.rationalize_coefficients(&(1, 10000).into());
        assert_eq!(expr, parse!("1/2*x+137/201"));
    }
}
