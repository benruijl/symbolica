use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul},
    sync::Arc,
};

use ahash::HashMap;
use rug::{
    integer::Order,
    ops::{NegAssign, Pow as RPow},
    Integer as MultiPrecisionInteger, Rational as MultiPrecisionRational,
};
use smallvec::{smallvec, SmallVec};

use crate::{
    domains::{
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, ToFiniteField,
        },
        integer::{Integer, IntegerRing, Z},
        rational::{Rational, RationalField},
        rational_polynomial::RationalPolynomial,
        EuclideanDomain, Field, Ring,
    },
    poly::{polynomial::MultivariatePolynomial, Variable, INLINED_EXPONENTS},
    representations::{Atom, AtomView},
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
/// a rational polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coefficient {
    Rational(Rational),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(RationalPolynomial<IntegerRing, u16>),
}

impl From<i64> for Coefficient {
    fn from(value: i64) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<i32> for Coefficient {
    fn from(value: i32) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<(i64, i64)> for Coefficient {
    #[inline]
    fn from(r: (i64, i64)) -> Self {
        Coefficient::Rational(r.into())
    }
}

impl From<Integer> for Coefficient {
    fn from(value: Integer) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<(Integer, Integer)> for Coefficient {
    fn from(value: (Integer, Integer)) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<MultiPrecisionInteger> for Coefficient {
    fn from(value: MultiPrecisionInteger) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<MultiPrecisionRational> for Coefficient {
    fn from(value: MultiPrecisionRational) -> Self {
        Coefficient::Rational(value.into())
    }
}

impl From<Rational> for Coefficient {
    fn from(value: Rational) -> Self {
        Coefficient::Rational(value)
    }
}

impl Default for Coefficient {
    fn default() -> Self {
        Coefficient::zero()
    }
}

impl Coefficient {
    pub fn new() -> Coefficient {
        Coefficient::zero()
    }

    pub fn zero() -> Coefficient {
        Coefficient::Rational(Rational::zero())
    }

    pub fn one() -> Coefficient {
        Coefficient::Rational(Rational::one())
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Coefficient::Rational(r) => r.is_zero(),
            Coefficient::FiniteField(num, _field) => num.0 == 0,
            Coefficient::RationalPolynomial(r) => r.numerator.is_zero(),
        }
    }
}

impl Add for Coefficient {
    type Output = Coefficient;

    fn add(self, rhs: Coefficient) -> Coefficient {
        match (self, rhs) {
            (Coefficient::Rational(r1), Coefficient::Rational(r2)) => {
                Coefficient::Rational(r1 + r2)
            }
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
            (Coefficient::FiniteField(_, _), _) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (_, Coefficient::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (Coefficient::Rational(r), Coefficient::RationalPolynomial(rp))
            | (Coefficient::RationalPolynomial(rp), Coefficient::Rational(r)) => {
                let r2 = RationalPolynomial {
                    numerator: rp.numerator.constant(r.numerator()),
                    denominator: rp.denominator.constant(r.denominator()),
                };
                Coefficient::RationalPolynomial(&rp + &r2)
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
        }
    }
}

impl Mul for Coefficient {
    type Output = Coefficient;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        match (self, rhs) {
            (Coefficient::Rational(r1), Coefficient::Rational(r2)) => {
                Coefficient::Rational(r1 * r2)
            }
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
            (Coefficient::FiniteField(_, _), _) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (_, Coefficient::FiniteField(_, _)) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (Coefficient::Rational(r), Coefficient::RationalPolynomial(mut rp))
            | (Coefficient::RationalPolynomial(mut rp), Coefficient::Rational(r)) => {
                let gcd1 = Z.gcd(&r.numerator(), &rp.denominator.content());
                let gcd2 = Z.gcd(&r.denominator(), &rp.numerator.content());
                rp.numerator = rp
                    .numerator
                    .div_coeff(&gcd2)
                    .mul_coeff(r.numerator().div(&gcd1));
                rp.denominator = rp
                    .denominator
                    .div_coeff(&gcd1)
                    .mul_coeff(r.denominator().div(&gcd2));
                Coefficient::RationalPolynomial(rp)
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
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerializedRational<'a> {
    pub(crate) is_negative: bool,
    pub(crate) num_digits: &'a [u8],
    pub(crate) den_digits: &'a [u8],
}

impl<'a> SerializedRational<'a> {
    pub fn is_negative(&self) -> bool {
        self.is_negative
    }

    pub fn to_rat(&self) -> MultiPrecisionRational {
        let mut num = MultiPrecisionInteger::from_digits(self.num_digits, Order::Lsf);
        let den = MultiPrecisionInteger::from_digits(self.den_digits, Order::Lsf);
        if self.is_negative {
            num.neg_assign();
        }

        MultiPrecisionRational::from((num, den))
    }
}

/// A view of a coefficient that keeps GMP rationals serialized.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CoefficientView<'a> {
    Natural(i64, i64),
    Large(SerializedRational<'a>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(&'a RationalPolynomial<IntegerRing, u16>),
}

impl ConvertToRing for RationalField {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number.into()
    }

    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        match number {
            Coefficient::Rational(r) => r,
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to rational"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Rational {
        match number {
            CoefficientView::Natural(r, d) => Rational::Natural(r, d),
            CoefficientView::Large(r) => Rational::Large(r.to_rat()),
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
            Coefficient::Rational(r) => {
                debug_assert!(r.is_integer());
                r.numerator()
            }
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to integer"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Integer {
        match number {
            CoefficientView::Natural(r, d) => {
                debug_assert!(d == 1);
                Integer::Natural(r)
            }
            CoefficientView::Large(r) => {
                let r = r.to_rat();
                debug_assert!(r.denom() == &1);
                Integer::from_large(r.numer().clone())
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
            Coefficient::Rational(r) => self.div(
                &r.numerator().to_finite_field(self),
                &r.denominator().to_finite_field(self),
            ),
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
            CoefficientView::Natural(n, d) => self.div(
                &Integer::new(n).to_finite_field(self),
                &Integer::new(d).to_finite_field(self),
            ),
            CoefficientView::Large(r) => {
                let (n, d) = r.to_rat().into_numer_denom();
                self.div(
                    &Integer::Large(n).to_finite_field(self),
                    &Integer::Large(d).to_finite_field(self),
                )
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

impl CoefficientView<'_> {
    pub fn normalize(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(num, den) => match Rational::new(*num, *den) {
                Rational::Natural(n, d) => Coefficient::Rational((n, d).into()),
                Rational::Large(l) => Coefficient::Rational(l.into()),
            },
            CoefficientView::Large(_)
            | CoefficientView::FiniteField(_, _)
            | CoefficientView::RationalPolynomial(_) => self.to_owned(),
        }
    }

    pub fn to_owned(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(num, den) => Coefficient::Rational((*num, *den).into()),
            CoefficientView::Large(r) => Coefficient::Rational(r.to_rat().into()),
            CoefficientView::FiniteField(num, field) => Coefficient::FiniteField(*num, *field),
            CoefficientView::RationalPolynomial(p) => Coefficient::RationalPolynomial((*p).clone()),
        }
    }

    pub fn pow(&self, other: &CoefficientView<'_>) -> (Coefficient, Coefficient) {
        // TODO: normalize 4^1/3 to 2^(2/3)?
        match (self, other) {
            (&CoefficientView::Natural(mut n1, mut d1), &CoefficientView::Natural(mut n2, d2)) => {
                if n2 < 0 {
                    if n1 == 0 {
                        panic!("Division by 0");
                    }

                    n2 = n2.saturating_abs();
                    (n1, d1) = (d1, n1);
                }

                if n2 <= u32::MAX as i64 {
                    if let Some(pn) = n1.checked_pow(n2 as u32) {
                        if let Some(pd) = d1.checked_pow(n2 as u32) {
                            // TODO: simplify 4^(1/2)
                            return ((pn, pd).into(), (1, d2).into());
                        }
                    }

                    (
                        MultiPrecisionRational::from((n1, d1)).pow(n2 as u32).into(),
                        (1, d2).into(),
                    )
                } else {
                    panic!("Power is too large: {}", n2);
                }
            }
            (&CoefficientView::RationalPolynomial(r), &CoefficientView::Natural(n2, d2)) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.clone().inv();
                    (
                        Coefficient::RationalPolynomial(r.pow(n2.unsigned_abs())),
                        (1, d2).into(),
                    )
                } else {
                    (
                        Coefficient::RationalPolynomial(r.pow(n2 as u64)),
                        (1, d2).into(),
                    )
                }
            }
            (&CoefficientView::Large(r), &CoefficientView::Natural(n2, d2)) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.to_rat().clone().recip();
                    (r.pow(n2.unsigned_abs() as u32).into(), (1, d2).into())
                } else {
                    (r.to_rat().pow(n2 as u32).into(), (1, d2).into())
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
            CoefficientView::Natural(_, d) => *d == 1,
            CoefficientView::Large(r) => r.to_rat().is_integer(),
            CoefficientView::FiniteField(_, _) => true,
            CoefficientView::RationalPolynomial(_) => false,
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
            (&CoefficientView::Natural(n1, d1), &CoefficientView::Natural(n2, d2)) => {
                // TODO: improve
                if n1 < 0 && n2 > 0 {
                    return Ordering::Less;
                }
                if n1 > 0 && n2 < 0 {
                    return Ordering::Greater;
                }

                match n1.checked_mul(d2) {
                    Some(a1) => match n2.checked_mul(d1) {
                        Some(a2) => a1.cmp(&a2),
                        None => MultiPrecisionInteger::from(a1).cmp(
                            &(MultiPrecisionInteger::from(n2) * MultiPrecisionInteger::from(d1)),
                        ),
                    },
                    None => (MultiPrecisionInteger::from(n1) * MultiPrecisionInteger::from(d2))
                        .cmp(&(MultiPrecisionInteger::from(n2) * MultiPrecisionInteger::from(d1))),
                }
            }
            (CoefficientView::Large(n1), CoefficientView::Large(n2)) => {
                n1.to_rat().cmp(&n2.to_rat())
            }
            (CoefficientView::FiniteField(n1, _), CoefficientView::FiniteField(n2, _)) => {
                n1.0.cmp(&n2.0)
            }
            (&CoefficientView::Natural(n1, d1), CoefficientView::Large(n2)) => {
                MultiPrecisionRational::from((n1, d1)).cmp(&n2.to_rat())
            }
            (CoefficientView::Large(n1), &CoefficientView::Natural(n2, d2)) => {
                n1.to_rat().cmp(&MultiPrecisionRational::from((n2, d2)))
            }
            _ => unreachable!(),
        }
    }
}

impl Add<CoefficientView<'_>> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (CoefficientView::Natural(n1, d1), CoefficientView::Natural(n2, d2)) => {
                Coefficient::Rational(Rational::Natural(n1, d1) + &Rational::Natural(n2, d2))
            }
            (CoefficientView::Natural(n1, d1), CoefficientView::Large(r2))
            | (CoefficientView::Large(r2), CoefficientView::Natural(n1, d1)) => {
                Coefficient::Rational(Rational::Natural(n1, d1) + &Rational::Large(r2.to_rat()))
            }
            (CoefficientView::Large(r1), CoefficientView::Large(r2)) => {
                (r1.to_rat() + r2.to_rat()).into()
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
            (CoefficientView::Natural(n, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d)) => {
                let r = (*p).clone();
                let r2 = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Natural(n)),
                    denominator: p.denominator.constant(Integer::Natural(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::Large(l), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l)) => {
                let r = (*p).clone();
                let (n, d) = l.to_rat().into_numer_denom();
                let r2 = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::from_large(n)),
                    denominator: p.denominator.constant(Integer::from_large(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let r = if p1.get_variables() != p2.get_variables() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_variables(&mut p2);
                    &p1 + &p2
                } else {
                    p1 + p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
        }
    }
}

impl Mul for CoefficientView<'_> {
    type Output = Coefficient;

    fn mul(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (CoefficientView::Natural(n1, d1), CoefficientView::Natural(n2, d2)) => {
                Coefficient::Rational(Rational::Natural(n1, d1) * &Rational::Natural(n2, d2))
            }
            (CoefficientView::Natural(n1, d1), CoefficientView::Large(r2))
            | (CoefficientView::Large(r2), CoefficientView::Natural(n1, d1)) => {
                Coefficient::Rational(Rational::Natural(n1, d1) * &Rational::Large(r2.to_rat()))
            }
            (CoefficientView::Large(r1), CoefficientView::Large(r2)) => {
                (r1.to_rat() * r2.to_rat()).into()
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
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (CoefficientView::Natural(n, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d)) => {
                let mut r = (*p).clone();
                let (n, d) = (Integer::Natural(n), Integer::Natural(d));

                let gcd1 = Z.gcd(&n, &r.denominator.content());
                let gcd2 = Z.gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::Large(l), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l)) => {
                let mut r = (*p).clone();
                let (n, d) = l.to_rat().into_numer_denom();
                let (n, d) = (Integer::from_large(n), Integer::from_large(d));

                let gcd1 = Z.gcd(&n, &r.denominator.content());
                let gcd2 = Z.gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let r = if p1.get_variables() != p2.get_variables() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_variables(&mut p2);
                    &p1 * &p2
                } else {
                    p1 * p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
        }
    }
}

impl Add<i64> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: i64) -> Coefficient {
        match self {
            CoefficientView::Natural(n1, d1) => {
                Coefficient::Rational(Rational::Natural(n1, d1) + &other.into())
            }
            CoefficientView::Large(r1) => (r1.to_rat() + other).into(),
            CoefficientView::FiniteField(n1, i1) => {
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &f.element_from_coefficient(other.into())), i1)
            }
            CoefficientView::RationalPolynomial(p) => {
                let a = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Natural(other)),
                    denominator: p.denominator.constant(Integer::Natural(1)),
                };

                Coefficient::RationalPolynomial(p + &a)
            }
        }
    }
}

impl Atom {
    pub fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Atom {
        self.as_view().set_coefficient_ring(vars)
    }
}

impl<'a> AtomView<'a> {
    pub fn set_coefficient_ring(&self, vars: &Arc<Vec<Variable>>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.set_coefficient_ring_with_ws_into(vars, ws, &mut out);
            out.into_inner()
        })
    }

    pub fn set_coefficient_ring_with_ws_into(
        &self,
        vars: &Arc<Vec<Variable>>,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        match self {
            AtomView::Num(n) => {
                if let CoefficientView::RationalPolynomial(r) = n.get_coeff_view() {
                    let old_var_map = r.get_variables();
                    if old_var_map != vars {
                        if old_var_map.iter().all(|x| vars.contains(x)) {
                            // upgrade the polynomial if no variables got removed
                            let mut r = r.clone();
                            let order: SmallVec<[Option<usize>; INLINED_EXPONENTS]> = vars
                                .iter()
                                .map(|x| old_var_map.iter().position(|xx| xx == x))
                                .collect();

                            r.numerator = r.numerator.rearrange_with_growth(&order);
                            r.denominator = r.denominator.rearrange_with_growth(&order);
                            r.numerator.variables = vars.clone();
                            r.denominator.variables = r.numerator.variables.clone();
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
                            exp.to_num(Coefficient::Rational((-1i64).into()));
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
                for arg in m.iter() {
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
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a.iter() {
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
}
