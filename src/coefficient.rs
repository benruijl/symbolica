use std::{cmp::Ordering, ops::Div, sync::Arc};

use ahash::HashMap;
use rug::{
    integer::Order,
    ops::{NegAssign, Pow as RPow},
    Integer as ArbitraryPrecisionInteger, Rational as ArbitraryPrecisionRational,
};
use smallvec::{smallvec, SmallVec};

use crate::{
    domains::{
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, ToFiniteField,
        },
        integer::{Integer, IntegerRing},
        rational::{Rational, RationalField},
        rational_polynomial::RationalPolynomial,
        EuclideanDomain, Field, Ring,
    },
    poly::{polynomial::MultivariatePolynomial, Variable, INLINED_EXPONENTS},
    representations::{
        Add, Atom, AtomSet, AtomView, Mul, Num, OwnedAdd, OwnedMul, OwnedNum, OwnedPow, Pow, Var,
    },
    state::{FiniteFieldIndex, ResettableBuffer, State, Workspace},
};

pub trait ConvertToRing: Ring {
    /// Convert from a Symbolica `Number` to a Ring.
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element;

    /// Convert from a Symbolica `BorrowedCoefficient` to a Ring.
    fn element_from_borrowed_coefficient(&self, number: BorrowedCoefficient<'_>) -> Self::Element;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Coefficient {
    Natural(i64, i64), // TODO: change to Rational
    Large(ArbitraryPrecisionRational),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(RationalPolynomial<IntegerRing, u16>),
}

impl From<i64> for Coefficient {
    fn from(value: i64) -> Self {
        Coefficient::Natural(value, 1)
    }
}

impl From<Integer> for Coefficient {
    fn from(value: Integer) -> Self {
        match value {
            Integer::Natural(n) => Coefficient::Natural(n, 1),
            Integer::Double(r) => Coefficient::Large(ArbitraryPrecisionRational::from(r)),
            Integer::Large(r) => Coefficient::Large(ArbitraryPrecisionRational::from(r)),
        }
    }
}

impl From<(Integer, Integer)> for Coefficient {
    fn from(value: (Integer, Integer)) -> Self {
        match (value.0, value.1) {
            (Integer::Natural(n), Integer::Natural(d)) => Coefficient::Natural(n, d),
            (Integer::Large(n), Integer::Natural(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Natural(n), Integer::Large(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Large(n), Integer::Large(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Natural(n), Integer::Double(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Double(n), Integer::Natural(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Double(n), Integer::Double(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Double(n), Integer::Large(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
            (Integer::Large(n), Integer::Double(d)) => {
                Coefficient::Large(ArbitraryPrecisionRational::from((n, d)))
            }
        }
    }
}

impl From<Rational> for Coefficient {
    fn from(value: Rational) -> Self {
        match value {
            Rational::Natural(n, d) => Coefficient::Natural(n, d),
            Rational::Large(r) => Coefficient::Large(r),
        }
    }
}

impl Coefficient {
    pub fn is_zero(&self) -> bool {
        match self {
            Coefficient::Natural(num, _den) => *num == 0,
            Coefficient::Large(_r) => false,
            Coefficient::FiniteField(num, _field) => num.0 == 0,
            Coefficient::RationalPolynomial(r) => r.numerator.is_zero(),
        }
    }

    pub fn add(self, other: Coefficient, state: &State) -> Coefficient {
        match (self, other) {
            (Coefficient::Natural(n1, d1), Coefficient::Natural(n2, d2)) => {
                let r = &Rational::Natural(n1, d1) + &Rational::Natural(n2, d2);
                match r {
                    Rational::Natural(n, d) => Coefficient::Natural(n, d),
                    Rational::Large(r) => Coefficient::Large(r),
                }
            }
            // TODO: check downcast
            (Coefficient::Natural(n1, d1), Coefficient::Large(r2))
            | (Coefficient::Large(r2), Coefficient::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((n1, d1));
                Coefficient::Large(r1 + r2)
            }
            (Coefficient::Large(r1), Coefficient::Large(r2)) => Coefficient::Large(r1 + r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(i1).get_prime(),
                        state.get_finite_field(i2).get_prime()
                    );
                }
                let f = state.get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (_, Coefficient::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (Coefficient::Natural(n, d), Coefficient::RationalPolynomial(r))
            | (Coefficient::RationalPolynomial(r), Coefficient::Natural(n, d)) => {
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(Integer::Natural(n)),
                    denominator: r.denominator.constant(Integer::Natural(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (Coefficient::Large(l), Coefficient::RationalPolynomial(r))
            | (Coefficient::RationalPolynomial(r), Coefficient::Large(l)) => {
                let (n, d) = l.into_numer_denom();
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(Integer::Large(n)),
                    denominator: r.denominator.constant(Integer::Large(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_var_map() != p2.get_var_map() {
                    p1.unify_var_map(&mut p2);
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

    pub fn mul(self, other: Coefficient, state: &State) -> Coefficient {
        match (self, other) {
            (Coefficient::Natural(n1, d1), Coefficient::Natural(n2, d2)) => {
                let r = &Rational::Natural(n1, d1) * &Rational::Natural(n2, d2);
                match r {
                    Rational::Natural(n, d) => Coefficient::Natural(n, d),
                    Rational::Large(r) => Coefficient::Large(r),
                }
            }
            // TODO: check downcast
            (Coefficient::Natural(n1, d1), Coefficient::Large(r2))
            | (Coefficient::Large(r2), Coefficient::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((n1, d1));
                Coefficient::Large(r1 * r2)
            }
            (Coefficient::Large(r1), Coefficient::Large(r2)) => Coefficient::Large(r1 * r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(i1).get_prime(),
                        state.get_finite_field(i2).get_prime()
                    );
                }
                let f = state.get_finite_field(i1);
                Coefficient::FiniteField(f.mul(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (_, Coefficient::FiniteField(_, _)) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (Coefficient::Natural(n, d), Coefficient::RationalPolynomial(mut r))
            | (Coefficient::RationalPolynomial(mut r), Coefficient::Natural(n, d)) => {
                let (n, d) = (Integer::Natural(n), Integer::Natural(d));

                let gcd1 = IntegerRing::new().gcd(&n, &r.denominator.content());
                let gcd2 = IntegerRing::new().gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (Coefficient::Large(l), Coefficient::RationalPolynomial(mut r))
            | (Coefficient::RationalPolynomial(mut r), Coefficient::Large(l)) => {
                let (n, d) = l.into_numer_denom();
                let (n, d) = (Integer::Large(n), Integer::Large(d));

                let gcd1 = IntegerRing::new().gcd(&n, &r.denominator.content());
                let gcd2 = IntegerRing::new().gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_var_map() != p2.get_var_map() {
                    p1.unify_var_map(&mut p2);
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

    pub fn to_rat(&self) -> ArbitraryPrecisionRational {
        let mut num = ArbitraryPrecisionInteger::from_digits(self.num_digits, Order::Lsf);
        let den = ArbitraryPrecisionInteger::from_digits(self.den_digits, Order::Lsf);
        if self.is_negative {
            num.neg_assign();
        }

        ArbitraryPrecisionRational::from((num, den))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BorrowedCoefficient<'a> {
    Natural(i64, i64),
    Large(SerializedRational<'a>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(&'a RationalPolynomial<IntegerRing, u16>),
}

impl ConvertToRing for RationalField {
    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        match number {
            Coefficient::Natural(r, d) => Rational::Natural(r, d),
            Coefficient::Large(r) => Rational::Large(r),
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to rational"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_borrowed_coefficient(&self, number: BorrowedCoefficient<'_>) -> Rational {
        match number {
            BorrowedCoefficient::Natural(r, d) => Rational::Natural(r, d),
            BorrowedCoefficient::Large(r) => Rational::Large(r.to_rat()),
            BorrowedCoefficient::FiniteField(_, _) => {
                panic!("Cannot convert finite field to rational")
            }
            BorrowedCoefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }
}

impl ConvertToRing for IntegerRing {
    #[inline]
    fn element_from_coefficient(&self, number: Coefficient) -> Integer {
        match number {
            Coefficient::Natural(r, d) => {
                debug_assert!(d == 1);
                Integer::Natural(r)
            }
            Coefficient::Large(r) => {
                let (n, d) = r.into_numer_denom();
                debug_assert!(d == 1);
                Integer::Large(n)
            }
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to integer"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to rational")
            }
        }
    }

    #[inline]
    fn element_from_borrowed_coefficient(&self, number: BorrowedCoefficient<'_>) -> Integer {
        match number {
            BorrowedCoefficient::Natural(r, d) => {
                debug_assert!(d == 1);
                Integer::Natural(r)
            }
            BorrowedCoefficient::Large(r) => {
                let r = r.to_rat();
                debug_assert!(r.denom() == &1);
                Integer::Large(r.numer().clone())
            }
            BorrowedCoefficient::FiniteField(_, _) => {
                panic!("Cannot convert finite field to integer")
            }
            BorrowedCoefficient::RationalPolynomial(_) => {
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
    fn element_from_coefficient(
        &self,
        number: Coefficient,
    ) -> <FiniteField<UField> as Ring>::Element {
        match number {
            Coefficient::Natural(n, d) => self.div(
                &Integer::new(n).to_finite_field(self),
                &Integer::new(d).to_finite_field(self),
            ),
            Coefficient::Large(r) => {
                let (n, d) = r.into_numer_denom();
                self.div(
                    &Integer::Large(n).to_finite_field(self),
                    &Integer::Large(d).to_finite_field(self),
                )
            }
            Coefficient::FiniteField(_, _) => panic!("Cannot convert finite field to other one"),
            Coefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }

    #[inline]
    fn element_from_borrowed_coefficient(
        &self,
        number: BorrowedCoefficient<'_>,
    ) -> <FiniteField<UField> as Ring>::Element {
        match number {
            BorrowedCoefficient::Natural(n, d) => self.div(
                &Integer::new(n).to_finite_field(self),
                &Integer::new(d).to_finite_field(self),
            ),
            BorrowedCoefficient::Large(r) => {
                let (n, d) = r.to_rat().into_numer_denom();
                self.div(
                    &Integer::Large(n).to_finite_field(self),
                    &Integer::Large(d).to_finite_field(self),
                )
            }
            BorrowedCoefficient::FiniteField(_, _) => {
                panic!("Cannot convert finite field to other one")
            }
            BorrowedCoefficient::RationalPolynomial(_) => {
                panic!("Cannot convert rational polynomial to finite field")
            }
        }
    }
}

impl BorrowedCoefficient<'_> {
    pub fn normalize(&self) -> Coefficient {
        match self {
            BorrowedCoefficient::Natural(num, den) => match Rational::new(*num, *den) {
                Rational::Natural(n, d) => Coefficient::Natural(n, d),
                Rational::Large(l) => Coefficient::Large(l),
            },
            BorrowedCoefficient::Large(_)
            | BorrowedCoefficient::FiniteField(_, _)
            | BorrowedCoefficient::RationalPolynomial(_) => self.to_owned(),
        }
    }

    pub fn to_owned(&self) -> Coefficient {
        match self {
            BorrowedCoefficient::Natural(num, den) => Coefficient::Natural(*num, *den),
            BorrowedCoefficient::Large(r) => Coefficient::Large(r.to_rat()),
            BorrowedCoefficient::FiniteField(num, field) => Coefficient::FiniteField(*num, *field),
            BorrowedCoefficient::RationalPolynomial(p) => {
                Coefficient::RationalPolynomial((*p).clone())
            }
        }
    }

    pub fn add(&self, other: &BorrowedCoefficient<'_>, state: &State) -> Coefficient {
        match (self, other) {
            (BorrowedCoefficient::Natural(n1, d1), BorrowedCoefficient::Natural(n2, d2)) => {
                let r = &Rational::Natural(*n1, *d1) + &Rational::Natural(*n2, *d2);
                match r {
                    Rational::Natural(n, d) => Coefficient::Natural(n, d),
                    Rational::Large(r) => Coefficient::Large(r),
                }
            }
            // TODO: check downcast
            (BorrowedCoefficient::Natural(n1, d1), BorrowedCoefficient::Large(r2))
            | (BorrowedCoefficient::Large(r2), BorrowedCoefficient::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Coefficient::Large(r1 + r2.to_rat())
            }
            (BorrowedCoefficient::Large(r1), BorrowedCoefficient::Large(r2)) => {
                Coefficient::Large(r1.to_rat() + r2.to_rat())
            }
            (
                BorrowedCoefficient::FiniteField(n1, i1),
                BorrowedCoefficient::FiniteField(n2, i2),
            ) => {
                if i1 != i2 {
                    panic!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(*i1).get_prime(),
                        state.get_finite_field(*i2).get_prime()
                    );
                }
                let f = state.get_finite_field(*i1);
                Coefficient::FiniteField(f.add(n1, n2), *i1)
            }
            (BorrowedCoefficient::FiniteField(_, _), _) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (_, BorrowedCoefficient::FiniteField(_, _)) => {
                panic!("Cannot add finite field to non-finite number. Convert other number first?");
            }
            (BorrowedCoefficient::Natural(n, d), BorrowedCoefficient::RationalPolynomial(p))
            | (BorrowedCoefficient::RationalPolynomial(p), BorrowedCoefficient::Natural(n, d)) => {
                let r = (*p).clone();
                let r2 = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Natural(*n)),
                    denominator: p.denominator.constant(Integer::Natural(*d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (BorrowedCoefficient::Large(l), BorrowedCoefficient::RationalPolynomial(p))
            | (BorrowedCoefficient::RationalPolynomial(p), BorrowedCoefficient::Large(l)) => {
                let r = (*p).clone();
                let (n, d) = l.to_rat().into_numer_denom();
                let r2 = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Large(n)),
                    denominator: p.denominator.constant(Integer::Large(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (
                BorrowedCoefficient::RationalPolynomial(p1),
                BorrowedCoefficient::RationalPolynomial(p2),
            ) => {
                let r = if p1.get_var_map() != p2.get_var_map() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_var_map(&mut p2);
                    &p1 + &p2
                } else {
                    *p1 + *p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
        }
    }

    pub fn mul(&self, other: &BorrowedCoefficient<'_>, state: &State) -> Coefficient {
        match (self, other) {
            (BorrowedCoefficient::Natural(n1, d1), BorrowedCoefficient::Natural(n2, d2)) => {
                let r = &Rational::Natural(*n1, *d1) * &Rational::Natural(*n2, *d2);
                match r {
                    Rational::Natural(n, d) => Coefficient::Natural(n, d),
                    Rational::Large(r) => Coefficient::Large(r),
                }
            }
            // TODO: check downcast
            (BorrowedCoefficient::Natural(n1, d1), BorrowedCoefficient::Large(r2))
            | (BorrowedCoefficient::Large(r2), BorrowedCoefficient::Natural(n1, d1)) => {
                let r1 = ArbitraryPrecisionRational::from((*n1, *d1));
                Coefficient::Large(r1 * r2.to_rat())
            }
            (BorrowedCoefficient::Large(r1), BorrowedCoefficient::Large(r2)) => {
                Coefficient::Large(r1.to_rat() * r2.to_rat())
            }
            (
                BorrowedCoefficient::FiniteField(n1, i1),
                BorrowedCoefficient::FiniteField(n2, i2),
            ) => {
                if i1 != i2 {
                    panic!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        state.get_finite_field(*i1).get_prime(),
                        state.get_finite_field(*i2).get_prime()
                    );
                }
                let f = state.get_finite_field(*i1);
                Coefficient::FiniteField(f.mul(n1, n2), *i1)
            }
            (BorrowedCoefficient::FiniteField(_, _), _) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (_, BorrowedCoefficient::FiniteField(_, _)) => {
                panic!("Cannot multiply finite field to non-finite number. Convert other number first?");
            }
            (BorrowedCoefficient::Natural(n, d), BorrowedCoefficient::RationalPolynomial(p))
            | (BorrowedCoefficient::RationalPolynomial(p), BorrowedCoefficient::Natural(n, d)) => {
                let mut r = (*p).clone();
                let (n, d) = (Integer::Natural(*n), Integer::Natural(*d));

                let gcd1 = IntegerRing::new().gcd(&n, &r.denominator.content());
                let gcd2 = IntegerRing::new().gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (BorrowedCoefficient::Large(l), BorrowedCoefficient::RationalPolynomial(p))
            | (BorrowedCoefficient::RationalPolynomial(p), BorrowedCoefficient::Large(l)) => {
                let mut r = (*p).clone();
                let (n, d) = l.to_rat().into_numer_denom();
                let (n, d) = (Integer::Large(n), Integer::Large(d));

                let gcd1 = IntegerRing::new().gcd(&n, &r.denominator.content());
                let gcd2 = IntegerRing::new().gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (
                BorrowedCoefficient::RationalPolynomial(p1),
                BorrowedCoefficient::RationalPolynomial(p2),
            ) => {
                let r = if p1.get_var_map() != p2.get_var_map() {
                    let mut p1 = (*p1).clone();
                    let mut p2 = (*p2).clone();
                    p1.unify_var_map(&mut p2);
                    &p1 * &p2
                } else {
                    *p1 * *p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
        }
    }

    pub fn pow(
        &self,
        other: &BorrowedCoefficient<'_>,
        _state: &State,
    ) -> (Coefficient, Coefficient) {
        // TODO: normalize 4^1/3 to 2^(2/3)?
        match (self, other) {
            (
                &BorrowedCoefficient::Natural(mut n1, mut d1),
                &BorrowedCoefficient::Natural(mut n2, d2),
            ) => {
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
                            return (Coefficient::Natural(pn, pd), Coefficient::Natural(1, d2));
                        }
                    }

                    (
                        Coefficient::Large(
                            ArbitraryPrecisionRational::from((n1, d1)).pow(n2 as u32),
                        ),
                        Coefficient::Natural(1, d2),
                    )
                } else {
                    panic!("Power is too large: {}", n2);
                }
            }
            (
                &BorrowedCoefficient::RationalPolynomial(r),
                &BorrowedCoefficient::Natural(n2, d2),
            ) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.clone().inv();
                    (
                        Coefficient::RationalPolynomial(r.pow(n2.unsigned_abs())),
                        Coefficient::Natural(1, d2),
                    )
                } else {
                    (
                        Coefficient::RationalPolynomial(r.pow(n2 as u64)),
                        Coefficient::Natural(1, d2),
                    )
                }
            }
            (&BorrowedCoefficient::Large(r), &BorrowedCoefficient::Natural(n2, d2)) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    panic!("Power is too large: {}", n2);
                }

                if n2 < 0 {
                    let r = r.to_rat().clone().recip();
                    (
                        Coefficient::Large(r.pow(n2.unsigned_abs() as u32)),
                        Coefficient::Natural(1, d2),
                    )
                } else {
                    (
                        Coefficient::Large(r.to_rat().pow(n2 as u32)),
                        Coefficient::Natural(1, d2),
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

    pub fn cmp(&self, other: &BorrowedCoefficient) -> Ordering {
        match (self, other) {
            (&BorrowedCoefficient::Natural(n1, d1), &BorrowedCoefficient::Natural(n2, d2)) => {
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
                        None => ArbitraryPrecisionInteger::from(a1).cmp(
                            &(ArbitraryPrecisionInteger::from(n2)
                                * ArbitraryPrecisionInteger::from(d1)),
                        ),
                    },
                    None => (ArbitraryPrecisionInteger::from(n1)
                        * ArbitraryPrecisionInteger::from(d2))
                    .cmp(
                        &(ArbitraryPrecisionInteger::from(n2)
                            * ArbitraryPrecisionInteger::from(d1)),
                    ),
                }
            }
            (BorrowedCoefficient::Large(n1), BorrowedCoefficient::Large(n2)) => {
                n1.to_rat().cmp(&n2.to_rat())
            }
            (BorrowedCoefficient::FiniteField(n1, _), BorrowedCoefficient::FiniteField(n2, _)) => {
                n1.0.cmp(&n2.0)
            }
            (&BorrowedCoefficient::Natural(n1, d1), BorrowedCoefficient::Large(n2)) => {
                ArbitraryPrecisionRational::from((n1, d1)).cmp(&n2.to_rat())
            }
            (BorrowedCoefficient::Large(n1), &BorrowedCoefficient::Natural(n2, d2)) => {
                n1.to_rat().cmp(&ArbitraryPrecisionRational::from((n2, d2)))
            }
            _ => unreachable!(),
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            BorrowedCoefficient::Natural(_, d) => *d == 1,
            BorrowedCoefficient::Large(r) => r.to_rat().is_integer(),
            BorrowedCoefficient::FiniteField(_, _) => true,
            BorrowedCoefficient::RationalPolynomial(_) => false,
        }
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
    pub fn set_coefficient_ring(
        &self,
        vars: &Arc<Vec<Variable>>,
        state: &State,
        workspace: &Workspace<P>,
        out: &mut Atom<P>,
    ) -> bool {
        match self {
            AtomView::Num(n) => {
                if let BorrowedCoefficient::RationalPolynomial(r) = n.get_number_view() {
                    let old_var_map = r.get_var_map().unwrap();
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
                            r.numerator.var_map = Some(vars.clone());
                            r.denominator.var_map = r.numerator.var_map.clone();
                            out.to_num()
                                .set_from_number(Coefficient::RationalPolynomial(r));
                            true
                        } else {
                            let mut n1 = workspace.new_atom();
                            r.numerator.to_expression(
                                workspace,
                                state,
                                &HashMap::default(),
                                &mut n1,
                            );

                            let mut n1_conv = workspace.new_atom();
                            n1.as_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n1_conv);

                            let mut n2 = workspace.new_atom();
                            r.denominator.to_expression(
                                workspace,
                                state,
                                &HashMap::default(),
                                &mut n2,
                            );

                            let mut n2_conv = workspace.new_atom();
                            n2.as_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n2_conv);

                            // create n1/n2
                            let mut n3 = workspace.new_atom();
                            let mut exp = workspace.new_atom();
                            exp.to_num().set_from_number(Coefficient::Natural(-1, 1));
                            let n3p = n3.to_pow();
                            n3p.set_from_base_and_exp(n2_conv.as_view(), exp.as_view());
                            n3p.set_dirty(true);

                            let mut m = workspace.new_atom();
                            let mm = m.to_mul();
                            mm.extend(n1_conv.as_view());
                            mm.extend(n3.as_view());
                            mm.set_dirty(true);
                            m.as_view().normalize(workspace, state, out);
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
                let id = v.get_name();
                if vars.contains(&id.into()) {
                    // change variable into coefficient
                    let mut poly = MultivariatePolynomial::new(
                        vars.len(),
                        &IntegerRing::new(),
                        None,
                        Some(vars.clone()),
                    );
                    let mut e: SmallVec<[u16; INLINED_EXPONENTS]> = smallvec![0; vars.len()];
                    e[vars.iter().position(|x| *x == id.into()).unwrap()] = 1;
                    poly.append_monomial(Integer::one(), &e);
                    let den = poly.one();

                    out.to_num()
                        .set_from_number(Coefficient::RationalPolynomial(RationalPolynomial {
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
                if base.set_coefficient_ring(vars, state, workspace, &mut nb) {
                    let mut o = workspace.new_atom();
                    let pow = o.to_pow();
                    pow.set_from_base_and_exp(nb.as_view(), exp);
                    pow.set_dirty(true);

                    o.as_view().normalize(workspace, state, out);
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
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, state, out);
                    true
                }
            }
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a.iter() {
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, state, out);
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
