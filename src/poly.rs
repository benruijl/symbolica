pub mod gcd;
pub mod polynomial;

use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add as OpAdd, AddAssign, Sub};

use smallvec::{smallvec, SmallVec};

use crate::representations::number::{BorrowedNumber, Number};
use crate::representations::{
    Add, Atom, AtomView, Identifier, Mul, Num, OwnedAdd, OwnedAtom, OwnedMul, OwnedNum, OwnedPow,
    OwnedVar, Pow, Var,
};
use crate::rings::rational::{Rational, RationalField};
use crate::rings::rational_polynomial::{RationalPolynomial, RationalPolynomialField};
use crate::rings::{Field, Ring};
use crate::state::{State, Workspace};

use self::polynomial::MultivariatePolynomial;

pub const INLINED_EXPONENTS: usize = 6;

pub trait Exponent:
    Hash + Debug + Display + Ord + Sub<Output = Self> + OpAdd<Output = Self> + AddAssign + Clone + Copy
{
    fn zero() -> Self;
    /// Convert the exponent to `u32`. This is always possible, as `u32` is the largest supported exponent type.
    fn to_u32(&self) -> u32;
    /// Convert from `u32`. This function may panic if the exponent is too large.
    fn from_u32(n: u32) -> Self;
    fn is_zero(&self) -> bool;
    fn checked_add(&self, other: &Self) -> Option<Self>;
}

impl Exponent for u32 {
    fn zero() -> Self {
        0
    }

    fn to_u32(&self) -> u32 {
        *self
    }

    fn from_u32(n: u32) -> Self {
        n
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn checked_add(&self, other: &Self) -> Option<Self> {
        u32::checked_add(*self, *other)
    }
}

/// An exponent limited to 255 for efficiency
impl Exponent for u8 {
    fn zero() -> Self {
        0
    }

    fn to_u32(&self) -> u32 {
        *self as u32
    }

    fn from_u32(n: u32) -> Self {
        if n < u8::MAX as u32 {
            n as u8
        } else {
            panic!("Exponent {} too large for u8", n);
        }
    }

    fn is_zero(&self) -> bool {
        *self == 0
    }

    fn checked_add(&self, other: &Self) -> Option<Self> {
        u8::checked_add(*self, *other)
    }
}

impl<'a, P: Atom> AtomView<'a, P> {
    /// Convert an expression to a polynomial.
    ///
    /// This function requires an expanded polynomial. If this yields too many terms, consider using
    /// calling `to_rational_polynomial` instead.
    pub fn to_polynomial<E: Exponent>(
        &self,
        var_map: Option<&[Identifier]>,
    ) -> Result<MultivariatePolynomial<RationalField, E>, &'static str> {
        fn check_factor<P: Atom>(
            factor: &AtomView<'_, P>,
            vars: &mut SmallVec<[Identifier; INLINED_EXPONENTS]>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match factor {
                AtomView::Num(n) => match n.get_number_view() {
                    BorrowedNumber::FiniteField(_, _) => {
                        Err("Finite field not supported in conversion routine")
                    }
                    _ => Ok(()),
                },
                AtomView::Var(v) => {
                    let name = v.get_name();
                    if !vars.contains(&name) {
                        if !allow_new_vars {
                            return Err("Expression contains variable that is not in variable map");
                        } else {
                            vars.push(v.get_name());
                        }
                    }
                    Ok(())
                }
                AtomView::Fun(_) => Err("function not supported in polynomial"),
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();
                    match base {
                        AtomView::Var(v) => {
                            let name = v.get_name();
                            if !vars.contains(&name) {
                                if !allow_new_vars {
                                    return Err(
                                        "Expression contains variable that is not in variable map",
                                    );
                                } else {
                                    vars.push(v.get_name());
                                }
                            }
                        }
                        _ => return Err("base must be a variable"),
                    }

                    match exp {
                        AtomView::Num(n) => match n.get_number_view() {
                            BorrowedNumber::FiniteField(_, _) => {
                                Err("Finite field not supported in conversion routine")
                            }
                            BorrowedNumber::Natural(n, d) => {
                                if d == 1 && n >= 0 && n <= u32::MAX as i64 {
                                    Ok(())
                                } else {
                                    Err("Exponent negative or a fraction")
                                }
                            }
                            BorrowedNumber::Large(r) => {
                                if r.denom().to_u8() == Some(1) && r.numer().to_u32().is_some() {
                                    Ok(())
                                } else {
                                    Err("Exponent too large or negative or a fraction")
                                }
                            }
                        },
                        _ => return Err("base must be a variable"),
                    }
                }
                AtomView::Add(_) => Err("Expression may not contain subexpressions"),
                AtomView::Mul(_) => unreachable!("Mul inside mul found"),
            }
        }

        fn check_term<P: Atom>(
            term: &AtomView<'_, P>,
            vars: &mut SmallVec<[Identifier; INLINED_EXPONENTS]>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match term {
                AtomView::Mul(m) => {
                    for factor in m.into_iter() {
                        check_factor(&factor, vars, allow_new_vars)?;
                    }
                    Ok(())
                }
                _ => check_factor(term, vars, allow_new_vars),
            }
        }

        // get all variables and check structure
        let mut vars: SmallVec<[Identifier; INLINED_EXPONENTS]> =
            var_map.map(|v| v.into()).unwrap_or(SmallVec::new());
        let mut n_terms = 0;
        match self {
            AtomView::Add(a) => {
                for term in a.into_iter() {
                    check_term(&term, &mut vars, var_map.is_none())?;
                    n_terms += 1;
                }
            }
            _ => {
                check_term(self, &mut vars, var_map.is_none())?;
                n_terms += 1;
            }
        }

        fn parse_factor<P: Atom, E: Exponent>(
            factor: &AtomView<'_, P>,
            vars: &[Identifier],
            coefficient: &mut Rational,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
        ) {
            match factor {
                AtomView::Num(n) => match n.get_number_view() {
                    BorrowedNumber::Natural(r, d) => {
                        *coefficient = Rational::Natural(r, d);
                    }
                    BorrowedNumber::Large(r) => *coefficient = Rational::Large(r.clone()),
                    BorrowedNumber::FiniteField(_, _) => unreachable!(),
                },
                AtomView::Var(v) => {
                    let id = v.get_name();
                    exponents[vars.iter().position(|v| *v == id).unwrap()] += E::from_u32(1);
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    let var_index = match base {
                        AtomView::Var(v) => {
                            let id = v.get_name();
                            vars.iter().position(|v| *v == id).unwrap()
                        }
                        _ => unreachable!(),
                    };

                    match exp {
                        AtomView::Num(n) => match n.get_number_view() {
                            BorrowedNumber::Natural(r, _) => {
                                exponents[var_index] += E::from_u32(r as u32)
                            }
                            BorrowedNumber::Large(r) => {
                                exponents[var_index] += E::from_u32(r.numer().to_u32().unwrap())
                            }
                            BorrowedNumber::FiniteField(_, _) => unreachable!(),
                        },
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!("Unsupported expression"),
            }
        }

        fn parse_term<P: Atom, E: Exponent>(
            term: &AtomView<'_, P>,
            vars: &[Identifier],
            poly: &mut MultivariatePolynomial<RationalField, E>,
        ) {
            let mut coefficient = Rational::Natural(1, 1);
            let mut exponents = smallvec![E::zero(); vars.len()];

            match term {
                AtomView::Mul(m) => {
                    for factor in m.into_iter() {
                        parse_factor(&factor, vars, &mut coefficient, &mut exponents);
                    }
                }
                _ => parse_factor(term, vars, &mut coefficient, &mut exponents),
            }

            poly.append_monomial(coefficient, &exponents);
        }

        let mut poly = MultivariatePolynomial::<RationalField, E>::new(
            vars.len(),
            RationalField::new(),
            Some(n_terms),
            Some(vars.clone()),
        );

        match self {
            AtomView::Add(a) => {
                for term in a.into_iter() {
                    parse_term(&term, &vars, &mut poly);
                }
            }
            _ => parse_term(self, &vars, &mut poly),
        }

        Ok(poly)
    }

    /// Convert an expression to a rational polynomial if possible.
    pub fn to_rational_polynomial<E: Exponent>(
        &self,
        workspace: &Workspace<P>,
        state: &State,
        var_map: Option<&[Identifier]>,
    ) -> Result<RationalPolynomial<E>, &'static str> {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(var_map) {
            let den = MultivariatePolynomial::one(RationalField::new());
            return Ok(RationalPolynomial::from_rat_num_and_den(&num, &den));
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                let num = self.to_polynomial(var_map)?;
                let den = MultivariatePolynomial::one(RationalField::new());
                Ok(RationalPolynomial::from_rat_num_and_den(&num, &den))
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_number_view();

                    if let BorrowedNumber::Natural(nn, nd) = num_n {
                        if nd != 1 {
                            Err("Exponent cannot be a faction")?
                        }

                        if nn != -1 && nn != 1 {
                            let mut h = workspace.new_atom();
                            if !self.expand(workspace, state, &mut h.get_mut()) {
                                // expansion did not change the input, so we are in a case of x^-3 or x^3
                                let r: RationalPolynomial<E> =
                                    base.to_rational_polynomial(workspace, state, var_map)?;

                                if nn < 0 {
                                    let r_inv = RationalPolynomialField::new().inv(&r);
                                    Ok(RationalPolynomialField::new().pow(&r_inv, -nn as u64))
                                } else {
                                    Ok(RationalPolynomialField::new().pow(&r, nn as u64))
                                }
                            } else {
                                h.get()
                                    .to_view()
                                    .to_rational_polynomial(workspace, state, var_map)
                            }
                        } else if nn < 0 {
                            let r: RationalPolynomial<E> =
                                base.to_rational_polynomial(workspace, state, var_map)?;
                            Ok(RationalPolynomialField::new().inv(&r))
                        } else {
                            base.to_rational_polynomial(workspace, state, var_map)
                        }
                    } else {
                        Err("Exponent needs to be an integer")?
                    }
                } else {
                    Err("Power needs to be a number")?
                }
            }
            AtomView::Fun(_) => Err("Functions not allowed")?,
            AtomView::Mul(m) => {
                let mut r = RationalPolynomialField::<E>::new().one();
                for arg in m.into_iter() {
                    let mut arg_r = arg.to_rational_polynomial(workspace, state, var_map)?;
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = RationalPolynomial::new(var_map);
                for arg in a.into_iter() {
                    let mut arg_r = arg.to_rational_polynomial(workspace, state, var_map)?;
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                return Ok(r);
            }
        }
    }
}

impl<P: Atom> OwnedAtom<P> {
    pub fn from_polynomial(
        &mut self,
        workspace: &Workspace<P>,
        poly: &MultivariatePolynomial<RationalField, u32>,
    ) {
        let var_map = poly
            .var_map
            .as_ref()
            .expect("No variable map present in polynomial");

        let add = self.transform_to_add();

        for monomial in poly {
            let mut mul_h = workspace.new_atom();
            let mul = mul_h.get_mut().transform_to_mul();

            for (&var_id, &pow) in var_map.iter().zip(monomial.exponents) {
                if pow > 0 {
                    let mut var_h = workspace.new_atom();
                    let var = var_h.get_mut().transform_to_var();
                    var.from_id(var_id);

                    if pow > 1 {
                        let mut num_h = workspace.new_atom();
                        let num = num_h.get_mut().transform_to_num();
                        num.from_number(Number::Natural(pow as i64, 1));

                        let mut pow_h = workspace.new_atom();
                        let pow = pow_h.get_mut().transform_to_pow();
                        pow.from_base_and_exp(var_h.get().to_view(), num_h.get().to_view());
                        mul.extend(pow_h.get().to_view());
                    } else {
                        mul.extend(var_h.get().to_view());
                    }
                }
            }

            let mut num_h = workspace.new_atom();
            let num = num_h.get_mut().transform_to_num();
            let number = match monomial.coefficient {
                Rational::Natural(n, d) => Number::Natural(*n as i64, *d as i64),
                Rational::Large(r) => Number::Large(r.clone()),
            };
            num.from_number(number);
            mul.extend(num_h.get().to_view());

            add.extend(mul_h.get().to_view());
        }
    }
}
