pub mod evaluate;
pub mod factor;
pub mod gcd;
pub mod groebner;
pub mod polynomial;

use std::borrow::Cow;
use std::cmp::Ordering::{self, Equal};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add as OpAdd, AddAssign, Div, Mul as OpMul, Neg, Rem, Sub};
use std::sync::Arc;

use ahash::HashMap;
use smallvec::{smallvec, SmallVec};
use smartstring::{LazyCompact, SmartString};

use crate::coefficient::{Coefficient, CoefficientView, ConvertToRing};
use crate::domains::factorized_rational_polynomial::{
    FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
};
use crate::domains::integer::Integer;
use crate::domains::rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial};
use crate::domains::{EuclideanDomain, Ring};
use crate::parser::{Operator, Token};
use crate::representations::{
    Add, Atom, AtomSet, AtomView, Fun, Identifier, Mul, Num, OwnedAdd, OwnedFun, OwnedMul,
    OwnedNum, OwnedPow, OwnedVar, Pow, Var,
};
use crate::state::{BufferHandle, State, Workspace};
use crate::utils;

use self::factor::Factorize;
use self::gcd::PolynomialGCD;
use self::polynomial::MultivariatePolynomial;

pub const INLINED_EXPONENTS: usize = 6;

pub trait Exponent:
    Hash
    + Debug
    + Display
    + Ord
    + OpMul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Sub<Output = Self>
    + OpAdd<Output = Self>
    + Sum<Self>
    + AddAssign
    + Clone
    + Copy
    + PartialEq
    + Eq
{
    fn zero() -> Self;
    fn one() -> Self;
    /// Convert the exponent to `u32`. This is always possible, as `u32` is the largest supported exponent type.
    fn to_u32(&self) -> u32;
    /// Convert from `u32`. This function may panic if the exponent is too large.
    fn from_u32(n: u32) -> Self;
    fn is_zero(&self) -> bool;
    fn checked_add(&self, other: &Self) -> Option<Self>;
    fn gcd(&self, other: &Self) -> Self;

    /// Pack a list of exponents into a number, such that arithmetic and
    /// comparisons can be performed. The caller must guarantee that:
    /// - the list is no longer than 8 entries
    /// - each entry is not larger than 255
    fn pack(list: &[Self]) -> u64;
    fn unpack(n: u64, out: &mut [Self]);

    /// Pack a list of exponents into a number, such that arithmetic and
    /// comparisons can be performed. The caller must guarantee that:
    /// - the list is no longer than 4 entries
    /// - each entry is not larger than 2^16 - 1
    fn pack_u16(list: &[Self]) -> u64;
    fn unpack_u16(n: u64, out: &mut [Self]);
}

impl Exponent for u32 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_u32(&self) -> u32 {
        *self
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        n
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        u32::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        utils::gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as u32;
        }
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as u32;
        }
    }
}

impl Exponent for u16 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_u32(&self) -> u32 {
        *self as u32
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        if n <= u16::MAX as u32 {
            n as u16
        } else {
            panic!("Exponent {} too large for u16", n);
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        u16::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        utils::gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as u16;
        }
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + x.to_be() as u64;
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes();
        }
    }
}

/// An exponent limited to 255 for efficiency
impl Exponent for u8 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_u32(&self) -> u32 {
        *self as u32
    }

    #[inline]
    fn from_u32(n: u32) -> Self {
        if n <= u8::MAX as u32 {
            n as u8
        } else {
            panic!("Exponent {} too large for u8", n);
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        u8::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        utils::gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        out.copy_from_slice(s);
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as u8;
        }
    }
}

/// A well-order of monomials.
pub trait MonomialOrder: Clone {
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering;
}

/// Graded reverse lexicographic ordering of monomials.
#[derive(Clone)]
pub struct GrevLexOrder {}

impl MonomialOrder for GrevLexOrder {
    #[inline]
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering {
        let deg: E = a.iter().cloned().sum();
        let deg2: E = b.iter().cloned().sum();

        match deg.cmp(&deg2) {
            Equal => {}
            x => {
                return x;
            }
        }

        for (a1, a2) in a.iter().rev().zip(b.iter().rev()) {
            match a1.cmp(a2) {
                Equal => {}
                x => {
                    return x.reverse();
                }
            }
        }

        Equal
    }
}

/// Lexicographic ordering of monomials.
#[derive(Clone)]
pub struct LexOrder {}

impl MonomialOrder for LexOrder {
    #[inline]
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering {
        a.cmp(b)
    }
}

/// A polynomial variable. It is either a (global) identifier
/// or a temporary variable.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Variable {
    Identifier(Identifier),
    Temporary(usize),
    Array(Identifier, usize), // an array entry, e.g. a[0]
}

impl From<Identifier> for Variable {
    fn from(i: Identifier) -> Variable {
        Variable::Identifier(i)
    }
}

impl Variable {
    pub fn to_id(&self) -> Option<Identifier> {
        match self {
            Variable::Identifier(s) => Some(*s),
            _ => None,
        }
    }

    pub fn fmt(&self, state: &State, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Variable::Identifier(v) => f.write_str(state.get_name(*v)),
            Variable::Temporary(t) => f.write_fmt(format_args!("_TMP_{}", *t)),
            Variable::Array(t, i) => f.write_fmt(format_args!("{}[{}]", state.get_name(*t), i)),
        }
    }

    pub fn to_string(&self, state: &State) -> String {
        match self {
            Variable::Identifier(v) => format!("{}", state.get_name(*v)),
            Variable::Temporary(t) => format!("_TMP_{}", *t),
            Variable::Array(t, i) => format!("{}[{}]", state.get_name(*t), i),
        }
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Convert an expression to a polynomial.
    ///
    /// This function requires an expanded polynomial. If this yields too many terms, consider using
    /// calling `to_rational_polynomial` instead.
    pub fn to_polynomial<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<&Arc<Vec<Variable>>>,
    ) -> Result<MultivariatePolynomial<R, E>, &'static str> {
        fn check_factor<P: AtomSet>(
            factor: &AtomView<'_, P>,
            vars: &mut Vec<Variable>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match factor {
                AtomView::Num(n) => match n.get_coeff_view() {
                    CoefficientView::FiniteField(_, _) => {
                        Err("Finite field not supported in conversion routine")
                    }
                    _ => Ok(()),
                },
                AtomView::Var(v) => {
                    let name = v.get_name();
                    if !vars.contains(&name.into()) {
                        if !allow_new_vars {
                            return Err("Expression contains variable that is not in variable map");
                        } else {
                            vars.push(v.get_name().into());
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
                            if !vars.contains(&name.into()) {
                                if !allow_new_vars {
                                    return Err(
                                        "Expression contains variable that is not in variable map",
                                    );
                                } else {
                                    vars.push(v.get_name().into());
                                }
                            }
                        }
                        _ => return Err("base must be a variable"),
                    }

                    match exp {
                        AtomView::Num(n) => match n.get_coeff_view() {
                            CoefficientView::Natural(n, d) => {
                                if d == 1 && n >= 0 && n <= u32::MAX as i64 {
                                    Ok(())
                                } else {
                                    Err("Exponent negative or a fraction")
                                }
                            }
                            CoefficientView::Large(r) => {
                                let r = r.to_rat();
                                if r.denom().to_u8() == Some(1) && r.numer().to_u32().is_some() {
                                    Ok(())
                                } else {
                                    Err("Exponent too large or negative or a fraction")
                                }
                            }
                            CoefficientView::FiniteField(_, _) => {
                                Err("Finite field not supported in conversion routine")
                            }
                            CoefficientView::RationalPolynomial(_) => {
                                Err("Rational polynomial not supported in conversion routine")
                            }
                        },
                        _ => Err("base must be a variable"),
                    }
                }
                AtomView::Add(_) => Err("Expression may not contain subexpressions"),
                AtomView::Mul(_) => unreachable!("Mul inside mul found"),
            }
        }

        fn check_term<P: AtomSet>(
            term: &AtomView<'_, P>,
            vars: &mut Vec<Variable>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match term {
                AtomView::Mul(m) => {
                    for factor in m.iter() {
                        check_factor(&factor, vars, allow_new_vars)?;
                    }
                    Ok(())
                }
                _ => check_factor(term, vars, allow_new_vars),
            }
        }

        // get all variables and check structure
        let mut vars = var_map.map(|v| (**v).clone()).unwrap_or_default();
        let mut n_terms = 0;
        match self {
            AtomView::Add(a) => {
                for term in a.iter() {
                    check_term(&term, &mut vars, var_map.is_none())?;
                    n_terms += 1;
                }
            }
            _ => {
                check_term(self, &mut vars, var_map.is_none())?;
                n_terms += 1;
            }
        }

        fn parse_factor<P: AtomSet, R: Ring + ConvertToRing, E: Exponent>(
            factor: &AtomView<'_, P>,
            vars: &[Variable],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: &R,
        ) {
            match factor {
                AtomView::Num(n) => {
                    field.mul_assign(
                        coefficient,
                        &field.element_from_coefficient_view(n.get_coeff_view()),
                    );
                }
                AtomView::Var(v) => {
                    let id = v.get_name();
                    exponents[vars.iter().position(|v| *v == id.into()).unwrap()] += E::one();
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    let var_index = match base {
                        AtomView::Var(v) => {
                            let id = v.get_name();
                            vars.iter().position(|v| *v == id.into()).unwrap()
                        }
                        _ => unreachable!(),
                    };

                    match exp {
                        AtomView::Num(n) => match n.get_coeff_view() {
                            CoefficientView::Natural(r, _) => {
                                exponents[var_index] += E::from_u32(r as u32)
                            }
                            CoefficientView::Large(r) => {
                                exponents[var_index] +=
                                    E::from_u32(r.to_rat().numer().to_u32().unwrap())
                            }
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!("Unsupported expression"),
            }
        }

        fn parse_term<P: AtomSet, R: Ring + ConvertToRing, E: Exponent>(
            term: &AtomView<'_, P>,
            vars: &[Variable],
            poly: &mut MultivariatePolynomial<R, E>,
            field: &R,
        ) {
            let mut coefficient = poly.field.one();
            let mut exponents = smallvec![E::zero(); vars.len()];

            match term {
                AtomView::Mul(m) => {
                    for factor in m.iter() {
                        parse_factor(&factor, vars, &mut coefficient, &mut exponents, field);
                    }
                }
                _ => parse_factor(term, vars, &mut coefficient, &mut exponents, field),
            }

            poly.append_monomial(coefficient, &exponents);
        }

        let mut poly = MultivariatePolynomial::<R, E>::new(
            vars.len(),
            field,
            Some(n_terms),
            Some(Arc::new(vars.clone())),
        );

        match self {
            AtomView::Add(a) => {
                for term in a.iter() {
                    parse_term(&term, &vars, &mut poly, field);
                }
            }
            _ => parse_term(self, &vars, &mut poly, field),
        }

        Ok(poly)
    }

    /// Convert an expression to a rational polynomial if possible.
    pub fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace<P>,
        state: &State,
        field: &R,
        out_field: &RO,
        var_map: Option<&Arc<Vec<Variable>>>,
    ) -> Result<RationalPolynomial<RO, E>, Cow<'static, str>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map) {
            let den = num.one();
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                let num = self.to_polynomial(field, var_map)?;
                let den = num.one();
                Ok(RationalPolynomial::from_num_den(num, den, out_field, false))
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd != 1 {
                            Err("Exponent cannot be a fraction")?
                        }

                        if nn != -1 {
                            let mut h = workspace.new_atom();
                            if !self.expand(workspace, state, h.get_mut()) {
                                // expansion did not change the input, so we are in a case of x^-3 or x^3
                                let r = base.to_rational_polynomial(
                                    workspace, state, field, out_field, var_map,
                                )?;

                                if nn < 0 {
                                    let r_inv = r.inv();
                                    Ok(r_inv.pow(-nn as u64))
                                } else {
                                    Ok(r.pow(nn as u64))
                                }
                            } else {
                                h.get().as_view().to_rational_polynomial(
                                    workspace, state, field, out_field, var_map,
                                )
                            }
                        } else if nn < 0 {
                            let r = base.to_rational_polynomial(
                                workspace, state, field, out_field, var_map,
                            )?;
                            Ok(r.inv())
                        } else {
                            base.to_rational_polynomial(workspace, state, field, out_field, var_map)
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
                let mut r = RationalPolynomial::new(out_field, var_map.cloned());
                r.numerator = r.numerator.add_monomial(out_field.one());
                for arg in m.iter() {
                    let mut arg_r =
                        arg.to_rational_polynomial(workspace, state, field, out_field, var_map)?;
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = RationalPolynomial::new(out_field, var_map.cloned());
                for arg in a.iter() {
                    let mut arg_r =
                        arg.to_rational_polynomial(workspace, state, field, out_field, var_map)?;
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
        }
    }

    /// Convert an expression to a rational polynomial if possible.
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace<P>,
        state: &State,
        field: &R,
        out_field: &RO,
        var_map: Option<&Arc<Vec<Variable>>>,
    ) -> Result<FactorizedRationalPolynomial<RO, E>, Cow<'static, str>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map) {
            let den = vec![(num.one(), 1)];
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, den, out_field, false,
            ));
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                let num = self.to_polynomial(field, var_map)?;
                let den = vec![(num.one(), 1)];
                Ok(FactorizedRationalPolynomial::from_num_den(
                    num, den, out_field, false,
                ))
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd != 1 {
                            Err("Exponent cannot be a fraction")?
                        }

                        if nn != -1 {
                            let mut h = workspace.new_atom();
                            if !self.expand(workspace, state, h.get_mut()) {
                                // expansion did not change the input, so we are in a case of x^-3 or x^3
                                let r = base.to_factorized_rational_polynomial(
                                    workspace, state, field, out_field, var_map,
                                )?;

                                if nn < 0 {
                                    let r_inv = r.inv();
                                    Ok(r_inv.pow(-nn as u64))
                                } else {
                                    Ok(r.pow(nn as u64))
                                }
                            } else {
                                h.get().as_view().to_factorized_rational_polynomial(
                                    workspace, state, field, out_field, var_map,
                                )
                            }
                        } else if nn < 0 {
                            let r = base.to_factorized_rational_polynomial(
                                workspace, state, field, out_field, var_map,
                            )?;
                            Ok(r.inv())
                        } else {
                            base.to_factorized_rational_polynomial(
                                workspace, state, field, out_field, var_map,
                            )
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
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.cloned());
                r.numerator = r.numerator.add_monomial(out_field.one());
                for arg in m.iter() {
                    let mut arg_r = arg.to_factorized_rational_polynomial(
                        workspace, state, field, out_field, var_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.cloned());
                for arg in a.iter() {
                    let mut arg_r = arg.to_factorized_rational_polynomial(
                        workspace, state, field, out_field, var_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
        }
    }

    /// Convert an expression to a polynomial, creating new temporary variables for
    /// all non-polynomial subexpressions. These are stored in `map`.
    pub fn to_polynomial_with_map<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        map: &mut HashMap<AtomView<'a, P>, Variable>,
    ) -> MultivariatePolynomial<R, E> {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, None) {
            return num;
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                // done by simple routine above
                unreachable!()
            }
            AtomView::Pow(p) => {
                // the case var^exp is already treated, so this must be a case that requires a map
                // check if the exponent is a positive integer, if so the base must be mapped
                // otherwise, map the entire power

                // TODO: make sure that this coefficient does not depend on any of the variables in var_map

                let (base, exp) = p.get_base_exp();

                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();
                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd == 1 && nn > 0 && nn < u32::MAX as i64 {
                            let id = if let Some(x) = map.get(&base) {
                                *x
                            } else {
                                let new_id = Variable::Temporary(map.len());
                                map.insert(base, new_id);
                                new_id
                            };

                            // generate id^pow
                            let mut r = MultivariatePolynomial::new(
                                1,
                                field,
                                None,
                                Some(Arc::new(vec![id])),
                            );
                            r.append_monomial(field.one(), &[E::from_u32(nn as u32)]);
                            return r;
                        }
                    }
                }

                let id = if let Some(x) = map.get(self) {
                    *x
                } else {
                    let new_id = Variable::Temporary(map.len());
                    map.insert(*self, new_id);
                    new_id
                };

                let mut r = MultivariatePolynomial::new(1, field, None, Some(Arc::new(vec![id])));
                r.append_monomial(field.one(), &[E::one()]);
                r
            }
            AtomView::Fun(f) => {
                // TODO: make sure that this coefficient does not depend on any of the variables in var_map

                // check if the argument consists of a single positive integer. If so, treat the function as an array
                if f.get_nargs() == 1 {
                    let arg = f.iter().next().unwrap();

                    if let AtomView::Num(n) = arg {
                        if let CoefficientView::Natural(n, 1) = n.get_coeff_view() {
                            if n >= 0 {
                                let id = Variable::Array(f.get_name(), n as usize);
                                let mut r = MultivariatePolynomial::new(
                                    1,
                                    field,
                                    None,
                                    Some(Arc::new(vec![id])),
                                );
                                r.append_monomial(field.one(), &[E::one()]);
                                return r;
                            }
                        }
                    }
                }

                let id = if let Some(x) = map.get(self) {
                    *x
                } else {
                    let new_id = Variable::Temporary(map.len());
                    map.insert(*self, new_id);
                    new_id
                };

                let mut r = MultivariatePolynomial::new(1, field, None, Some(Arc::new(vec![id])));
                r.append_monomial(field.one(), &[E::one()]);
                r
            }
            AtomView::Mul(m) => {
                let mut r = MultivariatePolynomial::new(0, field, None, None);
                r = r.add_monomial(field.one());
                for arg in m.iter() {
                    let mut arg_r = arg.to_polynomial_with_map(field, map);
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = MultivariatePolynomial::new(0, field, None, None);
                for arg in a.iter() {
                    let mut arg_r = arg.to_polynomial_with_map(field, map);
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }

    /// Convert an expression to a rational polynomial, creating new temporary variables for
    /// all non-rational subexpressions. These are stored in `map`.
    pub fn to_rational_polynomial_with_map<
        'b,
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &'b Workspace<P>,
        state: &State,
        field: &R,
        out_field: &RO,
        map: &mut HashMap<BufferHandle<'b, Atom<P>>, Variable>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, None) {
            let den = num.one();
            return RationalPolynomial::from_num_den(num, den, out_field, false);
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                // should be covered by the easy check
                unreachable!()
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd != 1 {
                            // convert base^(1/nd) to a new variable
                            let mut pow_h = workspace.new_atom();
                            pow_h.to_pow().set_from_base_and_exp(
                                base,
                                workspace
                                    .new_num(Coefficient::Rational((1, nd).into()))
                                    .as_view(),
                            );

                            let id = if let Some(x) = map.get(&pow_h) {
                                *x
                            } else {
                                let new_id = Variable::Temporary(map.len());
                                map.insert(pow_h, new_id);
                                new_id
                            };

                            let mut p = MultivariatePolynomial::new(
                                1,
                                field,
                                None,
                                Some(Arc::new(vec![id])),
                            );
                            p.append_monomial(
                                field.one(),
                                &[E::from_u32(nn.unsigned_abs() as u32)],
                            );
                            let den = p.one();
                            let r = RationalPolynomial::from_num_den(p, den, out_field, false);

                            if nn < 0 {
                                r.inv()
                            } else {
                                r
                            }
                        } else if nn != -1 {
                            let mut h = workspace.new_atom();
                            if !self.expand(workspace, state, h.get_mut()) {
                                // expansion did not change the input, so we are in a case of x^-3 or x^3
                                let r = base.to_rational_polynomial_with_map(
                                    workspace, state, field, out_field, map,
                                );

                                if nn < 0 {
                                    let r_inv = r.inv();
                                    r_inv.pow(-nn as u64)
                                } else {
                                    r.pow(nn as u64)
                                }
                            } else {
                                h.as_view().to_rational_polynomial_with_map(
                                    workspace, state, field, out_field, map,
                                )
                            }
                        } else if nn < 0 {
                            let r = base.to_rational_polynomial_with_map(
                                workspace, state, field, out_field, map,
                            );
                            r.inv()
                        } else {
                            base.to_rational_polynomial_with_map(
                                workspace, state, field, out_field, map,
                            )
                        }
                    } else {
                        // non-integer exponent, convert to new variable
                        let mut a = workspace.new_atom();
                        a.set_from_view(self);
                        let id = if let Some(x) = map.get(&a) {
                            *x
                        } else {
                            let new_id = Variable::Temporary(map.len());
                            map.insert(a, new_id);
                            new_id
                        };

                        let mut r =
                            MultivariatePolynomial::new(1, field, None, Some(Arc::new(vec![id])));
                        r.append_monomial(field.one(), &[E::one()]);
                        let den = r.one();
                        RationalPolynomial::from_num_den(r, den, out_field, false)
                    }
                } else {
                    // non-number exponent, convert to new variable
                    let mut a = workspace.new_atom();
                    a.set_from_view(self);
                    let id = if let Some(x) = map.get(&a) {
                        *x
                    } else {
                        let new_id = Variable::Temporary(map.len());
                        map.insert(a, new_id);
                        new_id
                    };

                    let mut r =
                        MultivariatePolynomial::new(1, field, None, Some(Arc::new(vec![id])));
                    r.append_monomial(field.one(), &[E::one()]);
                    let den = r.one();
                    RationalPolynomial::from_num_den(r, den, out_field, false)
                }
            }
            AtomView::Fun(_) => {
                // non-integer exponent, convert to new variable
                let mut a = workspace.new_atom();
                a.set_from_view(self);
                let id = if let Some(x) = map.get(&a) {
                    *x
                } else {
                    let new_id = Variable::Temporary(map.len());
                    map.insert(a, new_id);
                    new_id
                };

                let mut r = MultivariatePolynomial::new(1, field, None, Some(Arc::new(vec![id])));
                r.append_monomial(field.one(), &[E::one()]);
                let den = r.one();
                RationalPolynomial::from_num_den(r, den, out_field, false)
            }
            AtomView::Mul(m) => {
                let mut r = RationalPolynomial::new(out_field, None);
                r.numerator = r.numerator.add_monomial(out_field.one());
                for arg in m.iter() {
                    let mut arg_r = arg
                        .to_rational_polynomial_with_map(workspace, state, field, out_field, map);
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = RationalPolynomial::new(out_field, None);
                for arg in a.iter() {
                    let mut arg_r = arg
                        .to_rational_polynomial_with_map(workspace, state, field, out_field, map);
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }
}

impl<R: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<R, E, O> {
    pub fn to_expression<P: AtomSet>(
        &self,
        workspace: &Workspace<P>,
        state: &State,
        map: &HashMap<Variable, AtomView<P>>,
        out: &mut Atom<P>,
    ) where
        R::Element: Into<Coefficient>,
    {
        let var_map = self
            .var_map
            .as_ref()
            .expect("No variable map present in polynomial");

        if self.is_zero() {
            out.set_from_view(&workspace.new_num(0).as_view());
            return;
        }

        let add = out.to_add();

        let mut mul_h = workspace.new_atom();
        let mut var_h = workspace.new_atom();
        let mut num_h = workspace.new_atom();
        let mut pow_h = workspace.new_atom();

        for monomial in self {
            let mul = mul_h.to_mul();

            for (&var_id, &pow) in var_map.iter().zip(monomial.exponents) {
                if pow > E::zero() {
                    match var_id {
                        Variable::Identifier(v) => {
                            let var = var_h.to_var();
                            var.set_from_id(v);
                        }
                        Variable::Temporary(_) => {
                            let a = map.get(&var_id).expect("Variable missing from map");
                            var_h.set_from_view(a);
                        }
                        Variable::Array(n, i) => {
                            let fun = var_h.to_fun();
                            fun.set_from_name(n);
                            fun.add_arg(workspace.new_num(i as i64).as_view());
                        }
                    }

                    if pow > E::one() {
                        let num = num_h.to_num();
                        num.set_from_coeff((pow.to_u32() as i64).into());

                        let pow = pow_h.to_pow();
                        pow.set_from_base_and_exp(var_h.get().as_view(), num_h.get().as_view());
                        mul.extend(pow_h.get().as_view());
                    } else {
                        mul.extend(var_h.get().as_view());
                    }
                }
            }

            let num = num_h.to_num();
            let number = monomial.coefficient.clone().into();
            num.set_from_coeff(number);
            mul.extend(num_h.get().as_view());
            mul.set_dirty(true);

            add.extend(mul_h.get().as_view());
            add.set_dirty(true);
        }

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, state, &mut norm);
        std::mem::swap(norm.get_mut(), out);
    }
}

impl<R: Ring, E: Exponent> RationalPolynomial<R, E> {
    /// Convert from a rational polynomial to an atom. The `map` maps all
    /// temporary variables back to atoms.
    pub fn to_expression<P: AtomSet>(
        &self,
        workspace: &Workspace<P>,
        state: &State,
        map: &HashMap<Variable, AtomView<P>>,
        out: &mut Atom<P>,
    ) where
        R::Element: Into<Coefficient>,
    {
        if self.denominator.is_one() {
            self.numerator.to_expression(workspace, state, map, out);
            return;
        }

        let mul = out.to_mul();

        let mut poly = workspace.new_atom();
        self.numerator
            .to_expression(workspace, state, map, &mut poly);
        mul.extend(poly.as_view());

        self.denominator
            .to_expression(workspace, state, map, &mut poly);

        let mut pow_h = workspace.new_atom();
        let pow = pow_h.to_pow();
        pow.set_from_base_and_exp(poly.as_view(), workspace.new_num(-1).as_view());
        pow.set_dirty(true);
        mul.extend(pow_h.as_view());
        mul.set_dirty(true);

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, state, &mut norm);
        std::mem::swap(norm.get_mut(), out);
    }
}

impl Token {
    pub fn to_polynomial<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: &Arc<Vec<Variable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<MultivariatePolynomial<R, E>, Cow<'static, str>> {
        fn parse_factor<R: Ring + ConvertToRing, E: Exponent>(
            factor: &Token,
            var_name_map: &[SmartString<LazyCompact>],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: &R,
        ) -> Result<(), Cow<'static, str>> {
            match factor {
                Token::Number(n) => match n.parse::<Integer>() {
                    Ok(x) => {
                        field.mul_assign(coefficient, &field.element_from_integer(x));
                    }
                    Err(e) => Err(format!("Could not parse number: {}", e))?,
                },
                Token::ID(x) => {
                    let Some(index) = var_name_map.iter().position(|v| v == x) else {
                        Err(format!("Variable {} not specified in variable map", x))?
                    };
                    exponents[index] += E::one();
                }
                Token::Op(_, _, Operator::Neg, args) => {
                    if args.len() != 1 {
                        Err("Wrong args for neg")?;
                    }

                    *coefficient = field.neg(coefficient);
                    parse_factor(&args[0], var_name_map, coefficient, exponents, field)?;
                }
                Token::Op(_, _, Operator::Pow, args) => {
                    if args.len() != 2 {
                        Err("Wrong args for pow")?;
                    }

                    let var_index = match &args[0] {
                        Token::ID(v) => match var_name_map.iter().position(|v1| v == v1) {
                            Some(p) => p,
                            None => Err(format!("Variable {} not specified in variable map", v))?,
                        },
                        _ => Err("Unsupported base")?,
                    };

                    match &args[1] {
                        Token::Number(n) => {
                            if let Ok(x) = n.parse::<u32>() {
                                exponents[var_index] += E::from_u32(x);
                            } else {
                                Err("Invalid exponent")?
                            };
                        }
                        _ => Err("Unsupported exponent")?,
                    }
                }
                _ => Err("Unsupported expression")?,
            }

            Ok(())
        }

        fn parse_term<R: Ring + ConvertToRing, E: Exponent>(
            term: &Token,
            var_name_map: &[SmartString<LazyCompact>],
            poly: &mut MultivariatePolynomial<R, E>,
            field: &R,
        ) -> Result<(), Cow<'static, str>> {
            let mut coefficient = poly.field.one();
            let mut exponents = smallvec![E::zero(); var_name_map.len()];

            match term {
                Token::Op(_, _, Operator::Mul, args) => {
                    for factor in args {
                        parse_factor(
                            factor,
                            var_name_map,
                            &mut coefficient,
                            &mut exponents,
                            field,
                        )?;
                    }
                }
                Token::Op(_, _, Operator::Neg, args) => {
                    if args.len() != 1 {
                        Err("Wrong args for neg")?;
                    }

                    coefficient = field.neg(&coefficient);

                    match &args[0] {
                        Token::Op(_, _, Operator::Mul, args) => {
                            for factor in args {
                                parse_factor(
                                    factor,
                                    var_name_map,
                                    &mut coefficient,
                                    &mut exponents,
                                    field,
                                )?;
                            }
                        }
                        _ => parse_factor(
                            &args[0],
                            var_name_map,
                            &mut coefficient,
                            &mut exponents,
                            field,
                        )?,
                    }
                }
                _ => parse_factor(term, var_name_map, &mut coefficient, &mut exponents, field)?,
            }

            poly.append_monomial(coefficient, &exponents);
            Ok(())
        }

        match self {
            Token::Op(_, _, Operator::Add, args) => {
                let mut poly = MultivariatePolynomial::<R, E>::new(
                    var_map.len(),
                    field,
                    Some(args.len()),
                    Some(var_map.clone()),
                );

                for term in args {
                    parse_term(term, var_name_map, &mut poly, field)?;
                }
                Ok(poly)
            }
            _ => {
                let mut poly = MultivariatePolynomial::<R, E>::new(
                    var_map.len(),
                    field,
                    Some(1),
                    Some(var_map.clone()),
                );
                parse_term(self, var_name_map, &mut poly, field)?;
                Ok(poly)
            }
        }
    }

    /// Convert a parsed expression to a rational polynomial if possible,
    /// skipping the conversion to a Symbolica expression. This method
    /// is faster if the parsed expression is already in the same format
    /// i.e. the ordering is the same
    pub fn to_rational_polynomial<
        P: AtomSet,
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace<P>,
        state: &mut State,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<Variable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<RationalPolynomial<RO, E>, Cow<'static, str>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // use a faster routine to parse the rational polynomial
        if let Token::RationalPolynomial(r) = self {
            let mut iter = r.split(',');
            let Some(num) = iter.next() else {
                Err("Empty [] in input")?
            };

            let num = Token::parse_polynomial(num.as_bytes(), var_map, var_name_map, field).1;
            let den = if let Some(den) = iter.next() {
                Token::parse_polynomial(den.as_bytes(), var_map, var_name_map, field).1
            } else {
                num.one()
            };

            // in the fast format [a,b], the gcd of a and b should always be 1
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map, var_name_map) {
            let den = num.one();
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            Token::Number(_) | Token::ID(_) => {
                let num = self.to_polynomial(field, var_map, var_name_map)?;
                let den = num.one();
                Ok(RationalPolynomial::from_num_den(num, den, out_field, false))
            }
            Token::Op(_, _, Operator::Inv, args) => {
                assert!(args.len() == 1);
                let r = args[0].to_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;
                Ok(r.inv())
            }
            Token::Op(_, _, Operator::Pow, args) => {
                // we have a pow that could not be parsed by to_polynomial
                // if the exponent is not -1, we pass the subexpression to
                // the general routine
                if Token::Number("-1".into()) == args[1] {
                    let r = args[0].to_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    Ok(r.inv())
                } else {
                    let mut atom = workspace.new_atom();
                    self.to_atom_with_output(state, workspace, &mut atom)?;
                    atom.as_view().to_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        Some(var_map),
                    )
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = RationalPolynomial::new(out_field, Some(var_map.clone()));
                r.numerator = r.numerator.add_monomial(out_field.one());
                for arg in args {
                    let mut arg_r = arg.to_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = RationalPolynomial::new(out_field, Some(var_map.clone()));
                for arg in args {
                    let mut arg_r = arg.to_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;

                Ok(r.neg())
            }
            _ => {
                let mut atom = workspace.new_atom();
                self.to_atom_with_output(state, workspace, &mut atom)?;
                atom.as_view().to_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    Some(var_map),
                )
            }
        }
    }

    /// Convert a parsed expression to a rational polynomial if possible,
    /// skipping the conversion to a Symbolica expression. This method
    /// is faster if the parsed expression is already in the same format
    /// i.e. the ordering is the same
    pub fn to_factorized_rational_polynomial<
        P: AtomSet,
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace<P>,
        state: &mut State,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<Variable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<FactorizedRationalPolynomial<RO, E>, Cow<'static, str>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        // use a faster routine to parse the rational polynomial
        if let Token::RationalPolynomial(r) = self {
            let mut iter = r.split(',');
            let Some(num) = iter.next() else {
                Err("Empty [] in input")?
            };

            let num = Token::parse_polynomial(num.as_bytes(), var_map, var_name_map, field).1;

            let mut dens = vec![];

            let den = if let Some(den) = iter.next() {
                Token::parse_polynomial(den.as_bytes(), var_map, var_name_map, field).1
            } else {
                num.one()
            };

            if let Some(p1) = iter.next() {
                if !den.is_one() {
                    dens.push((
                        den,
                        p1.parse::<usize>()
                            .map_err(|e| format!("Could not parse power: {}", e))?,
                    ));
                }

                while let Some(p) = iter.next() {
                    let den = Token::parse_polynomial(p.as_bytes(), var_map, var_name_map, field).1;

                    let p = iter.next().ok_or("Missing power")?;
                    let p = p
                        .parse::<usize>()
                        .map_err(|e| format!("Could not parse power: {}", e))?;

                    dens.push((den, p));
                }
            } else if !den.is_one() {
                dens.push((den, 1));
            }

            // in the fast format [n,d1,p1,d2,p2,...] every denominator is irreducible and unique
            // TODO: set do_factor to true for [n,d] as this may have just the gcd being 1
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, dens, out_field, false,
            ));
        }

        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map, var_name_map) {
            let den = vec![(num.one(), 1)];
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, den, out_field, false,
            ));
        }

        match self {
            Token::Number(_) | Token::ID(_) => {
                let num = self.to_polynomial(field, var_map, var_name_map)?;
                let den = vec![(num.one(), 1)];
                Ok(FactorizedRationalPolynomial::from_num_den(
                    num, den, out_field, false,
                ))
            }
            Token::Op(_, _, Operator::Inv, args) => {
                assert!(args.len() == 1);
                let r = args[0].to_factorized_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;
                Ok(r.inv())
            }
            Token::Op(_, _, Operator::Pow, args) => {
                // we have a pow that could not be parsed by to_polynomial
                // if the exponent is not -1, we pass the subexpression to
                // the general routine
                if Token::Number("-1".into()) == args[1] {
                    let r = args[0].to_factorized_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    Ok(r.inv())
                } else {
                    let mut atom = workspace.new_atom();
                    self.to_atom_with_output(state, workspace, &mut atom)?;
                    atom.as_view().to_factorized_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        Some(var_map),
                    )
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, Some(var_map.clone()));
                r.numerator = r.numerator.add_monomial(out_field.one());
                for arg in args {
                    let mut arg_r = arg.to_factorized_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, Some(var_map.clone()));
                for arg in args {
                    let mut arg_r = arg.to_factorized_rational_polynomial(
                        workspace,
                        state,
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    r.unify_var_map(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_factorized_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;

                Ok(r.neg())
            }
            _ => {
                let mut atom = workspace.new_atom();
                self.to_atom_with_output(state, workspace, &mut atom)?;
                atom.as_view().to_factorized_rational_polynomial(
                    workspace,
                    state,
                    field,
                    out_field,
                    Some(var_map),
                )
            }
        }
    }
}
