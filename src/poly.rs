pub mod evaluate;
pub mod factor;
pub mod gcd;
pub mod groebner;
pub mod polynomial;
pub mod resultant;
pub mod univariate;

use std::borrow::Cow;
use std::cmp::Ordering::{self, Equal};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add as OpAdd, AddAssign, DerefMut, Div, Mul as OpMul, Neg, Rem, Sub};
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
use crate::representations::{Atom, AtomView, Symbol};
use crate::state::{State, Workspace};
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

/// A polynomial variable. It is either a (global) symbol
/// a temporary variable (for internal use), an array entry,
/// a function or any other non-polynomial part.
#[derive(Clone, Hash, Eq, Debug)]
pub enum Variable {
    Symbol(Symbol),
    Temporary(usize), // a temporary variable, for internal use
    Function(Symbol, Arc<Atom>),
    Other(Arc<Atom>), // any other non-polynomial part, for example x^-1, x^y, etc.
}

impl PartialEq for Variable {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Variable::Symbol(a), Variable::Symbol(b)) => a == b,
            (Variable::Temporary(a), Variable::Temporary(b)) => a == b,
            (Variable::Function(a, b), Variable::Function(c, d)) => a == c && b == d,
            (Variable::Other(a), Variable::Other(b)) => a == b,
            _ => false,
        }
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Variable::Symbol(v) => f.write_str(State::get_name(*v)),
            Variable::Temporary(t) => f.write_fmt(format_args!("_TMP_{}", *t)),
            Variable::Function(_, a) | Variable::Other(a) => std::fmt::Display::fmt(a, f),
        }
    }
}

impl From<Symbol> for Variable {
    fn from(i: Symbol) -> Variable {
        Variable::Symbol(i)
    }
}

impl Variable {
    pub fn to_id(&self) -> Option<Symbol> {
        match self {
            Variable::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Variable::Symbol(v) => format!("{}", State::get_name(*v)),
            Variable::Temporary(t) => format!("_TMP_{}", *t),
            Variable::Function(_, a) | Variable::Other(a) => format!("{}", a),
        }
    }
}

impl Atom {
    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    pub fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> MultivariatePolynomial<R, E> {
        self.as_view().to_polynomial(field, var_map)
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.as_view()
            .to_rational_polynomial(field, out_field, var_map)
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.as_view()
            .to_factorized_rational_polynomial(field, out_field, var_map)
    }
}

impl<'a> AtomView<'a> {
    /// Convert an expanded expression to a polynomial.
    fn to_polynomial_expanded<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<&Arc<Vec<Variable>>>,
        allow_new_vars: bool,
    ) -> Result<MultivariatePolynomial<R, E>, &'static str> {
        fn check_factor(
            factor: &AtomView<'_>,
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
                    let name = v.get_symbol();
                    if !vars.contains(&name.into()) {
                        if !allow_new_vars {
                            return Err("Expression contains variable that is not in variable map");
                        } else {
                            vars.push(v.get_symbol().into());
                        }
                    }
                    Ok(())
                }
                AtomView::Fun(_) => Err("function not supported in polynomial"),
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();
                    match base {
                        AtomView::Var(v) => {
                            let name = v.get_symbol();
                            if !vars.contains(&name.into()) {
                                if !allow_new_vars {
                                    return Err(
                                        "Expression contains variable that is not in variable map",
                                    );
                                } else {
                                    vars.push(v.get_symbol().into());
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

        fn check_term(
            term: &AtomView<'_>,
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
                    check_term(&term, &mut vars, allow_new_vars)?;
                    n_terms += 1;
                }
            }
            _ => {
                check_term(self, &mut vars, allow_new_vars)?;
                n_terms += 1;
            }
        }

        fn parse_factor<R: Ring + ConvertToRing, E: Exponent>(
            factor: &AtomView<'_>,
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
                    let id = v.get_symbol();
                    exponents[vars.iter().position(|v| *v == id.into()).unwrap()] += E::one();
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    let var_index = match base {
                        AtomView::Var(v) => {
                            let id = v.get_symbol();
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

        fn parse_term<R: Ring + ConvertToRing, E: Exponent>(
            term: &AtomView<'_>,
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

        let mut poly =
            MultivariatePolynomial::<R, E>::new(field, Some(n_terms), Arc::new(vars.clone()));

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

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    pub fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> MultivariatePolynomial<R, E> {
        self.to_polynomial_impl(field, var_map.as_ref().unwrap_or(&Arc::new(Vec::new())))
    }

    pub fn to_polynomial_impl<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: &Arc<Vec<Variable>>,
    ) -> MultivariatePolynomial<R, E> {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            return num;
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
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
                            return base.to_polynomial_impl(field, var_map).pow(nn as usize);
                        }
                    }
                }

                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Other(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Other(Arc::new(self.to_owned())));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp)
                }
            }
            AtomView::Fun(f) => {
                // TODO: make sure that this coefficient does not depend on any of the variables in var_map

                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Function(
                        f.get_symbol(),
                        Arc::new(self.to_owned()),
                    ));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp)
                }
            }
            AtomView::Mul(m) => {
                let mut r =
                    MultivariatePolynomial::new(field, None, var_map.clone()).constant(field.one());
                for arg in m.iter() {
                    let mut arg_r = arg.to_polynomial_impl(field, &r.variables);
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = MultivariatePolynomial::new(field, None, var_map.clone());
                for arg in a.iter() {
                    let mut arg_r = arg.to_polynomial_impl(field, &r.variables);
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        Workspace::get_local().with(|ws| {
            self.to_rational_polynomial_impl(
                ws,
                field,
                out_field,
                var_map.as_ref().unwrap_or(&Arc::new(Vec::new())),
            )
        })
    }

    fn to_rational_polynomial_impl<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<Variable>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            let den = num.one();
            return RationalPolynomial::from_num_den(num, den, out_field, false);
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd == 1 {
                            let b = base
                                .to_rational_polynomial_impl(workspace, field, out_field, var_map);

                            return if nn < 0 {
                                let b_inv = b.inv();
                                b_inv.pow(-nn as u64)
                            } else {
                                b.pow(nn as u64)
                            };
                        }
                    }
                }

                // non-integer exponent, convert to new variable
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Other(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    let den = r.one();
                    RationalPolynomial::from_num_den(r, den, out_field, false)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Other(Arc::new(self.to_owned())));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    let den = r.one();
                    RationalPolynomial::from_num_den(r, den, out_field, false)
                }
            }
            AtomView::Fun(f) => {
                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    let den = r.one();
                    RationalPolynomial::from_num_den(r, den, out_field, false)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Function(
                        f.get_symbol(),
                        Arc::new(self.to_owned()),
                    ));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);

                    let den = r.one();
                    RationalPolynomial::from_num_den(r, den, out_field, false)
                }
            }
            AtomView::Mul(m) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                for arg in m.iter() {
                    let mut arg_r = arg.to_rational_polynomial_impl(
                        workspace,
                        field,
                        out_field,
                        &r.numerator.variables,
                    );
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                for arg in a.iter() {
                    let mut arg_r = arg.to_rational_polynomial_impl(
                        workspace,
                        field,
                        out_field,
                        &r.numerator.variables,
                    );
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<Variable>>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        Workspace::get_local().with(|ws| {
            self.to_factorized_rational_polynomial_impl(
                ws,
                field,
                out_field,
                var_map.as_ref().unwrap_or(&Arc::new(Vec::new())),
            )
        })
    }

    pub fn to_factorized_rational_polynomial_impl<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
        workspace: &Workspace,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<Variable>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            let den = vec![(num.one(), 1)];
            return FactorizedRationalPolynomial::from_num_den(num, den, out_field, false);
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd) = num_n {
                        if nd == 1 {
                            let b = base.to_factorized_rational_polynomial_impl(
                                workspace, field, out_field, var_map,
                            );

                            return if nn < 0 {
                                let b_inv = b.inv();
                                b_inv.pow(-nn as u64)
                            } else {
                                b.pow(nn as u64)
                            };
                        }
                    }
                }

                // non-integer exponent, convert to new variable
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Other(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    FactorizedRationalPolynomial::from_num_den(r, vec![], out_field, false)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Other(Arc::new(self.to_owned())));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    FactorizedRationalPolynomial::from_num_den(r, vec![], out_field, false)
                }
            }
            AtomView::Fun(f) => {
                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    Variable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    FactorizedRationalPolynomial::from_num_den(r, vec![], out_field, false)
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(Variable::Function(
                        f.get_symbol(),
                        Arc::new(self.to_owned()),
                    ));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    FactorizedRationalPolynomial::from_num_den(r, vec![], out_field, false)
                }
            }
            AtomView::Mul(m) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                r.numer_coeff = out_field.one();
                for arg in m.iter() {
                    let mut arg_r = arg.to_factorized_rational_polynomial_impl(
                        workspace,
                        field,
                        out_field,
                        &r.numerator.variables,
                    );
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                for arg in a.iter() {
                    let mut arg_r = arg.to_factorized_rational_polynomial_impl(
                        workspace,
                        field,
                        out_field,
                        &r.numerator.variables,
                    );
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }
}

impl<R: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<R, E, O> {
    pub fn to_expression(&self) -> Atom
    where
        R::Element: Into<Coefficient>,
    {
        let mut out = Atom::default();
        self.to_expression_into(&mut out);
        out
    }

    pub fn to_expression_into(&self, out: &mut Atom)
    where
        R::Element: Into<Coefficient>,
    {
        Workspace::get_local().with(|ws| self.to_expression_with_map(ws, &HashMap::default(), out));
    }

    pub(crate) fn to_expression_with_map(
        &self,
        workspace: &Workspace,
        map: &HashMap<Variable, AtomView>,
        out: &mut Atom,
    ) where
        R::Element: Into<Coefficient>,
    {
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

            for (var_id, &pow) in self.variables.iter().zip(monomial.exponents) {
                if pow > E::zero() {
                    match var_id {
                        Variable::Symbol(v) => {
                            var_h.to_var(*v);
                        }
                        Variable::Temporary(_) => {
                            let a = map.get(var_id).expect("Variable missing from map");
                            var_h.set_from_view(a);
                        }
                        Variable::Function(_, a) | Variable::Other(a) => {
                            var_h.set_from_view(&a.as_view());
                        }
                    }

                    if pow > E::one() {
                        num_h.to_num((pow.to_u32() as i64).into());
                        pow_h.to_pow(var_h.as_view(), num_h.as_view());
                        mul.extend(pow_h.as_view());
                    } else {
                        mul.extend(var_h.as_view());
                    }
                }
            }

            let number = monomial.coefficient.clone().into();
            num_h.to_num(number);
            mul.extend(num_h.as_view());
            add.extend(mul_h.as_view());
        }

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
    }
}

impl<R: Ring, E: Exponent> RationalPolynomial<R, E> {
    pub fn to_expression(&self) -> Atom
    where
        R::Element: Into<Coefficient>,
    {
        let mut out = Atom::default();
        self.to_expression_into(&mut out);
        out
    }

    pub fn to_expression_into(&self, out: &mut Atom)
    where
        R::Element: Into<Coefficient>,
    {
        Workspace::get_local().with(|ws| self.to_expression_with_map(ws, &HashMap::default(), out));
    }

    /// Convert from a rational polynomial to an atom. The `map` maps all
    /// temporary variables back to atoms.
    pub(crate) fn to_expression_with_map(
        &self,
        workspace: &Workspace,
        map: &HashMap<Variable, AtomView>,
        out: &mut Atom,
    ) where
        R::Element: Into<Coefficient>,
    {
        if self.denominator.is_one() {
            self.numerator.to_expression_with_map(workspace, map, out);
            return;
        }

        let mul = out.to_mul();

        let mut poly = workspace.new_atom();
        self.numerator
            .to_expression_with_map(workspace, map, &mut poly);
        mul.extend(poly.as_view());

        self.denominator
            .to_expression_with_map(workspace, map, &mut poly);

        let mut pow_h = workspace.new_atom();
        pow_h.to_pow(poly.as_view(), workspace.new_num(-1).as_view());
        mul.extend(pow_h.as_view());

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
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
                let mut poly =
                    MultivariatePolynomial::<R, E>::new(field, Some(args.len()), var_map.clone());

                for term in args {
                    parse_term(term, var_name_map, &mut poly, field)?;
                }
                Ok(poly)
            }
            _ => {
                let mut poly = MultivariatePolynomial::<R, E>::new(field, Some(1), var_map.clone());
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
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + ConvertToRing + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
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
                let r = args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                Ok(r.inv())
            }
            Token::Op(_, _, Operator::Pow, args) => {
                // we have a pow that could not be parsed by to_polynomial
                // if the exponent is not -1, we pass the subexpression to
                // the general routine
                if Token::Number("-1".into()) == args[1] {
                    let r =
                        args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    Ok(r.inv())
                } else {
                    Workspace::get_local().with(|ws| {
                        let mut atom = ws.new_atom();
                        self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                        Ok(atom
                            .as_view()
                            .to_rational_polynomial_impl(ws, field, out_field, var_map))
                    })
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                for arg in args {
                    let mut arg_r =
                        arg.to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                for arg in args {
                    let mut arg_r =
                        arg.to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;

                Ok(r.neg())
            }
            _ => Workspace::get_local().with(|ws| {
                let mut atom = ws.new_atom();
                self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                Ok(atom
                    .as_view()
                    .to_rational_polynomial_impl(ws, field, out_field, var_map))
            }),
        }
    }

    /// Convert a parsed expression to a rational polynomial if possible,
    /// skipping the conversion to a Symbolica expression. This method
    /// is faster if the parsed expression is already in the same format
    /// i.e. the ordering is the same
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + ConvertToRing + PolynomialGCD<E>,
        E: Exponent,
    >(
        &self,
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
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    Ok(r.inv())
                } else {
                    Workspace::get_local().with(|ws| {
                        let mut atom = ws.new_atom();
                        self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                        Ok(atom
                            .as_view()
                            .to_factorized_rational_polynomial_impl(ws, field, out_field, var_map))
                    })
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                r.numer_coeff = out_field.one();
                for arg in args {
                    if let Token::Op(_, _, Operator::Inv, inv_args) = arg {
                        debug_assert!(inv_args.len() == 1);
                        let mut arg_r = inv_args[0].to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )?;

                        r.unify_variables(&mut arg_r);
                        r = &r / &arg_r;
                    } else {
                        let mut arg_r = arg.to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )?;
                        r.unify_variables(&mut arg_r);
                        r = &r * &arg_r;
                    }
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());

                // sort based on length, as this may improve performance
                let mut polys: Vec<FactorizedRationalPolynomial<_, _>> = args
                    .iter()
                    .map(|arg| {
                        arg.to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )
                    })
                    .collect::<Result<_, _>>()?;

                polys.sort_by_key(|p| {
                    p.numerator.nterms()
                        + p.denominators
                            .iter()
                            .map(|(x, _)| x.nterms())
                            .sum::<usize>()
                });

                for mut p in polys {
                    r.unify_variables(&mut p);
                    r = &r + &p;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_factorized_rational_polynomial(
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;

                Ok(r.neg())
            }
            _ => Workspace::get_local().with(|ws| {
                let mut atom = ws.new_atom();
                self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                Ok(atom
                    .as_view()
                    .to_factorized_rational_polynomial_impl(ws, field, out_field, var_map))
            }),
        }
    }
}
