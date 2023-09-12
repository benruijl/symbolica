pub mod evaluate;
pub mod gcd;
pub mod polynomial;

use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{Add as OpAdd, AddAssign, Div, Mul as OpMul, Neg, Sub};

use rug::{Complete, Integer as ArbitraryPrecisionInteger};
use smallvec::{smallvec, SmallVec};
use smartstring::{LazyCompact, SmartString};

use crate::parser::{Operator, Token};
use crate::representations::number::{BorrowedNumber, ConvertToRing, Number};
use crate::representations::{
    Add, Atom, AtomSet, AtomView, Identifier, Mul, Num, OwnedAdd, OwnedMul, OwnedNum, OwnedPow,
    OwnedVar, Pow, Var,
};
use crate::rings::integer::{Integer, IntegerRing};
use crate::rings::rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial};
use crate::rings::{EuclideanDomain, Ring};
use crate::state::{State, Workspace};
use crate::utils;

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
    + Sub<Output = Self>
    + OpAdd<Output = Self>
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

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Convert an expression to a polynomial.
    ///
    /// This function requires an expanded polynomial. If this yields too many terms, consider using
    /// calling `to_rational_polynomial` instead.
    pub fn to_polynomial<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: R,
        var_map: Option<&[Identifier]>,
    ) -> Result<MultivariatePolynomial<R, E>, &'static str> {
        fn check_factor<P: AtomSet>(
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
                            BorrowedNumber::Natural(n, d) => {
                                if d == 1 && n >= 0 && n <= u32::MAX as i64 {
                                    Ok(())
                                } else {
                                    Err("Exponent negative or a fraction")
                                }
                            }
                            BorrowedNumber::Large(r) => {
                                let r = r.to_rat();
                                if r.denom().to_u8() == Some(1) && r.numer().to_u32().is_some() {
                                    Ok(())
                                } else {
                                    Err("Exponent too large or negative or a fraction")
                                }
                            }
                            BorrowedNumber::FiniteField(_, _) => {
                                Err("Finite field not supported in conversion routine")
                            }
                            BorrowedNumber::RationalPolynomial(_) => {
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
            vars: &mut SmallVec<[Identifier; INLINED_EXPONENTS]>,
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
        let mut vars: SmallVec<[Identifier; INLINED_EXPONENTS]> =
            var_map.map(|v| v.into()).unwrap_or(SmallVec::new());
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
            vars: &[Identifier],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: R,
        ) {
            match factor {
                AtomView::Num(n) => {
                    field.mul_assign(
                        coefficient,
                        &field.element_from_borrowed_number(n.get_number_view()),
                    );
                }
                AtomView::Var(v) => {
                    let id = v.get_name();
                    exponents[vars.iter().position(|v| *v == id).unwrap()] += E::one();
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
            vars: &[Identifier],
            poly: &mut MultivariatePolynomial<R, E>,
            field: R,
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
            MultivariatePolynomial::<R, E>::new(vars.len(), field, Some(n_terms), Some(&vars));

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
        field: R,
        out_field: RO,
        var_map: Option<&[Identifier]>,
    ) -> Result<RationalPolynomial<RO, E>, Cow<'static, str>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map) {
            let den = num.new_from_constant(field.one());
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                let num = self.to_polynomial(field, var_map)?;
                let den = num.new_from_constant(field.one());
                Ok(RationalPolynomial::from_num_den(num, den, out_field, false))
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
                let mut r = RationalPolynomial::new(out_field, var_map);
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
                let mut r = RationalPolynomial::new(out_field, var_map);
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
}

impl<P: AtomSet> Atom<P> {
    pub fn from_polynomial<E: Exponent>(
        &mut self,
        workspace: &Workspace<P>,
        state: &State,
        poly: &MultivariatePolynomial<IntegerRing, E>,
    ) {
        let var_map = poly
            .var_map
            .as_ref()
            .expect("No variable map present in polynomial");

        let add = self.to_add();

        for monomial in poly {
            let mut mul_h = workspace.new_atom();
            let mul = mul_h.to_mul();

            for (&var_id, &pow) in var_map.iter().zip(monomial.exponents) {
                if pow > E::zero() {
                    let mut var_h = workspace.new_atom();
                    let var = var_h.to_var();
                    var.set_from_id(var_id);

                    if pow > E::one() {
                        let mut num_h = workspace.new_atom();
                        let num = num_h.to_num();
                        num.set_from_number(Number::Natural(pow.to_u32() as i64, 1));

                        let mut pow_h = workspace.new_atom();
                        let pow = pow_h.to_pow();
                        pow.set_from_base_and_exp(var_h.get().as_view(), num_h.get().as_view());
                        mul.extend(pow_h.get().as_view());
                    } else {
                        mul.extend(var_h.get().as_view());
                    }
                }
            }

            let mut num_h = workspace.new_atom();
            let num = num_h.to_num();
            let number = match monomial.coefficient {
                Integer::Natural(n) => Number::Natural(*n, 1),
                Integer::Large(r) => Number::Large(r.into()),
            };
            num.set_from_number(number);
            mul.extend(num_h.get().as_view());
            mul.set_dirty(true);

            add.extend(mul_h.get().as_view());
        }
        add.set_dirty(true);

        let mut norm = workspace.new_atom();
        self.as_view().normalize(workspace, state, &mut norm);
        std::mem::swap(norm.get_mut(), self);
    }
}

impl Token {
    pub fn to_polynomial<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: R,
        var_map: &[Identifier],
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<MultivariatePolynomial<R, E>, Cow<'static, str>> {
        fn parse_factor<R: Ring + ConvertToRing, E: Exponent>(
            factor: &Token,
            var_name_map: &[SmartString<LazyCompact>],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: R,
        ) -> Result<(), Cow<'static, str>> {
            match factor {
                Token::Number(n) => {
                    let num = if let Ok(x) = n.parse::<i64>() {
                        field.element_from_number(Number::Natural(x, 1))
                    } else {
                        match ArbitraryPrecisionInteger::parse(n) {
                            Ok(x) => {
                                let p = x.complete().into();
                                field.element_from_number(Number::Large(p))
                            }
                            Err(e) => Err(format!("Could not parse number: {}", e))?,
                        }
                    };
                    field.mul_assign(coefficient, &num);
                }
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
                            if let Ok(x) = n.parse::<i64>() {
                                if x < 1 || x > u32::MAX as i64 {
                                    Err("Invalid exponent")?;
                                }
                                exponents[var_index] += E::from_u32(x as u32);
                            } else {
                                match ArbitraryPrecisionInteger::parse(n) {
                                    Ok(x) => {
                                        let p: ArbitraryPrecisionInteger = x.complete();
                                        let exp = p.to_u32().ok_or("Cannot convert to u32")?;
                                        exponents[var_index] += E::from_u32(exp);
                                    }
                                    Err(e) => Err(format!("Could not parse number: {}", e))?,
                                }
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
            field: R,
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
                    Some(var_map),
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
                    Some(var_map),
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
        field: R,
        out_field: RO,
        var_map: &[Identifier],
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
                num.new_from_constant(field.one())
            };

            // in the fast format [a,b], the gcd of a and b should always be 1
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map, var_name_map) {
            let den = num.new_from_constant(field.one());
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            Token::Number(_) | Token::ID(_) => {
                let num = self.to_polynomial(field, var_map, var_name_map)?;
                let den = num.new_from_constant(field.one());
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
                let mut r = RationalPolynomial::new(out_field, Some(var_map));
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
                let mut r = RationalPolynomial::new(out_field, Some(var_map));
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
}
