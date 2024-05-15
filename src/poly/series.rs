use core::panic;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    atom::{Atom, AtomView, FunctionBuilder, Symbol},
    coefficient::CoefficientView,
    domains::{
        atom::AtomField, integer::Integer, rational::Rational, EuclideanDomain, Ring, RingPrinter,
    },
    printer::PrintOptions,
    state::State,
};

use super::Variable;

/// A Puiseux series. The truncation order is
/// relative to the lowest degree: i.e., a series in `x` with depth `d` is viewed as
/// `a*x^n*(1+bx+.. + O(x^(d+1)))`.
#[derive(Clone)]
pub struct Series<F: Ring> {
    coefficients: Vec<F::Element>,
    variable: Arc<Variable>,
    field: F,
    shift: isize, // a shift in units of the ramification that indicates the starting power of the series
    order: usize, // the order of truncation in units of the ramification
    ramification: usize, // the factor that makes the exponents integer
}

impl<F: Ring + std::fmt::Debug> std::fmt::Debug for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for c in self.coefficients.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{{ {:?} }}", c)?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + std::fmt::Display> std::fmt::Display for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let v = self.variable.to_string();

        if self.is_zero() {
            write!(f, "0")?;
            return write!(f, "+O({}^{})", v, self.absolute_order());
        }

        let mut first = true;
        for (e, c) in self.coefficients.iter().enumerate() {
            if F::is_zero(c) {
                continue;
            }

            if first {
                first = false;
            } else {
                write!(f, "+")?;
            }

            let p = RingPrinter {
                element: c,
                ring: &self.field,
                opts: PrintOptions::default(),
                in_product: true,
            };

            let e = self.get_exponent(e);

            if e.is_zero() {
                write!(f, "{}", p)?;
            } else if e.is_one() {
                if self.field.is_one(c) {
                    write!(f, "{}", v)?;
                } else {
                    write!(f, "{}*{}", p, v)?;
                }
            } else if self.field.is_one(c) {
                if e.is_integer() {
                    write!(f, "{}^{}", v, e)?;
                } else {
                    write!(f, "{}^({})", v, e)?;
                }
            } else if e.is_integer() {
                    write!(f, "{}*{}^{}", p, v, e)?;
            } else {
                write!(f, "{}*{}^({})", p, v, e)?;
            }
        }

        let o = self.absolute_order();
        if o.is_integer() {
            write!(f, "+O({}^{})", v, o)
        } else {
            write!(f, "+O({}^({}))", v, o)
        }
    }
}

impl<F: Ring> Series<F> {
    /// Constructs a zero series. Instead of using this constructor,
    /// prefer to create new series from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(field: &F, cap: Option<usize>, variable: Arc<Variable>, order: Rational) -> Self {
        if order.is_negative() {
            panic!("The order of the series must be positive.");
        }

        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            field: field.clone(),
            variable,
            shift: 0,
            order: order.numerator().to_i64().unwrap() as usize
                * order.denominator().to_i64().unwrap() as usize,
            ramification: order.denominator().to_i64().unwrap() as usize,
        }
    }

    /// Constructs a zero series, inheriting the field and variable from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a zero series with the given number of variables and capacity,
    /// inheriting the field and variable from `self`.
    #[inline]
    pub fn zero_with_capacity(&self, cap: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap),
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a constant series,
    /// inheriting the field and variable from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        if F::is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a series that is one, inheriting the field and variable map from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a series with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponent: Rational) -> Self {
        if F::is_zero(&coeff) {
            return self.zero();
        }

        let n = exponent.numerator().to_i64().unwrap() as isize;
        let d = exponent.denominator().to_i64().unwrap() as usize;

        let ram = Integer::from(self.ramification as i64)
            .lcm(&exponent.denominator())
            .to_i64()
            .unwrap() as usize;

        Self {
            coefficients: vec![coeff],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: n * d as isize,
            order: self.order * ram / d,
            ramification: ram,
        }
    }

    /// Constructs a series that is `x+a`, where `a` is a constant.
    #[inline]
    pub fn shifted_variable(&self, coeff: F::Element) -> Self {
        if F::is_zero(&coeff) {
            return self.monomial(self.field.one(), (1, 1).into());
        }

        Self {
            coefficients: vec![coeff, self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    // Map an index in the coefficient array to its power.
    #[inline]
    fn get_exponent(&self, index: usize) -> Rational {
        (Rational::from(index as i64) + Rational::from(self.shift as i64))
            / Rational::from(self.ramification as i64)
    }

    // Map a power to an index in the coefficient array.
    #[inline]
    fn get_index(&self, p: Rational) -> usize {
        let r = p * &Rational::from(self.ramification as i64);
        debug_assert!(r.is_integer());

        let i = r.numerator().to_i64().unwrap() - self.shift as i64;
        debug_assert!(i >= 0);
        i as usize
    }

    /// Get the relative order of the series expansion.
    #[inline]
    pub fn relative_order(&self) -> Rational {
        (self.order as i64, self.ramification as i64).into()
    }

    /// Get the absolute order of the series expansion.
    #[inline]
    pub fn absolute_order(&self) -> Rational {
        (
            self.order as i64 + self.shift as i64,
            self.ramification as i64,
        )
            .into()
    }

    fn change_ramification(&mut self, ram: usize) {
        let ram = Integer::from(self.ramification as i64)
            .lcm(&Integer::from(ram as i64))
            .to_i64()
            .unwrap() as usize;

        if ram == self.ramification {
            return;
        }

        let mut s = Self {
            coefficients: vec![
                self.field.zero();
                self.coefficients.len() * ram / self.ramification
            ],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: self.shift * (ram / self.ramification) as isize,
            order: self.order * (ram / self.ramification),
            ramification: ram,
        };

        for (i, c) in std::mem::take(&mut self.coefficients)
            .into_iter()
            .enumerate()
        {
            let index = s.get_index(self.get_exponent(i));
            s.coefficients[index] = c;
        }

        *self = s;
    }

    /// Truncate the series to the desired order.
    /// If the new order is larger, nothing happens.
    #[inline]
    pub fn truncate_absolute_order(&mut self, order: Rational) {
        if self.absolute_order() < order {
            return;
        }

        self.change_ramification(order.denominator().to_i64().unwrap() as usize);

        for i in (0..self.coefficients.len()).rev() {
            if self.get_exponent(i) >= order {
                self.coefficients.pop();
            } else {
                break;
            }
        }

        self.order = self.get_index(order);
        self.truncate();
    }

    #[inline]
    fn joint_ramification(&self, other: &Self) -> usize {
        Integer::from(self.ramification as i64)
            .lcm(&Integer::from(other.ramification as i64))
            .to_i64()
            .unwrap() as usize
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.field.is_one(&self.coefficients[0]) && self.shift == 0
    }

    /// Returns true if the series is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1 && self.shift == 0
    }

    #[inline]
    pub fn get_trailing_coefficient(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.field.zero();
        } else {
            self.coefficients[0].clone()
        }
    }

    #[inline]
    pub fn get_trailing_exponent(&self) -> Rational {
        self.get_exponent(0)
    }

    /// Get a copy of the variable/
    pub fn get_vars(&self) -> Arc<Variable> {
        self.variable.clone()
    }

    /// Get a reference to the variables
    pub fn get_vars_ref(&self) -> &Variable {
        self.variable.as_ref()
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        self.coefficients
            .last()
            .unwrap_or(&self.field.zero())
            .clone()
    }

    pub fn degree(&self) -> Rational {
        if self.is_zero() {
            return 0.into();
        }

        self.get_exponent(self.coefficients.len() - 1)
    }

    /// Compute `self^pow`.
    pub fn npow(&self, mut pow: usize) -> Self {
        if pow == 0 {
            return self.one();
        }

        let mut x = self.clone();
        let mut y = self.one();
        while pow != 1 {
            if pow % 2 == 1 {
                y = &y * &x;
                pow -= 1;
            }

            x = &x * &x;
            pow /= 2;
        }

        x * &y
    }

    /// Multiply the exponents `exp` units of the ramification.
    pub fn mul_exp_units(mut self, exp: isize) -> Self {
        if exp == 0 || self.is_zero() {
            return self.clone();
        }

        self.shift += exp;
        self
    }

    pub fn mul_coeff(mut self, coeff: &F::Element) -> Self {
        for c in &mut self.coefficients {
            if !F::is_zero(c) {
                self.field.mul_assign(c, coeff);
            }
        }

        self.truncate(); // zeros may have occurred

        self
    }

    /// Map a coefficient using the function `f`.
    pub fn map_coeff<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        f: T,
        field: U,
    ) -> Series<U> {
        let mut r = Series {
            coefficients: vec![],
            field,
            variable: self.variable.clone(),
            shift: self.shift,
            order: self.order,
            ramification: self.ramification,
        };
        r.coefficients = self.coefficients.iter().map(f).collect::<Vec<_>>();
        r.truncate();
        r
    }

    /// Truncate the leading and trailing non-zeroes.
    fn truncate(&mut self) {
        if self.coefficients.is_empty() {
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .rev()
            .position(|c| !F::is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.coefficients.truncate(self.coefficients.len() - d);

        if self.coefficients.is_empty() {
            self.order = (self.order as isize + self.shift) as usize;
            self.shift = 0;
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .position(|c| !F::is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.shift += d as isize;
        self.order -= d;
        self.coefficients.drain(0..d);
    }

    /// Remove the constant term, if it is first and it exists.
    fn remove_constant(mut self) -> Self {
        if !self.is_zero() && self.shift == 0 {
            self.coefficients[0] = self.field.zero();
            self.truncate();
        }

        self
    }
}

impl<F: Ring> PartialEq for Series<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable != other.variable {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.is_zero() != other.is_zero() {
                return false;
            }

            if self.is_zero() {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            // TODO: what is expected here?
            unimplemented!("Cannot compare non-constant series with different variable maps yet");
        }

        if self.degree() != other.degree() {
            return false;
        }
        self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring> std::hash::Hash for Series<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.variable.hash(state);
    }
}

impl<F: Ring> Eq for Series<F> {}

impl<R: Ring> PartialOrd for Series<R> {
    /// An ordering of seriess that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coefficients.partial_cmp(&other.coefficients)
    }
}

impl<F: Ring> Add for Series<F> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.field, other.field);
        assert_eq!(self.variable, other.variable);

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        if self.shift == other.shift && self.ramification == other.ramification {
            if self.coefficients.len() < other.coefficients.len() {
                std::mem::swap(&mut self, &mut other);
            }

            self.order = self.order.min(other.order);
            self.coefficients.truncate(self.order);

            for (i, c) in other.coefficients.iter().enumerate() {
                if i < self.coefficients.len() {
                    self.field.add_assign(&mut self.coefficients[i], c);
                }
            }

            self.truncate();
            self
        } else {
            let r = self.joint_ramification(&other);
            let r_d_s = r / self.ramification;
            let r_d_o = r / other.ramification;

            let shift = (self.shift * r_d_s as isize).min(other.shift * r_d_o as isize);
            let order_s = (self.order * r_d_s) as isize + self.shift * r_d_s as isize;
            let order_o = (other.order * r_d_o) as isize + other.shift * r_d_o as isize;

            let mut s = Self {
                coefficients: vec![],
                variable: self.variable.clone(),
                field: self.field.clone(),
                shift,
                order: (order_s.min(order_o) - shift) as usize,
                ramification: r,
            };

            s.coefficients = vec![self.field.zero(); s.order]; // TODO: tighten bound
            for (i, c) in self.coefficients.iter().enumerate() {
                let index = s.get_index(self.get_exponent(i));
                if index < s.order {
                    s.coefficients[index] = c.clone(); // TODO: prevent copy
                }
            }
            for (i, c) in other.coefficients.iter().enumerate() {
                let index = s.get_index(other.get_exponent(i));
                if index < s.order {
                    s.field.add_assign(&mut s.coefficients[index], c);
                }
            }

            s.truncate();
            s
        }
    }
}

impl<'a, 'b, F: Ring> Add<&'a Series<F>> for &'b Series<F> {
    type Output = Series<F>;

    fn add(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring> Sub for Series<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, 'b, F: Ring> Sub<&'a Series<F>> for &'b Series<F> {
    type Output = Series<F>;

    fn sub(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring> Neg for Series<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(c);
        }
        self
    }
}

impl<'a, 'b, F: Ring> Mul<&'a Series<F>> for &'b Series<F> {
    type Output = Series<F>;

    #[inline]
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return self.zero();
        }

        let r = self.joint_ramification(&rhs);
        let r_d_s = r / self.ramification;
        let r_d_o = r / rhs.ramification;
        let mut res = Series {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
            shift: self.shift * r_d_s as isize + rhs.shift * r_d_o as isize,
            order: (self.order * r_d_s).min(rhs.order * r_d_o),
            ramification: r,
        };

        res.coefficients = vec![self.field.zero(); res.order as usize];

        for (e1, c1) in self.coefficients.iter().enumerate() {
            if F::is_zero(c1) {
                continue;
            }

            let p1 = self.get_exponent(e1);

            for (e2, c2) in rhs.coefficients.iter().enumerate() {
                let p = &p1 + &rhs.get_exponent(e2);

                if !F::is_zero(c2) {
                    let index = res.get_index(p);
                    if index < res.coefficients.len() {
                        self.field
                            .add_mul_assign(&mut res.coefficients[index], c1, c2);
                    }
                }
            }
        }

        res.truncate();
        res
    }
}

impl<'a, F: Ring> Mul<&'a Series<F>> for Series<F> {
    type Output = Series<F>;

    #[inline]
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a, 'b> Div<&'a Series<AtomField>> for &'b Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        other.rpow((-1, 1).into()) * self
    }
}

impl<'a> Div<&'a Series<AtomField>> for Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: EuclideanDomain> Series<F> {
    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.field.zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.field.is_one(&c) {
                break;
            }

            c = self.field.gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        for c in &mut self.coefficients {
            let (quot, rem) = self.field.quot_rem(c, other);
            debug_assert!(F::is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the series primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }
}

impl Series<AtomField> {
    /// Extract x^a from an expression that comes from simplifying an exponential with logs
    /// i.e.: exp(c + 3 Log[x^5]]) = exp(c)*x^15.
    fn extract_exp_log(&self, e: AtomView, s: Symbol) -> Result<Self, &'static str> {
        if !e.contains_symbol(s) {
            return Ok(self.constant(e.to_owned()));
        }

        match e {
            AtomView::Pow(p) => {
                let (b, exp) = p.get_base_exp();

                if let AtomView::Var(v) = b {
                    if v.get_symbol() == s {
                        if let AtomView::Num(n) = exp {
                            if let CoefficientView::Natural(n, d) = n.get_coeff_view() {
                                Ok(self.monomial(self.field.one(), (n, d).into()))
                            } else {
                                unimplemented!("Cannot series expand with large exponents yet")
                            }
                        } else {
                            Err("Power of variable must be rational")
                        }
                    } else {
                        // s appears in the power
                        Err("Unexpected term in exp-log simplification")
                    }
                } else {
                    Err("Unexpected term in exp-log simplification")
                }
            }
            AtomView::Var(_) => Ok(self.monomial(self.field.one(), (1, 1).into())),
            AtomView::Mul(m) => {
                let mut shift_series = self.one();
                for a in m.iter() {
                    shift_series = &shift_series * &self.extract_exp_log(a, s)?;
                }

                Ok(shift_series)
            }
            _ => Err("Unexpected term in exp-log simplification"),
        }
    }

    pub fn exp(&self) -> Result<Self, &'static str> {
        if self.shift < 0 {
            return Err("Cannot compute the exponential of a series with poles");
        }

        if self.is_zero() {
            return Ok(self.one());
        }

        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        // construct the constant term, log(x) in the argument will be turned into x
        let e = FunctionBuilder::new(State::EXP).add_arg(&c).finish();

        // split the true constant part and the x-dependent part
        let shift_series = if let Variable::Symbol(s) = self.variable.as_ref() {
            self.extract_exp_log(e.as_view(), *s)?
        } else {
            unreachable!("Expansion variable is not a variable");
        };

        let p = self.clone().remove_constant();

        let mut r = self.one();
        let mut sp = p.clone();
        for i in 1..=self.order {
            let s = sp
                .clone()
                .div_coeff(&Atom::new_num(Integer::factorial(i as u32)));

            sp = sp * &p;

            r = r + s;
        }

        Ok(r * &shift_series)
    }

    pub fn log(&self) -> Result<Self, &'static str> {
        if self.is_zero() {
            return Err("Log yields minus infinity");
        }

        // construct the log argument, which may contain x
        let c = self.variable.to_atom().npow(self.get_exponent(0)) * &self.coefficients[0];
        // normalize the series to 1 + ..
        let p = self
            .clone()
            .div_coeff(&self.coefficients[0])
            .mul_exp_units(-self.shift)
            - self.one();

        let mut e = self.constant(FunctionBuilder::new(State::LOG).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let s = sp.clone().div_coeff(&Atom::new_num(i as i64));

            sp = sp * &p;

            if i % 2 == 0 {
                e = e - s;
            } else {
                e = e + s;
            }
        }

        Ok(e)
    }

    pub fn sin(&self) -> Result<Self, &'static str> {
        if self.shift < 0 {
            return Err("Cannot compute the sine of a series with poles");
        }

        if self.is_zero() {
            return Ok(self.zero());
        }

        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        if let Variable::Symbol(s) = self.variable.as_ref() {
            if c.contains_symbol(*s) {
                return Err(
                    "Cannot compute the sin of a series with a constant term that depends on x",
                );
            }
        }

        let p = self.clone().remove_constant();

        let mut e = self.constant(FunctionBuilder::new(State::SIN).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let mut b = if i % 2 == 1 {
                FunctionBuilder::new(State::COS).add_arg(&c).finish()
            } else {
                FunctionBuilder::new(State::SIN).add_arg(&c).finish()
            };

            if i % 4 >= 2 {
                b = b.neg();
            }

            let s = sp
                .clone()
                .mul_coeff(&b)
                .div_coeff(&Atom::new_num(Integer::factorial(i as u32)));

            sp = sp * &p;

            e = e + s;
        }

        Ok(e)
    }

    pub fn cos(&self) -> Result<Self, &'static str> {
        if self.shift < 0 {
            return Err("Cannot compute the sine of a series with poles");
        }

        if self.is_zero() {
            return Ok(self.zero());
        }

        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        if let Variable::Symbol(s) = self.variable.as_ref() {
            if c.contains_symbol(*s) {
                return Err(
                    "Cannot compute the cosine of a series with a constant term that depends on x",
                );
            }
        }

        let p = self.clone().remove_constant();

        let mut e = self.constant(FunctionBuilder::new(State::COS).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let mut b = if i % 2 == 1 {
                FunctionBuilder::new(State::SIN).add_arg(&c).finish()
            } else {
                -FunctionBuilder::new(State::COS).add_arg(&c).finish()
            };

            if i % 4 < 2 {
                b = b.neg();
            }

            let s = sp
                .clone()
                .mul_coeff(&b)
                .div_coeff(&Atom::new_num(Integer::factorial(i as u32)));

            sp = sp * &p;

            e = e + s;
        }

        Ok(e)
    }

    /// Take the series to the power of another series.
    pub fn pow(&self, pow: &Self) -> Result<Self, &'static str> {
        (self.log()? * pow).exp()
    }

    /// Take the series to the power of a rational number.
    pub fn rpow(&self, pow: Rational) -> Self {
        if pow.is_zero() {
            return self.one();
        }

        if pow.is_integer() && !pow.is_negative() {
            return self.npow(pow.numerator().to_i64().unwrap() as usize);
        }

        if pow.is_negative() && self.is_zero() {
            panic!("Cannot invert series with a zero constant term");
        }

        if self.is_zero() {
            return self.zero();
        }

        let c = self.coefficients[0].clone();

        // normalize the series to 1 + ...
        let cc = self.clone().div_coeff(&c).mul_exp_units(-self.shift) - self.one();

        let mut r = self.one();
        let mut x = self.one();

        let mut num = Rational::one();

        for i in 1..=self.order {
            num = &num * &(&pow - &(i as i64 - 1).into());

            x = x * &cc;

            let p = x
                .clone()
                .mul_coeff(&Atom::new_num(num.clone()))
                .div_coeff(&Atom::new_num(Integer::factorial(i as u32)));

            r = r + p;
        }

        let pow_ram = pow.denominator().to_i64().unwrap() as usize;
        r.change_ramification(pow_ram);

        let p = &(self.shift as i64, self.ramification as i64).into() * &pow;

        let shift = p.numerator().to_i64().unwrap() as isize
            * (r.ramification / p.denominator().to_i64().unwrap() as usize) as isize;

        r.mul_coeff(&c.npow(pow)).mul_exp_units(shift)
    }

    pub fn to_atom(&self) -> Atom {
        let mut a = Atom::new();
        self.to_atom_into(&mut a);
        a
    }

    pub fn to_atom_into(&self, out: &mut Atom) {
        out.to_num(0.into());

        if self.is_zero() {
            return;
        }

        let v = self.variable.to_atom();
        for (e, c) in self.coefficients.iter().enumerate() {
            let p = self.get_exponent(e);
            if !c.is_zero() {
                *out = &*out + &(v.npow(p) * c);
            }
        }
    }
}
