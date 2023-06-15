use std::fmt::{self, Display, Write};

use colored::Colorize;

use crate::{
    poly::{polynomial::MultivariatePolynomial, Exponent},
    representations::{number::BorrowedNumber, Add, Atom, AtomView, Fun, Mul, Num, Pow, Var},
    rings::{
        finite_field::FiniteFieldCore, rational_polynomial::RationalPolynomial, Ring, RingPrinter,
    },
    state::State,
};

#[derive(Debug, Copy, Clone)]
pub struct SymbolicaPrintOptions {
    pub terms_on_new_line: bool,
    pub color_top_level_sum: bool,
    pub print_finite_field: bool,
    pub explicit_rational_polynomial: bool,
}

impl Default for SymbolicaPrintOptions {
    fn default() -> Self {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: true,
            print_finite_field: true,
            explicit_rational_polynomial: false,
        }
    }
}

// TODO: make the printer generic over the print mode,
// as the modes will deviate quite a bit
#[derive(Debug, Copy, Clone)]
pub enum PrintMode {
    Symbolica(SymbolicaPrintOptions),
    Mathematica,
}

impl PrintMode {
    pub fn get_terms_on_new_line(&self) -> bool {
        match self {
            PrintMode::Symbolica(options) => options.terms_on_new_line,
            PrintMode::Mathematica => false,
        }
    }

    pub fn set_terms_on_new_line(self, terms_on_new_line: bool) -> PrintMode {
        match self {
            PrintMode::Symbolica(mut options) => {
                options.terms_on_new_line = terms_on_new_line;
                PrintMode::Symbolica(options)
            }
            PrintMode::Mathematica => PrintMode::Mathematica,
        }
    }

    pub fn get_color_top_level_sum(&self) -> bool {
        match self {
            PrintMode::Symbolica(options) => options.color_top_level_sum,
            PrintMode::Mathematica => false,
        }
    }

    pub fn set_color_top_level_sum(self, color_top_level_sum: bool) -> PrintMode {
        match self {
            PrintMode::Symbolica(mut options) => {
                options.color_top_level_sum = color_top_level_sum;
                PrintMode::Symbolica(options)
            }
            PrintMode::Mathematica => PrintMode::Mathematica,
        }
    }
}

impl Default for PrintMode {
    fn default() -> Self {
        Self::Symbolica(SymbolicaPrintOptions::default())
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PrintState {
    pub level: usize,
}

macro_rules! define_formatters {
    ($($a:ident),*) => {
        $(
        trait $a {
            fn fmt_debug(
                &self,
                f: &mut fmt::Formatter,
            ) -> fmt::Result;

            fn fmt_output(
                &self,
                f: &mut fmt::Formatter,
                print_mode: PrintMode,
                state: &State,
                print_state: PrintState,
            ) -> fmt::Result;
        })+
    };
}

define_formatters!(
    FormattedPrintVar,
    FormattedPrintNum,
    FormattedPrintFn,
    FormattedPrintPow,
    FormattedPrintMul,
    FormattedPrintAdd
);

pub struct AtomPrinter<'a, 'b, P: Atom> {
    pub atom: AtomView<'a, P>,
    pub state: &'b State,
    pub print_mode: PrintMode,
}

impl<'a, 'b, P: Atom> AtomPrinter<'a, 'b, P> {
    pub fn new(
        atom: AtomView<'a, P>,
        print_mode: PrintMode,
        state: &'b State,
    ) -> AtomPrinter<'a, 'b, P> {
        AtomPrinter {
            atom,
            state,
            print_mode,
        }
    }
}

impl<'a, 'b, P: Atom> fmt::Display for AtomPrinter<'a, 'b, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_state = PrintState { level: 0 };
        self.atom
            .fmt_output(f, self.print_mode, self.state, print_state)
    }
}

impl<'a, P: Atom> AtomView<'a, P> {
    fn fmt_debug(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_debug(fmt),
            AtomView::Var(v) => v.fmt_debug(fmt),
            AtomView::Fun(f) => f.fmt_debug(fmt),
            AtomView::Pow(p) => p.fmt_debug(fmt),
            AtomView::Mul(m) => m.fmt_debug(fmt),
            AtomView::Add(a) => a.fmt_debug(fmt),
        }
    }

    fn fmt_output(
        &self,
        fmt: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        print_state: PrintState,
    ) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, print_mode, state, print_state),
            AtomView::Var(v) => v.fmt_output(fmt, print_mode, state, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, print_mode, state, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, print_mode, state, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, print_mode, state, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, print_mode, state, print_state),
        }
    }
}

impl<'a, P: Atom> fmt::Debug for AtomView<'a, P> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_debug(fmt)
    }
}

impl<'a, A: Var<'a>> FormattedPrintVar for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        _print_mode: PrintMode,
        state: &State,
        _print_state: PrintState,
    ) -> fmt::Result {
        let name = state.get_name(self.get_name()).unwrap();
        if name.ends_with('_') {
            f.write_fmt(format_args!("{}", name.as_str().cyan().italic()))
        } else {
            f.write_str(name)
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("v_{}", self.get_name().to_u32()))
    }
}

impl<'a, A: Num<'a>> FormattedPrintNum for A {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let d = self.get_number_view();

        match d {
            BorrowedNumber::Natural(num, den) => {
                if den != 1 {
                    f.write_fmt(format_args!("{}/{}", num, den))
                } else {
                    f.write_fmt(format_args!("{}", num))
                }
            }
            BorrowedNumber::Large(r) => f.write_fmt(format_args!("{}", r)),
            BorrowedNumber::FiniteField(num, fi) => {
                f.write_fmt(format_args!("[m_{}%f_{}]", num.0, fi.0))
            }
            BorrowedNumber::RationalPolynomial(p) => f.write_fmt(format_args!("{}", p,)),
        }
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        _print_state: PrintState,
    ) -> fmt::Result {
        let d = self.get_number_view();

        match d {
            BorrowedNumber::Natural(num, den) => {
                if den != 1 {
                    f.write_fmt(format_args!("{}/{}", num, den))
                } else {
                    f.write_fmt(format_args!("{}", num))
                }
            }
            BorrowedNumber::Large(r) => f.write_fmt(format_args!("{}", r)),
            BorrowedNumber::FiniteField(num, fi) => {
                let ff = state.get_finite_field(fi);
                f.write_fmt(format_args!(
                    "[{}%{}]",
                    ff.from_element(num),
                    ff.get_prime()
                ))
            }
            BorrowedNumber::RationalPolynomial(p) => f.write_fmt(format_args!(
                "({})",
                RationalPolynomialPrinter {
                    poly: p,
                    state,
                    print_mode,
                }
            )),
        }
    }
}

impl<'a, A: Mul<'a>> FormattedPrintMul for A {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char('*')?;
            }
            first = false;

            if let AtomView::Add(_) = x {
                f.write_char('(')?;
                x.fmt_debug(f)?;
                f.write_char(')')?;
            } else {
                x.fmt_debug(f)?;
            }
        }
        Ok(())
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        let mut first = true;
        print_state.level += 1;
        for x in self.into_iter() {
            if !first {
                f.write_char('*')?;
            }
            first = false;

            if let AtomView::Add(_) = x {
                f.write_char('(')?;
                x.fmt_output(f, print_mode, state, print_state)?;
                f.write_char(')')?;
            } else {
                x.fmt_output(f, print_mode, state, print_state)?;
            }
        }
        Ok(())
    }
}

impl<'a, A: Fun<'a>> FormattedPrintFn for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())?;
        f.write_char('(')?;

        print_state.level += 1;
        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char(',')?;
            }
            first = false;

            x.fmt_output(f, print_mode, state, print_state)?;
        }

        f.write_char(')')
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("f_{}(", self.get_name().to_u32()))?;

        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char(',')?;
            }
            first = false;

            x.fmt_debug(f)?;
        }

        f.write_char(')')
    }
}

impl<'a, A: Pow<'a>> FormattedPrintPow for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        let b = self.get_base();

        print_state.level += 1;
        if let AtomView::Add(_) | AtomView::Mul(_) | AtomView::Pow(_) = b {
            f.write_char('(')?;
            b.fmt_output(f, print_mode, state, print_state)?;
            f.write_char(')')?;
        } else {
            b.fmt_output(f, print_mode, state, print_state)?;
        }

        f.write_char('^')?;

        let e = self.get_exp();
        if let AtomView::Add(_) | AtomView::Mul(_) = e {
            f.write_char('(')?;
            e.fmt_output(f, print_mode, state, print_state)?;
            f.write_char(')')
        } else {
            e.fmt_output(f, print_mode, state, print_state)
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let b = self.get_base();
        if let AtomView::Add(_) | AtomView::Mul(_) | AtomView::Pow(_) = b {
            f.write_char('(')?;
            b.fmt_debug(f)?;
            f.write_char(')')?;
        } else {
            b.fmt_debug(f)?;
        }

        f.write_char('^')?;

        let e = self.get_exp();
        if let AtomView::Add(_) | AtomView::Mul(_) = e {
            f.write_char('(')?;
            e.fmt_debug(f)?;
            f.write_char(')')
        } else {
            e.fmt_debug(f)
        }
    }
}

impl<'a, A: Add<'a>> FormattedPrintAdd for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        let mut first = true;
        print_state.level += 1;
        for x in self.into_iter() {
            if !first {
                if print_state.level == 1 && print_mode.get_terms_on_new_line() {
                    f.write_char('\n')?;
                    f.write_char('\t')?;
                }

                if print_state.level == 1 && print_mode.get_color_top_level_sum() {
                    f.write_fmt(format_args!("{}", "+".yellow()))?;
                } else {
                    f.write_char('+')?;
                }
            }
            first = false;

            x.fmt_output(f, print_mode, state, print_state)?;
        }
        Ok(())
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char('+')?;
            }
            first = false;

            x.fmt_debug(f)?;
        }
        Ok(())
    }
}

pub struct RationalPolynomialPrinter<'a, 'b, R: Ring, E: Exponent> {
    pub poly: &'a RationalPolynomial<R, E>,
    pub state: &'b State,
    pub print_mode: PrintMode,
}

impl<'a, 'b, R: Ring, E: Exponent> RationalPolynomialPrinter<'a, 'b, R, E> {
    pub fn new(
        poly: &'a RationalPolynomial<R, E>,
        state: &'b State,
        print_mode: PrintMode,
    ) -> RationalPolynomialPrinter<'a, 'b, R, E> {
        RationalPolynomialPrinter {
            poly,
            state,
            print_mode,
        }
    }
}

impl<'a, 'b, R: Ring, E: Exponent> Display for RationalPolynomialPrinter<'a, 'b, R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_explicit = match self.print_mode {
            PrintMode::Symbolica(s) => s.explicit_rational_polynomial,
            PrintMode::Mathematica => false,
        };

        if print_explicit {
            if self.poly.denominator.is_one() {
                if self.poly.numerator.is_zero() {
                    f.write_char('0')?;
                } else {
                    f.write_fmt(format_args!(
                        "[{}]",
                        PolynomialPrinter {
                            poly: &self.poly.numerator,
                            state: self.state,
                            print_mode: self.print_mode,
                        }
                    ))?;
                }
            } else {
                f.write_fmt(format_args!(
                    "[{},{}]",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        print_mode: self.print_mode,
                    },
                    PolynomialPrinter {
                        poly: &self.poly.denominator,
                        state: self.state,
                        print_mode: self.print_mode,
                    }
                ))?;
            }

            return Ok(());
        }

        if self.poly.denominator.is_one() {
            f.write_fmt(format_args!(
                "{}",
                PolynomialPrinter {
                    poly: &self.poly.numerator,
                    state: self.state,
                    print_mode: self.print_mode,
                }
            ))
        } else {
            if self.poly.numerator.nterms < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        print_mode: self.print_mode,
                    }
                ))?;
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        print_mode: self.print_mode,
                    }
                ))?;
            }

            if self.poly.denominator.nterms == 1 {
                let var_count = self
                    .poly
                    .denominator
                    .exponents
                    .iter()
                    .filter(|x| !x.is_zero())
                    .count();

                if var_count == 0
                    || self
                        .poly
                        .denominator
                        .field
                        .is_one(&self.poly.denominator.coefficients[0])
                        && var_count == 1
                {
                    return f.write_fmt(format_args!(
                        "/{}",
                        PolynomialPrinter {
                            poly: &self.poly.denominator,
                            state: self.state,
                            print_mode: self.print_mode,
                        }
                    ));
                }
            }

            f.write_fmt(format_args!(
                "/({})",
                PolynomialPrinter {
                    poly: &self.poly.denominator,
                    state: self.state,
                    print_mode: self.print_mode,
                }
            ))
        }
    }
}
pub struct PolynomialPrinter<'a, 'b, F: Ring + Display, E: Exponent> {
    pub poly: &'a MultivariatePolynomial<F, E>,
    pub state: &'b State,
    pub print_mode: PrintMode,
}

impl<'a, 'b, F: Ring + Display, E: Exponent> PolynomialPrinter<'a, 'b, F, E> {
    pub fn new(
        poly: &'a MultivariatePolynomial<F, E>,
        state: &'b State,
        print_mode: PrintMode,
    ) -> PolynomialPrinter<'a, 'b, F, E> {
        PolynomialPrinter {
            poly,
            state,
            print_mode,
        }
    }
}

impl<'a, 'b, F: Ring + Display, E: Exponent> Display for PolynomialPrinter<'a, 'b, F, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let var_map = match self.poly.var_map.as_ref() {
            Some(v) => v,
            None => {
                return write!(f, "{}", self.poly);
            }
        };

        let mut is_first_term = true;
        for monomial in self.poly {
            let mut is_first_factor = true;
            if self.poly.field.is_one(&monomial.coefficient) {
                if !is_first_term {
                    write!(f, "+")?;
                }
            } else if monomial
                .coefficient
                .eq(&self.poly.field.neg(&self.poly.field.one()))
            {
                write!(f, "-")?;
            } else {
                if is_first_term {
                    self.poly.field.fmt_display(monomial.coefficient, f)?;
                } else {
                    write!(
                        f,
                        "{:+}",
                        RingPrinter {
                            ring: &self.poly.field,
                            element: &monomial.coefficient
                        }
                    )?;
                }
                is_first_factor = false;
            }
            is_first_term = false;
            for (var_id, e) in var_map.iter().zip(monomial.exponents) {
                if e.is_zero() {
                    continue;
                }
                if is_first_factor {
                    is_first_factor = false;
                } else {
                    write!(f, "*")?;
                }

                f.write_str(self.state.get_name(*var_id).unwrap())?;

                if e.to_u32() != 1 {
                    write!(f, "^{}", e)?;
                }
            }
            if is_first_factor {
                write!(f, "1")?;
            }
        }
        if is_first_term {
            write!(f, "0")?;
        }

        if let PrintMode::Symbolica(s) = self.print_mode {
            if s.print_finite_field {
                Display::fmt(&self.poly.field, f)?;
            }
        }

        Ok(())
    }
}
