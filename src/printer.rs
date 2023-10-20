use std::fmt::{self, Display, Write};

use colored::Colorize;

use crate::{
    poly::{polynomial::MultivariatePolynomial, Exponent, MonomialOrder},
    representations::{number::BorrowedNumber, Add, AtomSet, AtomView, Fun, Mul, Num, Pow, Var},
    rings::{
        finite_field::FiniteFieldCore, rational_polynomial::RationalPolynomial, Ring, RingPrinter,
    },
    state::State,
};

#[derive(Debug, Copy, Clone)]
pub struct PrintOptions {
    pub terms_on_new_line: bool,
    pub color_top_level_sum: bool,
    pub color_builtin_functions: bool,
    pub print_finite_field: bool,
    pub explicit_rational_polynomial: bool,
    pub number_thousands_separator: Option<char>,
    pub multiplication_operator: char,
    pub square_brackets_for_function: bool,
    pub num_exp_as_superscript: bool,
    pub latex: bool,
}

impl PrintOptions {
    /// Print the output in a Mathematica-readable format.
    pub fn mathematica() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_functions: false,
            print_finite_field: true,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            square_brackets_for_function: true,
            num_exp_as_superscript: false,
            latex: false,
        }
    }

    /// Print the output in a Latex input format.
    pub fn latex() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_functions: false,
            print_finite_field: true,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: true,
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: true,
            color_builtin_functions: true,
            print_finite_field: true,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: false,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct PrintState {
    pub level: usize,
    pub explicit_sign: bool,
    pub superscript: bool,
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
                print_opts: PrintOptions,
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

pub struct AtomPrinter<'a, 'b, P: AtomSet> {
    pub atom: AtomView<'a, P>,
    pub state: &'b State,
    pub print_opts: PrintOptions,
}

impl<'a, 'b, P: AtomSet> AtomPrinter<'a, 'b, P> {
    /// Create a new atom printer with default printing options.
    pub fn new(atom: AtomView<'a, P>, state: &'b State) -> AtomPrinter<'a, 'b, P> {
        AtomPrinter {
            atom,
            state,
            print_opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(
        atom: AtomView<'a, P>,
        print_opts: PrintOptions,
        state: &'b State,
    ) -> AtomPrinter<'a, 'b, P> {
        AtomPrinter {
            atom,
            state,
            print_opts,
        }
    }
}

impl<'a, 'b, P: AtomSet> fmt::Display for AtomPrinter<'a, 'b, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_state = PrintState {
            level: 0,
            explicit_sign: false,
            superscript: false,
        };
        self.atom
            .fmt_output(f, self.print_opts, self.state, print_state)
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
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
        opts: PrintOptions,
        state: &State,
        print_state: PrintState,
    ) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, opts, state, print_state),
            AtomView::Var(v) => v.fmt_output(fmt, opts, state, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, opts, state, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, opts, state, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, opts, state, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, opts, state, print_state),
        }
    }
}

impl<'a, P: AtomSet> fmt::Debug for AtomView<'a, P> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_debug(fmt)
    }
}

impl<'a, A: Var<'a>> FormattedPrintVar for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: PrintOptions,
        state: &State,
        print_state: PrintState,
    ) -> fmt::Result {
        if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

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
            BorrowedNumber::Large(r) => f.write_fmt(format_args!("{}", r.to_rat())),
            BorrowedNumber::FiniteField(num, fi) => {
                f.write_fmt(format_args!("[m_{}%f_{}]", num.0, fi.0))
            }
            BorrowedNumber::RationalPolynomial(p) => f.write_fmt(format_args!("{}", p,)),
        }
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: PrintOptions,
        state: &State,
        print_state: PrintState,
    ) -> fmt::Result {
        /// Input must be digits only.
        fn format_num(
            mut s: String,
            opts: &PrintOptions,
            print_state: &PrintState,
            f: &mut fmt::Formatter,
        ) -> fmt::Result {
            if print_state.superscript {
                let map = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
                s = s
                    .as_bytes()
                    .iter()
                    .map(|x| map[(x - b'0') as usize])
                    .collect();

                return f.write_str(&s);
            }

            if let Some(c) = opts.number_thousands_separator {
                let mut first = true;
                for triplet in s.as_bytes().chunks(3) {
                    if !first {
                        f.write_char(c)?;
                    }
                    f.write_str(std::str::from_utf8(triplet).unwrap())?;
                    first = false;
                }

                Ok(())
            } else {
                f.write_str(&s)
            }
        }

        let d = self.get_number_view();

        let is_negative = match d {
            BorrowedNumber::Natural(n, _) => n < 0,
            BorrowedNumber::Large(r) => r.is_negative(),
            _ => false,
        };

        if is_negative {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "-".yellow()))?;
            } else if print_state.superscript {
                f.write_char('⁻')?;
            } else {
                f.write_char('-')?;
            }
        } else if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        match d {
            BorrowedNumber::Natural(num, den) => {
                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(num.unsigned_abs().to_string(), &opts, &print_state, f)?;
                    if den != 1 {
                        f.write_char('/')?;
                        format_num(den.to_string(), &opts, &print_state, f)?;
                    }
                    Ok(())
                } else if den != 1 {
                    if opts.latex {
                        f.write_fmt(format_args!("\\frac{{{}}}{{{}}}", num.unsigned_abs(), den))
                    } else {
                        f.write_fmt(format_args!("{}/{}", num.unsigned_abs(), den))
                    }
                } else {
                    f.write_fmt(format_args!("{}", num.unsigned_abs()))
                }
            }
            BorrowedNumber::Large(r) => {
                let rat = r.to_rat().abs();
                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(rat.numer().to_string(), &opts, &print_state, f)?;
                    if !rat.is_integer() {
                        f.write_char('/')?;
                        format_num(rat.denom().to_string(), &opts, &print_state, f)?;
                    }
                    Ok(())
                } else if !rat.is_integer() {
                    if opts.latex {
                        f.write_fmt(format_args!("\\frac{{{}}}{{{}}}", rat.numer(), rat.denom(),))
                    } else {
                        f.write_fmt(format_args!("{}/{}", rat.numer(), rat.denom()))
                    }
                } else {
                    f.write_fmt(format_args!("{}", rat.numer()))
                }
            }
            BorrowedNumber::FiniteField(num, fi) => {
                let ff = state.get_finite_field(fi);
                f.write_fmt(format_args!(
                    "[{}%{}]",
                    ff.from_element(num),
                    ff.get_prime()
                ))
            }
            BorrowedNumber::RationalPolynomial(p) => f.write_fmt(format_args!(
                "{}",
                RationalPolynomialPrinter {
                    poly: p,
                    state,
                    opts,
                    add_parentheses: true,
                }
            )),
        }
    }
}

impl<'a, A: Mul<'a>> FormattedPrintMul for A {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;
        for x in self.iter() {
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
        opts: PrintOptions,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().last() {
            // write -1*x as -x
            if n.get_number_view() == BorrowedNumber::Natural(-1, 1) {
                if print_state.level == 1 && opts.color_top_level_sum {
                    f.write_fmt(format_args!("{}", "-".yellow()))?;
                } else {
                    f.write_char('-')?;
                }

                first = true;
            } else {
                n.fmt_output(f, opts, state, print_state)?;
                first = false;
            }

            skip_num = true;
        } else if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        print_state.level += 1;
        print_state.explicit_sign = false;
        for x in self.iter().take(if skip_num {
            self.get_nargs() - 1
        } else {
            self.get_nargs()
        }) {
            if !first {
                if opts.latex {
                    f.write_char(' ')?;
                } else {
                    f.write_char(opts.multiplication_operator)?;
                }
            }
            first = false;

            if let AtomView::Add(_) = x {
                f.write_char('(')?;
                x.fmt_output(f, opts, state, print_state)?;
                f.write_char(')')?;
            } else {
                x.fmt_output(f, opts, state, print_state)?;
            }
        }
        Ok(())
    }
}

impl<'a, A: Fun<'a>> FormattedPrintFn for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: PrintOptions,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_name();
        let name = state.get_name(id).unwrap();
        if name.ends_with('_') {
            f.write_fmt(format_args!("{}", name.as_str().cyan().italic()))?;
        } else {
            // check if the function name is built in
            if opts.color_builtin_functions && State::is_builtin(id) {
                f.write_fmt(format_args!("{}", name.as_str().purple()))?;
            } else {
                f.write_str(name)?;
            }
        }

        if opts.square_brackets_for_function {
            f.write_char('[')?;
        } else {
            f.write_char('(')?;
        }

        print_state.level += 1;
        print_state.explicit_sign = false;
        let mut first = true;
        for x in self.iter() {
            if !first {
                f.write_char(',')?;
            }
            first = false;

            x.fmt_output(f, opts, state, print_state)?;
        }

        if opts.square_brackets_for_function {
            f.write_char(']')
        } else {
            f.write_char(')')
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("f_{}(", self.get_name().to_u32()))?;

        let mut first = true;
        for x in self.iter() {
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
        opts: PrintOptions,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let b = self.get_base();
        let e = self.get_exp();

        print_state.level += 1;
        print_state.explicit_sign = false;

        let mut superscript_exponent = false;
        if opts.latex {
            if let AtomView::Num(n) = e {
                if n.get_number_view() == BorrowedNumber::Natural(-1, 1) {
                    // TODO: construct the numerator
                    f.write_str("\\frac{1}{")?;
                    b.fmt_output(f, opts, state, print_state)?;
                    return f.write_char('}');
                }
            }
        } else if opts.num_exp_as_superscript {
            if let AtomView::Num(n) = e {
                superscript_exponent = n.get_number_view().is_integer()
            }
        }

        let base_needs_parentheses =
            matches!(b, AtomView::Add(_) | AtomView::Mul(_) | AtomView::Pow(_))
                || if let AtomView::Num(n) = b {
                    !n.get_number_view().is_integer()
                } else {
                    false
                };

        if base_needs_parentheses {
            f.write_char('(')?;
            b.fmt_output(f, opts, state, print_state)?;
            f.write_char(')')?;
        } else {
            b.fmt_output(f, opts, state, print_state)?;
        }

        if !superscript_exponent {
            f.write_char('^')?;
        }

        if opts.latex {
            f.write_char('{')?;
            e.fmt_output(f, opts, state, print_state)?;
            f.write_char('}')
        } else {
            let exp_needs_parentheses = matches!(e, AtomView::Add(_) | AtomView::Mul(_))
                || if let AtomView::Num(n) = e {
                    !n.get_number_view().is_integer()
                } else {
                    false
                };

            if exp_needs_parentheses {
                f.write_char('(')?;
                e.fmt_output(f, opts, state, print_state)?;
                f.write_char(')')
            } else {
                print_state.superscript = superscript_exponent;
                e.fmt_output(f, opts, state, print_state)
            }
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
        opts: PrintOptions,
        state: &State,
        mut print_state: PrintState,
    ) -> fmt::Result {
        let mut first = true;
        print_state.level += 1;

        for x in self.iter() {
            if !first && print_state.level == 1 && opts.terms_on_new_line {
                f.write_char('\n')?;
                f.write_char('\t')?;
            }
            print_state.explicit_sign = !first;
            first = false;

            x.fmt_output(f, opts, state, print_state)?;
        }
        Ok(())
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut first = true;
        for x in self.iter() {
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
    pub opts: PrintOptions,
    pub add_parentheses: bool,
}

impl<'a, 'b, R: Ring, E: Exponent> RationalPolynomialPrinter<'a, 'b, R, E> {
    pub fn new(
        poly: &'a RationalPolynomial<R, E>,
        state: &'b State,
    ) -> RationalPolynomialPrinter<'a, 'b, R, E> {
        RationalPolynomialPrinter {
            poly,
            state,
            opts: PrintOptions::default(),
            add_parentheses: false,
        }
    }

    pub fn new_with_options(
        poly: &'a RationalPolynomial<R, E>,
        state: &'b State,
        opts: PrintOptions,
    ) -> RationalPolynomialPrinter<'a, 'b, R, E> {
        RationalPolynomialPrinter {
            poly,
            state,
            opts,
            add_parentheses: false,
        }
    }
}

impl<'a, 'b, R: Ring, E: Exponent> Display for RationalPolynomialPrinter<'a, 'b, R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.opts.explicit_rational_polynomial {
            if self.poly.denominator.is_one() {
                if self.poly.numerator.is_zero() {
                    f.write_char('0')?;
                } else {
                    f.write_fmt(format_args!(
                        "[{}]",
                        PolynomialPrinter {
                            poly: &self.poly.numerator,
                            state: self.state,
                            opts: self.opts,
                        }
                    ))?;
                }
            } else {
                f.write_fmt(format_args!(
                    "[{},{}]",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
                    },
                    PolynomialPrinter {
                        poly: &self.poly.denominator,
                        state: self.state,
                        opts: self.opts,
                    }
                ))?;
            }

            return Ok(());
        }

        if self.poly.denominator.is_one() {
            if !self.add_parentheses || self.poly.numerator.nterms < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
                    }
                ))
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
                    }
                ))
            }
        } else {
            if self.opts.latex {
                return f.write_fmt(format_args!(
                    "\\frac{{{}}}{{{}}}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
                    },
                    PolynomialPrinter {
                        poly: &self.poly.denominator,
                        state: self.state,
                        opts: self.opts,
                    }
                ));
            }

            if self.poly.numerator.nterms < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
                    }
                ))?;
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        state: self.state,
                        opts: self.opts,
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
                            opts: self.opts,
                        }
                    ));
                }
            }

            f.write_fmt(format_args!(
                "/({})",
                PolynomialPrinter {
                    poly: &self.poly.denominator,
                    state: self.state,
                    opts: self.opts,
                }
            ))
        }
    }
}
pub struct PolynomialPrinter<'a, 'b, F: Ring + Display, E: Exponent, O: MonomialOrder> {
    pub poly: &'a MultivariatePolynomial<F, E, O>,
    pub state: &'b State,
    pub opts: PrintOptions,
}

impl<'a, 'b, R: Ring + Display, E: Exponent, O: MonomialOrder> PolynomialPrinter<'a, 'b, R, E, O> {
    pub fn new(
        poly: &'a MultivariatePolynomial<R, E, O>,
        state: &'b State,
    ) -> PolynomialPrinter<'a, 'b, R, E, O> {
        PolynomialPrinter {
            poly,
            state,
            opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(
        poly: &'a MultivariatePolynomial<R, E, O>,
        state: &'b State,
        opts: PrintOptions,
    ) -> PolynomialPrinter<'a, 'b, R, E, O> {
        PolynomialPrinter { poly, state, opts }
    }
}

impl<'a, 'b, F: Ring + Display, E: Exponent, O: MonomialOrder> Display
    for PolynomialPrinter<'a, 'b, F, E, O>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        let var_map = match self.poly.var_map.as_ref() {
            Some(v) => v,
            None => {
                return write!(f, "{}", self.poly);
            }
        };

        let mut is_first_term = true;
        for monomial in self.poly {
            let mut is_first_factor = true;
            if self.poly.field.is_one(monomial.coefficient) {
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
                    self.poly
                        .field
                        .fmt_display(monomial.coefficient, Some(self.state), true, f)?;
                } else {
                    write!(
                        f,
                        "{:+}",
                        RingPrinter {
                            ring: &self.poly.field,
                            element: monomial.coefficient,
                            state: Some(self.state),
                            in_product: true
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
                } else if !self.opts.latex {
                    write!(f, "*")?;
                }

                f.write_str(&var_id.to_string(self.state))?;

                if e.to_u32() != 1 {
                    if self.opts.latex {
                        write!(f, "^{{{}}}", e)?;
                    } else {
                        write!(f, "^{}", e)?;
                    }
                }
            }
            if is_first_factor {
                write!(f, "1")?;
            }
        }
        if is_first_term {
            write!(f, "0")?;
        }

        if self.opts.print_finite_field {
            Display::fmt(&self.poly.field, f)?;
        }

        Ok(())
    }
}
