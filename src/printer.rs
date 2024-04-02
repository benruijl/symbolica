use std::fmt::{self, Display, Write};

use colored::Colorize;

use crate::{
    coefficient::CoefficientView,
    domains::{
        factorized_rational_polynomial::FactorizedRationalPolynomial,
        finite_field::FiniteFieldCore, rational_polynomial::RationalPolynomial, Ring, RingPrinter,
    },
    poly::{polynomial::MultivariatePolynomial, Exponent, MonomialOrder},
    representations::{
        default::FunView, AddView, AtomView, MulView, NumView, PowView, Symbol, VarView,
    },
    state::State,
    tensors::matrix::Matrix,
};

#[derive(Debug, Copy, Clone)]
pub struct PrintOptions {
    pub terms_on_new_line: bool,
    pub color_top_level_sum: bool,
    pub color_builtin_symbols: bool,
    pub print_finite_field: bool,
    pub symmetric_representation_for_finite_field: bool,
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
            color_builtin_symbols: false,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
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
            color_builtin_symbols: false,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: true,
        }
    }

    /// Print the output suitable for a file.
    pub fn file() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_finite_field: false,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: false,
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: true,
            color_builtin_symbols: true,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
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
                print_opts: &PrintOptions,
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

pub struct AtomPrinter<'a> {
    pub atom: AtomView<'a>,
    pub print_opts: PrintOptions,
}

impl<'a> AtomPrinter<'a> {
    /// Create a new atom printer with default printing options.
    pub fn new(atom: AtomView<'a>) -> AtomPrinter<'a> {
        AtomPrinter {
            atom,
            print_opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(atom: AtomView<'a>, print_opts: PrintOptions) -> AtomPrinter<'a> {
        AtomPrinter { atom, print_opts }
    }
}

impl<'a> fmt::Display for AtomPrinter<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let print_state = PrintState {
            level: 0,
            explicit_sign: false,
            superscript: false,
        };
        self.atom.fmt_output(f, &self.print_opts, print_state)
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(State::get_name(*self))
    }
}

impl<'a> AtomView<'a> {
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
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, opts, print_state),
            AtomView::Var(v) => v.fmt_output(fmt, opts, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, opts, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, opts, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, opts, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, opts, print_state),
        }
    }
}

impl<'a> fmt::Debug for AtomView<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_debug(fmt)
    }
}

impl<'a> FormattedPrintVar for VarView<'a> {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> fmt::Result {
        if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        let name = State::get_name(id);

        if opts.latex {
            match id {
                State::E => f.write_char('e'),
                State::PI => f.write_str("\\pi"),
                State::I => f.write_char('i'),
                _ => f.write_str(name),
            }
        } else if opts.color_builtin_symbols && name.ends_with('_') {
            f.write_fmt(format_args!("{}", name.cyan().italic()))
        } else if opts.color_builtin_symbols && State::is_builtin(id) {
            f.write_fmt(format_args!("{}", name.purple()))
        } else {
            f.write_str(name)
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintNum for NumView<'a> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
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

        let d = self.get_coeff_view();

        let is_negative = match d {
            CoefficientView::Natural(n, _) => n < 0,
            CoefficientView::Large(r) => r.is_negative(),
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
            CoefficientView::Natural(num, den) => {
                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(num.unsigned_abs().to_string(), opts, &print_state, f)?;
                    if den != 1 {
                        f.write_char('/')?;
                        format_num(den.to_string(), opts, &print_state, f)?;
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
            CoefficientView::Large(r) => {
                let rat = r.to_rat().abs();
                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(rat.numer().to_string(), opts, &print_state, f)?;
                    if !rat.is_integer() {
                        f.write_char('/')?;
                        format_num(rat.denom().to_string(), opts, &print_state, f)?;
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
            CoefficientView::FiniteField(num, fi) => {
                let ff = State::get_finite_field(fi);
                f.write_fmt(format_args!(
                    "[{}%{}]",
                    ff.from_element(&num),
                    ff.get_prime()
                ))
            }
            CoefficientView::RationalPolynomial(p) => f.write_fmt(format_args!(
                "[{}]",
                RationalPolynomialPrinter {
                    poly: p,
                    opts: *opts,
                    add_parentheses: false,
                }
            )),
        }
    }
}

impl<'a> FormattedPrintMul for MulView<'a> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> fmt::Result {
        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().last() {
            // write -1*x as -x
            if n.get_coeff_view() == CoefficientView::Natural(-1, 1) {
                if print_state.level == 1 && opts.color_top_level_sum {
                    f.write_fmt(format_args!("{}", "-".yellow()))?;
                } else {
                    f.write_char('-')?;
                }

                first = true;
            } else {
                n.fmt_output(f, opts, print_state)?;
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
                if opts.latex {
                    f.write_str("\\left(")?;
                } else {
                    f.write_char('(')?;
                }
                x.fmt_output(f, opts, print_state)?;
                if opts.latex {
                    f.write_str("\\right)")?;
                } else {
                    f.write_char(')')?;
                }
            } else {
                x.fmt_output(f, opts, print_state)?;
            }
        }
        Ok(())
    }
}

impl<'a> FormattedPrintFn for FunView<'a> {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> fmt::Result {
        if print_state.explicit_sign {
            if print_state.level == 1 && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        let name = State::get_name(id);

        if opts.latex {
            if name == "cos" || name == "sin" || name == "exp" || name == "log" {
                f.write_fmt(format_args!("\\{}\\!\\left(", name))?;
            } else {
                f.write_fmt(format_args!("{}\\!\\left(", name))?;
            }
        } else {
            if opts.color_builtin_symbols && name.ends_with('_') {
                f.write_fmt(format_args!("{}", name.cyan().italic()))?;
            } else if opts.color_builtin_symbols && State::is_builtin(id) {
                f.write_fmt(format_args!("{}", name.purple()))?;
            } else {
                f.write_str(name)?;
            }

            if opts.square_brackets_for_function {
                f.write_char('[')?;
            } else {
                f.write_char('(')?;
            }
        }

        print_state.level += 1;
        print_state.explicit_sign = false;
        let mut first = true;
        for x in self.iter() {
            if !first {
                f.write_char(',')?;
            }
            first = false;

            x.fmt_output(f, opts, print_state)?;
        }

        if opts.latex {
            f.write_str("\\right)")
        } else if opts.square_brackets_for_function {
            f.write_char(']')
        } else {
            f.write_char(')')
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintPow for PowView<'a> {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
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
                if n.get_coeff_view() == CoefficientView::Natural(-1, 1) {
                    // TODO: construct the numerator
                    f.write_str("\\frac{1}{")?;
                    b.fmt_output(f, opts, print_state)?;
                    return f.write_char('}');
                }
            }
        } else if opts.num_exp_as_superscript {
            if let AtomView::Num(n) = e {
                superscript_exponent = n.get_coeff_view().is_integer()
            }
        }

        let base_needs_parentheses =
            matches!(b, AtomView::Add(_) | AtomView::Mul(_) | AtomView::Pow(_))
                || if let AtomView::Num(n) = b {
                    !n.get_coeff_view().is_integer()
                } else {
                    false
                };

        if base_needs_parentheses {
            if opts.latex {
                f.write_str("\\left(")?;
            } else {
                f.write_char('(')?;
            }
            b.fmt_output(f, opts, print_state)?;
            if opts.latex {
                f.write_str("\\right)")?;
            } else {
                f.write_char(')')?;
            }
        } else {
            b.fmt_output(f, opts, print_state)?;
        }

        if !superscript_exponent {
            f.write_char('^')?;
        }

        if opts.latex {
            f.write_char('{')?;
            e.fmt_output(f, opts, print_state)?;
            f.write_char('}')
        } else {
            let exp_needs_parentheses = matches!(e, AtomView::Add(_) | AtomView::Mul(_))
                || if let AtomView::Num(n) = e {
                    !n.get_coeff_view().is_integer()
                } else {
                    false
                };

            if exp_needs_parentheses {
                f.write_char('(')?;
                e.fmt_output(f, opts, print_state)?;
                f.write_char(')')
            } else {
                print_state.superscript = superscript_exponent;
                e.fmt_output(f, opts, print_state)
            }
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintAdd for AddView<'a> {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        opts: &PrintOptions,
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

            x.fmt_output(f, opts, print_state)?;
        }
        Ok(())
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

pub struct FactorizedRationalPolynomialPrinter<'a, R: Ring, E: Exponent> {
    pub poly: &'a FactorizedRationalPolynomial<R, E>,
    pub opts: PrintOptions,
    pub add_parentheses: bool,
}

impl<'a, R: Ring, E: Exponent> FactorizedRationalPolynomialPrinter<'a, R, E> {
    pub fn new(
        poly: &'a FactorizedRationalPolynomial<R, E>,
    ) -> FactorizedRationalPolynomialPrinter<'a, R, E> {
        FactorizedRationalPolynomialPrinter {
            poly,
            opts: PrintOptions::default(),
            add_parentheses: false,
        }
    }

    pub fn new_with_options(
        poly: &'a FactorizedRationalPolynomial<R, E>,
        opts: PrintOptions,
    ) -> FactorizedRationalPolynomialPrinter<'a, R, E> {
        FactorizedRationalPolynomialPrinter {
            poly,
            opts,
            add_parentheses: false,
        }
    }
}

impl<'a, R: Ring, E: Exponent> Display for FactorizedRationalPolynomialPrinter<'a, R, E> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.opts.explicit_rational_polynomial {
            if !R::is_zero(&self.poly.numer_coeff)
                && !self.poly.numerator.field.is_one(&self.poly.numer_coeff)
            {
                f.write_fmt(format_args!(
                    "[{}]*",
                    RingPrinter {
                        ring: &self.poly.numerator.field,
                        element: &self.poly.numer_coeff,
                        opts: self.opts,
                        in_product: false
                    }
                ))?;
            }

            if self.poly.denominators.is_empty()
                && self.poly.numerator.field.is_one(&self.poly.denom_coeff)
            {
                if self.poly.numerator.is_zero() {
                    f.write_char('0')?;
                } else {
                    f.write_fmt(format_args!(
                        "[{}]",
                        PolynomialPrinter {
                            poly: &self.poly.numerator,
                            opts: self.opts,
                        }
                    ))?;
                }
            } else {
                f.write_fmt(format_args!(
                    "[{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    },
                ))?;

                if !self.poly.numerator.field.is_one(&self.poly.denom_coeff) {
                    f.write_fmt(format_args!(
                        ",{},1",
                        RingPrinter {
                            ring: &self.poly.numerator.field,
                            element: &self.poly.denom_coeff,
                            opts: self.opts,
                            in_product: false
                        },
                    ))?;
                }

                for (d, p) in &self.poly.denominators {
                    f.write_fmt(format_args!(
                        ",{}",
                        PolynomialPrinter {
                            poly: d,

                            opts: self.opts,
                        }
                    ))?;
                    f.write_fmt(format_args!(",{}", p))?;
                }

                f.write_char(']')?;
            }

            return Ok(());
        }

        if R::is_zero(&self.poly.numer_coeff) {
            return f.write_char('0');
        }

        if self.poly.denominators.is_empty()
            && self.poly.numerator.field.is_one(&self.poly.denom_coeff)
        {
            if !self.poly.numerator.field.is_one(&self.poly.numer_coeff) {
                f.write_fmt(format_args!(
                    "{}",
                    RingPrinter {
                        ring: &self.poly.numerator.field,
                        element: &self.poly.numer_coeff,
                        opts: self.opts,
                        in_product: false
                    }
                ))?;
            }

            if (self.poly.numerator.field.is_one(&self.poly.numer_coeff) && !self.add_parentheses)
                || self.poly.numerator.nterms() < 2
            {
                if !self.poly.numerator.field.is_one(&self.poly.numer_coeff) {
                    if self.poly.numerator.is_one() {
                        return Ok(());
                    }

                    f.write_char('*')?;
                }

                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    }
                ))
            } else {
                if !self.poly.numerator.field.is_one(&self.poly.numer_coeff) {
                    if self.poly.numerator.is_one() {
                        return Ok(());
                    }

                    f.write_char('*')?;
                }

                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    }
                ))
            }
        } else {
            if self.opts.latex {
                if !self.poly.numerator.field.is_one(&self.poly.numer_coeff) {
                    f.write_fmt(format_args!(
                        "{} ",
                        RingPrinter {
                            ring: &self.poly.numerator.field,
                            element: &self.poly.numer_coeff,
                            opts: self.opts,
                            in_product: false
                        }
                    ))?;
                }

                f.write_fmt(format_args!(
                    "\\frac{{{}}}{{",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    },
                ))?;

                if !self.poly.numerator.field.is_one(&self.poly.denom_coeff) {
                    f.write_fmt(format_args!(
                        "{}",
                        RingPrinter {
                            ring: &self.poly.numerator.field,
                            element: &self.poly.denom_coeff,
                            opts: self.opts,
                            in_product: false
                        }
                    ))?;
                }

                for (d, p) in &self.poly.denominators {
                    if *p == 1 {
                        f.write_fmt(format_args!(
                            "({})",
                            PolynomialPrinter {
                                poly: d,

                                opts: self.opts,
                            }
                        ))?;
                    } else {
                        f.write_fmt(format_args!(
                            "({})^{}",
                            PolynomialPrinter {
                                poly: d,

                                opts: self.opts,
                            },
                            p
                        ))?;
                    }
                }

                return f.write_str("}}");
            }

            if !self.poly.numerator.field.is_one(&self.poly.numer_coeff) {
                f.write_fmt(format_args!(
                    "{}*",
                    RingPrinter {
                        ring: &self.poly.numerator.field,
                        element: &self.poly.numer_coeff,
                        opts: self.opts,
                        in_product: false
                    }
                ))?;
            }

            if self.poly.numerator.nterms() < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    }
                ))?;
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,

                        opts: self.opts,
                    }
                ))?;
            }

            f.write_char('/')?;

            if self.poly.denominators.is_empty() {
                return f.write_fmt(format_args!(
                    "{}",
                    RingPrinter {
                        ring: &self.poly.numerator.field,
                        element: &self.poly.denom_coeff,
                        opts: self.opts,
                        in_product: true
                    }
                ));
            }

            if self.poly.numerator.field.is_one(&self.poly.denom_coeff)
                && self.poly.denominators.len() == 1
                && self.poly.denominators[0].0.nterms() == 1
                && self.poly.denominators[0].1 == 1
            {
                let (d, _) = &self.poly.denominators[0];
                let var_count = d.exponents.iter().filter(|x| !x.is_zero()).count();

                if var_count == 0 || d.field.is_one(&d.coefficients[0]) && var_count == 1 {
                    return f.write_fmt(format_args!(
                        "{}",
                        PolynomialPrinter {
                            poly: d,

                            opts: self.opts,
                        }
                    ));
                }
            }

            f.write_char('(')?; // TODO: add special cases for 1 argument

            if !self.poly.numerator.field.is_one(&self.poly.denom_coeff) {
                f.write_fmt(format_args!(
                    "{}",
                    RingPrinter {
                        ring: &self.poly.numerator.field,
                        element: &self.poly.denom_coeff,
                        opts: self.opts,
                        in_product: true
                    }
                ))?;
            }

            for (d, p) in &self.poly.denominators {
                if *p == 1 {
                    f.write_fmt(format_args!(
                        "({})",
                        PolynomialPrinter {
                            poly: d,

                            opts: self.opts,
                        }
                    ))?;
                } else {
                    f.write_fmt(format_args!(
                        "({})^{}",
                        PolynomialPrinter {
                            poly: d,

                            opts: self.opts,
                        },
                        p
                    ))?;
                }
            }

            f.write_char(')')
        }
    }
}

pub struct RationalPolynomialPrinter<'a, R: Ring, E: Exponent> {
    pub poly: &'a RationalPolynomial<R, E>,
    pub opts: PrintOptions,
    pub add_parentheses: bool,
}

impl<'a, R: Ring, E: Exponent> RationalPolynomialPrinter<'a, R, E> {
    pub fn new(poly: &'a RationalPolynomial<R, E>) -> RationalPolynomialPrinter<'a, R, E> {
        RationalPolynomialPrinter {
            poly,
            opts: PrintOptions::default(),
            add_parentheses: false,
        }
    }

    pub fn new_with_options(
        poly: &'a RationalPolynomial<R, E>,
        opts: PrintOptions,
    ) -> RationalPolynomialPrinter<'a, R, E> {
        RationalPolynomialPrinter {
            poly,
            opts,
            add_parentheses: false,
        }
    }
}

impl<'a, R: Ring, E: Exponent> Display for RationalPolynomialPrinter<'a, R, E> {
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
                            opts: self.opts,
                        }
                    ))?;
                }
            } else {
                f.write_fmt(format_args!(
                    "[{},{}]",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        opts: self.opts,
                    },
                    PolynomialPrinter {
                        poly: &self.poly.denominator,
                        opts: self.opts,
                    }
                ))?;
            }

            return Ok(());
        }

        if self.poly.denominator.is_one() {
            if !self.add_parentheses || self.poly.numerator.nterms() < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        opts: self.opts,
                    }
                ))
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
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
                        opts: self.opts,
                    },
                    PolynomialPrinter {
                        poly: &self.poly.denominator,
                        opts: self.opts,
                    }
                ));
            }

            if self.poly.numerator.nterms() < 2 {
                f.write_fmt(format_args!(
                    "{}",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        opts: self.opts,
                    }
                ))?;
            } else {
                f.write_fmt(format_args!(
                    "({})",
                    PolynomialPrinter {
                        poly: &self.poly.numerator,
                        opts: self.opts,
                    }
                ))?;
            }

            if self.poly.denominator.nterms() == 1 {
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
                            opts: self.opts,
                        }
                    ));
                }
            }

            f.write_fmt(format_args!(
                "/({})",
                PolynomialPrinter {
                    poly: &self.poly.denominator,

                    opts: self.opts,
                }
            ))
        }
    }
}
pub struct PolynomialPrinter<'a, F: Ring + Display, E: Exponent, O: MonomialOrder> {
    pub poly: &'a MultivariatePolynomial<F, E, O>,
    pub opts: PrintOptions,
}

impl<'a, R: Ring + Display, E: Exponent, O: MonomialOrder> PolynomialPrinter<'a, R, E, O> {
    pub fn new(poly: &'a MultivariatePolynomial<R, E, O>) -> PolynomialPrinter<'a, R, E, O> {
        PolynomialPrinter {
            poly,
            opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(
        poly: &'a MultivariatePolynomial<R, E, O>,
        opts: PrintOptions,
    ) -> PolynomialPrinter<'a, R, E, O> {
        PolynomialPrinter { poly, opts }
    }
}

impl<'a, F: Ring + Display, E: Exponent, O: MonomialOrder> Display
    for PolynomialPrinter<'a, F, E, O>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.sign_plus() {
            f.write_char('+')?;
        }

        let var_map: Vec<String> = self
            .poly
            .variables
            .as_ref()
            .iter()
            .map(|v| v.to_string())
            .collect();

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
                        .fmt_display(monomial.coefficient, &self.opts, true, f)?;
                } else {
                    write!(
                        f,
                        "{:+}",
                        RingPrinter {
                            ring: &self.poly.field,
                            element: monomial.coefficient,
                            opts: self.opts,
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

                f.write_str(var_id)?;

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

pub struct MatrixPrinter<'a, F: Ring + Display> {
    pub matrix: &'a Matrix<F>,
    pub opts: PrintOptions,
}

impl<'a, F: Ring + Display> MatrixPrinter<'a, F> {
    pub fn new(matrix: &'a Matrix<F>) -> MatrixPrinter<'a, F> {
        MatrixPrinter {
            matrix,

            opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(matrix: &'a Matrix<F>, opts: PrintOptions) -> MatrixPrinter<'a, F> {
        MatrixPrinter { matrix, opts }
    }
}

impl<'a, F: Ring + Display> Display for MatrixPrinter<'a, F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.opts.latex {
            f.write_str("\\begin{pmatrix}")?;

            for (ri, r) in self.matrix.row_iter().enumerate() {
                for (ci, c) in r.iter().enumerate() {
                    f.write_fmt(format_args!(
                        "{}",
                        RingPrinter {
                            ring: &self.matrix.field,
                            element: c,
                            opts: self.opts,
                            in_product: false,
                        },
                    ))?;

                    if ci + 1 < self.matrix.ncols as usize {
                        f.write_str(" & ")?;
                    }
                }
                if ri + 1 < self.matrix.nrows as usize {
                    f.write_str(r" \\ ")?;
                }
            }

            f.write_str("\\end{pmatrix}")
        } else {
            f.write_char('{')?;
            for (ri, r) in self.matrix.row_iter().enumerate() {
                f.write_char('{')?;
                for (ci, c) in r.iter().enumerate() {
                    f.write_fmt(format_args!(
                        "{}",
                        RingPrinter {
                            ring: &self.matrix.field,
                            element: c,
                            opts: self.opts,
                            in_product: false,
                        },
                    ))?;

                    if ci + 1 < self.matrix.ncols as usize {
                        f.write_char(',')?;
                    }
                }
                f.write_char('}')?;
                if ri + 1 < self.matrix.nrows as usize {
                    f.write_char(',')?;
                }
            }
            f.write_char('}')
        }
    }
}
