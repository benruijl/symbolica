//! Methods for printing atoms and polynomials.

use std::fmt::{self, Error, Write};

use colored::Colorize;

use crate::{
    atom::{representation::FunView, AddView, AtomView, MulView, NumView, PowView, VarView},
    coefficient::CoefficientView,
    domains::{finite_field::FiniteFieldCore, SelfRing},
    state::State,
};

/// A function that takes an atom and prints it in a custom way.
/// If the function returns `None`, the default printing is used.
pub type PrintFunction = Box<dyn Fn(AtomView, &PrintOptions) -> Option<String> + Send + Sync>;

/// Various options for printing expressions.
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
    pub double_star_for_exponentiation: bool,
    pub square_brackets_for_function: bool,
    pub num_exp_as_superscript: bool,
    pub latex: bool,
    pub precision: Option<usize>,
    pub pretty_matrix: bool,
    pub hide_namespace: Option<&'static str>,
    pub hide_all_namespaces: bool,
    pub color_namespace: bool,
    pub max_terms: Option<usize>,
}

impl PrintOptions {
    pub const fn new() -> Self {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: true,
            color_builtin_symbols: true,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: false,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            color_namespace: true,
            max_terms: None,
        }
    }

    /// Print the output in a Mathematica-readable format.
    pub const fn mathematica() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            square_brackets_for_function: true,
            num_exp_as_superscript: false,
            latex: false,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            color_namespace: false,
            max_terms: None,
        }
    }

    /// Print the output in a Latex input format.
    pub const fn latex() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_finite_field: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: true,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            color_namespace: false,
            max_terms: None,
        }
    }

    /// Print the output suitable for a file.
    pub const fn file() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_finite_field: false,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            latex: false,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: false,
            color_namespace: false,
            max_terms: None,
        }
    }

    /// Print the output suitable for a file without namespaces.
    pub const fn file_no_namespace() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::file()
        }
    }

    /// Print the output with namespaces suppressed.
    pub const fn short() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::new()
        }
    }

    /// Print the output in a sympy input format.
    pub const fn sympy() -> PrintOptions {
        Self {
            double_star_for_exponentiation: true,
            ..Self::file()
        }
    }

    pub fn from_fmt(f: &std::fmt::Formatter) -> PrintOptions {
        PrintOptions {
            precision: f.precision(),
            hide_all_namespaces: !f.alternate(),
            terms_on_new_line: f.align() == Some(std::fmt::Alignment::Right),
            ..Default::default()
        }
    }

    pub fn update_with_fmt(mut self, f: &std::fmt::Formatter) -> Self {
        self.precision = f.precision();

        if f.alternate() {
            self.hide_all_namespaces = false;
        }

        if let Some(a) = f.align() {
            self.terms_on_new_line = a == std::fmt::Alignment::Right;
        }
        self
    }

    pub const fn hide_namespace(mut self, namespace: &'static str) -> Self {
        self.hide_namespace = Some(namespace);
        self
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// The current state useful for printing. These
/// settings will control, for example, if parentheses are needed
/// (e.g., a sum in a product),
/// and if 1 should be suppressed (e.g. in a product).
#[derive(Debug, Copy, Clone)]
pub struct PrintState {
    pub in_sum: bool,
    pub in_product: bool,
    pub suppress_one: bool,
    pub in_exp: bool,
    pub top_level_add_child: bool,
    pub superscript: bool,
    pub level: u16,
}

impl Default for PrintState {
    fn default() -> Self {
        Self::new()
    }
}

impl PrintState {
    pub const fn new() -> PrintState {
        Self {
            in_sum: false,
            in_product: false,
            in_exp: false,
            suppress_one: false,
            top_level_add_child: true,
            superscript: false,
            level: 0,
        }
    }

    pub fn from_fmt(f: &std::fmt::Formatter) -> PrintState {
        PrintState {
            in_sum: f.sign_plus(),
            ..Default::default()
        }
    }

    pub fn update_with_fmt(mut self, f: &std::fmt::Formatter) -> Self {
        self.in_sum = f.sign_plus();
        self
    }

    pub fn step(self, in_sum: bool, in_product: bool, in_exp: bool) -> Self {
        Self {
            in_sum,
            in_product,
            in_exp,
            level: self.level + 1,
            ..self
        }
    }
}

macro_rules! define_formatters {
    ($($a:ident),*) => {
        $(
        trait $a {
            fn fmt_debug(
                &self,
                f: &mut fmt::Formatter,
            ) -> fmt::Result;

            fn fmt_output<W: std::fmt::Write>(
                &self,
                f: &mut W,
                print_opts: &PrintOptions,
                print_state: PrintState,
            ) -> Result<bool, Error>;
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

/// A printer for atoms, useful in a [format!].
///
/// # Examples
///
/// ```
/// use symbolica::{atom::AtomCore, parse};
/// use symbolica::printer::PrintOptions;
/// let a = parse!("x + y").unwrap();
/// println!("{}", a.printer(PrintOptions::latex()));
/// ```
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
        self.atom
            .format(
                f,
                &self.print_opts.update_with_fmt(f),
                PrintState::from_fmt(f),
            )
            .map(|_| ())
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

    pub(crate) fn format<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, Error> {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, opts, print_state),
            AtomView::Var(v) => v.fmt_output(fmt, opts, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, opts, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, opts, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, opts, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, opts, print_state),
        }
    }

    /// Construct a printer for the atom with special options.
    pub(crate) fn printer(&self, opts: PrintOptions) -> AtomPrinter {
        AtomPrinter::new_with_options(*self, opts)
    }

    /// Print the atom in a form that is unique and independent of any implementation details.
    ///
    /// Anti-symmetric functions are not supported.
    pub(crate) fn to_canonical_string(&self) -> String {
        let mut s = String::new();
        self.to_canonical_view_impl(&mut s);
        s
    }

    fn to_canonical_view_impl(&self, out: &mut String) {
        fn add_paren(cur: AtomView, s: AtomView) -> bool {
            if let AtomView::Pow(_) = cur {
                matches!(s, AtomView::Add(_) | AtomView::Mul(_))
            } else if let AtomView::Mul(_) = cur {
                matches!(s, AtomView::Add(_))
            } else {
                false
            }
        }

        match self {
            AtomView::Num(_) => write!(out, "{}", self.printer(PrintOptions::file())).unwrap(),
            AtomView::Var(v) => v.get_symbol().format(&PrintOptions::file(), out).unwrap(),
            AtomView::Fun(f) => {
                f.get_symbol().format(&PrintOptions::file(), out).unwrap();
                out.push('(');

                let mut args = vec![];

                for x in f.iter() {
                    let mut arg = String::new();
                    x.to_canonical_view_impl(&mut arg);
                    args.push(arg);
                }

                // TODO: anti-symmetric may generate minus sign...
                if f.is_symmetric() {
                    args.sort();
                }

                if f.is_antisymmetric() {
                    unimplemented!(
                        "Antisymmetric functions are not supported yet for canonical view"
                    );
                }

                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(out, ",").unwrap();
                    }
                    write!(out, "{}", arg).unwrap();
                }

                write!(out, ")").unwrap();
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if add_paren(*self, b) {
                    write!(out, "(").unwrap();
                    b.to_canonical_view_impl(out);
                    write!(out, ")").unwrap();
                } else {
                    b.to_canonical_view_impl(out);
                }

                if add_paren(*self, e) {
                    write!(out, "^(").unwrap();
                    e.to_canonical_view_impl(out);
                    write!(out, ")").unwrap();
                } else {
                    write!(out, "^").unwrap();
                    e.to_canonical_view_impl(out);
                }
            }
            AtomView::Mul(m) => {
                let mut terms = vec![];

                for x in m.iter() {
                    let mut term = if add_paren(*self, x) {
                        "(".to_string()
                    } else {
                        String::new()
                    };

                    x.to_canonical_view_impl(&mut term);

                    if add_paren(*self, x) {
                        term.push(')');
                    }

                    terms.push(term);
                }

                terms.sort();

                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(out, "*").unwrap();
                    }

                    write!(out, "{}", term).unwrap();
                }
            }
            AtomView::Add(a) => {
                let mut terms = vec![];

                for x in a.iter() {
                    let mut term = if add_paren(*self, x) {
                        "(".to_string()
                    } else {
                        String::new()
                    };

                    x.to_canonical_view_impl(&mut term);

                    if add_paren(*self, x) {
                        term.push(')');
                    }

                    terms.push(term);
                }

                terms.sort();

                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(out, "+").unwrap();
                    }
                    write!(out, "{}", term).unwrap();
                }
            }
        }
    }
}

impl<'a> fmt::Debug for AtomView<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.fmt_debug(fmt)
    }
}

impl<'a> FormattedPrintVar for VarView<'a> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();

        if let Some(custom_print) = &State::get_symbol_data(id).custom_print {
            if let Some(s) = custom_print(self.as_view(), opts) {
                f.write_str(&s)?;
                return Ok(false);
            }
        }

        id.format(opts, f)?;
        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintNum for NumView<'a> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }

    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        /// Input must be digits only.
        fn format_num<W: std::fmt::Write>(
            mut s: String,
            opts: &PrintOptions,
            print_state: &PrintState,
            f: &mut W,
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
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "-".yellow()))?;
            } else if print_state.superscript {
                f.write_char('⁻')?;
            } else {
                f.write_char('-')?;
            }

            print_state.in_sum = false;
        } else if print_state.in_sum {
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }

            print_state.in_sum = false;
        }

        match d {
            CoefficientView::Natural(num, den) => {
                if den == 1 && print_state.suppress_one && (num == 1 || num == -1) {
                    return Ok(true);
                }

                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(num.unsigned_abs().to_string(), opts, &print_state, f)?;
                    if den != 1 {
                        f.write_char('/')?;
                        format_num(den.to_string(), opts, &print_state, f)?;
                    }
                } else if den != 1 {
                    if opts.latex {
                        f.write_fmt(format_args!("\\frac{{{}}}{{{}}}", num.unsigned_abs(), den))?;
                    } else {
                        f.write_fmt(format_args!("{}/{}", num.unsigned_abs(), den))?;
                    }
                } else {
                    f.write_fmt(format_args!("{}", num.unsigned_abs()))?;
                }

                Ok(false)
            }
            CoefficientView::Float(fl) => {
                fl.to_float().format(opts, print_state, f)?;
                Ok(false)
            }
            CoefficientView::Large(r) => {
                let rat = r.to_rat().abs();
                if !opts.latex
                    && (opts.number_thousands_separator.is_some() || print_state.superscript)
                {
                    format_num(rat.numerator().to_string(), opts, &print_state, f)?;
                    if !rat.is_integer() {
                        f.write_char('/')?;
                        format_num(rat.denominator().to_string(), opts, &print_state, f)?;
                    }
                } else if !rat.is_integer() {
                    if opts.latex {
                        f.write_fmt(format_args!(
                            "\\frac{{{}}}{{{}}}",
                            rat.numerator(),
                            rat.denominator(),
                        ))?;
                    } else {
                        f.write_fmt(format_args!(
                            "{}/{}",
                            rat.numerator_ref(),
                            rat.denominator_ref()
                        ))?;
                    }
                } else {
                    f.write_fmt(format_args!("{}", rat.numerator_ref()))?;
                }
                Ok(false)
            }
            CoefficientView::FiniteField(num, fi) => {
                let ff = State::get_finite_field(fi);
                f.write_fmt(format_args!(
                    "[{}%{}]",
                    ff.from_element(&num),
                    ff.get_prime()
                ))?;
                Ok(false)
            }
            CoefficientView::RationalPolynomial(p) => {
                f.write_char('[')?;
                p.deserialize().format(opts, print_state, f)?;
                f.write_char(']').map(|_| false)
            }
        }
    }
}

impl<'a> FormattedPrintMul for MulView<'a> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }

    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let add_paren = print_state.in_exp;
        if add_paren {
            if print_state.in_sum {
                print_state.in_sum = false;
                f.write_char('+')?;
            }

            f.write_char('(')?;
            print_state.in_exp = false;
        }

        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().last() {
            // write -1*x as -x
            if n.get_coeff_view() == CoefficientView::Natural(-1, 1) {
                if print_state.top_level_add_child && opts.color_top_level_sum {
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
        } else if print_state.in_sum {
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
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
                x.format(f, opts, print_state)?;
                if opts.latex {
                    f.write_str("\\right)")?;
                } else {
                    f.write_char(')')?;
                }
            } else {
                x.format(f, opts, print_state)?;
            }
        }

        if add_paren {
            f.write_char(')')?;
        }
        Ok(false)
    }
}

impl<'a> FormattedPrintFn for FunView<'a> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        if let Some(custom_print) = &State::get_symbol_data(id).custom_print {
            if let Some(s) = custom_print(self.as_view(), opts) {
                f.write_str(&s)?;
                return Ok(false);
            }
        }

        id.format(opts, f)?;

        if opts.latex {
            f.write_str("\\!\\left(")?;
        } else {
            if opts.square_brackets_for_function {
                f.write_char('[')?;
            } else {
                f.write_char('(')?;
            }
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.suppress_one = false;
        let mut first = true;
        for x in self.iter() {
            if !first {
                f.write_char(',')?;
            }
            first = false;

            x.format(f, opts, print_state)?;
        }

        if opts.latex {
            f.write_str("\\right)")?;
        } else if opts.square_brackets_for_function {
            f.write_char(']')?;
        } else {
            f.write_char(')')?;
        }

        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintPow for PowView<'a> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let b = self.get_base();
        let e = self.get_exp();

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.suppress_one = false;

        let mut superscript_exponent = false;
        if opts.latex {
            if let AtomView::Num(n) = e {
                if n.get_coeff_view() == CoefficientView::Natural(-1, 1) {
                    // TODO: construct the numerator
                    f.write_str("\\frac{1}{")?;
                    b.format(f, opts, print_state)?;
                    f.write_char('}')?;
                    return Ok(false);
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
                    match n.get_coeff_view() {
                        CoefficientView::Natural(n, d) => n < 0 || d != 1,
                        CoefficientView::Float(_) => true, // TODO
                        CoefficientView::Large(r) => r.is_negative() || !r.to_rat().is_integer(),
                        CoefficientView::FiniteField(n, i) => {
                            opts.symmetric_representation_for_finite_field
                                && n.0 * 2 > State::get_finite_field(i).get_prime()
                        }
                        CoefficientView::RationalPolynomial(_) => true,
                    }
                } else {
                    false
                };

        if base_needs_parentheses {
            if opts.latex {
                f.write_str("\\left(")?;
            } else {
                f.write_char('(')?;
            }
            b.format(f, opts, print_state)?;
            if opts.latex {
                f.write_str("\\right)")?;
            } else {
                f.write_char(')')?;
            }
        } else {
            b.format(f, opts, print_state)?;
        }

        if !superscript_exponent {
            if !opts.latex && opts.double_star_for_exponentiation {
                f.write_str("**")?;
            } else {
                f.write_char('^')?;
            }
        }

        if opts.latex {
            f.write_char('{')?;
            e.format(f, opts, print_state)?;
            f.write_char('}')?;
        } else {
            let exp_needs_parentheses = matches!(e, AtomView::Add(_) | AtomView::Mul(_))
                || if let AtomView::Num(n) = e {
                    !n.get_coeff_view().is_integer()
                } else {
                    false
                };

            if exp_needs_parentheses {
                f.write_char('(')?;
                e.format(f, opts, print_state)?;
                f.write_char(')')?;
            } else {
                print_state.superscript = superscript_exponent;
                e.format(f, opts, print_state)?;
            }
        }

        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

impl<'a> FormattedPrintAdd for AddView<'a> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let mut first = true;
        print_state.top_level_add_child = print_state.level == 0;
        print_state.level += 1;
        print_state.suppress_one = false;

        let add_paren = print_state.in_product || print_state.in_exp;
        if add_paren {
            if print_state.in_sum {
                if print_state.top_level_add_child && opts.color_top_level_sum {
                    f.write_fmt(format_args!("{}", "+".yellow()))?;
                } else {
                    f.write_char('+')?;
                }
            }

            print_state.in_sum = false;
            print_state.in_product = false;
            print_state.in_exp = false;
            f.write_char('(')?;
        }

        let mut count = 0;
        for x in self.iter() {
            if let Some(max_terms) = opts.max_terms {
                if count >= max_terms {
                    break;
                }
            }

            if !first && print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            first = false;

            x.format(f, opts, print_state)?;
            print_state.in_sum = true;
            count += 1;
        }

        if opts.max_terms.is_some() && count < self.get_nargs() {
            if print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            if print_state.top_level_add_child && opts.color_top_level_sum {
                f.write_fmt(format_args!("{0}...", "+".yellow()))?;
            } else {
                f.write_str("+...")?;
            }
        }

        if add_paren {
            f.write_char(')')?;
        }
        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        <Self as std::fmt::Debug>::fmt(self, f)
    }
}

#[cfg(test)]
mod test {
    use colored::control::ShouldColorize;

    use crate::{
        atom::{AtomCore, AtomView},
        domains::{finite_field::Zp, integer::Z, SelfRing},
        parse,
        printer::{AtomPrinter, PrintOptions, PrintState},
        symbol,
    };

    #[test]
    fn atoms() {
        let a = parse!("f(x,y^2)^(x+z)/5+3").unwrap();

        if ShouldColorize::from_env().should_colorize() {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "1/5*f(x,y^2)^(x+z)\u{1b}[33m+\u{1b}[0m3"
            );
        } else {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "1/5*f(x,y^2)^(x+z)+3"
            );
        }

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::latex())
            ),
            "\\frac{1}{5} f\\!\\left(x,y^{2}\\right)^{x+z}+3"
        );

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::mathematica())
            ),
            "1/5 f[x,y^2]^(x+z)+3"
        );

        let a = parse!("8127389217 x^2").unwrap();
        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(
                    a.as_view(),
                    PrintOptions {
                        number_thousands_separator: Some('_'),
                        multiplication_operator: ' ',
                        num_exp_as_superscript: true,
                        ..PrintOptions::file()
                    }
                )
            ),
            "812_738_921_7 symbolica::x²"
        );
    }

    #[test]
    fn polynomials() {
        let a = parse!("15 x^2")
            .unwrap()
            .to_polynomial::<_, u8>(&Zp::new(17), None);

        let mut s = String::new();
        a.format(
            &PrintOptions {
                print_finite_field: true,
                symmetric_representation_for_finite_field: true,
                ..PrintOptions::file()
            },
            PrintState::new(),
            &mut s,
        )
        .unwrap();

        assert_eq!(s, "-2*x^2 % 17");
    }

    #[test]
    fn rational_polynomials() {
        let a = parse!("15 x^2 / (1+x)")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert_eq!(format!("{}", a), "15*x^2/(1+x)");

        let a = parse!("(15 x^2 + 6) / (1+x)")
            .unwrap()
            .to_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert_eq!(format!("{}", a), "(6+15*x^2)/(1+x)");
    }

    #[test]
    fn factorized_rational_polynomials() {
        let a = parse!("15 x^2 / ((1+x)(x+2))")
            .unwrap()
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(
            format!("{}", a) == "15*x^2/((1+x)*(2+x))"
                || format!("{}", a) == "15*x^2/((2+x)*(1+x))"
        );

        let a = parse!("(15 x^2 + 6) / ((1+x)(x+2))")
            .unwrap()
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(
            format!("{}", a) == "3*(2+5*x^2)/((1+x)*(2+x))"
                || format!("{}", a) == "3*(2+5*x^2)/((2+x)*(1+x))"
        );

        let a = parse!("1/(v1*v2)")
            .unwrap()
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(format!("{}", a) == "1/(v1*v2)" || format!("{}", a) == "1/(v2*v1)");

        let a = parse!("-1/(2+v1)")
            .unwrap()
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(format!("{}", a) == "-1/(2+v1)");
    }

    #[test]
    fn base_parentheses() {
        let a = parse!("(-1)^(x+1)-(1/2)^x").unwrap();
        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::file_no_namespace())
            ),
            "(-1)^(x+1)-(1/2)^x"
        )
    }

    #[test]
    fn canon() {
        let _ = symbol!("canon_f"; Symmetric).unwrap();
        let _ = symbol!("canon_y");
        let _ = symbol!("canon_x");

        let a = parse!("canon_x^2 + 2*canon_x*canon_y + canon_y^2*(canon_x+canon_y) + canon_f(canon_x,canon_y)").unwrap();
        assert_eq!(
            a.to_canonical_string(),
            "(symbolica::canon_x+symbolica::canon_y)*symbolica::canon_y^2+2*symbolica::canon_x*symbolica::canon_y+symbolica::canon_f(symbolica::canon_x,symbolica::canon_y)+symbolica::canon_x^2"
        );
    }

    #[test]
    fn custom_print() {
        let _ = symbol!("mu";;;|a, opt| {
            if !opt.latex {
                return None; // use default printer
            }

            let mut fmt = String::new();
            fmt.push_str("\\mu");
            if let AtomView::Fun(f) = a {
                fmt.push_str("_{");
                let n_args = f.get_nargs();

                for (i, a) in f.iter().enumerate() {
                    a.format(&mut fmt, opt, PrintState::new()).unwrap();
                    if i < n_args - 1 {
                        fmt.push_str(",");
                    }
                }

                fmt.push_str("}");
            }

            Some(fmt)
        })
        .unwrap();

        let e = crate::parse!("mu^2 + mu(1) + mu(1,2)").unwrap();
        let s = format!("{}", e.printer(PrintOptions::latex()));
        assert_eq!(s, "\\mu^{2}+\\mu_{1}+\\mu_{1,2}");
    }
}
