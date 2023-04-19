use std::fmt::{self, Write};

use colored::Colorize;

use crate::{
    representations::{number::BorrowedNumber, Add, Atom, AtomView, Fun, Mul, Num, Pow, Var},
    state::State,
};

#[derive(Debug, Copy, Clone)]
pub enum PrintMode {
    Form,
    Mathematica,
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
        self.atom.fmt_output(f, self.print_mode, self.state)
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
    ) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, print_mode, state),
            AtomView::Var(v) => v.fmt_output(fmt, print_mode, state),
            AtomView::Fun(f) => f.fmt_output(fmt, print_mode, state),
            AtomView::Pow(p) => p.fmt_output(fmt, print_mode, state),
            AtomView::Mul(t) => t.fmt_output(fmt, print_mode, state),
            AtomView::Add(e) => e.fmt_output(fmt, print_mode, state),
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
    ) -> fmt::Result {
        let name = state.get_name(self.get_name()).unwrap();
        if name.ends_with('_') {
            f.write_fmt(format_args!("{}", name.as_str().cyan()))
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
        }
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        _print_mode: PrintMode,
        state: &State,
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
                f.write_fmt(format_args!("[{}%{}]", ff.from_element(num), ff.get_prime()))
            }
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
    ) -> fmt::Result {
        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char('*')?;
            }
            first = false;

            if let AtomView::Add(_) = x {
                f.write_char('(')?;
                x.fmt_output(f, print_mode, state)?;
                f.write_char(')')?;
            } else {
                x.fmt_output(f, print_mode, state)?;
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
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())?;
        f.write_char('(')?;

        let mut first = true;
        for x in self.into_iter() {
            if !first {
                print!(",");
            }
            first = false;

            x.fmt_output(f, print_mode, state)?;
        }

        f.write_char(')')
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("f_{}(", self.get_name().to_u32()))?;

        let mut first = true;
        for x in self.into_iter() {
            if !first {
                print!(",");
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
    ) -> fmt::Result {
        let b = self.get_base();
        if let AtomView::Add(_) | AtomView::Mul(_) = b {
            f.write_char('(')?;
            b.fmt_output(f, print_mode, state)?;
            f.write_char(')')?;
        } else {
            b.fmt_output(f, print_mode, state)?;
        }

        f.write_char('^')?;

        let e = self.get_exp();
        if let AtomView::Add(_) | AtomView::Mul(_) = e {
            f.write_char('(')?;
            e.fmt_output(f, print_mode, state)?;
            f.write_char(')')
        } else {
            e.fmt_output(f, print_mode, state)
        }
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let b = self.get_base();
        if let AtomView::Add(_) | AtomView::Mul(_) = b {
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
    ) -> fmt::Result {
        let mut first = true;
        for x in self.into_iter() {
            if !first {
                f.write_char('+')?;
            }
            first = false;

            x.fmt_output(f, print_mode, state)?;
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
