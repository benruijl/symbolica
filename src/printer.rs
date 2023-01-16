use std::fmt::{self, Write};

use crate::{
    representations::{Add, Atom, AtomView, Fn, ListIteratorT, Mul, Num, Pow, Var},
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
    fn fmt_output(
        &self,
        fmt: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, print_mode, state),
            AtomView::Var(v) => v.fmt_output(fmt, print_mode, state),
            AtomView::Fn(f) => f.fmt_output(fmt, print_mode, state),
            AtomView::Pow(p) => p.fmt_output(fmt, print_mode, state),
            AtomView::Mul(t) => t.fmt_output(fmt, print_mode, state),
            AtomView::Add(e) => e.fmt_output(fmt, print_mode, state),
        }
    }
}

impl<'a, A: Var<'a>> FormattedPrintVar for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        _print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())
    }
}

impl<'a, A: Num<'a>> FormattedPrintNum for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        _print_mode: PrintMode,
        _state: &State,
    ) -> fmt::Result {
        let d = self.get_numden();
        if d.1 != 1 {
            f.write_fmt(format_args!("{}/{}", d.0, d.1))
        } else {
            f.write_fmt(format_args!("{}", d.0))
        }
    }
}

impl<'a, A: Mul<'a>> FormattedPrintMul for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                f.write_char('*').unwrap();
            }
            first = false;

            x.fmt_output(f, print_mode, state).unwrap();
        }
        Ok(())
    }
}

impl<'a, A: Fn<'a>> FormattedPrintFn for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())
            .unwrap();
        f.write_char('(').unwrap();

        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                print!(",");
            }
            first = false;

            x.fmt_output(f, print_mode, state).unwrap();
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
        if let AtomView::Add(_) = b {
            f.write_char('(').unwrap();
            b.fmt_output(f, print_mode, state).unwrap();
            f.write_char(')').unwrap();
        } else {
            b.fmt_output(f, print_mode, state).unwrap();
        }

        f.write_char('^').unwrap();

        let e = self.get_exp();
        if let AtomView::Add(_) = b {
            f.write_char('(').unwrap();
            e.fmt_output(f, print_mode, state).unwrap();
            f.write_char(')')
        } else {
            e.fmt_output(f, print_mode, state)
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
        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                f.write_char('+').unwrap();
            }
            first = false;

            x.fmt_output(f, print_mode, state).unwrap();
        }
        Ok(())
    }
}
