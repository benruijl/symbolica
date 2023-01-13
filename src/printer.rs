use std::fmt::{self, Write};

use crate::{
    representations::{AtomT, AtomView, FunctionT, ListIteratorT, NumberT, TermT, VarT},
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
    FormattedPrintNumber,
    FormattedPrintFunction,
    FormattedPrintTerm
);

pub struct AtomPrinter<'a, 'b, P: AtomT> {
    pub atom: AtomView<'a, P>,
    pub state: &'b State,
    pub print_mode: PrintMode,
}

impl<'a, 'b, P: AtomT> AtomPrinter<'a, 'b, P> {
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

impl<'a, 'b, P: AtomT> fmt::Display for AtomPrinter<'a, 'b, P> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.atom.fmt_output(f, self.print_mode, self.state)
    }
}

impl<'a, P: AtomT> AtomView<'a, P> {
    fn fmt_output(
        &self,
        fmt: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        match self {
            AtomView::Number(n) => n.fmt_output(fmt, print_mode, state),
            AtomView::Var(v) => v.fmt_output(fmt, print_mode, state),
            AtomView::Function(f) => f.fmt_output(fmt, print_mode, state),
            AtomView::Term(t) => t.fmt_output(fmt, print_mode, state),
        }
    }
}

impl<'a, A: VarT<'a>> FormattedPrintVar for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())
            .unwrap();

        let pow = self.get_pow();

        if !pow.is_one() {
            f.write_char('^').unwrap();
            pow.fmt_output(f, print_mode, state)
        } else {
            Ok(())
        }
    }
}

impl<'a, A: NumberT<'a>> FormattedPrintNumber for A {
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

impl<'a, A: TermT<'a>> FormattedPrintTerm for A {
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

impl<'a, A: FunctionT<'a>> FormattedPrintFunction for A {
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
