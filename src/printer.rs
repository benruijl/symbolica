use std::fmt::{self, Write};

use crate::{
    representations::{
        number::BorrowedNumber, Add, Atom, AtomView, Fun, ListIterator, Mul, Num, Pow, Var,
    },
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
            fn print(&self);

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
    pub fn print(&self) {
        match self {
            AtomView::Num(n) => n.print(),
            AtomView::Var(v) => v.print(),
            AtomView::Fun(f) => f.print(),
            AtomView::Pow(p) => p.print(),
            AtomView::Mul(m) => m.print(),
            AtomView::Add(a) => a.print(),
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

impl<'a, A: Var<'a>> FormattedPrintVar for A {
    fn fmt_output(
        &self,
        f: &mut fmt::Formatter,
        _print_mode: PrintMode,
        state: &State,
    ) -> fmt::Result {
        f.write_str(state.get_name(self.get_name()).unwrap())
    }

    fn print(&self) {
        print!("v_{}", self.get_name().to_u32());
    }
}

impl<'a, A: Num<'a>> FormattedPrintNum for A {
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
                f.write_fmt(format_args!("[{}%{}]", ff.to_u64(num), ff.get_prime()))
            }
        }
    }

    fn print(&self) {
        let d = self.get_number_view();

        match d {
            BorrowedNumber::Natural(num, den) => {
                if den != 1 {
                    print!("{}/{}", num, den)
                } else {
                    print!("{}", num)
                }
            }
            BorrowedNumber::Large(r) => print!("{}", r),
            BorrowedNumber::FiniteField(num, fi) => {
                print!("[m_{}%f_{}]", num.0, fi.0);
            }
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

    fn print(&self) {
        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                print!("*");
            }
            first = false;

            x.print();
        }
    }
}

impl<'a, A: Fun<'a>> FormattedPrintFn for A {
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

    fn print(&self) {
        print!("f_{}(", self.get_name().to_u32());

        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                print!(",");
            }
            first = false;

            x.print();
        }

        print!(")")
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

    fn print(&self) {
        let b = self.get_base();
        if let AtomView::Add(_) = b {
            print!("(");
            b.print();
            print!(")");
        } else {
            b.print();
        }

        print!("^");

        let e = self.get_exp();
        if let AtomView::Add(_) = b {
            print!("(");
            e.print();
            print!(")");
        } else {
            e.print();
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

    fn print(&self) {
        let mut it = self.into_iter();
        let mut first = true;
        while let Some(x) = it.next() {
            if !first {
                print!("+");
            }
            first = false;

            x.print();
        }
    }
}
