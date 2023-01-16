use smallvec::SmallVec;

use crate::{
    representations::{
        Atom, AtomView, Fn, ListIteratorT, Mul, Num, OwnedMul, OwnedNum, OwnedPow, Pow,
        Var,
    },
    state::{ResettableBuffer, Workspace},
};

impl<'a, P: Atom> AtomView<'a, P> {
    /// Normalize a term.
    pub fn normalize(&self, workspace: &mut Workspace<P>, out: &mut P::O) {
        let mut new_term_handle = workspace.get_mul_buf();
        let new_term = new_term_handle.get_buf();

        // TODO: move to global workspace? hard because of references
        let mut number_buf: SmallVec<[P::N<'a>; 20]> = SmallVec::new();
        let mut symbol_buf: SmallVec<[P::V<'a>; 20]> = SmallVec::new();
        let mut func_buf: SmallVec<[P::F<'a>; 20]> = SmallVec::new();
        let mut pow_buf: SmallVec<[P::P<'a>; 20]> = SmallVec::new();

        match self {
            AtomView::Mul(t) => {
                let mut it = t.into_iter();
                while let Some(a) = it.next() {
                    match a {
                        AtomView::Var(v) => symbol_buf.push(v),
                        AtomView::Num(n) => number_buf.push(n),
                        AtomView::Fn(f) => func_buf.push(f.clone()),
                        AtomView::Pow(p) => pow_buf.push(p.clone()),
                        _ => {}
                    }
                }
            }
            _ => unreachable!("Can only normalize term"),
        }

        pow_buf.sort_by(|a, b| a.get_base().partial_cmp(&b.get_base()).unwrap());

        // TODO: merge var(x) and pow(var(x),2)
        symbol_buf.sort_by_key(|k| k.get_name());

        if !symbol_buf.is_empty() {
            let mut last_symbol = &symbol_buf[0];

            let mut last_pow = 1;
            for x in &symbol_buf[1..] {
                if x == last_symbol {
                    last_pow += 1;
                } else {
                    if last_pow > 1 {
                        let mut new_pow_br = workspace.get_pow_buf();
                        let new_pow = new_pow_br.get_buf();

                        let mut new_num_br = workspace.get_num_buf();
                        let new_num = new_num_br.get_buf();
                        P::ON::from_u64_frac(new_num, last_pow, 1);

                        new_pow.from_base_and_exp(
                            AtomView::Var(last_symbol.clone()),
                            AtomView::Num(new_num.to_num_view()),
                        );
                        new_term.extend(AtomView::Pow(new_pow.to_pow_view()));
                    } else {
                        new_term.extend(AtomView::Var(last_symbol.clone()));
                    }

                    last_symbol = x;
                    last_pow = 1;
                }
            }

            if last_pow > 1 {
                let mut new_pow_br = workspace.get_pow_buf();
                let new_pow = new_pow_br.get_buf();

                let mut new_num_br = workspace.get_num_buf();
                let new_num = new_num_br.get_buf();
                P::ON::from_u64_frac(new_num, last_pow, 1);

                new_pow.from_base_and_exp(
                    AtomView::Var(last_symbol.clone()),
                    AtomView::Num(new_num.to_num_view()),
                );
                new_term.extend(AtomView::Pow(new_pow.to_pow_view()));
            } else {
                new_term.extend(AtomView::Var(last_symbol.clone()));
            }
        }

        if !pow_buf.is_empty() {
            let (mut last_base, exp) = pow_buf[0].get_base_exp();

            let mut new_num_br = workspace.get_num_buf();
            let mut last_pow = new_num_br.get_buf();

            // merge all numerical powers
            if let AtomView::Num(n) = exp {
                P::ON::from_view(&mut last_pow, n);
            } else {
                unimplemented!()
            };

            for x in &pow_buf[1..] {
                let (base, exp) = x.get_base_exp();
                if base == last_base {
                    if let AtomView::Num(n) = exp {
                        last_pow.add(&n);
                    } else {
                        unimplemented!()
                    }
                } else {
                    if last_pow.to_num_view().is_one() {
                        new_term.extend(last_base);
                    } else {
                        let mut new_pow_br = workspace.get_pow_buf();
                        let new_pow = new_pow_br.get_buf();
                        new_pow.from_base_and_exp(last_base, AtomView::Num(last_pow.to_num_view()));
                        new_term.extend(AtomView::Pow(new_pow.to_pow_view()))
                    }

                    last_base = base;
                    if let AtomView::Num(n) = exp {
                        last_pow.reset(); // TODO: needed?
                        P::ON::from_view(&mut last_pow, n);
                    } else {
                        unimplemented!()
                    };
                }
            }

            if last_pow.to_num_view().is_one() {
                new_term.extend(last_base);
            } else {
                let mut new_pow_br = workspace.get_pow_buf();
                let new_pow = new_pow_br.get_buf();
                new_pow.from_base_and_exp(last_base, AtomView::Num(last_pow.to_num_view()));

                new_term.extend(AtomView::Pow(new_pow.to_pow_view()));
            }
        }

        func_buf.sort_by(|a, b| a.cmp(b));

        // TODO: normalise each function, checking the dirty flag first
        for x in func_buf {
            new_term.extend(AtomView::Fn(x));
        }

        if !number_buf.is_empty() {
            let mut new_num_br = workspace.get_num_buf();
            let mut new_num = new_num_br.get_buf();
            P::ON::from_view(&mut new_num, number_buf.pop().unwrap());

            for x in &number_buf {
                new_num.add(x);
            }

            new_term.extend(AtomView::Num(new_num.to_num_view()));
        }

        new_term.to_atom(out);
    }
}
