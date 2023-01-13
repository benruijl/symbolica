use smallvec::SmallVec;

use crate::{
    representations::{
        AtomT, AtomView, FunctionT, ListIteratorT, OwnedNumberT, OwnedTermT, OwnedVarT, TermT, VarT,
    },
    state::{ResettableBuffer, Workspace},
};

impl<'a, P: AtomT> AtomView<'a, P> {
    /// Normalize a term.
    pub fn normalize(&self, workspace: &mut Workspace<P>, out: &mut P::O) {
        let mut new_term_br = workspace.term_buf.get_buf_ref();
        let new_term = new_term_br.get_buf();

        // TODO: move to global workspace? hard because of references
        let mut number_buf: SmallVec<[P::N<'a>; 20]> = SmallVec::new();
        let mut symbol_buf: SmallVec<[P::V<'a>; 20]> = SmallVec::new();
        let mut func_buf: SmallVec<[P::F<'a>; 20]> = SmallVec::new();

        match self {
            AtomView::Term(t) => {
                let mut it = t.into_iter();
                while let Some(a) = it.next() {
                    match a {
                        AtomView::Var(v) => symbol_buf.push(v),
                        AtomView::Number(n) => number_buf.push(n),
                        AtomView::Function(f) => func_buf.push(f.clone()),
                        _ => {}
                    }
                }
            }
            _ => unreachable!("Can only normalize term"),
        }

        symbol_buf.sort_by_key(|k| k.get_name());

        if !symbol_buf.is_empty() {
            let mut last_name = symbol_buf[0].get_name();
            let mut last_pow: <P as AtomT>::ON = P::ON::from_view(symbol_buf[0].get_pow());
            for x in &symbol_buf[1..] {
                let name = x.get_name();
                let pow = x.get_pow();
                if name == last_name {
                    last_pow.add(&pow);
                } else {
                    let mut new_var = P::OV::new();
                    new_var.from_id_pow(last_name, last_pow);
                    new_term.extend(AtomView::Var(new_var.to_var_view()));

                    last_name = name;
                    last_pow = P::ON::from_view(pow);
                }
            }

            let mut new_var_br = workspace.var_buf.get_buf_ref();
            let new_var = new_var_br.get_buf();
            new_var.from_id_pow(last_name, last_pow);
            new_term.extend(AtomView::Var(new_var.to_var_view()));
        }

        // TODO
        func_buf.sort_by(|a, b| a.cmp(b));

        for x in func_buf {
            new_term.extend(AtomView::Function(x));
        }

        if !number_buf.is_empty() {
            let mut out: <P as AtomT>::ON = P::ON::from_view(number_buf.pop().unwrap());

            for x in &number_buf {
                out.add(x);
            }

            let k = out.to_num_view();
            new_term.extend(AtomView::Number(k));
        }

        new_term.to_atom(out);
    }
}
