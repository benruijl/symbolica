use crate::representations::{
    AtomT, AtomView, FunctionT, ListIteratorT, OwnedNumberT, OwnedTermT, OwnedVarT, TermT, VarT,
};

impl<'a, P: AtomT> AtomView<'a, P> {
    pub fn normalize(&self) -> P::O {
        let mut new_term = P::OT::new();

        // TODO: global workspace
        let mut number_buf: Vec<P::N<'a>> = vec![];
        let mut symbol_buf: Vec<P::V<'a>> = vec![];
        let mut func_buf: Vec<P::F<'a>> = vec![];

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
                    let new_var = P::OV::from_id_pow(last_name, last_pow);
                    new_term.extend(AtomView::Var(new_var.to_var_view()));

                    last_name = name;
                    last_pow = P::ON::from_view(pow);
                }
            }

            let new_var = P::OV::from_id_pow(last_name, last_pow);
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

        new_term.to_atom()
    }
}
