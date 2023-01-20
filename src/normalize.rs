use std::cmp::Ordering;

use smallvec::SmallVec;

use crate::{
    representations::{
        Atom, AtomView, Fun, ListIterator, Mul, Num, OwnedAtom, OwnedMul, OwnedNum, OwnedPow, Pow,
        Var,
    },
    state::{BufferHandle, Workspace},
};

impl<'a, P: Atom> AtomView<'a, P> {
    /// Sort factors in a term. `x` and `pow(x,2)` are placed next to each other by sorting a pow based on the base only.
    fn partial_cmp<'b>(&self, other: &AtomView<'b, P>) -> Option<Ordering> {
        match (&self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Some(Ordering::Equal),
            (AtomView::Num(_), _) => Some(Ordering::Greater),
            (_, AtomView::Num(_)) => Some(Ordering::Less),

            (AtomView::Var(v1), AtomView::Var(v2)) => Some(v1.get_name().cmp(&v2.get_name())),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                // TODO: inline partial_cmp call by creating an inlined version
                Some(p1.get_base().partial_cmp(&p2.get_base()).unwrap())
            }
            (_, AtomView::Pow(p2)) => {
                let base = p2.get_base();
                Some(self.partial_cmp(&base).unwrap())
            }
            (AtomView::Pow(p1), _) => {
                let base = p1.get_base();
                Some(base.partial_cmp(other).unwrap())
            }
            (AtomView::Var(_), _) => Some(Ordering::Less),
            (_, AtomView::Var(_)) => Some(Ordering::Greater),

            (AtomView::Mul(_), AtomView::Mul(_)) => Some(Ordering::Equal), // TODO
            (AtomView::Mul(_), _) => Some(Ordering::Less),
            (_, AtomView::Mul(_)) => Some(Ordering::Greater),

            (AtomView::Add(_), AtomView::Add(_)) => Some(Ordering::Equal), // TODO
            (AtomView::Add(_), _) => Some(Ordering::Less),
            (_, AtomView::Add(_)) => Some(Ordering::Greater),

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                // TODO: on equality the arguments have to be compared too
                Some(f1.get_name().cmp(&f2.get_name()))
            }
        }
    }
}

impl<P: Atom> OwnedAtom<P> {
    /// Merge two factors if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    fn merge_factors(&mut self, other: &mut Self, helper: &mut Self) -> bool {
        // x^a * x^b = x^(a + b)
        if let OwnedAtom::Pow(p1) = self {
            if let OwnedAtom::Pow(p2) = other {
                let new_exp = helper.transform_to_num();

                let (base2, exp2) = p2.to_pow_view().get_base_exp();

                // help the borrow checker out by encapsulating base1 and exp1
                {
                    let (base1, exp1) = p1.to_pow_view().get_base_exp();

                    if base1 != base2 {
                        return false;
                    }

                    if let AtomView::Num(n) = &exp1 {
                        new_exp.from_view(n);
                    } else {
                        unimplemented!("No support for non-numerical powers yet");
                    }
                }

                if let AtomView::Num(n2) = &exp2 {
                    new_exp.add(n2);

                    if new_exp.to_num_view().is_one() {
                        self.from_view(&base2);
                    } else {
                        p1.from_base_and_exp(base2, AtomView::Num(new_exp.to_num_view()));
                    }

                    return true;
                } else {
                    unimplemented!("No support for non-numerical powers yet");
                }
            }
        }

        // x * x^n = x^(n+1)
        if let OwnedAtom::Pow(p) = other {
            let pv = p.to_pow_view();
            let (base, exp) = pv.get_base_exp();

            if self.to_view() == base {
                if let AtomView::Num(n) = &exp {
                    let num = helper.transform_to_num();
                    num.from_u64_frac(1, 1);
                    num.add(n);
                    let op = self.transform_to_pow();
                    op.from_base_and_exp(base, AtomView::Num(num.to_num_view()));

                    return true;
                } else {
                    unimplemented!("No support for non-numerical powers yet");
                };
            } else {
                return false;
            }
        }

        // simplify num1 * num2
        if let OwnedAtom::Num(n1) = self {
            if let OwnedAtom::Num(n2) = other {
                n1.mul(&n2.to_num_view());
                return true;
            } else {
                return false;
            }
        }

        // x * x => x^2
        if self.to_view() == other.to_view() {
            // add powers
            let exp = other.transform_to_num();
            exp.from_u64_frac(2, 1);

            //let mut a = workspace.get_atom_test_buf();
            let new_pow = helper.transform_to_pow();
            new_pow.from_base_and_exp(self.to_view(), AtomView::Num(exp.to_num_view()));

            // overwrite self with the new power view
            let pow_handle = self.transform_to_pow();
            pow_handle.from_view(&new_pow.to_pow_view());

            return true;
        }

        false
    }
}

impl<'a, P: Atom> AtomView<'a, P> {
    /// Normalize a term.
    pub fn normalize(&self, workspace: &Workspace<P>, out: &mut P::OM) {
        let mut atom_test_buf: SmallVec<[BufferHandle<OwnedAtom<P>>; 20]> = SmallVec::new();
        match self {
            AtomView::Mul(t) => {
                let mut it = t.into_iter();
                while let Some(a) = it.next() {
                    let mut handle = workspace.get_atom_stack();
                    let new_at = handle.get_buf_mut();
                    new_at.from_view(&a);

                    // TODO: check dirty flag and normalize

                    atom_test_buf.push(handle);
                }
            }
            _ => unreachable!("Can only normalize term"),
        }

        atom_test_buf.sort_by(|a, b| {
            a.get_buf()
                .to_view()
                .partial_cmp(&b.get_buf().to_view())
                .unwrap()
        });

        if !atom_test_buf.is_empty() {
            let mut last_buf = atom_test_buf.remove(0);

            let mut handle = workspace.get_atom_stack();
            let helper = handle.get_buf_mut();

            for mut cur_buf in atom_test_buf.drain(..) {
                if !last_buf
                    .get_buf_mut()
                    .merge_factors(cur_buf.get_buf_mut(), helper)
                {
                    // we are done merging
                    out.extend(last_buf.get_buf().to_view());
                    last_buf = cur_buf;
                }
            }

            out.extend(last_buf.get_buf().to_view());
        }
    }
}
