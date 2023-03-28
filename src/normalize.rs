use std::cmp::Ordering;

use smallvec::SmallVec;

use crate::{
    representations::{
        number::{BorrowedNumber, Number},
        Add, Atom, AtomView, Fun, ListIterator, ListSlice, Mul, Num, OwnedAdd, OwnedAtom, OwnedFun,
        OwnedMul, OwnedNum, OwnedPow, OwnedVar, Pow, Var,
    },
    state::{BufferHandle, ResettableBuffer, State, Workspace},
};

impl<'a, P: Atom> AtomView<'a, P> {
    /// Sort factors in a term. `x` and `pow(x,2)` are placed next to each other by sorting a pow based on the base only.
    /// TODO: sort x and x*2 next to each other by ignoring coefficient
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
    fn merge_factors(&mut self, other: &mut Self, helper: &mut Self, state: &State) -> bool {
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
                    new_exp.add(n2, state);

                    if new_exp.to_num_view().is_zero() {
                        let num = self.transform_to_num();
                        num.from_number(Number::Natural(1, 1));
                    } else if new_exp.to_num_view().is_one() {
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
                    num.from_number(Number::Natural(1, 1));
                    num.add(n, state);
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
                n1.mul(&n2.to_num_view(), state);
                return true;
            } else {
                return false;
            }
        }

        // x * x => x^2
        if self.to_view() == other.to_view() {
            // add powers
            let exp = other.transform_to_num();
            exp.from_number(Number::Natural(2, 1));

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

    /// Merge two terms if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    fn merge_terms(&mut self, other: &mut Self, helper: &mut Self, state: &State) -> bool {
        if let OwnedAtom::Num(n1) = self {
            if let OwnedAtom::Num(n2) = other {
                n1.add(&n2.to_num_view(), state);
                return true;
            } else {
                return false;
            }
        }

        // compare the non-coefficient part of terms and add the coefficients if they are the same
        if let OwnedAtom::Mul(m) = self {
            let slice = m.to_mul_view().to_slice();

            let last_elem = slice.get(slice.len() - 1);

            let (non_coeff1, has_coeff) = if let AtomView::Num(_) = &last_elem {
                (slice.get_subslice(0..slice.len() - 1), true)
            } else {
                (m.to_mul_view().to_slice(), false)
            };

            if let OwnedAtom::Mul(m2) = other {
                let slice2 = m2.to_mul_view().to_slice();
                let last_elem2 = slice2.get(slice2.len() - 1);

                let non_coeff2 = if let AtomView::Num(_) = &last_elem2 {
                    slice2.get_subslice(0..slice2.len() - 1)
                } else {
                    m2.to_mul_view().to_slice()
                };

                if non_coeff1.eq(&non_coeff2) {
                    // TODO: not correct for finite fields!
                    let num = if let AtomView::Num(n) = &last_elem {
                        n.get_number_view()
                    } else {
                        BorrowedNumber::Natural(1, 1)
                    };

                    let new_coeff = if let AtomView::Num(n) = &last_elem2 {
                        num.add(&n.get_number_view(), state)
                    } else {
                        num.add(&BorrowedNumber::Natural(1, 1), state)
                    };

                    // help the borrow checker by dropping all references
                    drop(last_elem);
                    drop(non_coeff1);
                    drop(non_coeff2);
                    drop(last_elem2);
                    drop(slice2);
                    drop(slice);

                    if new_coeff.is_zero() {
                        let num = self.transform_to_num();
                        num.from_number(new_coeff);

                        return true;
                    }

                    let on = helper.transform_to_num();
                    on.from_number(new_coeff);

                    if has_coeff {
                        m.replace_last(on.to_num_view().to_view());
                    } else {
                        m.extend(on.to_num_view().to_view());
                    }

                    return true;
                }
            }
        } else {
            if let OwnedAtom::Mul(m) = other {
                let slice = m.to_mul_view().to_slice();

                if slice.len() != 2 {
                    return false; // no match
                }

                let last_elem = slice.get(slice.len() - 1);

                if self.to_view() == slice.get(0) {
                    let (new_coeff, has_num) = if let AtomView::Num(n) = &last_elem {
                        (
                            n.get_number_view()
                                .add(&BorrowedNumber::Natural(1, 1), state),
                            true,
                        )
                    } else {
                        (Number::Natural(2, 1), false)
                    };

                    // help the borrow checker by dropping all references
                    drop(last_elem);
                    drop(slice);

                    if new_coeff.is_zero() {
                        let num = self.transform_to_num();
                        num.from_number(new_coeff);

                        return true;
                    }

                    let on = helper.transform_to_num();
                    on.from_number(new_coeff);

                    if has_num {
                        m.replace_last(on.to_num_view().to_view());
                    } else {
                        m.extend(on.to_num_view().to_view());
                    }

                    std::mem::swap(self, other);

                    return true;
                }
            } else {
                if self.to_view() == other.to_view() {
                    let mul = helper.transform_to_mul();

                    let num = other.transform_to_num();
                    num.from_number(Number::Natural(2, 1));

                    mul.extend(self.to_view());
                    mul.extend(other.to_view());

                    std::mem::swap(self, helper);
                    return true;
                }
            }
        };

        false
    }
}

impl<'a, P: Atom> AtomView<'a, P> {
    #[inline(always)]
    pub fn is_dirty(&self) -> bool {
        match self {
            AtomView::Num(n) => n.is_dirty(),
            AtomView::Var(_) => false,
            AtomView::Fun(f) => f.is_dirty(),
            AtomView::Pow(p) => p.is_dirty(),
            AtomView::Mul(m) => m.is_dirty(),
            AtomView::Add(a) => a.is_dirty(),
        }
    }

    /// Normalize an atom.
    pub fn normalize(&self, workspace: &Workspace<P>, state: &State, out: &mut OwnedAtom<P>) {
        // TODO: check dirty flag here too

        match self {
            AtomView::Mul(t) => {
                let mut atom_test_buf: SmallVec<[BufferHandle<OwnedAtom<P>>; 20]> = SmallVec::new();

                let mut it = t.into_iter();
                while let Some(a) = it.next() {
                    let mut handle = workspace.new_atom();
                    let new_at = handle.get_buf_mut();

                    if a.is_dirty() {
                        a.normalize(workspace, state, new_at);
                    } else {
                        new_at.from_view(&a);
                    }

                    atom_test_buf.push(handle);
                }

                atom_test_buf.sort_by(|a, b| {
                    a.get_buf()
                        .to_view()
                        .partial_cmp(&b.get_buf().to_view())
                        .unwrap()
                });

                let out = out.transform_to_mul();

                if !atom_test_buf.is_empty() {
                    let mut last_buf = atom_test_buf.remove(0);

                    let mut handle = workspace.new_atom();
                    let helper = handle.get_buf_mut();

                    for mut cur_buf in atom_test_buf.drain(..) {
                        if !last_buf.get_buf_mut().merge_factors(
                            cur_buf.get_buf_mut(),
                            helper,
                            state,
                        ) {
                            // we are done merging
                            {
                                let v = last_buf.get_buf().to_view();
                                if let AtomView::Num(n) = v {
                                    if !n.is_one() {
                                        out.extend(last_buf.get_buf().to_view());
                                    }
                                } else {
                                    out.extend(last_buf.get_buf().to_view());
                                }
                            }
                            last_buf = cur_buf;
                        }
                    }

                    out.extend(last_buf.get_buf().to_view());
                }
            }
            AtomView::Num(n) => {
                // TODO: normalize and remove dirty flag
                let nn = out.transform_to_num();
                nn.from_view(n);
            }
            AtomView::Var(v) => {
                let vv = out.transform_to_var();
                vv.from_view(v);
            }
            AtomView::Fun(f) => {
                let out = out.transform_to_fun();
                out.from_name(f.get_name());

                let mut it = f.into_iter();

                let mut handle = workspace.new_atom();
                let new_at = handle.get_buf_mut();
                while let Some(a) = it.next() {
                    if a.is_dirty() {
                        new_at.reset(); // TODO: needed?
                        a.normalize(workspace, state, new_at);
                        out.add_arg(new_at.to_view());
                    } else {
                        out.add_arg(a);
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                if base.is_dirty() || exp.is_dirty() {
                    let mut base_handle = workspace.new_atom();
                    let mut exp_handle = workspace.new_atom();

                    let new_base = base_handle.get_buf_mut();
                    let new_exp = exp_handle.get_buf_mut();

                    base.normalize(workspace, state, new_base);
                    exp.normalize(workspace, state, new_exp);

                    // simplyify a number to a power
                    if let AtomView::Num(_n) = base {
                        if let AtomView::Num(_e) = exp {
                            //let out = out.transform_to_num();
                            //out.from_view(&n);
                            // TODO: implement pow
                            //out.pow(&e, state);

                            let out = out.transform_to_pow();
                            out.from_base_and_exp(new_base.to_view(), new_exp.to_view());
                        } else {
                            let out = out.transform_to_pow();
                            out.from_base_and_exp(new_base.to_view(), new_exp.to_view());
                        }
                    } else {
                        let out = out.transform_to_pow();
                        out.from_base_and_exp(new_base.to_view(), new_exp.to_view());
                    }
                } else {
                    let pp = out.transform_to_pow();
                    pp.from_view(p);
                    pp.set_dirty(false);
                }
            }
            AtomView::Add(a) => {
                let mut atom_test_buf: SmallVec<[BufferHandle<OwnedAtom<P>>; 20]> = SmallVec::new();

                let mut it = a.into_iter();
                while let Some(a) = it.next() {
                    let mut handle = workspace.new_atom();
                    let new_at = handle.get_buf_mut();

                    if a.is_dirty() {
                        a.normalize(workspace, state, new_at);
                    } else {
                        new_at.from_view(&a);
                    }

                    atom_test_buf.push(handle);
                }

                atom_test_buf.sort_by(|a, b| {
                    a.get_buf()
                        .to_view()
                        .partial_cmp(&b.get_buf().to_view())
                        .unwrap()
                });

                let out = out.transform_to_add();

                if !atom_test_buf.is_empty() {
                    let mut last_buf = atom_test_buf.remove(0);

                    let mut handle = workspace.new_atom();
                    let helper = handle.get_buf_mut();

                    for mut cur_buf in atom_test_buf.drain(..) {
                        if !last_buf
                            .get_buf_mut()
                            .merge_terms(cur_buf.get_buf_mut(), helper, state)
                        {
                            // we are done merging
                            {
                                let v = last_buf.get_buf().to_view();
                                if let AtomView::Num(n) = v {
                                    if !n.is_zero() {
                                        out.extend(last_buf.get_buf().to_view());
                                    }
                                } else {
                                    out.extend(last_buf.get_buf().to_view());
                                }
                            }
                            last_buf = cur_buf;
                        }
                    }

                    out.extend(last_buf.get_buf().to_view());
                }
            }
        }
    }
}
