use std::cmp::Ordering;

use smallvec::SmallVec;

use crate::{
    representations::{
        number::{BorrowedNumber, Number},
        Add, Atom, AtomSet, AtomView, Fun, ListSlice, Mul, Num, OwnedAdd, OwnedFun, OwnedMul,
        OwnedNum, OwnedPow, OwnedVar, Pow, Var,
    },
    state::{BufferHandle, FunctionAttribute::Symmetric, State, Workspace},
};

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Compare two atoms.
    pub fn cmp(&self, other: &AtomView<'_, P>) -> Ordering {
        if self == other {
            // equality comparison is a fast check
            return Ordering::Equal;
        }

        match (&self, other) {
            (AtomView::Num(n1), AtomView::Num(n2)) => {
                n1.get_number_view().cmp(&n2.get_number_view())
            }
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_name().cmp(&v2.get_name()),
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                let (b1, e1) = p1.get_base_exp();
                let (b2, e2) = p2.get_base_exp();
                b1.cmp(&b2).then_with(|| e1.cmp(&e2))
            }
            (_, AtomView::Pow(_)) => Ordering::Greater,
            (AtomView::Pow(_), _) => Ordering::Less,
            (AtomView::Mul(m1), AtomView::Mul(m2)) => {
                let it1 = m1.to_slice();
                let it2 = m2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
            (AtomView::Mul(_), _) => Ordering::Less,
            (_, AtomView::Mul(_)) => Ordering::Greater,
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_name().cmp(&f2.get_name());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }

                if cfg!(feature = "full_fn_cmp") {
                    let len_cmp = f1.get_nargs().cmp(&f2.get_nargs());
                    if len_cmp != Ordering::Equal {
                        return len_cmp;
                    }

                    for (arg1, arg2) in f1.iter().zip(f2.iter()) {
                        let argcmp = arg1.cmp(&arg2);
                        if argcmp != Ordering::Equal {
                            return argcmp;
                        }
                    }

                    Ordering::Equal
                } else {
                    f1.fast_cmp(*f2)
                }
            }
        }
    }

    /// Compare factors in a term. `x` and `x^2` are placed next to each other by sorting a power based on the base only.
    pub(crate) fn cmp_factors(&self, other: &AtomView<'_, P>) -> Ordering {
        match (&self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,

            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_name().cmp(&v2.get_name()),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                // TODO: inline partial_cmp call by creating an inlined version
                p1.get_base().cmp(&p2.get_base())
            }
            (_, AtomView::Pow(p2)) => {
                let base = p2.get_base();
                self.cmp(&base).then(Ordering::Less) // sort x^2*x -> x*x^2
            }
            (AtomView::Pow(p1), _) => {
                let base = p1.get_base();
                base.cmp(other).then(Ordering::Greater)
            }
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,

            (AtomView::Mul(_), _) | (_, AtomView::Mul(_)) => {
                unreachable!("Cannot have a submul in a factor");
            }
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_name().cmp(&f2.get_name());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }

                if cfg!(feature = "full_fn_cmp") {
                    let len_cmp = f1.get_nargs().cmp(&f2.get_nargs());
                    if len_cmp != Ordering::Equal {
                        return len_cmp;
                    }

                    for (arg1, arg2) in f1.iter().zip(f2.iter()) {
                        let argcmp = arg1.cmp(&arg2);
                        if argcmp != Ordering::Equal {
                            return argcmp;
                        }
                    }

                    Ordering::Equal
                } else {
                    f1.fast_cmp(*f2)
                }
            }
        }
    }

    /// Compare terms in an expression. `x` and `x*2` are placed next to each other.
    pub(crate) fn cmp_terms(&self, other: &AtomView<'_, P>) -> Ordering {
        debug_assert!(!matches!(self, AtomView::Add(_)));
        debug_assert!(!matches!(other, AtomView::Add(_)));
        match (self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,

            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_name().cmp(&v2.get_name()),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                let (b1, e1) = p1.get_base_exp();
                let (b2, e2) = p2.get_base_exp();
                b1.cmp(&b2).then_with(|| e1.cmp(&e2))
            }
            (AtomView::Mul(m1), AtomView::Mul(m2)) => {
                let actual_len1 = if m1.has_coefficient() {
                    m1.get_nargs() - 1
                } else {
                    m1.get_nargs()
                };

                let actual_len2 = if m2.has_coefficient() {
                    m2.get_nargs() - 1
                } else {
                    m2.get_nargs()
                };

                let len_cmp = actual_len1.cmp(&actual_len2);
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in m1.iter().zip(m2.iter()) {
                    if let AtomView::Num(_) = t1 {
                        break;
                    }
                    if let AtomView::Num(_) = t2 {
                        break;
                    }

                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
            (AtomView::Mul(m1), a2) => {
                if !m1.has_coefficient() || m1.get_nargs() != 2 {
                    return Ordering::Greater;
                }

                let it1 = m1.to_slice();
                it1.get(0).cmp(a2)
            }
            (a1, AtomView::Mul(m2)) => {
                if !m2.has_coefficient() || m2.get_nargs() != 2 {
                    return Ordering::Less;
                }

                let it2 = m2.to_slice();
                a1.cmp(&it2.get(0))
            }
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (_, AtomView::Pow(_)) => Ordering::Greater,
            (AtomView::Pow(_), _) => Ordering::Less,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_name().cmp(&f2.get_name());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }

                if cfg!(feature = "full_fn_cmp") {
                    let len_cmp = f1.get_nargs().cmp(&f2.get_nargs());
                    if len_cmp != Ordering::Equal {
                        return len_cmp;
                    }

                    for (arg1, arg2) in f1.iter().zip(f2.iter()) {
                        let argcmp = arg1.cmp(&arg2);
                        if argcmp != Ordering::Equal {
                            return argcmp;
                        }
                    }

                    Ordering::Equal
                } else {
                    f1.fast_cmp(*f2)
                }
            }
            (AtomView::Add(_), _) | (_, AtomView::Add(_)) => unreachable!("Cannot have nested add"),
        }
    }
}

impl<P: AtomSet> Atom<P> {
    /// Merge two factors if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    fn merge_factors(
        &mut self,
        other: &mut Self,
        helper: &mut Self,
        state: &State,
        workspace: &Workspace<P>,
    ) -> bool {
        // x^a * x^b = x^(a + b)
        if let Atom::Pow(p1) = self {
            if let Atom::Pow(p2) = other {
                let (base2, exp2) = p2.to_pow_view().get_base_exp();

                let (base1, exp1) = p1.to_pow_view().get_base_exp();

                if base1 != base2 {
                    return false;
                }

                if let AtomView::Num(n) = &exp1 {
                    if let AtomView::Num(n2) = &exp2 {
                        let new_exp = helper.to_num();
                        new_exp.set_from_view(n);
                        new_exp.add(n2, state);

                        if new_exp.to_num_view().is_zero() {
                            let num = self.to_num();
                            num.set_from_number(Number::Natural(1, 1));
                        } else if new_exp.to_num_view().is_one() {
                            self.set_from_view(&base2);
                        } else {
                            p1.set_from_base_and_exp(base2, helper.as_view());
                        }

                        return true;
                    }
                }

                let new_exp = helper.to_add();
                new_exp.extend(exp1);
                new_exp.extend(exp2);
                new_exp.set_dirty(true);
                let mut helper2 = workspace.new_atom();
                helper.as_view().normalize(workspace, state, &mut helper2);
                p1.set_from_base_and_exp(base2, helper2.as_view());
                return true;
            }
        }

        // x * x^n = x^(n+1)
        if let Atom::Pow(p) = other {
            let pv = p.to_pow_view();
            let (base, exp) = pv.get_base_exp();

            if self.as_view() == base {
                if let AtomView::Num(n) = &exp {
                    let num = helper.to_num();

                    let new_exp = n
                        .get_number_view()
                        .add(&BorrowedNumber::Natural(1, 1), state);

                    if new_exp.is_zero() {
                        let num = self.to_num();
                        num.set_from_number(Number::Natural(1, 1));
                    } else if Number::Natural(1, 1) == new_exp {
                        self.set_from_view(&base);
                    } else {
                        num.set_from_number(new_exp);
                        self.to_pow()
                            .set_from_base_and_exp(base, AtomView::Num(num.to_num_view()));
                    }
                } else {
                    self.to_num().set_from_number(Number::Natural(1, 1));

                    let new_exp = helper.to_add();
                    new_exp.extend(self.as_view());
                    new_exp.extend(exp);
                    new_exp.set_dirty(true);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize(workspace, state, &mut helper2);
                    self.to_pow().set_from_base_and_exp(base, helper2.as_view());
                }

                return true;
            } else {
                return false;
            }
        }

        // simplify num1 * num2
        if let Atom::Num(n1) = self {
            if let Atom::Num(n2) = other {
                n1.mul(&n2.to_num_view(), state);
                return true;
            } else {
                return false;
            }
        }

        // x * x => x^2
        if self.as_view() == other.as_view() {
            if let AtomView::Var(v) = self.as_view() {
                if v.get_name() == State::I {
                    self.to_num().set_from_number(Number::Natural(-1, 1));
                    return true;
                }
            }

            // add powers
            let exp = other.to_num();
            exp.set_from_number(Number::Natural(2, 1));

            //let mut a = workspace.get_atom_test_buf();
            let new_pow = helper.to_pow();
            new_pow.set_from_base_and_exp(self.as_view(), AtomView::Num(exp.to_num_view()));

            // overwrite self with the new power view
            let pow_handle = self.to_pow();
            pow_handle.set_from_view(&new_pow.to_pow_view());

            return true;
        }

        false
    }

    /// Merge two terms if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    pub fn merge_terms(&mut self, other: &mut Self, helper: &mut Self, state: &State) -> bool {
        if let Atom::Num(n1) = self {
            if let Atom::Num(n2) = other {
                n1.add(&n2.to_num_view(), state);
                return true;
            } else {
                return false;
            }
        }

        // compare the non-coefficient part of terms and add the coefficients if they are the same
        if let Atom::Mul(m) = self {
            let slice = m.to_mul_view().to_slice();

            let last_elem = slice.get(slice.len() - 1);

            let (non_coeff1, has_coeff) = if let AtomView::Num(_) = &last_elem {
                (slice.get_subslice(0..slice.len() - 1), true)
            } else {
                (m.to_mul_view().to_slice(), false)
            };

            if let Atom::Mul(m2) = other {
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

                    let len = slice.len();

                    // help the borrow checker by dropping all references
                    drop(non_coeff1);
                    drop(slice);

                    if new_coeff == Number::Natural(1, 1) {
                        assert!(has_coeff);

                        if len == 2 {
                            // downgrade
                            self.set_from_view(&slice2.get(0));
                        } else {
                            // remove coefficient
                            let m = self.to_mul();
                            for a in non_coeff2.iter() {
                                m.extend(a);
                            }
                            m.set_has_coefficient(false);
                        }

                        return true;
                    }

                    drop(slice2);
                    drop(non_coeff2);

                    if new_coeff.is_zero() {
                        let num = self.to_num();
                        num.set_from_number(new_coeff);

                        return true;
                    }

                    let on = helper.to_num();
                    on.set_from_number(new_coeff);

                    if has_coeff {
                        m.replace_last(on.to_num_view().as_view());
                    } else {
                        m.extend(on.to_num_view().as_view());
                    }

                    return true;
                }
            } else {
                if non_coeff1.len() != 1 || other.as_view() != slice.get(0) {
                    return false;
                }

                let new_coeff = if let AtomView::Num(n) = &last_elem {
                    n.get_number_view()
                        .add(&BorrowedNumber::Natural(1, 1), state)
                } else {
                    return false;
                };

                // help the borrow checker by dropping all references
                drop(slice);
                drop(non_coeff1);

                assert!(new_coeff != Number::Natural(1, 1));
                if new_coeff.is_zero() {
                    let num = self.to_num();
                    num.set_from_number(new_coeff);

                    return true;
                }

                let on = helper.to_num();
                on.set_from_number(new_coeff);

                m.replace_last(on.to_num_view().as_view());

                return true;
            }
        } else if let Atom::Mul(m) = other {
            let slice = m.to_mul_view().to_slice();

            if slice.len() != 2 {
                return false; // no match
            }

            let last_elem = slice.get(slice.len() - 1);

            if self.as_view() == slice.get(0) {
                let (new_coeff, has_num) = if let AtomView::Num(n) = &last_elem {
                    (
                        n.get_number_view()
                            .add(&BorrowedNumber::Natural(1, 1), state),
                        true,
                    )
                } else {
                    return false; // last elem is not a coefficient
                };

                // help the borrow checker by dropping all references
                drop(slice);

                assert!(new_coeff != Number::Natural(1, 1));
                if new_coeff.is_zero() {
                    let num = self.to_num();
                    num.set_from_number(new_coeff);

                    return true;
                }

                let on = helper.to_num();
                on.set_from_number(new_coeff);

                if has_num {
                    m.replace_last(on.to_num_view().as_view());
                } else {
                    m.extend(on.to_num_view().as_view());
                }

                std::mem::swap(self, other);

                return true;
            }
        } else if self.as_view() == other.as_view() {
            let mul = helper.to_mul();

            let num = other.to_num();
            num.set_from_number(Number::Natural(2, 1));

            mul.extend(self.as_view());
            mul.extend(other.as_view());
            mul.set_has_coefficient(true);

            std::mem::swap(self, helper);
            return true;
        };

        false
    }
}

impl<'a, P: AtomSet> AtomView<'a, P> {
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
    pub fn normalize(&self, workspace: &Workspace<P>, state: &State, out: &mut Atom<P>) {
        if !self.is_dirty() {
            out.set_from_view(self);
            return;
        }

        match self {
            AtomView::Mul(t) => {
                let mut atom_test_buf: SmallVec<[BufferHandle<Atom<P>>; 20]> = SmallVec::new();

                for a in t.iter() {
                    let mut handle = workspace.new_atom();
                    let new_at = handle.get_mut();

                    if a.is_dirty() {
                        a.normalize(workspace, state, new_at);
                    } else {
                        new_at.set_from_view(&a);
                    }

                    if let Atom::Mul(mul) = new_at {
                        for c in mul.to_mul_view().iter() {
                            // TODO: remove this copy
                            let mut handle = workspace.new_atom();
                            let child_copy = handle.get_mut();
                            child_copy.set_from_view(&c);

                            if let AtomView::Num(n) = c {
                                if n.is_one() {
                                    continue;
                                }

                                if n.is_zero() {
                                    let on = out.to_num();
                                    on.set_from_number(Number::Natural(0, 1));
                                    return;
                                }
                            }

                            atom_test_buf.push(handle);
                        }
                    } else {
                        if let AtomView::Num(n) = handle.as_view() {
                            if n.is_one() {
                                continue;
                            }

                            if n.is_zero() {
                                let on = out.to_num();
                                on.set_from_number(Number::Natural(0, 1));
                                return;
                            }
                        }

                        atom_test_buf.push(handle);
                    }
                }

                atom_test_buf.sort_by(|a, b| a.as_view().cmp_factors(&b.as_view()));

                if !atom_test_buf.is_empty() {
                    let out_mul = out.to_mul();

                    let mut last_buf = atom_test_buf.remove(0);

                    let mut handle = workspace.new_atom();
                    let helper = handle.get_mut();
                    let mut cur_len = 0;

                    for mut cur_buf in atom_test_buf.drain(..) {
                        if !last_buf.get_mut().merge_factors(
                            cur_buf.get_mut(),
                            helper,
                            state,
                            workspace,
                        ) {
                            // we are done merging
                            {
                                let v = last_buf.as_view();
                                if let AtomView::Num(n) = v {
                                    out_mul.set_has_coefficient(!n.is_one());
                                    if !n.is_one() {
                                        out_mul.extend(v);
                                        cur_len += 1;
                                    }
                                } else {
                                    out_mul.extend(v);
                                    cur_len += 1;
                                }
                            }
                            last_buf = cur_buf;
                        }
                    }

                    if cur_len == 0 {
                        out.set_from_view(&last_buf.as_view());
                    } else {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            out_mul.set_has_coefficient(!n.is_one());
                            if !n.is_one() {
                                out_mul.extend(v);
                            } else if cur_len == 1 {
                                // downgrade
                                last_buf.set_from_view(&out_mul.to_mul_view().to_slice().get(0));
                                out.set_from_view(&last_buf.as_view());
                            }
                        } else {
                            out_mul.extend(v);
                            out_mul.set_has_coefficient(false);
                        }
                    }
                } else {
                    let on = out.to_num();
                    on.set_from_number(Number::Natural(1, 1));
                }
            }
            AtomView::Num(n) => {
                let normalized_num = n.get_number_view().normalize();
                let nn = out.to_num();
                nn.set_from_number(normalized_num);
            }
            AtomView::Var(v) => {
                let vv = out.to_var();
                vv.set_from_view(v);
            }
            AtomView::Fun(f) => {
                let name = f.get_name();
                let out_f = out.to_fun();
                out_f.set_from_name(name);

                /// Add an argument `a` to `f` and flatten nested `arg`s.
                #[inline(always)]
                fn add_arg<P: AtomSet>(f: &mut P::OF, a: AtomView<P>) {
                    if let AtomView::Fun(fa) = a {
                        if fa.get_name() == State::ARG {
                            // flatten f(arg(...)) = f(...)
                            for aa in fa.iter() {
                                f.add_arg(aa);
                            }

                            return;
                        }
                    }

                    f.add_arg(a);
                }

                let mut handle = workspace.new_atom();
                for a in f.iter() {
                    if a.is_dirty() {
                        a.normalize(workspace, state, &mut handle);
                        add_arg(out_f, handle.as_view());
                    } else {
                        add_arg(out_f, a);
                    }
                }

                if state.get_function_attributes(name).contains(&Symmetric) {
                    let mut arg_buf: SmallVec<[BufferHandle<Atom<P>>; 20]> = SmallVec::new();

                    for a in out_f.to_fun_view().iter() {
                        let mut handle = workspace.new_atom();
                        handle.set_from_view(&a);
                        arg_buf.push(handle);
                    }

                    arg_buf.sort_by(|a, b| a.as_view().cmp(&b.as_view()));

                    let out_f = out.to_fun();
                    out_f.set_from_name(name);
                    for a in arg_buf {
                        out_f.add_arg(a.as_view());
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_handle = workspace.new_atom();
                let mut exp_handle = workspace.new_atom();

                if base.is_dirty() {
                    base.normalize(workspace, state, &mut base_handle);
                } else {
                    // TODO: prevent copy
                    base_handle.get_mut().set_from_view(&base);
                };

                if exp.is_dirty() {
                    exp.normalize(workspace, state, &mut exp_handle);
                } else {
                    // TODO: prevent copy
                    exp_handle.get_mut().set_from_view(&exp);
                };

                'pow_simplify: {
                    if let AtomView::Num(e) = exp_handle.as_view() {
                        if let BorrowedNumber::Natural(0, 1) = &e.get_number_view() {
                            // x^0 = 1
                            let n = out.to_num();
                            n.set_from_number(Number::Natural(1, 1));
                            break 'pow_simplify;
                        } else if let BorrowedNumber::Natural(1, 1) = &e.get_number_view() {
                            // remove power of 1
                            out.set_from_view(&base_handle.as_view());
                            break 'pow_simplify;
                        } else if let AtomView::Num(n) = base_handle.as_view() {
                            // simplify a number to a numerical power
                            let (new_base_num, new_exp_num) =
                                n.get_number_view().pow(&e.get_number_view(), state);

                            if let Number::Natural(1, 1) = &new_exp_num {
                                let out = out.to_num();
                                out.set_from_number(new_base_num);
                                break 'pow_simplify;
                            }

                            let nb = base_handle.get_mut().to_num();
                            nb.set_from_number(new_base_num);

                            let ne = exp_handle.get_mut().to_num();
                            ne.set_from_number(new_exp_num);
                        } else if let AtomView::Var(v) = base_handle.as_view() {
                            if v.get_name() == State::I {
                                if let BorrowedNumber::Natural(n, d) = &e.get_number_view() {
                                    let mut new_base = workspace.new_atom();

                                    // the case n < 0 is handled automagically
                                    if *n % 2 == 0 {
                                        if *n % 4 == 0 {
                                            let n = new_base.to_num();
                                            n.set_from_number(Number::Natural(1, 1));
                                        } else {
                                            let n = new_base.to_num();
                                            n.set_from_number(Number::Natural(-1, 1));
                                        }
                                    } else if (*n - 1) % 4 == 0 {
                                        new_base.set_from_view(&base_handle.as_view());
                                    } else {
                                        let n = new_base.to_mul();
                                        n.extend(base_handle.as_view());
                                        let mut helper = workspace.new_atom();
                                        helper.to_num().set_from_number(Number::Natural(-1, 1));
                                        n.extend(helper.as_view());
                                        n.set_dirty(true);
                                        new_base.as_view().normalize(workspace, state, &mut helper);
                                        std::mem::swap(&mut new_base, &mut helper);
                                    }

                                    if *d == 1 {
                                        out.set_from_view(&new_base.as_view());
                                    } else {
                                        let mut new_exp = workspace.new_atom();
                                        new_exp.to_num().set_from_number(Number::Natural(1, *d));
                                        out.to_pow().set_from_base_and_exp(
                                            new_base.as_view(),
                                            new_exp.as_view(),
                                        );
                                    }

                                    break 'pow_simplify;
                                }
                            }
                        } else if let AtomView::Pow(p_base) = base_handle.as_view() {
                            // simplify x^2^3
                            let (p_base_base, p_base_exp) = p_base.get_base_exp();
                            if let AtomView::Num(n) = p_base_exp {
                                let new_exp = n.get_number_view().mul(&e.get_number_view(), state);

                                if let Number::Natural(1, 1) = &new_exp {
                                    out.set_from_view(&p_base_base);
                                    break 'pow_simplify;
                                }

                                let ne = exp_handle.get_mut().to_num();
                                ne.set_from_number(new_exp);

                                let out = out.to_pow();
                                out.set_from_base_and_exp(p_base_base, exp_handle.as_view());

                                break 'pow_simplify;
                            }
                        } else if let AtomView::Mul(_) = base_handle.as_view() {
                            // TODO: turn (x*y)^2 into x^2*y^2?
                            // for now, expand() needs to be used
                        }
                    }

                    let out = out.to_pow();
                    out.set_from_base_and_exp(base_handle.as_view(), exp_handle.as_view());
                }
            }
            AtomView::Add(a) => {
                let mut atom_test_buf: SmallVec<[BufferHandle<Atom<P>>; 20]> = SmallVec::new();

                for a in a.iter() {
                    let mut handle = workspace.new_atom();
                    let new_at = handle.get_mut();

                    if a.is_dirty() {
                        // TODO: if a is a nested addition, prevent a sort
                        a.normalize(workspace, state, new_at);
                    } else {
                        new_at.set_from_view(&a);
                    }

                    if let Atom::Add(new_add) = new_at {
                        for c in new_add.to_add_view().iter() {
                            // TODO: remove this copy
                            let mut handle = workspace.new_atom();
                            let child_copy = handle.get_mut();
                            child_copy.set_from_view(&c);

                            if let AtomView::Num(n) = c {
                                if n.is_zero() {
                                    continue;
                                }
                            }

                            atom_test_buf.push(handle);
                        }
                    } else {
                        if let AtomView::Num(n) = handle.as_view() {
                            if n.is_zero() {
                                continue;
                            }
                        }
                        atom_test_buf.push(handle);
                    }
                }

                atom_test_buf.sort_by(|a, b| a.as_view().cmp_terms(&b.as_view()));

                if !atom_test_buf.is_empty() {
                    let out_add = out.to_add();

                    let mut last_buf = atom_test_buf.remove(0);

                    let mut handle = workspace.new_atom();
                    let helper = handle.get_mut();
                    let mut cur_len = 0;

                    for mut cur_buf in atom_test_buf.drain(..) {
                        if !last_buf
                            .get_mut()
                            .merge_terms(cur_buf.get_mut(), helper, state)
                        {
                            // we are done merging
                            {
                                let v = last_buf.as_view();
                                if let AtomView::Num(n) = v {
                                    if !n.is_zero() {
                                        out_add.extend(last_buf.as_view());
                                        cur_len += 1;
                                    }
                                } else {
                                    out_add.extend(last_buf.as_view());
                                    cur_len += 1;
                                }
                            }
                            last_buf = cur_buf;
                        }
                    }

                    if cur_len == 0 {
                        out.set_from_view(&last_buf.as_view());
                    } else {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                out_add.extend(v);
                            } else if cur_len == 1 {
                                // downgrade
                                last_buf.set_from_view(&out_add.to_add_view().to_slice().get(0));
                                out.set_from_view(&last_buf.as_view());
                            }
                        } else {
                            out_add.extend(v);
                        }
                    }
                } else {
                    let on = out.to_num();
                    on.set_from_number(Number::Natural(0, 1));
                }
            }
        }
    }
}
