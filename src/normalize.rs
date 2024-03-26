use std::{cmp::Ordering, ops::DerefMut};

use smallvec::SmallVec;

use crate::{
    coefficient::{Coefficient, CoefficientView},
    domains::{integer::Z, rational::Q},
    poly::Variable,
    representations::{Atom, AtomView, Fun, Symbol},
    state::{RecycledAtom, State, Workspace},
};

impl<'a> AtomView<'a> {
    /// Compare two atoms.
    pub fn cmp(&self, other: &AtomView<'_>) -> Ordering {
        if self == other {
            // equality comparison is a fast check
            return Ordering::Equal;
        }

        match (&self, other) {
            (AtomView::Num(n1), AtomView::Num(n2)) => n1.get_coeff_view().cmp(&n2.get_coeff_view()),
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
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
                let name_comp = f1.get_symbol().cmp(&f2.get_symbol());
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
    pub(crate) fn cmp_factors(&self, other: &AtomView<'_>) -> Ordering {
        match (&self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,

            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
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
                let name_comp = f1.get_symbol().cmp(&f2.get_symbol());
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
    pub(crate) fn cmp_terms(&self, other: &AtomView<'_>) -> Ordering {
        debug_assert!(!matches!(self, AtomView::Add(_)));
        debug_assert!(!matches!(other, AtomView::Add(_)));
        match (self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Greater,
            (_, AtomView::Num(_)) => Ordering::Less,

            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol().cmp(&v2.get_symbol()),
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
                let name_comp = f1.get_symbol().cmp(&f2.get_symbol());
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

impl Atom {
    /// Merge two factors if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    fn merge_factors(
        &mut self,
        other: &mut Self,
        helper: &mut Self,

        workspace: &Workspace,
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
                        let new_exp = helper.to_num(n.get_coeff_view() + n2.get_coeff_view());

                        if new_exp.to_num_view().is_zero() {
                            self.to_num(1.into());
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
                let mut helper2 = workspace.new_atom();
                helper.as_view().normalize(workspace, &mut helper2);
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
                    let new_exp = n.get_coeff_view() + 1;

                    if new_exp.is_zero() {
                        self.to_num(1.into());
                    } else if new_exp == 1.into() {
                        self.set_from_view(&base);
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(base, AtomView::Num(num.to_num_view()));
                    }
                } else {
                    self.to_num(1.into());

                    let new_exp = helper.to_add();
                    new_exp.extend(self.as_view());
                    new_exp.extend(exp);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize(workspace, &mut helper2);
                    self.to_pow(base, helper2.as_view());
                }

                return true;
            } else {
                return false;
            }
        }

        // simplify num1 * num2
        if let Atom::Num(n1) = self {
            if let Atom::Num(n2) = other {
                n1.mul(&n2.to_num_view());
                return true;
            } else {
                return false;
            }
        }

        // x * x => x^2
        if self.as_view() == other.as_view() {
            if let AtomView::Var(v) = self.as_view() {
                if v.get_symbol() == State::I {
                    self.to_num((-1).into());
                    return true;
                }
            }

            // add powers
            let exp = other.to_num(2.into());
            helper.to_pow(self.as_view(), AtomView::Num(exp.to_num_view()));

            // overwrite self with the new power view
            std::mem::swap(self, helper);

            return true;
        }

        false
    }

    /// Merge two terms if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    pub fn merge_terms(&mut self, other: AtomView, helper: &mut Self) -> bool {
        if let Atom::Num(n1) = self {
            if let AtomView::Num(n2) = other {
                n1.add(&n2);
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

            if let AtomView::Mul(m2) = other {
                let slice2 = m2.to_slice();
                let last_elem2 = slice2.get(slice2.len() - 1);

                let non_coeff2 = if let AtomView::Num(_) = &last_elem2 {
                    slice2.get_subslice(0..slice2.len() - 1)
                } else {
                    m2.to_slice()
                };

                if non_coeff1.eq(&non_coeff2) {
                    // TODO: not correct for finite fields!
                    let num = if let AtomView::Num(n) = &last_elem {
                        n.get_coeff_view()
                    } else {
                        CoefficientView::Natural(1, 1)
                    };

                    let new_coeff = if let AtomView::Num(n) = &last_elem2 {
                        num + n.get_coeff_view()
                    } else {
                        num + 1
                    };

                    let len = slice.len();

                    if new_coeff == 1.into() {
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
                            m.set_normalized(true);
                        }

                        return true;
                    }

                    if new_coeff.is_zero() {
                        self.to_num(new_coeff);
                        return true;
                    }

                    let on = helper.to_num(new_coeff);

                    if has_coeff {
                        m.replace_last(on.to_num_view().as_view());
                    } else {
                        m.extend(on.to_num_view().as_view());
                    }

                    return true;
                }
            } else {
                if non_coeff1.len() != 1 || other != slice.get(0) {
                    return false;
                }

                let new_coeff = if let AtomView::Num(n) = &last_elem {
                    n.get_coeff_view() + 1
                } else {
                    return false;
                };

                assert!(new_coeff != 1.into());
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }

                let on = helper.to_num(new_coeff);

                m.replace_last(on.to_num_view().as_view());

                return true;
            }
        } else if let AtomView::Mul(m) = other {
            let slice = m.to_slice();

            if slice.len() != 2 {
                return false; // no match
            }

            let last_elem = slice.get(slice.len() - 1);

            if self.as_view() == slice.get(0) {
                let (new_coeff, has_num) = if let AtomView::Num(n) = &last_elem {
                    (n.get_coeff_view() + 1, true)
                } else {
                    return false; // last elem is not a coefficient
                };

                assert!(new_coeff != 1.into());
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }

                let on = helper.to_num(new_coeff);

                other.clone_into(self);

                if let Atom::Mul(m) = self {
                    if has_num {
                        m.replace_last(on.to_num_view().as_view());
                    } else {
                        m.extend(on.to_num_view().as_view());
                    }
                }

                return true;
            }
        } else if self.as_view() == other {
            let mul = helper.to_mul();
            mul.extend(self.as_view());
            self.to_num((2, 1).into());
            mul.extend(self.as_view());
            mul.set_has_coefficient(true);
            mul.set_normalized(true);

            std::mem::swap(self, helper);
            return true;
        };

        false
    }
}

impl<'a> AtomView<'a> {
    #[inline(always)]
    pub fn needs_normalization(&self) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => false,
            AtomView::Fun(f) => !f.is_normalized(),
            AtomView::Pow(p) => !p.is_normalized(),
            AtomView::Mul(m) => !m.is_normalized(),
            AtomView::Add(a) => !a.is_normalized(),
        }
    }

    /// Normalize an atom.
    pub fn normalize(&self, workspace: &Workspace, out: &mut Atom) {
        if !self.needs_normalization() {
            out.set_from_view(self);
            return;
        }

        match self {
            AtomView::Mul(t) => {
                let mut atom_test_buf: SmallVec<[_; 20]> = SmallVec::new();

                for a in t.iter() {
                    let mut handle = workspace.new_atom();

                    if a.needs_normalization() {
                        a.normalize(workspace, &mut handle);
                    } else {
                        handle.set_from_view(&a);
                    }

                    if let Atom::Mul(mul) = handle.deref_mut() {
                        for c in mul.to_mul_view().iter() {
                            // TODO: remove this copy
                            let mut handle = workspace.new_atom();
                            handle.set_from_view(&c);

                            if let AtomView::Num(n) = c {
                                if n.is_one() {
                                    continue;
                                }

                                if n.is_zero() {
                                    out.to_num(Coefficient::zero());
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
                                out.to_num(Coefficient::zero());
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

                    let mut tmp = workspace.new_atom();
                    let mut cur_len = 0;

                    atom_test_buf.reverse();
                    while let Some(mut cur_buf) = atom_test_buf.pop() {
                        if !last_buf.merge_factors(&mut cur_buf, &mut tmp, workspace) {
                            // we are done merging
                            {
                                let v = last_buf.as_view();
                                if let AtomView::Num(n) = v {
                                    if !n.is_one() {
                                        // the number is not in the final position, which only happens when i*i merges to -1
                                        // add it to the first position in the reversed buffer
                                        atom_test_buf.insert(0, last_buf);
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
                            if !n.is_one() {
                                out_mul.extend(v);
                                out_mul.set_has_coefficient(true);
                                out_mul.set_normalized(true);
                            } else if cur_len == 1 {
                                // downgrade
                                last_buf.set_from_view(&out_mul.to_mul_view().to_slice().get(0));
                                out.set_from_view(&last_buf.as_view());
                            }
                        } else {
                            out_mul.extend(v);
                            out_mul.set_normalized(true);
                        }
                    }
                } else {
                    out.to_num(1.into());
                }
            }
            AtomView::Num(n) => {
                let normalized_num = n.get_coeff_view().normalize();
                out.to_num(normalized_num);
            }
            AtomView::Var(_) => {
                self.clone_into(out);
            }
            AtomView::Fun(f) => {
                let id = f.get_symbol();
                let out_f = out.to_fun(id);

                /// Add an argument `a` to `f` and flatten nested `arg`s.
                #[inline(always)]
                fn add_arg(f: &mut Fun, a: AtomView) {
                    if let AtomView::Fun(fa) = a {
                        if fa.get_symbol() == State::ARG {
                            // flatten f(arg(...)) = f(...)
                            for aa in fa.iter() {
                                f.add_arg(aa);
                            }

                            return;
                        }
                    }

                    f.add_arg(a);
                }

                /// Take Cartesian product of arguments
                #[inline(always)]
                fn cartesian_product<'b>(
                    workspace: &Workspace,
                    list: &[Vec<AtomView<'b>>],
                    fun_name: Symbol,
                    cur: &mut Vec<AtomView<'b>>,
                    acc: &mut Vec<RecycledAtom>,
                ) {
                    if list.is_empty() {
                        let mut h = workspace.new_atom();
                        let f = h.to_fun(fun_name);
                        for a in cur.iter() {
                            add_arg(f, *a);
                        }
                        acc.push(h);
                        return;
                    }

                    for a in &list[0] {
                        cur.push(*a);
                        cartesian_product(workspace, &list[1..], fun_name, cur, acc);
                        cur.pop();
                    }
                }

                let mut handle = workspace.new_atom();
                for a in f.iter() {
                    if a.needs_normalization() {
                        a.normalize(workspace, &mut handle);
                        add_arg(out_f, handle.as_view());
                    } else {
                        add_arg(out_f, a);
                    }
                }

                out_f.set_normalized(true);

                if [State::COS, State::SIN, State::EXP, State::LOG].contains(&id)
                    && out_f.to_fun_view().get_nargs() == 1
                {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if let AtomView::Num(n) = arg {
                        if n.is_zero() && id != State::LOG || n.is_one() && id == State::LOG {
                            if id == State::COS || id == State::EXP {
                                let buffer = workspace.new_num(Coefficient::one());
                                out.set_from_view(&buffer.as_view());
                                return;
                            } else if id == State::SIN || id == State::LOG {
                                let buffer = workspace.new_num(Coefficient::zero());
                                out.set_from_view(&buffer.as_view());
                                return;
                            }
                        }
                    }
                }

                // try to turn the argument into a number
                if id == State::COEFF && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if let AtomView::Num(_) = arg {
                        let mut buffer = workspace.new_atom();
                        buffer.set_from_view(&arg);
                        out.set_from_view(&buffer.as_view());
                        return;
                    } else {
                        let r = arg.to_rational_polynomial(&Q, &Z, None);

                        // disallow wildcards as variables
                        if r.numerator.get_vars_ref().iter().all(|v| {
                            if let Variable::Symbol(v) = v {
                                v.get_wildcard_level() == 0
                            } else {
                                false
                            }
                        }) {
                            out.to_num(Coefficient::RationalPolynomial(r));
                            return;
                        }
                    }
                }

                if id.is_linear() {
                    // linearize sums
                    if out_f
                        .to_fun_view()
                        .iter()
                        .any(|a| matches!(a, AtomView::Add(_)))
                    {
                        let mut arg_buf = Vec::with_capacity(out_f.to_fun_view().get_nargs());

                        for a in out_f.to_fun_view().iter() {
                            let mut vec = vec![];
                            if let AtomView::Add(aa) = a {
                                for a in aa.iter() {
                                    vec.push(a);
                                }
                            } else {
                                vec.push(a);
                            }
                            arg_buf.push(vec);
                        }

                        let mut acc = Vec::new();
                        cartesian_product(workspace, &arg_buf, id, &mut vec![], &mut acc);

                        let mut add_h = workspace.new_atom();
                        let add = add_h.to_add();

                        let mut h = workspace.new_atom();
                        for a in acc {
                            a.as_view().normalize(workspace, &mut h);
                            add.extend(h.as_view());
                        }

                        add_h.as_view().normalize(workspace, out);
                        return;
                    }

                    // linearize products
                    if out_f.to_fun_view().iter().any(|a| {
                        if let AtomView::Mul(m) = a {
                            m.has_coefficient()
                        } else {
                            false
                        }
                    }) {
                        let mut new_term = workspace.new_atom();
                        let t = new_term.to_mul();
                        let mut new_fun = workspace.new_atom();
                        let nf = new_fun.to_fun(id);
                        let mut coeff: Coefficient = 1.into();
                        for a in out_f.to_fun_view().iter() {
                            if let AtomView::Mul(m) = a {
                                if m.has_coefficient() {
                                    let mut stripped = workspace.new_atom();
                                    let mul = stripped.to_mul();

                                    for a in m.iter() {
                                        if let AtomView::Num(n) = a {
                                            coeff = coeff * n.get_coeff_view().to_owned();
                                        } else {
                                            mul.extend(a);
                                        }
                                    }

                                    nf.add_arg(stripped.as_view());
                                } else {
                                    nf.add_arg(a);
                                }
                            } else {
                                nf.add_arg(a);
                            }
                        }

                        t.extend(new_fun.as_view());
                        t.extend(workspace.new_num(coeff).as_view());
                        t.as_view().normalize(workspace, out);
                        return;
                    }
                }

                if id.is_symmetric() || id.is_antisymmetric() {
                    let mut arg_buf: SmallVec<[(usize, _); 20]> = SmallVec::new();

                    for (i, a) in out_f.to_fun_view().iter().enumerate() {
                        let mut handle = workspace.new_atom();
                        handle.set_from_view(&a);
                        arg_buf.push((i, handle));
                    }

                    arg_buf.sort_by(|a, b| a.1.as_view().cmp(&b.1.as_view()));

                    if id.is_antisymmetric() {
                        if arg_buf
                            .windows(2)
                            .any(|w| w[0].1.as_view() == w[1].1.as_view())
                        {
                            out.to_num(Coefficient::zero());
                            return;
                        }

                        // find the number of swaps needed to sort the arguments
                        let mut order: SmallVec<[usize; 20]> = (0..arg_buf.len())
                            .map(|i| arg_buf.iter().position(|(j, _)| *j == i).unwrap())
                            .collect();
                        let mut swaps = 0;
                        for i in 0..order.len() {
                            let pos = order[i..].iter().position(|&x| x == i).unwrap();
                            order.copy_within(i..i + pos, i + 1);
                            swaps += pos;
                        }

                        if swaps % 2 == 1 {
                            let mut handle = workspace.new_atom();
                            let out_f = handle.to_fun(id);

                            for (_, a) in arg_buf {
                                out_f.add_arg(a.as_view());
                            }

                            out_f.set_normalized(true);

                            let m = out.to_mul();
                            m.extend(handle.as_view());
                            handle.to_num((-1).into());
                            m.extend(handle.as_view());
                            m.set_normalized(true);

                            return;
                        }
                    }

                    let out_f = out.to_fun(id);
                    for (_, a) in arg_buf {
                        out_f.add_arg(a.as_view());
                    }

                    out_f.set_normalized(true);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_handle = workspace.new_atom();
                let mut exp_handle = workspace.new_atom();

                if base.needs_normalization() {
                    base.normalize(workspace, &mut base_handle);
                } else {
                    // TODO: prevent copy
                    base_handle.set_from_view(&base);
                };

                if exp.needs_normalization() {
                    exp.normalize(workspace, &mut exp_handle);
                } else {
                    // TODO: prevent copy
                    exp_handle.set_from_view(&exp);
                };

                'pow_simplify: {
                    if let AtomView::Num(e) = exp_handle.as_view() {
                        let exp_num = e.get_coeff_view();
                        if exp_num == CoefficientView::Natural(0, 1) {
                            // x^0 = 1
                            out.to_num(1.into());
                            break 'pow_simplify;
                        } else if exp_num == CoefficientView::Natural(1, 1) {
                            // remove power of 1
                            out.set_from_view(&base_handle.as_view());
                            break 'pow_simplify;
                        } else if let AtomView::Num(n) = base_handle.as_view() {
                            // simplify a number to a numerical power
                            let (new_base_num, new_exp_num) = n.get_coeff_view().pow(&exp_num);

                            if new_exp_num == 1.into() {
                                out.to_num(new_base_num);
                                break 'pow_simplify;
                            }

                            base_handle.to_num(new_base_num);
                            exp_handle.to_num(new_exp_num);
                        } else if let AtomView::Var(v) = base_handle.as_view() {
                            if v.get_symbol() == State::I {
                                if let CoefficientView::Natural(n, d) = exp_num {
                                    let mut new_base = workspace.new_atom();

                                    // the case n < 0 is handled automagically
                                    if n % 2 == 0 {
                                        if n % 4 == 0 {
                                            new_base.to_num(1.into());
                                        } else {
                                            new_base.to_num((-1).into());
                                        }
                                    } else if (n - 1) % 4 == 0 {
                                        new_base.set_from_view(&base_handle.as_view());
                                    } else {
                                        let n = new_base.to_mul();
                                        n.extend(base_handle.as_view());
                                        let mut helper = workspace.new_atom();
                                        helper.to_num((-1).into());
                                        n.extend(helper.as_view());
                                        new_base.as_view().normalize(workspace, &mut helper);
                                        std::mem::swap(&mut new_base, &mut helper);
                                    }

                                    if d == 1 {
                                        out.set_from_view(&new_base.as_view());
                                    } else {
                                        let mut new_exp = workspace.new_atom();
                                        new_exp.to_num((1i64, d).into());
                                        out.to_pow(new_base.as_view(), new_exp.as_view());
                                    }

                                    break 'pow_simplify;
                                }
                            }
                        } else if let AtomView::Pow(p_base) = base_handle.as_view() {
                            // simplify x^2^3
                            let (p_base_base, p_base_exp) = p_base.get_base_exp();
                            if let AtomView::Num(n) = p_base_exp {
                                let new_exp = n.get_coeff_view() * exp_num;

                                if new_exp == 1.into() {
                                    out.set_from_view(&p_base_base);
                                    break 'pow_simplify;
                                }

                                exp_handle.to_num(new_exp);

                                let p = out.to_pow(p_base_base, exp_handle.as_view());
                                p.set_normalized(true);

                                break 'pow_simplify;
                            }
                        } else if let AtomView::Mul(_) = base_handle.as_view() {
                            // TODO: turn (x*y)^2 into x^2*y^2?
                            // for now, expand() needs to be used
                        }
                    }

                    out.to_pow(base_handle.as_view(), exp_handle.as_view());
                }

                out.set_normalized(true);
            }
            AtomView::Add(a) => {
                let mut new_sum = workspace.new_atom();
                let ns = new_sum.to_add();

                let mut atom_sort_buf: SmallVec<[_; 20]> = SmallVec::new();

                let mut norm_arg = workspace.new_atom();
                for a in a.iter() {
                    let r = if a.needs_normalization() {
                        // TODO: if a is a nested addition, prevent a sort
                        a.normalize(workspace, &mut norm_arg);
                        norm_arg.as_view()
                    } else {
                        a
                    };

                    if let AtomView::Add(new_add) = r {
                        for c in new_add.iter() {
                            if let AtomView::Num(n) = c {
                                if n.is_zero() {
                                    continue;
                                }
                            }

                            ns.extend(r);
                        }
                    } else {
                        if let AtomView::Num(n) = r {
                            if n.is_zero() {
                                continue;
                            }
                        }

                        ns.extend(r); // TODO: prevent copy?
                    }
                }

                for x in ns.to_add_view().iter() {
                    atom_sort_buf.push(x);
                }

                atom_sort_buf.sort_by(|a, b| a.cmp_terms(b));

                if atom_sort_buf.is_empty() {
                    out.to_num(Coefficient::zero());
                    return;
                }
                let out_add = out.to_add();

                let mut last_buf = workspace.new_atom();
                last_buf.set_from_view(&atom_sort_buf[0]);

                let mut helper = workspace.new_atom();
                let mut cur_len = 0;

                for cur in atom_sort_buf.iter().skip(1) {
                    if !last_buf.merge_terms(*cur, &mut helper) {
                        // we are done merging
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                out_add.extend(v);
                                cur_len += 1;
                            }
                        } else {
                            out_add.extend(v);
                            cur_len += 1;
                        }

                        // TODO: prevent this copy, as it occurs on every non-merge
                        cur.clone_into(&mut last_buf);
                    }
                }

                if cur_len == 0 {
                    out.set_from_view(&last_buf.as_view());
                } else {
                    let v = last_buf.as_view();
                    if let AtomView::Num(n) = v {
                        if !n.is_zero() {
                            out_add.extend(v);
                            out_add.set_normalized(true);
                        } else if cur_len == 1 {
                            // downgrade
                            last_buf.set_from_view(&out_add.to_add_view().to_slice().get(0));
                            out.set_from_view(&last_buf.as_view());
                        }
                    } else {
                        out_add.extend(v);
                        out_add.set_normalized(true);
                    }
                }
            }
        }
    }
}
