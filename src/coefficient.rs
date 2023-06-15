use smallvec::{smallvec, SmallVec};

use crate::{
    poly::{polynomial::MultivariatePolynomial, INLINED_EXPONENTS},
    representations::{
        number::{BorrowedNumber, Number},
        Add, Atom, AtomView, Identifier, Mul, Num, OwnedAdd, OwnedAtom, OwnedMul, OwnedNum,
        OwnedPow, Pow, Var,
    },
    rings::{
        integer::{Integer, IntegerRing},
        rational_polynomial::RationalPolynomial,
    },
    state::{ResettableBuffer, State, Workspace},
};

impl<'a, P: Atom> AtomView<'a, P> {
    pub fn set_coefficient_ring(
        &self,
        vars: &[Identifier],
        state: &State,
        workspace: &Workspace<P>,
        out: &mut OwnedAtom<P>,
    ) -> bool {
        match self {
            AtomView::Num(n) => {
                if let BorrowedNumber::RationalPolynomial(r) = n.get_number_view() {
                    let old_var_map = r.get_var_map().unwrap();
                    if old_var_map != vars {
                        if old_var_map.iter().all(|x| vars.contains(x)) {
                            // upgrade the polynomial if no variables got removed
                            let mut r = r.clone();
                            let order: SmallVec<[Option<usize>; INLINED_EXPONENTS]> = vars
                                .iter()
                                .map(|x| old_var_map.iter().position(|xx| xx == x))
                                .collect();

                            r.numerator = r.numerator.rearrange_with_growth(&order);
                            r.denominator = r.denominator.rearrange_with_growth(&order);
                            r.numerator.var_map = Some(vars.into());
                            r.denominator.var_map = Some(vars.into());
                            out.transform_to_num()
                                .from_number(Number::RationalPolynomial(r));
                            true
                        } else {
                            let mut n1 = workspace.new_atom();
                            n1.from_polynomial(workspace, state, &r.numerator);

                            let mut n1_conv = workspace.new_atom();
                            n1.to_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n1_conv);

                            let mut n2 = workspace.new_atom();
                            n2.from_polynomial(workspace, state, &r.denominator);

                            let mut n2_conv = workspace.new_atom();
                            n2.to_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n2_conv);

                            // create n1/n2
                            let mut n3 = workspace.new_atom();
                            let mut exp = workspace.new_atom();
                            exp.transform_to_num().from_number(Number::Natural(-1, 1));
                            let n3p = n3.transform_to_pow();
                            n3p.from_base_and_exp(n2_conv.to_view(), exp.to_view());
                            n3p.set_dirty(true);

                            let mut m = workspace.new_atom();
                            let mm = m.transform_to_mul();
                            mm.extend(n1_conv.to_view());
                            mm.extend(n3.to_view());
                            mm.set_dirty(true);
                            m.to_view().normalize(workspace, state, out);
                            true
                        }
                    } else {
                        out.from_view(self);
                        false
                    }
                } else {
                    out.from_view(self);
                    false
                }
            }
            AtomView::Var(v) => {
                let id = v.get_name();
                if vars.contains(&id) {
                    // change variable into coefficient
                    let mut poly = MultivariatePolynomial::new(
                        vars.len(),
                        IntegerRing::new(),
                        None,
                        Some(vars),
                    );
                    let mut e: SmallVec<[u16; INLINED_EXPONENTS]> = smallvec![0; vars.len()];
                    e[vars.iter().position(|x| *x == id).unwrap()] = 1;
                    poly.append_monomial(Integer::one(), &e);
                    let den = MultivariatePolynomial::new_from_constant(&poly, Integer::one());

                    out.transform_to_num()
                        .from_number(Number::RationalPolynomial(RationalPolynomial {
                            numerator: poly,
                            denominator: den,
                        }));
                    true
                } else {
                    out.from_view(self);
                    false
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = workspace.new_atom();
                if base.set_coefficient_ring(vars, state, workspace, &mut nb) {
                    let mut o = workspace.new_atom();
                    let pow = o.transform_to_pow();
                    pow.from_base_and_exp(nb.to_view(), exp);
                    pow.set_dirty(true);

                    o.to_view().normalize(workspace, state, out);
                    true
                } else {
                    out.from_view(self);
                    false
                }
            }
            AtomView::Mul(m) => {
                let mut o = workspace.new_atom();
                let mul = o.transform_to_mul();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in m.into_iter() {
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.to_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.to_view().normalize(workspace, state, out);
                    true
                }
            }
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.transform_to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a.into_iter() {
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.to_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.to_view().normalize(workspace, state, out);
                    true
                }
            }
            AtomView::Fun(_) => {
                // do not propagate into functions
                out.from_view(self);
                false
            }
        }
    }
}
