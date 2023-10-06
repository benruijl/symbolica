use ahash::HashMap;
use smallvec::{smallvec, SmallVec};

use crate::{
    poly::{polynomial::MultivariatePolynomial, Variable, INLINED_EXPONENTS},
    representations::{
        number::{BorrowedNumber, Number},
        Add, Atom, AtomSet, AtomView, Mul, Num, OwnedAdd, OwnedMul, OwnedNum, OwnedPow, Pow, Var,
    },
    rings::{
        integer::{Integer, IntegerRing},
        rational_polynomial::RationalPolynomial,
    },
    state::{ResettableBuffer, State, Workspace},
};

impl<'a, P: AtomSet> AtomView<'a, P> {
    pub fn set_coefficient_ring(
        &self,
        vars: &[Variable],
        state: &State,
        workspace: &Workspace<P>,
        out: &mut Atom<P>,
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
                            out.to_num().set_from_number(Number::RationalPolynomial(r));
                            true
                        } else {
                            let mut n1 = workspace.new_atom();
                            n1.from_polynomial(workspace, state, &r.numerator, &HashMap::default());

                            let mut n1_conv = workspace.new_atom();
                            n1.as_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n1_conv);

                            let mut n2 = workspace.new_atom();
                            n2.from_polynomial(
                                workspace,
                                state,
                                &r.denominator,
                                &HashMap::default(),
                            );

                            let mut n2_conv = workspace.new_atom();
                            n2.as_view()
                                .set_coefficient_ring(vars, state, workspace, &mut n2_conv);

                            // create n1/n2
                            let mut n3 = workspace.new_atom();
                            let mut exp = workspace.new_atom();
                            exp.to_num().set_from_number(Number::Natural(-1, 1));
                            let n3p = n3.to_pow();
                            n3p.set_from_base_and_exp(n2_conv.as_view(), exp.as_view());
                            n3p.set_dirty(true);

                            let mut m = workspace.new_atom();
                            let mm = m.to_mul();
                            mm.extend(n1_conv.as_view());
                            mm.extend(n3.as_view());
                            mm.set_dirty(true);
                            m.as_view().normalize(workspace, state, out);
                            true
                        }
                    } else {
                        out.set_from_view(self);
                        false
                    }
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Var(v) => {
                let id = v.get_name();
                if vars.contains(&id.into()) {
                    // change variable into coefficient
                    let mut poly = MultivariatePolynomial::new(
                        vars.len(),
                        IntegerRing::new(),
                        None,
                        Some(vars),
                    );
                    let mut e: SmallVec<[u16; INLINED_EXPONENTS]> = smallvec![0; vars.len()];
                    e[vars.iter().position(|x| *x == id.into()).unwrap()] = 1;
                    poly.append_monomial(Integer::one(), &e);
                    let den = MultivariatePolynomial::new_from_constant(&poly, Integer::one());

                    out.to_num()
                        .set_from_number(Number::RationalPolynomial(RationalPolynomial {
                            numerator: poly,
                            denominator: den,
                        }));
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = workspace.new_atom();
                if base.set_coefficient_ring(vars, state, workspace, &mut nb) {
                    let mut o = workspace.new_atom();
                    let pow = o.to_pow();
                    pow.set_from_base_and_exp(nb.as_view(), exp);
                    pow.set_dirty(true);

                    o.as_view().normalize(workspace, state, out);
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Mul(m) => {
                let mut o = workspace.new_atom();
                let mul = o.to_mul();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in m.iter() {
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, state, out);
                    true
                }
            }
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a.iter() {
                    arg_o.reset();

                    changed |= arg.set_coefficient_ring(vars, state, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_dirty(changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, state, out);
                    true
                }
            }
            AtomView::Fun(_) => {
                // do not propagate into functions
                out.set_from_view(self);
                false
            }
        }
    }
}
