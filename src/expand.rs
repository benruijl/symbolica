use std::ops::DerefMut;

use smallvec::SmallVec;

use crate::{
    atom::{Atom, AtomView, Symbol},
    coefficient::CoefficientView,
    combinatorics::CombinationWithReplacementIterator,
    domains::integer::Integer,
    state::{RecycledAtom, Workspace},
};

impl Atom {
    /// Expand an expression.
    pub fn expand(&self) -> Atom {
        self.as_view().expand()
    }

    /// Expand an expression in the variable `var`.
    pub fn expand_in(&self, var: Symbol) -> Atom {
        self.as_view().expand_in(var)
    }

    /// Expand an expression, returning `true` iff the expression changed.
    pub fn expand_into(&self, out: &mut Atom) -> bool {
        self.as_view().expand_into(None, out)
    }
}

impl<'a> AtomView<'a> {
    /// Expand an expression.
    pub fn expand(&self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.expand_with_ws_into(ws, None, &mut a);
            a.into_inner()
        })
    }

    /// Expand an expression.
    pub fn expand_in(&self, var: Symbol) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.expand_with_ws_into(ws, Some(var), &mut a);
            a.into_inner()
        })
    }

    /// Expand an expression, returning `true` iff the expression changed.
    pub fn expand_into(&self, var: Option<Symbol>, out: &mut Atom) -> bool {
        Workspace::get_local().with(|ws| self.expand_with_ws_into(ws, var, out))
    }

    /// Expand an expression, returning `true` iff the expression changed.
    pub fn expand_with_ws_into(
        &self,
        workspace: &Workspace,
        var: Option<Symbol>,
        out: &mut Atom,
    ) -> bool {
        let changed = self.expand_no_norm(workspace, var, out);

        if changed {
            let mut a = workspace.new_atom();
            out.as_view().normalize(workspace, &mut a);
            std::mem::swap(out, &mut a);
        }

        changed
    }

    /// Expand an expression, but do not normalize the result.
    fn expand_no_norm(&self, workspace: &Workspace, var: Option<Symbol>, out: &mut Atom) -> bool {
        if let Some(s) = var {
            if !self.contains_symbol(s) {
                out.set_from_view(self);
                return false;
            }
        }

        match self {
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut new_base = workspace.new_atom();
                let mut changed = base.expand_with_ws_into(workspace, var, &mut new_base);

                let mut new_exp = workspace.new_atom();
                changed |= exp.expand_with_ws_into(workspace, var, &mut new_exp);

                let (negative, num) = 'get_num: {
                    if let AtomView::Num(n) = new_exp.as_view() {
                        if let CoefficientView::Natural(n, 1) = n.get_coeff_view() {
                            if n.unsigned_abs() <= u32::MAX as u64 {
                                break 'get_num (n < 0, n.unsigned_abs() as u32);
                            }
                        }
                    }

                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.to_pow(new_base.as_view(), new_exp.as_view());
                    pow.set_normalized(!changed);
                    pow_h.as_view().normalize(workspace, out);
                    return changed;
                };

                if let AtomView::Add(a) = new_base.as_view() {
                    // expand (a+b+c+..)^n
                    let mut args: SmallVec<[AtomView; 10]> = SmallVec::with_capacity(a.get_nargs());
                    for arg in a.iter() {
                        args.push(arg);
                    }

                    let mut add_h = workspace.new_atom();
                    let add = add_h.to_add();

                    let mut ci = CombinationWithReplacementIterator::new(args.len(), num);

                    while let Some(new_term) = ci.next() {
                        let mut hh = workspace.new_atom();
                        let p = hh.to_mul();

                        let mut hhh = workspace.new_atom();
                        for (a, pow) in args.iter().zip(new_term) {
                            if *pow != 0 {
                                let mut new_exp_h = workspace.new_atom();
                                new_exp_h.to_num((*pow as i64).into());
                                hhh.to_pow(*a, new_exp_h.as_view());
                                p.extend(hhh.as_view());
                            }
                        }

                        let mut normalized_child = workspace.new_atom();
                        hh.as_view().normalize(workspace, &mut normalized_child);

                        let mut expanded_child = workspace.new_atom();
                        normalized_child.as_view().expand_with_ws_into(
                            workspace,
                            var,
                            &mut expanded_child,
                        );

                        let coeff_f = Integer::multinom(new_term);
                        if coeff_f != Integer::one() {
                            let mut coeff_h = workspace.new_atom();
                            coeff_h.to_num(coeff_f.into());

                            if let Atom::Mul(m) = expanded_child.deref_mut() {
                                m.extend(coeff_h.as_view());
                                add.extend(expanded_child.as_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(expanded_child.as_view());
                                mul.extend(coeff_h.as_view());
                                add.extend(mul_h.as_view());
                            }
                        } else {
                            add.extend(expanded_child.as_view());
                        }
                    }

                    if negative {
                        let mut num_h = workspace.new_atom();
                        num_h.to_num((-1i64).into());

                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(add_h.as_view(), num_h.as_view());

                        pow_h.as_view().normalize(workspace, out);
                    } else {
                        add_h.as_view().normalize(workspace, out);
                    }

                    true
                } else if let AtomView::Mul(m) = new_base.as_view() {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut exp_h = workspace.new_atom();
                    if negative {
                        exp_h.to_num((-(num as i64)).into());
                    } else {
                        exp_h.to_num((num as i64).into());
                    }

                    for arg in m.iter() {
                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(arg, exp_h.as_view());
                        mul.extend(pow_h.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);

                    true
                } else {
                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.to_pow(new_base.as_view(), new_exp.as_view());
                    pow.set_normalized(!changed);
                    pow_h.as_view().normalize(workspace, out);
                    changed
                }
            }
            AtomView::Mul(m) => {
                let mut changed = false;

                let mut sum: SmallVec<[RecycledAtom; 10]> = SmallVec::new();
                let mut new_sum: SmallVec<[RecycledAtom; 10]> = SmallVec::new();

                for arg in m.iter() {
                    let mut new_arg = workspace.new_atom();
                    changed |= arg.expand_with_ws_into(workspace, var, &mut new_arg);

                    // expand (1+x)*y
                    if let AtomView::Add(a) = new_arg.as_view() {
                        changed = true;

                        for child in a.iter() {
                            for s in &sum {
                                let mut b = workspace.new_atom();
                                b.set_from_view(&s.as_view());

                                if let Atom::Mul(m) = b.deref_mut() {
                                    m.extend(child);
                                    new_sum.push(b);
                                } else {
                                    let mut mul_h = workspace.new_atom();
                                    let mul = mul_h.to_mul();
                                    mul.extend(b.as_view());
                                    mul.extend(child);
                                    new_sum.push(mul_h);
                                }
                            }

                            if sum.is_empty() {
                                let mut b = workspace.new_atom();
                                b.set_from_view(&child);
                                new_sum.push(b);
                            }
                        }

                        std::mem::swap(&mut sum, &mut new_sum);
                        new_sum.clear();
                    } else if sum.is_empty() {
                        sum.push(new_arg);
                    } else {
                        for summand in &mut sum {
                            if let Atom::Mul(m) = summand.deref_mut() {
                                m.extend(new_arg.as_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(summand.as_view());
                                mul.extend(new_arg.as_view());
                                *summand = mul_h;
                            }
                        }
                    }
                }

                if !changed {
                    out.set_from_view(self);
                    return false;
                }

                debug_assert!(!sum.is_empty());

                if sum.len() == 1 {
                    sum[0].as_view().normalize(workspace, out);
                } else {
                    let add = out.to_add();
                    for x in sum {
                        add.extend(x.as_view());
                    }
                }

                changed
            }
            AtomView::Add(a) => {
                let mut changed = false;

                let add = out.to_add();

                let mut new_arg = workspace.new_atom();
                for arg in a.iter() {
                    changed |= arg.expand_no_norm(workspace, var, &mut new_arg);
                    add.extend(new_arg.as_view());
                }

                add.set_normalized(!changed);
                changed
            }
            _ => {
                out.set_from_view(self);
                false
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{atom::Atom, state::State};

    #[test]
    fn exponent() {
        let exp = Atom::parse("(1+v1+v2)^4").unwrap().expand();
        let res = Atom::parse("4*v1+4*v2+6*v1^2+4*v1^3+v1^4+6*v2^2+4*v2^3+v2^4+12*v1*v2+12*v1*v2^2+4*v1*v2^3+12*v1^2*v2+6*v1^2*v2^2+4*v1^3*v2+1").unwrap();
        assert_eq!(exp, res);
    }

    #[test]
    fn association() {
        let exp = Atom::parse("(1+v1)*(2+v2)*(3+v1)").unwrap().expand();
        let res = Atom::parse("8*v1+3*v2+2*v1^2+4*v1*v2+v1^2*v2+6").unwrap();
        assert_eq!(exp, res);
    }

    #[test]
    fn mul_pow() {
        let exp = Atom::parse("(v1*v2*2)^3*2").unwrap().expand();
        let res = Atom::parse("v1^3*v2^3*16").unwrap();
        assert_eq!(exp, res);
    }

    #[test]
    fn mul_pow_neg() {
        let exp = Atom::parse("(v1*v2*2)^-3").unwrap().expand();
        let res = Atom::parse("8^-1*v1^-3*v2^-3").unwrap();
        assert_eq!(exp, res);
    }

    #[test]
    fn expand_in_var() {
        let exp = Atom::parse("(1+v1)^2+(1+v2)^100")
            .unwrap()
            .expand_in(State::get_symbol("v1"));
        let res = Atom::parse("1+2*v1+v1^2+(v2+1)^100").unwrap();
        assert_eq!(exp, res);
    }
}
