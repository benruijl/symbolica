use rug::Rational;
use smallvec::SmallVec;

use crate::{
    representations::{
        number::{BorrowedNumber, Number},
        Add, Atom, AtomView, Mul, Num, OwnedAdd, OwnedAtom, OwnedMul, OwnedNum, OwnedPow, Pow,
    },
    rings::integer::Integer,
    state::{BufferHandle, State, Workspace},
    utils::CombinationIterator,
};

impl<'a, P: Atom> AtomView<'a, P> {
    pub fn expand(&self, workspace: &Workspace<P>, state: &State, out: &mut OwnedAtom<P>) -> bool {
        assert!(!self.is_dirty());

        match self {
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut new_base = workspace.new_atom();
                let mut changed = base.expand(workspace, state, new_base.get_mut());

                let mut new_exp = workspace.new_atom();
                changed |= exp.expand(workspace, state, new_exp.get_mut());

                let (negative, num) = 'get_num: {
                    if let AtomView::Num(n) = new_exp.get().to_view() {
                        if let BorrowedNumber::Natural(n, 1) = n.get_number_view() {
                            if n.abs() < u32::MAX as i64 {
                                break 'get_num (n < 0, n.abs() as u32);
                            }
                        }
                    }

                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.get_mut().transform_to_pow();
                    pow.from_base_and_exp(new_base.get().to_view(), new_exp.get().to_view());
                    pow.set_dirty(changed);
                    pow_h.get().to_view().normalize(workspace, state, out);
                    return changed;
                };

                if let AtomView::Add(a) = new_base.get().to_view() {
                    // expand (a+b+c+..)^n
                    let mut args: SmallVec<[AtomView<P>; 10]> =
                        SmallVec::with_capacity(a.get_nargs());
                    for arg in a.into_iter() {
                        args.push(arg);
                    }

                    let mut add_h = workspace.new_atom();
                    let add = add_h.get_mut().transform_to_add();

                    let mut ci = CombinationIterator::new(args.len(), num);

                    while let Some(new_term) = ci.next() {
                        let mut hh = workspace.new_atom();
                        let p = hh.get_mut().transform_to_mul();

                        for (a, pow) in args.iter().zip(new_term) {
                            let mut hhh = workspace.new_atom();
                            let new_pow = hhh.get_mut().transform_to_pow();
                            if *pow != 0 {
                                let mut new_exp_h = workspace.new_atom();
                                let new_exp = new_exp_h.get_mut().transform_to_num();
                                new_exp.from_number(Number::Natural(*pow as i64, 1));
                                new_pow.from_base_and_exp(*a, new_exp_h.get().to_view());
                                new_pow.set_dirty(true);
                                p.extend(hhh.get().to_view());
                            }
                        }
                        p.set_dirty(true);

                        let mut normalized_child = workspace.new_atom();
                        hh.get_mut().to_view().normalize(
                            workspace,
                            state,
                            normalized_child.get_mut(),
                        );

                        let mut expanded_child = workspace.new_atom();
                        normalized_child.get_mut().to_view().expand(
                            workspace,
                            state,
                            expanded_child.get_mut(),
                        );

                        let mut coeff_h = workspace.new_atom();
                        let coeff = coeff_h.get_mut().transform_to_num();
                        let coeff_f = Integer::multinom(new_term);

                        if coeff_f != Integer::one() {
                            coeff.from_number(match coeff_f {
                                Integer::Natural(n) => Number::Natural(n, 1),
                                Integer::Large(l) => Number::Large(Rational::from(l)),
                            });

                            if let OwnedAtom::Mul(m) = expanded_child.get_mut() {
                                m.extend(coeff_h.get().to_view());
                                m.set_dirty(true);
                                add.extend(expanded_child.get().to_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.get_mut().transform_to_mul();
                                mul.extend(expanded_child.get().to_view());
                                mul.extend(coeff_h.get().to_view());
                                mul.set_dirty(true);
                                add.extend(mul_h.get().to_view());
                            }
                        } else {
                            add.extend(expanded_child.get().to_view());
                        }
                    }
                    add.set_dirty(true);

                    if negative {
                        let mut pow_h = workspace.new_atom();
                        let pow = pow_h.get_mut().transform_to_pow();

                        let mut num_h = workspace.new_atom();
                        let num = num_h.get_mut().transform_to_num();
                        num.from_number(Number::Natural(-1, 1));

                        pow.from_base_and_exp(add_h.get().to_view(), num_h.get().to_view());
                        pow.set_dirty(true);

                        pow_h.get().to_view().normalize(workspace, state, out);
                    } else {
                        add_h.get().to_view().normalize(workspace, state, out);
                    }

                    true
                } else if let AtomView::Mul(m) = new_base.get().to_view() {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.get_mut().transform_to_mul();

                    let mut exp_h = workspace.new_atom();
                    let exp = exp_h.get_mut().transform_to_num();
                    exp.from_number(Number::Natural(num as i64, 1));

                    for arg in m.into_iter() {
                        let mut pow_h = workspace.new_atom();
                        let pow = pow_h.get_mut().transform_to_pow();
                        pow.from_base_and_exp(arg, exp_h.get().to_view());
                        pow.set_dirty(true);
                        mul.extend(pow_h.get().to_view());
                    }
                    mul.set_dirty(true);

                    if negative {
                        let mut pow_h = workspace.new_atom();
                        let pow = pow_h.get_mut().transform_to_pow();

                        let mut num_h = workspace.new_atom();
                        let num = num_h.get_mut().transform_to_num();
                        num.from_number(Number::Natural(-1, 1));

                        pow.from_base_and_exp(mul_h.get().to_view(), num_h.get().to_view());
                        pow.set_dirty(true);
                        pow_h.get().to_view().normalize(workspace, state, out);
                    } else {
                        mul_h.get().to_view().normalize(workspace, state, out);
                    }
                    true
                } else {
                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.get_mut().transform_to_pow();
                    pow.from_base_and_exp(new_base.get().to_view(), new_exp.get().to_view());
                    pow.set_dirty(changed);
                    pow_h.get().to_view().normalize(workspace, state, out);
                    changed
                }
            }
            AtomView::Mul(m) => {
                let mut changed = false;

                let mut sum: SmallVec<[BufferHandle<OwnedAtom<P>>; 10]> = SmallVec::new();
                let mut new_sum: SmallVec<[BufferHandle<OwnedAtom<P>>; 10]> = SmallVec::new();

                for arg in m.into_iter() {
                    let mut new_arg = workspace.new_atom();
                    changed |= arg.expand(workspace, state, new_arg.get_mut());

                    // expand (1+x)*y
                    if let AtomView::Add(a) = new_arg.get().to_view() {
                        changed = true;

                        for child in a.into_iter() {
                            for s in &sum {
                                let mut b = workspace.new_atom();
                                b.get_mut().from_view(&s.get().to_view());

                                if let OwnedAtom::Mul(m) = b.get_mut() {
                                    m.extend(child);
                                    m.set_dirty(true);
                                    new_sum.push(b);
                                } else {
                                    let mut mul_h = workspace.new_atom();
                                    let mul = mul_h.get_mut().transform_to_mul();
                                    mul.extend(b.get().to_view());
                                    mul.extend(child);
                                    mul.set_dirty(true);
                                    new_sum.push(mul_h);
                                }
                            }

                            if sum.is_empty() {
                                let mut b = workspace.new_atom();
                                b.get_mut().from_view(&child);
                                new_sum.push(b);
                            }
                        }

                        std::mem::swap(&mut sum, &mut new_sum);
                        new_sum.clear();
                    } else {
                        if sum.is_empty() {
                            sum.push(new_arg);
                        } else {
                            for summand in &mut sum {
                                if let OwnedAtom::Mul(m) = summand.get_mut() {
                                    m.extend(new_arg.get().to_view());
                                    m.set_dirty(true);
                                } else {
                                    let mut mul_h = workspace.new_atom();
                                    let mul = mul_h.get_mut().transform_to_mul();
                                    mul.extend(summand.get().to_view());
                                    mul.extend(new_arg.get().to_view());
                                    mul.set_dirty(true);
                                    *summand = mul_h;
                                }
                            }
                        }
                    }
                }

                debug_assert!(!sum.is_empty());

                if sum.len() == 1 {
                    sum[0].get().to_view().normalize(workspace, state, out);
                } else {
                    let mut add_h = workspace.new_atom();
                    let add = add_h.get_mut().transform_to_add();
                    for x in sum {
                        add.extend(x.get().to_view());
                    }

                    add.set_dirty(true);
                    add_h.get().to_view().normalize(workspace, state, out);
                }

                changed
            }
            AtomView::Add(a) => {
                let mut changed = false;

                let mut add_h = workspace.new_atom();
                let add = add_h.get_mut().transform_to_add();

                for arg in a.into_iter() {
                    let mut new_arg = workspace.new_atom();
                    changed |= arg.expand(workspace, state, new_arg.get_mut());
                    add.extend(new_arg.get().to_view());
                }

                add.set_dirty(changed);
                add_h.get().to_view().normalize(workspace, state, out);
                changed
            }
            _ => {
                out.from_view(self);
                false
            }
        }
    }
}
