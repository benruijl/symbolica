use ahash::HashMap;

use crate::{
    representations::{
        number::{BorrowedNumber, Number},
        Add, AsAtomView, Atom, AtomBuilder, AtomSet, AtomView, Fun, Identifier, Mul, Num, OwnedAdd,
        OwnedFun, OwnedMul, OwnedNum, OwnedPow, Pow, Var,
    },
    rings::{integer::Integer, rational::Rational},
    state::{State, Workspace, COS, DERIVATIVE, EXP, LOG, SIN},
};

impl<'a, P: AtomSet> AtomView<'a, P> {
    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub fn derivative(
        &self,
        x: Identifier,
        workspace: &Workspace<P>,
        state: &State,
        out: &mut Atom<P>,
    ) -> bool {
        match self {
            AtomView::Num(_) => {
                let n = out.to_num();
                n.set_from_number(Number::Natural(0, 1));
                false
            }
            AtomView::Var(v) => {
                if v.get_name() == x {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(1, 1));
                    true
                } else {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(0, 1));
                    false
                }
            }
            AtomView::Fun(f_orig) => {
                // detect if the function to derive is the derivative function itself
                // if so, derive the last argument of the derivative function and set
                // a flag to later accumulate previous derivatives
                let (to_derive, f, is_der) = if f_orig.get_name() == DERIVATIVE {
                    let to_derive = f_orig.iter().last().unwrap();
                    (
                        to_derive,
                        match to_derive {
                            AtomView::Fun(f) => f,
                            _ => panic!("Last argument of der function must be a function"),
                        },
                        true,
                    )
                } else {
                    (*self, *f_orig, false)
                };

                // take derivative of all the arguments and store it in a list
                let mut args_der = Vec::with_capacity(f.get_nargs());
                for (i, arg) in f.iter().enumerate() {
                    let mut arg_der = workspace.new_atom();
                    if arg.derivative(x, workspace, state, &mut arg_der) {
                        args_der.push((i, arg_der));
                    }
                }

                if args_der.is_empty() {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(0, 1));
                    return false;
                }

                // derive special functions
                if f.get_nargs() == 1 && [EXP, LOG, SIN, COS].contains(&f.get_name()) {
                    let mut fn_der = workspace.new_atom();
                    match f.get_name() {
                        EXP => {
                            fn_der.set_from_view(self);
                        }
                        LOG => {
                            let mut n = workspace.new_atom();
                            n.to_num().set_from_number(Number::Natural(-1, 1));

                            let p = fn_der.to_pow();
                            p.set_from_base_and_exp(f.iter().next().unwrap(), n.as_view());
                            p.set_dirty(true);
                        }
                        SIN => {
                            let p = fn_der.to_fun();
                            p.set_from_name(COS);
                            p.add_arg(f.iter().next().unwrap());
                            p.set_dirty(true);
                        }
                        COS => {
                            let mut n = workspace.new_atom();
                            n.to_num().set_from_number(Number::Natural(-1, 1));

                            let mut sin = workspace.new_atom();
                            let sin_fun = sin.to_fun();
                            sin_fun.set_from_name(SIN);
                            sin_fun.add_arg(f.iter().next().unwrap());

                            let m = fn_der.to_mul();
                            m.extend(sin.as_view());
                            m.extend(n.as_view());
                            m.set_dirty(true);
                        }
                        _ => unreachable!(),
                    }

                    let (_, mut arg_der) = args_der.pop().unwrap();
                    if let Atom::Mul(m) = arg_der.get_mut() {
                        m.extend(fn_der.as_view());
                        m.set_dirty(true);
                        arg_der.as_view().normalize(workspace, state, out);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(fn_der.as_view());
                        m.extend(arg_der.as_view());
                        m.set_dirty(true);
                        mul.as_view().normalize(workspace, state, out);
                    }

                    return true;
                }

                // create a derivative function that tags which index was derived
                let mut add = workspace.new_atom();
                let a = add.to_add();
                let mut fn_der = workspace.new_atom();
                let mut n = workspace.new_atom();
                let mut mul = workspace.new_atom();
                for (index, arg_der) in args_der {
                    let p = fn_der.to_fun();
                    p.set_from_name(DERIVATIVE);

                    if is_der {
                        for (i, x_orig) in f_orig.iter().take(f.get_nargs()).enumerate() {
                            if let AtomView::Num(nn) = x_orig {
                                let num = nn.get_number_view().add(
                                    &BorrowedNumber::Natural(if i == index { 1 } else { 0 }, 1),
                                    state,
                                );
                                n.to_num().set_from_number(num);
                                p.add_arg(n.as_view());
                            } else {
                                panic!("Derivative function must contain numbers for all but the last position");
                            }
                        }
                    } else {
                        for i in 0..f.get_nargs() {
                            n.to_num().set_from_number(Number::Natural(
                                if i == index { 1 } else { 0 },
                                1,
                            ));
                            p.add_arg(n.as_view());
                        }
                    }

                    p.add_arg(to_derive);
                    p.set_dirty(true);

                    let m = mul.to_mul();
                    m.extend(fn_der.as_view());
                    m.extend(arg_der.as_view());
                    m.set_dirty(true);
                    mul.as_view().normalize(workspace, state, out);

                    a.extend(mul.as_view());
                    a.set_dirty(true);
                }

                add.as_view().normalize(workspace, state, out);
                true
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut exp_der = workspace.new_atom();
                let exp_der_non_zero = exp.derivative(x, workspace, state, &mut exp_der);

                let mut base_der = workspace.new_atom();
                let base_der_non_zero = base.derivative(x, workspace, state, &mut base_der);

                if !exp_der_non_zero && !base_der_non_zero {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(0, 1));
                    return false;
                }

                let mut exp_der_contrib = workspace.new_atom();

                if exp_der_non_zero {
                    // create log(base)
                    let mut log_base = workspace.new_atom();
                    let lb = log_base.to_fun();
                    lb.set_from_name(LOG);
                    lb.add_arg(base);

                    if let Atom::Mul(m) = exp_der.get_mut() {
                        m.extend(*self);
                        m.extend(log_base.as_view());
                        m.set_dirty(true);
                        exp_der
                            .as_view()
                            .normalize(workspace, state, &mut exp_der_contrib);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(*self);
                        m.extend(exp_der.as_view());
                        m.extend(log_base.as_view());
                        m.set_dirty(true);
                        mul.as_view()
                            .normalize(workspace, state, &mut exp_der_contrib);
                    }

                    if !base_der_non_zero {
                        out.set_from_view(&exp_der_contrib.as_view());
                        return true;
                    }
                }

                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();
                mul.extend(base_der.as_view());

                let mut new_exp = workspace.new_atom();
                if let AtomView::Num(n) = exp {
                    mul.extend(exp);

                    let pow_min_one = new_exp.to_num();
                    let res = n
                        .get_number_view()
                        .add(&BorrowedNumber::Natural(-1, 1), state);
                    pow_min_one.set_from_number(res);
                } else {
                    mul.extend(exp);

                    let ao = new_exp.to_add();
                    ao.extend(exp);

                    let mut min_one = workspace.new_atom();
                    min_one.to_num().set_from_number(Number::Natural(-1, 1));

                    ao.extend(min_one.as_view());
                    ao.set_dirty(true);
                }

                let mut pow_h = workspace.new_atom();
                let pow = pow_h.to_pow();
                pow.set_from_base_and_exp(base, new_exp.as_view());
                pow.set_dirty(true);

                mul.extend(pow_h.as_view());
                mul.set_dirty(true);

                if exp_der_non_zero {
                    let mut add = workspace.new_atom();
                    let a = add.to_add();

                    a.extend(mul_h.as_view());
                    a.extend(exp_der_contrib.as_view());
                    a.set_dirty(true);

                    add.as_view().normalize(workspace, state, out);
                } else {
                    mul_h.as_view().normalize(workspace, state, out);
                }

                true
            }
            AtomView::Mul(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut mul_h = workspace.new_atom();
                let mut non_zero = false;
                for arg in args.iter() {
                    let mut arg_der = workspace.new_atom();
                    if arg.derivative(x, workspace, state, &mut arg_der) {
                        if let Atom::Mul(mm) = arg_der.get_mut() {
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                    mm.set_dirty(true);
                                }
                            }

                            add.extend(arg_der.as_view());
                            add.set_dirty(true);
                        } else {
                            let mm = mul_h.to_mul();
                            mm.extend(arg_der.as_view());
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                    mm.set_dirty(true);
                                }
                            }
                            add.extend(mul_h.as_view());
                            add.set_dirty(true);
                        }

                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, state, out);
                    true
                } else {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(0, 1));
                    false
                }
            }
            AtomView::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut arg_der = workspace.new_atom();
                let mut non_zero = false;
                for arg in args.iter() {
                    if arg.derivative(x, workspace, state, &mut arg_der) {
                        add.extend(arg_der.as_view());
                        non_zero = true;
                        add.set_dirty(true);
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, state, out);
                    true
                } else {
                    let n = out.to_num();
                    n.set_from_number(Number::Natural(0, 1));
                    false
                }
            }
        }
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    pub fn taylor_series(
        &self,
        x: Identifier,
        expansion_point: AtomView<P>,
        depth: u32,
        workspace: &Workspace<P>,
        state: &State,
        out: &mut Atom<P>,
    ) -> bool {
        let mut current_order = workspace.new_atom();
        current_order.set_from_view(self);

        let mut next_order = workspace.new_atom();

        let var = workspace.new_var(x);
        let var_pat = var.into_pattern(state);
        let expansion_point_pat = expansion_point.into_pattern(state);

        // construct x - expansion_point
        // TODO: check that expansion_point does not involve `x`
        let mut dist = AtomBuilder::new(var.as_view(), state, workspace, workspace.new_atom());
        dist = dist - expansion_point;

        let mut series = workspace.new_atom();
        let series_sum = series.to_add();

        let mut series_contrib = workspace.new_atom();

        for d in 0..=depth {
            // replace x by expansion_point
            var_pat.replace_all(
                current_order.as_view(),
                &expansion_point_pat,
                state,
                workspace,
                &HashMap::default(),
                &mut next_order,
            );

            if d > 0 {
                let m = series_contrib.to_mul();
                m.extend(next_order.as_view());
                if d > 1 {
                    let mut exp = workspace.new_atom();
                    let e = exp.to_pow();
                    e.set_from_base_and_exp(
                        dist.as_atom_view(),
                        workspace.new_num(d as i64).as_view(),
                    );
                    m.extend(exp.as_atom_view());
                } else if d == 1 {
                    m.extend(dist.as_atom_view());
                }

                let mut fact = workspace.new_atom();
                fact.to_num().set_from_number(
                    Rational::from_num_den(Integer::one(), Integer::factorial(d)).into(),
                );

                m.extend(fact.as_atom_view());
                m.set_dirty(true);

                series_sum.extend(series_contrib.as_view());
            } else {
                series_sum.extend(next_order.as_view());
            }

            if d < depth
                && current_order
                    .as_view()
                    .derivative(x, workspace, state, &mut next_order)
            {
                std::mem::swap(&mut current_order, &mut next_order);
            } else {
                if d == 0 {
                    out.set_from_view(&workspace.new_num(0).as_view());
                    return false;
                }

                break;
            }
        }

        series_sum.set_dirty(true);
        series.as_view().normalize(workspace, state, out);

        true
    }
}
