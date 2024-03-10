use std::ops::DerefMut;

use crate::{
    coefficient::Coefficient,
    domains::integer::Integer,
    representations::{Atom, AtomView, Symbol},
    state::{State, Workspace},
};

impl Atom {
    /// Take a derivative of the expression with respect to `x`.
    pub fn derivative(&self, x: Symbol) -> Atom {
        self.as_view().derivative(x)
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        self.as_view().derivative_into(x, out)
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    pub fn taylor_series(&self, x: Symbol, expansion_point: AtomView, depth: u32) -> Atom {
        self.as_view().taylor_series(x, expansion_point, depth)
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    /// Returns `true` iff the result is non-zero.
    pub fn taylor_series_into(
        &self,
        x: Symbol,
        expansion_point: AtomView,
        depth: u32,
        out: &mut Atom,
    ) -> bool {
        self.as_view()
            .taylor_series_into(x, expansion_point, depth, out)
    }
}

impl<'a> AtomView<'a> {
    /// Take a derivative of the expression with respect to `x`.
    pub fn derivative(&self, x: Symbol) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.derivative_with_ws_into(x, ws, &mut out);
            out.into_inner()
        })
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        Workspace::get_local().with(|ws| self.derivative_with_ws_into(x, ws, out))
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub fn derivative_with_ws_into(
        &self,
        x: Symbol,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        match self {
            AtomView::Num(_) => {
                out.to_num(Coefficient::zero());
                false
            }
            AtomView::Var(v) => {
                if v.get_symbol() == x {
                    out.to_num(1.into());
                    true
                } else {
                    out.to_num(Coefficient::zero());

                    false
                }
            }
            AtomView::Fun(f_orig) => {
                // detect if the function to derive is the derivative function itself
                // if so, derive the last argument of the derivative function and set
                // a flag to later accumulate previous derivatives
                let (to_derive, f, is_der) = if f_orig.get_symbol() == State::DERIVATIVE {
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
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        args_der.push((i, arg_der));
                    }
                }

                if args_der.is_empty() {
                    out.to_num(Coefficient::zero());
                    return false;
                }

                // derive special functions
                if f.get_nargs() == 1
                    && [State::EXP, State::LOG, State::SIN, State::COS].contains(&f.get_symbol())
                {
                    let mut fn_der = workspace.new_atom();
                    match f.get_symbol() {
                        State::EXP => {
                            fn_der.set_from_view(self);
                        }
                        State::LOG => {
                            let mut n = workspace.new_atom();
                            n.to_num((-1).into());

                            fn_der.to_pow(f.iter().next().unwrap(), n.as_view());
                        }
                        State::SIN => {
                            let p = fn_der.to_fun(State::COS);
                            p.add_arg(f.iter().next().unwrap());
                        }
                        State::COS => {
                            let mut n = workspace.new_atom();
                            n.to_num((-1).into());

                            let mut sin = workspace.new_atom();
                            let sin_fun = sin.to_fun(State::SIN);
                            sin_fun.add_arg(f.iter().next().unwrap());

                            let m = fn_der.to_mul();
                            m.extend(sin.as_view());
                            m.extend(n.as_view());
                        }
                        _ => unreachable!(),
                    }

                    let (_, mut arg_der) = args_der.pop().unwrap();
                    if let Atom::Mul(m) = arg_der.deref_mut() {
                        m.extend(fn_der.as_view());
                        arg_der.as_view().normalize(workspace, out);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(fn_der.as_view());
                        m.extend(arg_der.as_view());
                        mul.as_view().normalize(workspace, out);
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
                    let p = fn_der.to_fun(State::DERIVATIVE);

                    if is_der {
                        for (i, x_orig) in f_orig.iter().take(f.get_nargs()).enumerate() {
                            if let AtomView::Num(nn) = x_orig {
                                let num = nn.get_coeff_view() + (if i == index { 1 } else { 0 });
                                n.to_num(num);
                                p.add_arg(n.as_view());
                            } else {
                                panic!("Derivative function must contain numbers for all but the last position");
                            }
                        }
                    } else {
                        for i in 0..f.get_nargs() {
                            n.to_num((if i == index { 1 } else { 0 }, 1).into());
                            p.add_arg(n.as_view());
                        }
                    }

                    p.add_arg(to_derive);

                    let m = mul.to_mul();
                    m.extend(fn_der.as_view());
                    m.extend(arg_der.as_view());
                    mul.as_view().normalize(workspace, out);

                    a.extend(mul.as_view());
                }

                add.as_view().normalize(workspace, out);
                true
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut exp_der = workspace.new_atom();
                let exp_der_non_zero = exp.derivative_with_ws_into(x, workspace, &mut exp_der);

                let mut base_der = workspace.new_atom();
                let base_der_non_zero = base.derivative_with_ws_into(x, workspace, &mut base_der);

                if !exp_der_non_zero && !base_der_non_zero {
                    out.to_num(0.into());
                    return false;
                }

                let mut exp_der_contrib = workspace.new_atom();

                if exp_der_non_zero {
                    // create log(base)
                    let mut log_base = workspace.new_atom();
                    let lb = log_base.to_fun(State::LOG);
                    lb.add_arg(base);

                    if let Atom::Mul(m) = exp_der.deref_mut() {
                        m.extend(*self);
                        m.extend(log_base.as_view());
                        exp_der.as_view().normalize(workspace, &mut exp_der_contrib);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(*self);
                        m.extend(exp_der.as_view());
                        m.extend(log_base.as_view());
                        mul.as_view().normalize(workspace, &mut exp_der_contrib);
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

                    let res = n.get_coeff_view() + -1;
                    new_exp.to_num(res);
                } else {
                    mul.extend(exp);

                    let ao = new_exp.to_add();
                    ao.extend(exp);

                    let mut min_one = workspace.new_atom();
                    min_one.to_num((-1).into());

                    ao.extend(min_one.as_view());
                }

                let mut pow_h = workspace.new_atom();
                pow_h.to_pow(base, new_exp.as_view());

                mul.extend(pow_h.as_view());

                if exp_der_non_zero {
                    let mut add = workspace.new_atom();
                    let a = add.to_add();

                    a.extend(mul_h.as_view());
                    a.extend(exp_der_contrib.as_view());

                    add.as_view().normalize(workspace, out);
                } else {
                    mul_h.as_view().normalize(workspace, out);
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
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        if let Atom::Mul(mm) = arg_der.deref_mut() {
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                }
                            }

                            add.extend(arg_der.as_view());
                        } else {
                            let mm = mul_h.to_mul();
                            mm.extend(arg_der.as_view());
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                }
                            }
                            add.extend(mul_h.as_view());
                        }

                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, out);
                    true
                } else {
                    out.to_num(0.into());
                    false
                }
            }
            AtomView::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut arg_der = workspace.new_atom();
                let mut non_zero = false;
                for arg in args.iter() {
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        add.extend(arg_der.as_view());
                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, out);
                    true
                } else {
                    out.to_num(0.into());
                    false
                }
            }
        }
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    pub fn taylor_series(&self, x: Symbol, expansion_point: AtomView, depth: u32) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.taylor_series_with_ws_into(x, expansion_point, depth, ws, &mut out);
            out.into_inner()
        })
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    /// Returns `true` iff the result is non-zero.
    pub fn taylor_series_into(
        &self,
        x: Symbol,
        expansion_point: AtomView,
        depth: u32,
        out: &mut Atom,
    ) -> bool {
        Workspace::get_local()
            .with(|ws| self.taylor_series_with_ws_into(x, expansion_point, depth, ws, out))
    }

    /// Taylor expand in `x` around `expansion_point` to depth `depth`.
    /// Returns `true` iff the result is non-zero.
    pub fn taylor_series_with_ws_into(
        &self,
        x: Symbol,
        expansion_point: AtomView,
        depth: u32,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        let mut current_order = workspace.new_atom();
        current_order.set_from_view(self);

        let mut next_order = workspace.new_atom();

        let var = workspace.new_var(x);
        let var_pat = var.into_pattern();
        let expansion_point_pat = expansion_point.into_pattern();

        // construct x - expansion_point
        // TODO: check that expansion_point does not involve `x`

        let mut dist = workspace.new_atom();
        var.as_view()
            .sub_with_ws_into(workspace, expansion_point, &mut dist);

        let mut series = workspace.new_atom();
        let series_sum = series.to_add();

        let mut series_contrib = workspace.new_atom();

        for d in 0..=depth {
            // replace x by expansion_point
            var_pat.replace_all_with_ws_into(
                current_order.as_view(),
                &expansion_point_pat,
                workspace,
                None,
                None,
                &mut next_order,
            );

            if d > 0 {
                let m = series_contrib.to_mul();
                m.extend(next_order.as_view());
                if d > 1 {
                    let mut exp = workspace.new_atom();
                    exp.to_pow(dist.as_view(), workspace.new_num(d as i64).as_view());
                    m.extend(exp.as_view());
                } else if d == 1 {
                    m.extend(dist.as_view());
                }

                let mut fact = workspace.new_atom();
                fact.to_num((Integer::one(), Integer::factorial(d)).into());

                m.extend(fact.as_view());

                series_sum.extend(series_contrib.as_view());
            } else {
                series_sum.extend(next_order.as_view());
            }

            if d < depth
                && current_order
                    .as_view()
                    .derivative_with_ws_into(x, workspace, &mut next_order)
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

        series.as_view().normalize(workspace, out);

        true
    }
}
