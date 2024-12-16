use std::{
    ops::{Add, DerefMut, Div, Mul, Sub},
    sync::Arc,
};

use crate::{
    atom::{Atom, AtomView, FunctionBuilder, Symbol},
    coefficient::{Coefficient, CoefficientView},
    combinatorics::CombinationWithReplacementIterator,
    domains::{atom::AtomField, integer::Integer, rational::Rational},
    poly::{series::Series, Variable},
    state::Workspace,
};

impl<'a> AtomView<'a> {
    /// Take a derivative of the expression with respect to `x`.
    pub(crate) fn derivative(&self, x: Symbol) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.derivative_with_ws_into(x, ws, &mut out);
            out.into_inner()
        })
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub(crate) fn derivative_into(&self, x: Symbol, out: &mut Atom) -> bool {
        Workspace::get_local().with(|ws| self.derivative_with_ws_into(x, ws, out))
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub(crate) fn derivative_with_ws_into(
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
                let (to_derive, f, is_der) = if f_orig.get_symbol() == Atom::DERIVATIVE {
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
                    && [Atom::EXP, Atom::LOG, Atom::SIN, Atom::COS].contains(&f.get_symbol())
                {
                    let mut fn_der = workspace.new_atom();
                    match f.get_symbol() {
                        Atom::EXP => {
                            fn_der.set_from_view(self);
                        }
                        Atom::LOG => {
                            let mut n = workspace.new_atom();
                            n.to_num((-1).into());

                            fn_der.to_pow(f.iter().next().unwrap(), n.as_view());
                        }
                        Atom::SIN => {
                            let p = fn_der.to_fun(Atom::COS);
                            p.add_arg(f.iter().next().unwrap());
                        }
                        Atom::COS => {
                            let mut n = workspace.new_atom();
                            n.to_num((-1).into());

                            let mut sin = workspace.new_atom();
                            let sin_fun = sin.to_fun(Atom::SIN);
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
                    let p = fn_der.to_fun(Atom::DERIVATIVE);

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
                    let lb = log_base.to_fun(Atom::LOG);
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

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    pub fn series(
        &self,
        x: Symbol,
        expansion_point: AtomView,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> Result<Series<AtomField>, &'static str> {
        if !depth_is_absolute && (depth.is_negative() || depth.is_zero()) {
            return Err("Cannot series expand to negative or zero depth");
        }

        // heuristic current depth
        let mut current_depth = if depth.is_negative() || depth.is_zero() {
            Rational::one()
        } else {
            depth.clone()
        };

        loop {
            let info = Series::new(
                &AtomField::new(),
                None,
                Arc::new(Variable::Symbol(x)),
                expansion_point.to_owned(),
                &current_depth + &(1.into(), current_depth.denominator()).into(),
            );

            let mut series = self.series_impl(x, expansion_point, &info)?;
            if !depth_is_absolute && series.relative_order() >= depth {
                series.truncate_relative_order(depth);
                break Ok(series);
            } else if depth_is_absolute && series.absolute_order() > depth {
                series.truncate_absolute_order(&depth + &((1.into(), depth.denominator())).into());
                break Ok(series);
            } else {
                // increase the expansion depth
                // TODO: find better heuristic
                current_depth = &current_depth * &2.into();
            }
        }
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    pub(crate) fn series_impl(
        &self,
        x: Symbol,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Result<Series<AtomField>, &'static str> {
        if !self.contains_symbol(x) {
            return Ok(info.constant(self.to_owned()));
        }

        // TODO: optimize, appending a monomial using addition is slow
        match self {
            AtomView::Num(n) => Ok(info.constant(n.to_owned().into())),
            AtomView::Var(v) => {
                if v.get_symbol() == x {
                    Ok(info.shifted_variable(expansion_point.to_owned()))
                } else {
                    Ok(info.constant(v.to_owned().into()))
                }
            }
            AtomView::Fun(f) => {
                let mut args_series = Vec::with_capacity(f.get_nargs());
                for arg in f {
                    args_series.push(arg.series_impl(x, expansion_point, info)?);
                }

                if args_series.is_empty() {
                    return Ok(info.constant(f.to_owned().into()));
                }

                match f.get_symbol() {
                    Atom::COS => args_series[0].cos(),
                    Atom::SIN => args_series[0].sin(),
                    Atom::EXP => args_series[0].exp(),
                    Atom::LOG => args_series[0].log(),
                    Atom::SQRT => args_series[0].rpow((1, 2).into()),
                    _ => {
                        // TODO: also check for log(x)?
                        if args_series
                            .iter()
                            .any(|x| x.get_trailing_exponent().is_negative())
                        {
                            return Err("Cannot series expand custom function with poles");
                        }

                        // TODO: depth is an overestimate
                        let order = info.absolute_order();
                        let depth = order.numerator().to_i64().unwrap() as u32
                            * order.denominator().to_i64().unwrap() as u32;

                        // strip the constant terms
                        let mut constants = vec![];
                        for x in &mut args_series {
                            if x.get_trailing_exponent().is_zero() {
                                let c = x.get_trailing_coefficient();
                                *x = &*x - &x.constant(c.clone());
                                constants.push(c);
                            } else {
                                constants.push(Atom::new_num(0));
                            }
                        }

                        let mut f_eval = FunctionBuilder::new(f.get_symbol());
                        for c in &constants {
                            f_eval = f_eval.add_arg(c);
                        }
                        let constant = f_eval.finish();

                        let mut result = info.constant(constant.clone());
                        for i in 0..=depth {
                            let mut it =
                                CombinationWithReplacementIterator::new(args_series.len(), i);

                            while let Some(x) = it.next() {
                                let mut f_der = FunctionBuilder::new(Atom::DERIVATIVE);
                                let mut term = info.one();
                                for (arg, pow) in x.iter().enumerate() {
                                    if *pow > 0 {
                                        term = &term * &args_series[arg].npow(*pow as usize);
                                    }
                                    f_der = f_der.add_arg(&Atom::new_num(*pow as i64));
                                }

                                f_der = f_der.add_arg(&constant);

                                result = &result
                                    + &term
                                        .mul_coeff(&f_der.finish())
                                        .mul_coeff(&Atom::new_num(Integer::multinom(x)))
                                        .div_coeff(&Atom::new_num(Integer::factorial(i)));
                            }
                        }

                        Ok(result)
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_series = base.series_impl(x, expansion_point, info)?;

                if let AtomView::Num(n) = exp {
                    if let CoefficientView::Natural(n, d) = n.get_coeff_view() {
                        if n < 0 && base_series.is_zero() {
                            // in case of 1/0, grow the expansion depth of the base series
                            // it could be that the base series is exactly zero,
                            // to prevent an infinite loop, we stop the loop at ep^-1000
                            let mut current_depth = info.relative_order();
                            while base_series.is_zero() && current_depth < 1000.into() {
                                let info = Series::new(
                                    &AtomField::new(),
                                    None,
                                    info.get_variable().clone(),
                                    info.get_expansion_point().clone(),
                                    &current_depth
                                        + &(1.into(), current_depth.denominator()).into(),
                                );

                                base_series = base.series_impl(x, expansion_point, &info)?;
                                current_depth = &current_depth * &2.into();
                            }
                        }

                        base_series.rpow((n, d).into())
                    } else {
                        unimplemented!("Cannot series expand with large exponents yet")
                    }
                } else {
                    let e = exp.series_impl(x, expansion_point, info)?;
                    base_series.pow(&e)
                }
            }
            AtomView::Mul(args) => {
                let mut iter = args.iter();
                let mut series = iter.next().unwrap().series_impl(x, expansion_point, info)?;
                for arg in iter {
                    series = &series * &arg.series_impl(x, expansion_point, info)?;
                }

                Ok(series)
            }
            AtomView::Add(args) => {
                let mut iter = args.iter();
                let mut series = iter.next().unwrap().series_impl(x, expansion_point, info)?;
                for arg in iter {
                    series = &series + &arg.series_impl(x, expansion_point, info)?;
                }

                Ok(series)
            }
        }
    }
}

impl Mul<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn mul(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        (&self) * rhs
    }
}

impl Mul<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn mul(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs * self
    }
}

impl Mul<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn mul(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs * &self
    }
}

impl Mul<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn mul(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        let Variable::Symbol(x) = self.get_variable().as_ref().clone() else {
            panic!("Series variable is not a symbol");
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                &AtomField::new(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = rhs
                .as_view()
                .series_impl(x, expansion_point.as_view(), &info)?
                * self;
            if series.relative_order() >= self.relative_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Add<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn add(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        (&self) + rhs
    }
}

impl Add<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn add(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs + self
    }
}

impl Add<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn add(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs + &self
    }
}

impl Add<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn add(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        let Variable::Symbol(x) = self.get_variable().as_ref().clone() else {
            panic!("Series variable is not a symbol");
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                &AtomField::new(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = rhs
                .as_view()
                .series_impl(x, expansion_point.as_view(), &info)?
                + self.clone();
            if series.absolute_order() >= self.absolute_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Div<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn div(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        (&self) / rhs
    }
}

impl Div<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn div(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs.rpow((-1, 1).into()).unwrap() * self
    }
}

impl Div<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn div(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        rhs.rpow((-1, 1).into()).unwrap() * &self
    }
}

impl Div<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn div(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        let Variable::Symbol(x) = self.get_variable().as_ref().clone() else {
            panic!("Series variable is not a symbol");
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                &AtomField::new(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = self
                / &rhs
                    .as_view()
                    .series_impl(x, expansion_point.as_view(), &info)?;
            if series.relative_order() >= self.relative_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Sub<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn sub(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        (&self) + &(-rhs)
    }
}

impl Sub<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn sub(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        -rhs.clone() + self
    }
}

impl Sub<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, &'static str>;

    fn sub(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, &'static str> {
        -rhs.clone() + &self
    }
}

impl Sub<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, &'static str>;

    fn sub(self, rhs: &Atom) -> Result<Series<AtomField>, &'static str> {
        self + &(-rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::atom::{Atom, AtomCore, Symbol};

    #[test]
    fn derivative() {
        let v1 = Symbol::new("v1");
        let inputs = [
            "(1+2*v1)^(5+v1)",
            "log(2*v1) + exp(3*v1) + sin(4*v1) + cos(y*v1)",
            "f(v1^2,v1)",
            "der(0,1,f(v1,v1^3))",
        ];
        let r = inputs.map(|input| Atom::parse(input).unwrap().derivative(v1));

        let res = [
            "(2*v1+1)^(v1+5)*log(2*v1+1)+2*(v1+5)*(2*v1+1)^(v1+4)",
            "2*(2*v1)^-1+3*exp(3*v1)+4*cos(4*v1)-y*sin(v1*y)",
            "der(0,1,f(v1^2,v1))+2*v1*der(1,0,f(v1^2,v1))",
            "der(1,1,f(v1,v1^3))+3*v1^2*der(0,2,f(v1,v1^3))",
        ];
        let res = res.map(|input| Atom::parse(input).unwrap());

        assert_eq!(r, res);
    }

    #[test]
    fn series() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("exp(v1^2+1)*log(v1+3)/v1/(v1+1)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 2.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse(
            "1/3*exp(1)+v1*(-7/18*exp(1)+2*exp(1)*log(3))+v1^2*(119/162*exp(1)-2*exp(1)*log(3))-exp(1)*log(3)+v1^-1*exp(1)*log(3)",
        )
        .unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_shift() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("1/(v1+1)").unwrap();
        let t = input
            .series(v1, Atom::new_num(-1).as_view(), 5.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse("1/(v1+1)").unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_spurious_pole() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("(1-cos(v1))/sin(v1)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 5.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse("1/2*v1+1/24*v1^3+1/240*v1^5").unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_logx() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("log(v1)*(1+v1)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse("log(v1)+v1*log(v1)").unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_sqrt() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("(v1^3+v1+1)^(1/2)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse("1+1/2*v1-1/8*v1^2+9/16*v1^3-37/128*v1^4").unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_fractions() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("1/v1^5").unwrap();

        let t = input
            .series(v1, Atom::new_num(0).as_view(), 3.into(), true)
            .unwrap();

        let t2 = t.rpow((1, 3).into()).unwrap();

        assert_eq!(t2.absolute_order(), (22, 3).into());
    }

    #[test]
    fn series_zero() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("1/v1^2+1/v1+v1").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 0.into(), true)
            .unwrap();

        assert_eq!(t.to_atom().expand(), Atom::parse("v1^-2+v1^-1").unwrap());
    }

    #[test]
    fn series_poles() {
        let v1 = Symbol::new("v1");
        let input = Atom::parse("1/(v1^10+v1^20)").unwrap();

        let t = input
            .series(v1, Atom::new_num(0).as_view(), (-1).into(), true)
            .unwrap()
            .to_atom();

        assert_eq!(t, Atom::parse("v1^-10").unwrap())
    }

    #[test]
    fn series_user_function() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("f(exp(v1),sin(v1))").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 2.into(), true)
            .unwrap()
            .to_atom();

        let res = Atom::parse(
            "der(0,0,f(1,0))+f(1,0)+v1*(der(0,1,f(1,0))+der(1,0,f(1,0)))
            +v1^2*(1/2*der(0,2,f(1,0))+1/2*der(1,0,f(1,0))+der(1,1,f(1,0))+1/2*der(2,0,f(1,0)))",
        )
        .unwrap();
        assert_eq!(t, res);
    }

    #[test]
    fn series_exp_log() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("1+2*log(v1^4)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap()
            .exp()
            .unwrap();

        assert_eq!(t.to_atom().expand(), Atom::parse("v1^8*exp(1)",).unwrap());
    }

    #[test]
    fn series_sub_atom() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("1/(1-v1)").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap();

        let r = (t - &Atom::parse("1/v1+1").unwrap()).unwrap();

        assert_eq!(r.absolute_order(), (5, 1).into());
        assert_eq!(
            r.to_atom(),
            Atom::parse("-1*v1^-1+v1+v1^2+v1^3+v1^4").unwrap()
        );
    }

    #[test]
    fn series_div_atom() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("v1").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap();

        let r = ((t / &Atom::parse("exp(v1)-1").unwrap()).unwrap() * &Atom::parse("v1").unwrap())
            .unwrap();

        assert_eq!(r.relative_order(), (4, 1).into());
        assert_eq!(r.to_atom(), Atom::parse("v1+-1/2*v1^2+1/12*v1^3").unwrap());
    }

    #[test]
    fn series_relative_order() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("exp(v1)/v1-1/6*v1^2").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), false)
            .unwrap();

        assert_eq!(t.relative_order(), (4, 1).into());
        assert_eq!(t.to_atom(), Atom::parse("v1^-1+1+1/2*v1").unwrap());
    }

    #[test]
    fn series_truncate() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("v1^10").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap();
        assert_eq!(t.absolute_order(), (10, 1).into());
        assert_eq!(t.relative_order(), (0, 1).into());
        assert_eq!(t.to_atom(), Atom::parse("0").unwrap());

        let r = (&t * &input).unwrap();
        assert_eq!(r.absolute_order(), (20, 1).into());
    }

    #[test]
    fn series_empty() {
        let v1 = Symbol::new("v1");

        let input = Atom::parse("v1").unwrap();
        let t = input
            .series(v1, Atom::new_num(0).as_view(), 4.into(), true)
            .unwrap();

        let r = &t - &t;

        let t2 = Atom::parse("v1^6")
            .unwrap()
            .series(v1, Atom::new_num(0).as_view(), 4.into(), false)
            .unwrap();

        let x = (&r + &Atom::parse("v1^6").unwrap()).unwrap();
        assert_eq!(r.absolute_order(), (5, 1).into());

        let c = x.cos().unwrap();
        assert_eq!(c.absolute_order(), (10, 1).into());
        assert_eq!(c.relative_order(), (10, 1).into());

        let s = x.sin().unwrap();
        assert_eq!(s.absolute_order(), (5, 1).into());
        assert_eq!(s.relative_order(), (5, 1).into());

        let e = x.exp().unwrap();
        assert_eq!(e.absolute_order(), (5, 1).into());
        assert_eq!(e.relative_order(), (5, 1).into());

        let add = &r + &t2;
        assert_eq!(add.absolute_order(), (5, 1).into());
        let mul = &r * &t2;
        assert_eq!(mul.absolute_order(), (11, 1).into());
    }
}
