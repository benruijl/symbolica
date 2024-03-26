use dyn_clone::DynClone;

use crate::{
    representations::{default::ListSlice, Atom, AtomView, Num, SliceType, Symbol},
    state::{State, Workspace},
    transformer::{Transformer, TransformerError},
};

#[derive(Clone)]
pub enum Pattern {
    Literal(Atom),
    Wildcard(Symbol),
    Fn(Symbol, Vec<Pattern>),
    Pow(Box<[Pattern; 2]>),
    Mul(Vec<Pattern>),
    Add(Vec<Pattern>),
    Transformer(Box<(Option<Pattern>, Vec<Transformer>)>),
}

impl Atom {
    pub fn into_pattern(&self) -> Pattern {
        Pattern::from_view(self.as_view(), true)
    }
}

impl<'a> AtomView<'a> {
    pub fn into_pattern(self) -> Pattern {
        Pattern::from_view(self, true)
    }
}

impl Pattern {
    pub fn parse(input: &str) -> Result<Pattern, String> {
        // TODO: use workspace instead of owned atom
        Ok(Atom::parse(input)?.into_pattern())
    }

    /// Convert the pattern to an atom, if there are not transformers present.
    pub fn to_atom(&self) -> Result<Atom, &'static str> {
        Workspace::get_local().with(|ws| {
            let mut out = Atom::new();
            self.to_atom_impl(ws, &mut out)?;
            Ok(out)
        })
    }

    fn to_atom_impl(&self, ws: &Workspace, out: &mut Atom) -> Result<(), &'static str> {
        match self {
            Pattern::Literal(a) => {
                out.set_from_view(&a.as_view());
            }
            Pattern::Wildcard(s) => {
                out.to_var(*s);
            }
            Pattern::Fn(s, a) => {
                let mut f = ws.new_atom();
                let fun = f.to_fun(*s);

                for arg in a {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    fun.add_arg(arg_h.as_view());
                }

                f.as_view().normalize(ws, out);
            }
            Pattern::Pow(p) => {
                let mut base = ws.new_atom();
                p[0].to_atom_impl(ws, &mut base)?;

                let mut exp = ws.new_atom();
                p[1].to_atom_impl(ws, &mut exp)?;

                let mut pow_h = ws.new_atom();
                pow_h.to_pow(base.as_view(), exp.as_view());
                pow_h.as_view().normalize(ws, out);
            }
            Pattern::Mul(m) => {
                let mut mul_h = ws.new_atom();
                let mul = mul_h.to_mul();

                for arg in m {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    mul.extend(arg_h.as_view());
                }

                mul_h.as_view().normalize(ws, out);
            }
            Pattern::Add(a) => {
                let mut add_h = ws.new_atom();
                let add = add_h.to_add();

                for arg in a {
                    let mut arg_h = ws.new_atom();
                    arg.to_atom_impl(ws, &mut arg_h)?;
                    add.extend(arg_h.as_view());
                }

                add_h.as_view().normalize(ws, out);
            }
            Pattern::Transformer(_) => Err("Cannot convert transformer to atom")?,
        }

        Ok(())
    }

    pub fn add(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                // create new literal
                let mut e = workspace.new_atom();
                let a = e.to_add();

                a.extend(l1.as_view());
                a.extend(l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        let mut new_args = vec![];
        if let Pattern::Add(l1) = self {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(self.clone());
        }
        if let Pattern::Add(l1) = rhs {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(rhs.clone());
        }

        // TODO: fuse literal parts
        Pattern::Add(new_args)
    }

    pub fn mul(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                let mut e = workspace.new_atom();
                let a = e.to_mul();

                a.extend(l1.as_view());
                a.extend(l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        let mut new_args = vec![];
        if let Pattern::Mul(l1) = self {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(self.clone());
        }
        if let Pattern::Mul(l1) = rhs {
            new_args.extend_from_slice(l1);
        } else {
            new_args.push(rhs.clone());
        }

        // TODO: fuse literal parts
        Pattern::Mul(new_args)
    }

    pub fn div(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l2) = rhs {
            let mut pow = workspace.new_atom();
            pow.to_num((-1).into());

            let mut e = workspace.new_atom();
            e.to_pow(l2.as_view(), pow.as_view());

            let mut b = Atom::default();
            e.as_view().normalize(workspace, &mut b);

            match self {
                Pattern::Mul(m) => {
                    let mut new_args = m.clone();
                    new_args.push(Pattern::Literal(b));
                    Pattern::Mul(new_args)
                }
                Pattern::Literal(l1) => {
                    let mut m = workspace.new_atom();
                    let md = m.to_mul();

                    md.extend(l1.as_view());
                    md.extend(b.as_view());

                    let mut b = Atom::default();
                    m.as_view().normalize(workspace, &mut b);
                    Pattern::Literal(b)
                }
                _ => Pattern::Mul(vec![self.clone(), Pattern::Literal(b)]),
            }
        } else {
            let exp = Num::new((-1).into()).into();

            let rhs = Pattern::Mul(vec![
                self.clone(),
                Pattern::Pow(Box::new([rhs.clone(), Pattern::Literal(exp)])),
            ]);

            match self {
                Pattern::Mul(m) => {
                    let mut new_args = m.clone();
                    new_args.push(rhs);
                    Pattern::Mul(new_args)
                }
                _ => Pattern::Mul(vec![self.clone(), rhs]),
            }
        }
    }

    pub fn pow(&self, rhs: &Self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            if let Pattern::Literal(l2) = rhs {
                let mut e = workspace.new_atom();
                e.to_pow(l1.as_view(), l2.as_view());

                let mut b = Atom::default();
                e.as_view().normalize(workspace, &mut b);

                return Pattern::Literal(b);
            }
        }

        Pattern::Pow(Box::new([self.clone(), rhs.clone()]))
    }

    pub fn neg(&self, workspace: &Workspace) -> Self {
        if let Pattern::Literal(l1) = self {
            let mut e = workspace.new_atom();
            let a = e.to_mul();

            let mut sign = workspace.new_atom();
            sign.to_num((-1).into());

            a.extend(l1.as_view());
            a.extend(sign.as_view());

            let mut b = Atom::default();
            e.as_view().normalize(workspace, &mut b);

            Pattern::Literal(b)
        } else {
            let sign = Num::new((-1).into()).into();

            // TODO: simplify if a literal is already present
            Pattern::Mul(vec![self.clone(), Pattern::Literal(sign)])
        }
    }
}

impl Pattern {
    /// A quick check to see if a pattern can match.
    #[inline]
    pub fn could_match(&self, target: AtomView) -> bool {
        match (self, target) {
            (Pattern::Fn(f1, _), AtomView::Fun(f2)) => {
                f1.get_wildcard_level() > 0 || *f1 == f2.get_symbol()
            }
            (Pattern::Mul(_), AtomView::Mul(_)) => true,
            (Pattern::Add(_), AtomView::Add(_)) => true,
            (Pattern::Wildcard(_), _) => true,
            (Pattern::Pow(_), AtomView::Pow(_)) => true,
            (Pattern::Literal(p), _) => p.as_view() == target,
            (Pattern::Transformer(_), _) => unreachable!(),
            (_, _) => false,
        }
    }

    /// Check if the expression `atom` contains a wildcard.
    fn has_wildcard(atom: AtomView<'_>) -> bool {
        match atom {
            AtomView::Num(_) => false,
            AtomView::Var(v) => v.get_wildcard_level() > 0,
            AtomView::Fun(f) => {
                if f.get_symbol().get_wildcard_level() > 0 {
                    return true;
                }

                for arg in f.iter() {
                    if Self::has_wildcard(arg) {
                        return true;
                    }
                }
                false
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                Self::has_wildcard(base) || Self::has_wildcard(exp)
            }
            AtomView::Mul(m) => {
                for child in m.iter() {
                    if Self::has_wildcard(child) {
                        return true;
                    }
                }
                false
            }
            AtomView::Add(a) => {
                for child in a.iter() {
                    if Self::has_wildcard(child) {
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Create a pattern from an atom view.
    fn from_view(atom: AtomView<'_>, is_top_layer: bool) -> Pattern {
        // split up Add and Mul for literal patterns as well so that x+y can match to x+y+z
        if Self::has_wildcard(atom)
            || is_top_layer && matches!(atom, AtomView::Mul(_) | AtomView::Add(_))
        {
            match atom {
                AtomView::Var(v) => Pattern::Wildcard(v.get_symbol()),
                AtomView::Fun(f) => {
                    let name = f.get_symbol();

                    let mut args = Vec::with_capacity(f.get_nargs());
                    for arg in f.iter() {
                        args.push(Self::from_view(arg, false));
                    }

                    Pattern::Fn(name, args)
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    Pattern::Pow(Box::new([
                        Self::from_view(base, false),
                        Self::from_view(exp, false),
                    ]))
                }
                AtomView::Mul(m) => {
                    let mut args = Vec::with_capacity(m.get_nargs());

                    for child in m.iter() {
                        args.push(Self::from_view(child, false));
                    }

                    Pattern::Mul(args)
                }
                AtomView::Add(a) => {
                    let mut args = Vec::with_capacity(a.get_nargs());
                    for child in a.iter() {
                        args.push(Self::from_view(child, false));
                    }

                    Pattern::Add(args)
                }
                AtomView::Num(_) => unreachable!("Number cannot have wildcard"),
            }
        } else {
            let mut oa = Atom::default();
            oa.set_from_view(&atom);
            Pattern::Literal(oa)
        }
    }

    /// Substitute the wildcards in the pattern with the values in the match stack.
    pub fn substitute_wildcards(
        &self,
        workspace: &Workspace,
        out: &mut Atom,
        match_stack: &MatchStack,
    ) -> Result<(), TransformerError> {
        match self {
            Pattern::Wildcard(name) => {
                if let Some(w) = match_stack.get(*name) {
                    w.to_atom(out);
                } else {
                    panic!("Unsubstituted wildcard {}", name.get_id());
                }
            }
            Pattern::Fn(mut name, args) => {
                if name.get_wildcard_level() > 0 {
                    if let Some(w) = match_stack.get(name) {
                        if let Match::FunctionName(fname) = w {
                            name = *fname
                        } else {
                            unreachable!("Wildcard must be a function name")
                        }
                    } else {
                        panic!("Unsubstituted wildcard {}", name.get_id());
                    }
                }

                let mut func_h = workspace.new_atom();
                let func = func_h.to_fun(name);

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => func.add_arg(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Arg | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            func.add_arg(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom(&mut handle);
                                        func.add_arg(handle.as_view())
                                    }
                                },
                                Match::FunctionName(_) => {
                                    unreachable!("Wildcard cannot be function name")
                                }
                            }

                            continue;
                        } else {
                            panic!("Unsubstituted wildcard {}", name.get_id());
                        }
                    }

                    let mut handle = workspace.new_atom();
                    arg.substitute_wildcards(workspace, &mut handle, match_stack)?;
                    func.add_arg(handle.as_view());
                }

                func_h.as_view().normalize(workspace, out);
            }
            Pattern::Pow(base_and_exp) => {
                let mut base = workspace.new_atom();
                let mut exp = workspace.new_atom();
                let mut oas = [&mut base, &mut exp];

                for (out, arg) in oas.iter_mut().zip(base_and_exp.iter()) {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => out.set_from_view(s),
                                Match::Multiple(_, _) => {
                                    let mut handle = workspace.new_atom();
                                    w.to_atom(&mut handle);
                                    out.set_from_view(&handle.as_view())
                                }
                                Match::FunctionName(_) => {
                                    unreachable!("Wildcard cannot be function name")
                                }
                            }

                            continue;
                        } else {
                            panic!("Unsubstituted wildcard {}", w.get_id());
                        }
                    }

                    let mut handle = workspace.new_atom();
                    arg.substitute_wildcards(workspace, &mut handle, match_stack)?;
                    out.set_from_view(&handle.as_view());
                }

                let mut pow_h = workspace.new_atom();
                pow_h.to_pow(oas[0].as_view(), oas[1].as_view());
                pow_h.as_view().normalize(workspace, out);
            }
            Pattern::Mul(args) => {
                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => mul.extend(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Mul | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            mul.extend(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom(&mut handle);
                                        mul.extend(handle.as_view())
                                    }
                                },
                                Match::FunctionName(_) => {
                                    unreachable!("Wildcard cannot be function name")
                                }
                            }

                            continue;
                        } else {
                            panic!("Unsubstituted wildcard {}", w.get_id());
                        }
                    }

                    let mut handle = workspace.new_atom();
                    arg.substitute_wildcards(workspace, &mut handle, match_stack)?;
                    mul.extend(handle.as_view());
                }
                mul_h.as_view().normalize(workspace, out);
            }
            Pattern::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();

                for arg in args {
                    if let Pattern::Wildcard(w) = arg {
                        if let Some(w) = match_stack.get(*w) {
                            match w {
                                Match::Single(s) => add.extend(*s),
                                Match::Multiple(t, wargs) => match t {
                                    SliceType::Add | SliceType::Empty | SliceType::One => {
                                        for arg in wargs {
                                            add.extend(*arg);
                                        }
                                    }
                                    _ => {
                                        let mut handle = workspace.new_atom();
                                        w.to_atom(&mut handle);
                                        add.extend(handle.as_view())
                                    }
                                },
                                Match::FunctionName(_) => {
                                    unreachable!("Wildcard cannot be function name")
                                }
                            }

                            continue;
                        } else {
                            panic!("Unsubstituted wildcard {}", w.get_id());
                        }
                    }

                    let mut handle = workspace.new_atom();
                    arg.substitute_wildcards(workspace, &mut handle, match_stack)?;
                    add.extend(handle.as_view());
                }
                add_h.as_view().normalize(workspace, out);
            }
            Pattern::Literal(oa) => {
                out.set_from_view(&oa.as_view());
            }
            Pattern::Transformer(p) => {
                let (pat, ts) = &**p;
                let pat = pat.as_ref().ok_or_else(|| {
                    TransformerError::ValueError(
                        "Transformer is missing an expression to act on.".to_owned(),
                    )
                })?;

                let mut handle = workspace.new_atom();
                pat.substitute_wildcards(workspace, &mut handle, match_stack)?;

                Transformer::execute(handle.as_view(), ts, workspace, out)?;
            }
        }

        Ok(())
    }

    /// Return an iterator that replaces the pattern in the target once.
    pub fn replace_iter<'a>(
        &'a self,
        target: AtomView<'a>,
        rhs: &'a Pattern,
        conditions: &'a Condition<WildcardAndRestriction>,
        settings: &'a MatchSettings,
    ) -> ReplaceIterator<'a, 'a> {
        ReplaceIterator::new(self, target, rhs, conditions, settings)
    }

    /// Replace all occurrences of the pattern in the target
    /// For every matched atom, the first canonical match is used and then the atom is skipped.
    pub fn replace_all(
        &self,
        target: AtomView<'_>,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
    ) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.replace_all_with_ws_into(target, rhs, ws, conditions, settings, &mut out);
            out.into_inner()
        })
    }

    /// Replace all occurrences of the pattern in the target, returning `true` iff a match was found.
    /// For every matched atom, the first canonical match is used and then the atom is skipped.
    pub fn replace_all_into(
        &self,
        target: AtomView<'_>,
        rhs: &Pattern,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
        out: &mut Atom,
    ) -> bool {
        Workspace::get_local()
            .with(|ws| self.replace_all_with_ws_into(target, rhs, ws, conditions, settings, out))
    }

    /// Replace all occurrences of the pattern in the target, returning `true` iff a match was found.
    /// For every matched atom, the first canonical match is used and then the atom is skipped.
    pub fn replace_all_with_ws_into(
        &self,
        target: AtomView<'_>,
        rhs: &Pattern,
        workspace: &Workspace,
        conditions: Option<&Condition<WildcardAndRestriction>>,
        settings: Option<&MatchSettings>,
        out: &mut Atom,
    ) -> bool {
        let matched = self.replace_all_no_norm(
            target,
            rhs,
            workspace,
            conditions.unwrap_or(&Condition::default()),
            settings.unwrap_or(&MatchSettings::default()),
            0,
            out,
        );

        if matched {
            let mut norm = workspace.new_atom();
            out.as_view().normalize(workspace, &mut norm);
            std::mem::swap(out, &mut norm);
        }

        matched
    }

    /// Replace all occurrences of the pattern in the target, without normalizing the output.
    fn replace_all_no_norm(
        &self,
        target: AtomView<'_>,
        rhs: &Pattern,
        workspace: &Workspace,
        conditions: &Condition<WildcardAndRestriction>,
        settings: &MatchSettings,
        level: usize,
        out: &mut Atom,
    ) -> bool {
        if let Some(max_level) = settings.level_range.1 {
            if level > max_level {
                out.set_from_view(&target);
                return false;
            }
        }

        if level >= settings.level_range.0 && self.could_match(target) {
            let mut match_stack = MatchStack::new(conditions, settings);

            let mut it = AtomMatchIterator::new(self, target);
            //let mut it = SubSliceIterator::new(self, target, &match_stack, true);
            if let Some((_, used_flags)) = it.next(&mut match_stack) {
                let mut rhs_subs = workspace.new_atom();
                rhs.substitute_wildcards(workspace, &mut rhs_subs, &match_stack)
                    .unwrap(); // TODO: escalate?

                if used_flags.iter().all(|x| *x) {
                    // all used, return rhs
                    out.set_from_view(&rhs_subs.as_view());
                    return true;
                }

                match target {
                    AtomView::Mul(m) => {
                        let out = out.to_mul();

                        for (child, used) in m.iter().zip(used_flags) {
                            if !used {
                                out.extend(child);
                            }
                        }

                        out.extend(rhs_subs.as_view());
                    }
                    AtomView::Add(a) => {
                        let out = out.to_add();

                        for (child, used) in a.iter().zip(used_flags) {
                            if !used {
                                out.extend(child);
                            }
                        }

                        out.extend(rhs_subs.as_view());
                    }
                    _ => {
                        out.set_from_view(&rhs_subs.as_view());
                    }
                }

                return true;
            }
        }

        // no match found at this level, so check the children
        let submatch = match target {
            AtomView::Fun(f) => {
                let out = out.to_fun(f.get_symbol());

                let mut submatch = false;

                for child in f.iter() {
                    let mut child_buf = workspace.new_atom();

                    submatch |= self.replace_all_no_norm(
                        child,
                        rhs,
                        workspace,
                        conditions,
                        settings,
                        level + 1,
                        &mut child_buf,
                    );

                    out.add_arg(child_buf.as_view());
                }

                out.set_normalized(!submatch && f.is_normalized());
                submatch
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_out = workspace.new_atom();
                let mut submatch = self.replace_all_no_norm(
                    base,
                    rhs,
                    workspace,
                    conditions,
                    settings,
                    if settings.level_is_tree_depth {
                        level + 1
                    } else {
                        level
                    },
                    &mut base_out,
                );

                let mut exp_out = workspace.new_atom();
                submatch |= self.replace_all_no_norm(
                    exp,
                    rhs,
                    workspace,
                    conditions,
                    settings,
                    if settings.level_is_tree_depth {
                        level + 1
                    } else {
                        level
                    },
                    &mut exp_out,
                );

                let out = out.to_pow(base_out.as_view(), exp_out.as_view());
                out.set_normalized(!submatch && p.is_normalized());
                submatch
            }
            AtomView::Mul(m) => {
                let mul = out.to_mul();

                let mut submatch = false;
                for child in m.iter() {
                    let mut child_buf = workspace.new_atom();

                    submatch |= self.replace_all_no_norm(
                        child,
                        rhs,
                        workspace,
                        conditions,
                        settings,
                        if settings.level_is_tree_depth {
                            level + 1
                        } else {
                            level
                        },
                        &mut child_buf,
                    );

                    mul.extend(child_buf.as_view());
                }

                mul.set_has_coefficient(m.has_coefficient());
                mul.set_normalized(!submatch && m.is_normalized());
                submatch
            }
            AtomView::Add(a) => {
                let out = out.to_add();
                let mut submatch = false;
                for child in a.iter() {
                    let mut child_buf = workspace.new_atom();

                    submatch |= self.replace_all_no_norm(
                        child,
                        rhs,
                        workspace,
                        conditions,
                        settings,
                        if settings.level_is_tree_depth {
                            level + 1
                        } else {
                            level
                        },
                        &mut child_buf,
                    );

                    out.extend(child_buf.as_view());
                }
                out.set_normalized(!submatch && a.is_normalized());
                submatch
            }
            _ => {
                out.set_from_view(&target); // no children
                false
            }
        };

        submatch
    }

    pub fn pattern_match<'a>(
        &'a self,
        target: AtomView<'a>,
        conditions: &'a Condition<WildcardAndRestriction>,
        settings: &'a MatchSettings,
    ) -> PatternAtomTreeIterator<'a, 'a> {
        PatternAtomTreeIterator::new(self, target, conditions, settings)
    }
}

impl std::fmt::Debug for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Wildcard(arg0) => f.debug_tuple("Wildcard").field(arg0).finish(),
            Self::Fn(arg0, arg1) => f.debug_tuple("Fn").field(arg0).field(arg1).finish(),
            Self::Pow(arg0) => f.debug_tuple("Pow").field(arg0).finish(),
            Self::Mul(arg0) => f.debug_tuple("Mul").field(arg0).finish(),
            Self::Add(arg0) => f.debug_tuple("Add").field(arg0).finish(),
            Self::Literal(arg0) => f.debug_tuple("Literal").field(arg0).finish(),
            Self::Transformer(arg0) => f.debug_tuple("Transformer").field(arg0).finish(),
        }
    }
}

pub trait FilterFn: for<'a, 'b> Fn(&'a Match<'b>) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(FilterFn);
impl<T: Clone + Send + Sync + for<'a, 'b> Fn(&'a Match<'b>) -> bool> FilterFn for T {}

pub trait CmpFn: for<'a, 'b> Fn(&Match<'_>, &Match<'_>) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(CmpFn);
impl<T: Clone + Send + Sync + for<'a, 'b> Fn(&Match<'_>, &Match<'_>) -> bool> CmpFn for T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    Num,
    Var,
    Add,
    Mul,
    Pow,
    Fun,
}

/// Restrictions for a wildcard. Note that a length restriction
/// applies at any level and therefore
/// `x_*f(x_) : length(x) == 2`
/// does not match to `x*y*f(x*y)`, since the pattern `x_` has length
/// 1 inside the function argument.
pub enum PatternRestriction {
    Length(usize, Option<usize>), // min-max range
    IsAtomType(AtomType),
    IsLiteralWildcard(Symbol),
    Filter(Box<dyn FilterFn>),
    Cmp(Symbol, Box<dyn CmpFn>),
    NotGreedy,
}

pub type WildcardAndRestriction = (Symbol, PatternRestriction);

/// A logical expression.
#[derive(Clone, Debug, Default)]
pub enum Condition<T> {
    And(Box<(Condition<T>, Condition<T>)>),
    Or(Box<(Condition<T>, Condition<T>)>),
    Not(Box<Condition<T>>),
    Yield(T),
    #[default]
    True,
    False,
}

impl<T> From<T> for Condition<T> {
    fn from(value: T) -> Self {
        Condition::Yield(value)
    }
}

impl<T, R: Into<Condition<T>>> std::ops::BitOr<R> for Condition<T> {
    type Output = Condition<T>;

    fn bitor(self, rhs: R) -> Self::Output {
        Condition::Or(Box::new((self, rhs.into())))
    }
}

impl<T, R: Into<Condition<T>>> std::ops::BitAnd<R> for Condition<T> {
    type Output = Condition<T>;

    fn bitand(self, rhs: R) -> Self::Output {
        Condition::And(Box::new((self, rhs.into())))
    }
}

impl<T> std::ops::Not for Condition<T> {
    type Output = Condition<T>;

    fn not(self) -> Self::Output {
        Condition::Not(Box::new(self))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionResult {
    True,
    False,
    Inconclusive,
}

impl std::ops::BitOr<ConditionResult> for ConditionResult {
    type Output = ConditionResult;

    fn bitor(self, rhs: ConditionResult) -> Self::Output {
        match (self, rhs) {
            (ConditionResult::True, _) => ConditionResult::True,
            (_, ConditionResult::True) => ConditionResult::True,
            (ConditionResult::False, ConditionResult::False) => ConditionResult::False,
            _ => ConditionResult::Inconclusive,
        }
    }
}

impl std::ops::BitAnd<ConditionResult> for ConditionResult {
    type Output = ConditionResult;

    fn bitand(self, rhs: ConditionResult) -> Self::Output {
        match (self, rhs) {
            (ConditionResult::False, _) => ConditionResult::False,
            (_, ConditionResult::False) => ConditionResult::False,
            (ConditionResult::True, ConditionResult::True) => ConditionResult::True,
            _ => ConditionResult::Inconclusive,
        }
    }
}

impl std::ops::Not for ConditionResult {
    type Output = ConditionResult;

    fn not(self) -> Self::Output {
        match self {
            ConditionResult::True => ConditionResult::False,
            ConditionResult::False => ConditionResult::True,
            ConditionResult::Inconclusive => ConditionResult::Inconclusive,
        }
    }
}

impl From<bool> for ConditionResult {
    fn from(value: bool) -> Self {
        if value {
            ConditionResult::True
        } else {
            ConditionResult::False
        }
    }
}

impl Condition<WildcardAndRestriction> {
    /// Check if the conditions on `var` are met
    fn check_possible(&self, var: Symbol, value: &Match, stack: &MatchStack) -> ConditionResult {
        match self {
            Condition::And(a) => {
                a.0.check_possible(var, value, stack) & a.1.check_possible(var, value, stack)
            }
            Condition::Or(o) => {
                o.0.check_possible(var, value, stack) | o.1.check_possible(var, value, stack)
            }
            Condition::Not(n) => !n.check_possible(var, value, stack),
            Condition::True => ConditionResult::True,
            Condition::False => ConditionResult::False,
            Condition::Yield((v, r)) => {
                if *v != var {
                    match r {
                        PatternRestriction::Cmp(v, _) if *v == var => {}
                        _ => {
                            return ConditionResult::Inconclusive;
                        }
                    }
                }

                match r {
                    PatternRestriction::IsAtomType(t) => {
                        let is_type = match t {
                            AtomType::Num => matches!(value, Match::Single(AtomView::Num(_))),
                            AtomType::Var => matches!(value, Match::Single(AtomView::Var(_))),
                            AtomType::Add => matches!(
                                value,
                                Match::Single(AtomView::Add(_))
                                    | Match::Multiple(SliceType::Add, _)
                            ),
                            AtomType::Mul => matches!(
                                value,
                                Match::Single(AtomView::Mul(_))
                                    | Match::Multiple(SliceType::Mul, _)
                            ),
                            AtomType::Pow => matches!(
                                value,
                                Match::Single(AtomView::Pow(_))
                                    | Match::Multiple(SliceType::Pow, _)
                            ),
                            AtomType::Fun => matches!(value, Match::Single(AtomView::Fun(_))),
                        };

                        (is_type == matches!(r, PatternRestriction::IsAtomType(_))).into()
                    }
                    PatternRestriction::IsLiteralWildcard(wc) => {
                        if let Match::Single(AtomView::Var(v)) = value {
                            (wc == &v.get_symbol()).into()
                        } else {
                            false.into()
                        }
                    }
                    PatternRestriction::Length(min, max) => match &value {
                        Match::Single(_) | Match::FunctionName(_) => {
                            (*min <= 1 && max.map(|m| m >= 1).unwrap_or(true)).into()
                        }
                        Match::Multiple(_, slice) => (*min <= slice.len()
                            && max.map(|m| m >= slice.len()).unwrap_or(true))
                        .into(),
                    },
                    PatternRestriction::Filter(f) => f(value).into(),
                    PatternRestriction::Cmp(v2, f) => {
                        if *v == var {
                            if let Some((_, value2)) = stack.stack.iter().find(|(k, _)| k == v2) {
                                f(value, value2).into()
                            } else {
                                ConditionResult::Inconclusive
                            }
                        } else if let Some((_, value2)) = stack.stack.iter().find(|(k, _)| k == v) {
                            f(value2, value).into()
                        } else {
                            ConditionResult::Inconclusive
                        }
                    }
                    PatternRestriction::NotGreedy => true.into(),
                }
            }
        }
    }

    fn get_range_hint(&self, var: Symbol) -> (Option<usize>, Option<usize>) {
        match self {
            Condition::And(a) => {
                let (min1, max1) = a.0.get_range_hint(var);
                let (min2, max2) = a.1.get_range_hint(var);

                (
                    match (min1, min2) {
                        (None, None) => None,
                        (None, Some(m)) => Some(m),
                        (Some(m), None) => Some(m),
                        (Some(m1), Some(m2)) => Some(m1.max(m2)),
                    },
                    match (max1, max2) {
                        (None, None) => None,
                        (None, Some(m)) => Some(m),
                        (Some(m), None) => Some(m),
                        (Some(m1), Some(m2)) => Some(m1.min(m2)),
                    },
                )
            }
            Condition::Or(o) => {
                // take the extremes of the min and max
                let (min1, max1) = o.0.get_range_hint(var);
                let (min2, max2) = o.1.get_range_hint(var);

                (
                    if let (Some(m1), Some(m2)) = (min1, min2) {
                        Some(m1.min(m2))
                    } else {
                        None
                    },
                    if let (Some(m1), Some(m2)) = (max1, max2) {
                        Some(m1.max(m2))
                    } else {
                        None
                    },
                )
            }
            Condition::Not(_) => {
                // the range is disconnected and therefore cannot be described
                // using our range conditions
                (None, None)
            }
            Condition::True | Condition::False => (None, None),
            Condition::Yield((v, r)) => {
                if *v != var {
                    return (None, None);
                }

                match r {
                    PatternRestriction::Length(min, max) => (Some(*min), *max),
                    PatternRestriction::IsAtomType(
                        AtomType::Var | AtomType::Num | AtomType::Fun,
                    )
                    | PatternRestriction::IsLiteralWildcard(_) => (Some(1), Some(1)),
                    _ => (None, None),
                }
            }
        }
    }
}

impl Clone for PatternRestriction {
    fn clone(&self) -> Self {
        match self {
            Self::Length(min, max) => Self::Length(*min, *max),
            Self::IsAtomType(t) => Self::IsAtomType(*t),
            Self::IsLiteralWildcard(w) => Self::IsLiteralWildcard(*w),
            Self::Filter(f) => Self::Filter(dyn_clone::clone_box(f)),
            Self::Cmp(i, f) => Self::Cmp(*i, dyn_clone::clone_box(f)),
            Self::NotGreedy => Self::NotGreedy,
        }
    }
}

impl std::fmt::Debug for PatternRestriction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(arg0, arg1) => f.debug_tuple("Length").field(arg0).field(arg1).finish(),
            Self::IsAtomType(t) => write!(f, "Is{:?}", t),
            Self::IsLiteralWildcard(arg0) => {
                f.debug_tuple("IsLiteralWildcard").field(arg0).finish()
            }
            Self::Filter(_) => f.debug_tuple("Filter").finish(),
            Self::Cmp(arg0, _) => f.debug_tuple("Cmp").field(arg0).finish(),
            Self::NotGreedy => write!(f, "NotGreedy"),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Match<'a> {
    Single(AtomView<'a>),
    Multiple(SliceType, Vec<AtomView<'a>>),
    FunctionName(Symbol),
}

impl<'a> std::fmt::Debug for Match<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(a) => f.debug_tuple("").field(a).finish(),
            Self::Multiple(t, list) => f.debug_tuple("").field(t).field(list).finish(),
            Self::FunctionName(name) => f.debug_tuple("Fn").field(name).finish(),
        }
    }
}

impl<'a> Match<'a> {
    /// Create a new atom from a matched subexpression.
    /// Arguments lists are wrapped in the function `arg`.
    pub fn to_atom(&self, out: &mut Atom) {
        match self {
            Self::Single(v) => {
                out.set_from_view(v);
            }
            Self::Multiple(t, wargs) => match t {
                SliceType::Add => {
                    let add = out.to_add();
                    for arg in wargs {
                        add.extend(*arg);
                    }

                    add.set_normalized(true);
                }
                SliceType::Mul => {
                    let mul = out.to_mul();
                    for arg in wargs {
                        mul.extend(*arg);
                    }

                    // normalization may be needed, for example
                    // to update the coefficient flag
                }
                SliceType::Arg => {
                    let fun = out.to_fun(State::ARG);
                    for arg in wargs {
                        fun.add_arg(*arg);
                    }

                    fun.set_normalized(true);
                }
                SliceType::Pow => {
                    let p = out.to_pow(wargs[0], wargs[1]);
                    p.set_normalized(true);
                }
                SliceType::One => {
                    out.set_from_view(&wargs[0]);
                }
                SliceType::Empty => {
                    let f = out.to_fun(State::ARG);
                    f.set_normalized(true);
                }
            },
            Self::FunctionName(n) => {
                let f = out.to_fun(*n);
                f.set_normalized(true);
            }
        }
    }
}

/// Settings related to pattern matching.
#[derive(Default, Clone)]
pub struct MatchSettings {
    /// Specifies wildcards that try to match as little as possible.
    pub non_greedy_wildcards: Vec<Symbol>,
    /// Specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when entering a function, or going one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    pub level_range: (usize, Option<usize>),
    /// Determine whether a level reflects the expression tree depth or the function depth.
    pub level_is_tree_depth: bool,
}

/// An insertion-ordered map of wildcard identifiers to a subexpressions.
/// It keeps track of all conditions on wildcards and will check them
/// before inserting.
pub struct MatchStack<'a, 'b> {
    stack: Vec<(Symbol, Match<'a>)>,
    conditions: &'b Condition<WildcardAndRestriction>,
    settings: &'b MatchSettings,
}

impl<'a, 'b> std::fmt::Debug for MatchStack<'a, 'b> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatchStack")
            .field("stack", &self.stack)
            .finish()
    }
}

impl<'a, 'b> MatchStack<'a, 'b> {
    /// Create a new match stack.
    pub fn new(
        conditions: &'b Condition<WildcardAndRestriction>,
        settings: &'b MatchSettings,
    ) -> MatchStack<'a, 'b> {
        MatchStack {
            stack: Vec::new(),
            conditions,
            settings,
        }
    }

    /// Add a new map of identifier `key` to value `value` to the stack and return the size the stack had before inserting this new entry.
    /// If the entry `(key, value)` already exists, it is not inserted again and therefore the returned size is the actual size.
    /// If the `key` exists in the map, but the `value` is different, the insertion is ignored and `None` is returned.
    pub fn insert(&mut self, key: Symbol, value: Match<'a>) -> Option<usize> {
        for (rk, rv) in self.stack.iter() {
            if rk == &key {
                if rv == &value {
                    return Some(self.stack.len());
                } else {
                    return None;
                }
            }
        }

        // test whether the current value passes all conditions
        // or returns an inconclusive result
        if self.conditions.check_possible(key, &value, self) == ConditionResult::False {
            return None;
        }

        self.stack.push((key, value));
        Some(self.stack.len() - 1)
    }

    /// Get the mapped value for the wildcard `key`.
    pub fn get(&self, key: Symbol) -> Option<&Match<'a>> {
        for (rk, rv) in self.stack.iter() {
            if rk == &key {
                return Some(rv);
            }
        }
        None
    }

    /// Return the length of the stack.
    #[inline]
    pub fn len(&self) -> usize {
        self.stack.len()
    }

    /// Truncate the stack to `len`.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.stack.truncate(len)
    }

    /// Get the range of an identifier based on previous matches and based
    /// on conditions.
    pub fn get_range(&self, identifier: Symbol) -> (usize, Option<usize>) {
        if identifier.get_wildcard_level() == 0 {
            return (1, Some(1));
        }

        for (rk, rv) in self.stack.iter() {
            if rk == &identifier {
                return match rv {
                    Match::Single(_) => (1, Some(1)),
                    Match::Multiple(slice_type, slice) => {
                        match slice_type {
                            SliceType::Empty => (0, Some(0)),
                            SliceType::Arg => (slice.len(), Some(slice.len())),
                            _ => {
                                // the length needs to include 1 since for example x*y is only
                                // length one in f(x*y)
                                // TODO: the length can only be 1 or slice.len() and no values in between
                                // so we could optimize this
                                (1, Some(slice.len()))
                            }
                        }
                    }
                    Match::FunctionName(_) => (1, Some(1)),
                };
            }
        }

        let (minimal, maximal) = self.conditions.get_range_hint(identifier);

        match identifier.get_wildcard_level() {
            1 => (minimal.unwrap_or(1), Some(maximal.unwrap_or(1))), // x_
            2 => (minimal.unwrap_or(1), maximal),                    // x__
            _ => (minimal.unwrap_or(0), maximal),                    // x___
        }
    }
}

impl<'a, 'b, 'c> IntoIterator for &'c MatchStack<'a, 'b> {
    type Item = &'c (Symbol, Match<'a>);
    type IntoIter = std::slice::Iter<'c, (Symbol, Match<'a>)>;

    fn into_iter(self) -> Self::IntoIter {
        self.stack.iter()
    }
}

struct WildcardIter {
    initialized: bool,
    name: Symbol,
    indices: Vec<u32>,
    size_target: u32,
    min_size: u32,
    max_size: u32,
    greedy: bool,
}

enum PatternIter<'a, 'b> {
    Literal(Option<usize>, AtomView<'b>),
    Wildcard(WildcardIter),
    Fn(
        Option<usize>,
        Symbol,
        &'b [Pattern],
        Box<Option<SubSliceIterator<'a, 'b>>>,
    ), // index first
    Sequence(
        Option<usize>,
        SliceType,
        &'b [Pattern],
        Box<Option<SubSliceIterator<'a, 'b>>>,
    ),
}

/// An iterator that tries to match an entire atom or
/// a subslice to a pattern.
pub struct AtomMatchIterator<'a, 'b> {
    try_match_atom: bool,
    sl_it: Option<SubSliceIterator<'a, 'b>>,
    pattern: &'b Pattern,
    target: AtomView<'a>,
    old_match_stack_len: Option<usize>,
}

impl<'a, 'b> AtomMatchIterator<'a, 'b> {
    pub fn new(pattern: &'b Pattern, target: AtomView<'a>) -> AtomMatchIterator<'a, 'b> {
        let try_match_atom = matches!(pattern, Pattern::Wildcard(_) | Pattern::Literal(_));

        AtomMatchIterator {
            try_match_atom,
            sl_it: None,
            pattern,
            target,
            old_match_stack_len: None,
        }
    }

    pub fn next(&mut self, match_stack: &mut MatchStack<'a, 'b>) -> Option<(usize, &[bool])> {
        if self.try_match_atom {
            self.try_match_atom = false;

            if let Pattern::Wildcard(w) = self.pattern {
                let range = match_stack.get_range(*w);
                if range.0 <= 1 && range.1.map(|w| w >= 1).unwrap_or(true) {
                    // TODO: any problems with matching Single vs a list?
                    if let Some(new_stack_len) = match_stack.insert(*w, Match::Single(self.target))
                    {
                        self.old_match_stack_len = Some(new_stack_len);
                        return Some((new_stack_len, &[]));
                    }
                }
            } else if let Pattern::Literal(w) = self.pattern {
                if w.as_view() == self.target {
                    return Some((match_stack.len(), &[]));
                }
            }
            // TODO: also do type matches, Fn Fn, etc?
        }

        if let Some(oml) = self.old_match_stack_len {
            match_stack.truncate(oml);
            self.old_match_stack_len = None;
        }

        if matches!(self.pattern, Pattern::Literal(_)) {
            // TODO: also catch Pattern:Add(_) and Pattern:Mul(_) without any sub-wildcards
            return None;
        }

        if self.sl_it.is_none() {
            self.sl_it = Some(SubSliceIterator::new(
                self.pattern,
                self.target,
                match_stack,
                true,
                matches!(self.pattern, Pattern::Wildcard(_) | Pattern::Literal(_)),
            ));
        }

        self.sl_it.as_mut().unwrap().next(match_stack)
    }
}

/// An iterator that matches a slice of patterns to a slice of atoms.
/// Use the [`SubSliceIterator::next`] to get the next match, if any.
///  
/// The flag `complete` determines whether the pattern should match the entire
/// slice `target`. The flag `ordered_gapless` determines whether the the patterns
/// may match the slice of atoms in any order. For a non-symmetric function, this
/// flag should likely be set.
pub struct SubSliceIterator<'a, 'b> {
    pattern: &'b [Pattern], // input term
    target: ListSlice<'a>,
    iterators: Vec<PatternIter<'a, 'b>>,
    used_flag: Vec<bool>,
    initialized: bool,
    matches: Vec<usize>,   // track match stack length
    complete: bool,        // match needs to consume entire target
    ordered_gapless: bool, // pattern should appear ordered and have no gaps
    do_not_match_to_single_atom_in_list: bool,
    do_not_match_entire_slice: bool,
}

impl<'a, 'b> SubSliceIterator<'a, 'b> {
    /// Create an iterator over a pattern applied to a target.
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        match_stack: &MatchStack<'a, 'b>,
        do_not_match_to_single_atom_in_list: bool,
        do_not_match_entire_slice: bool,
    ) -> SubSliceIterator<'a, 'b> {
        let mut shortcut_done = false;

        // a pattern and target can either be a single atom or a list
        // for (list, list)  create a subslice iterator on the lists that is not complete
        // for (single, list), upgrade single to a slice with one element

        let (pat_list, target_list) = match (pattern, target) {
            (Pattern::Mul(m1), AtomView::Mul(m2)) => (m1.as_slice(), m2.to_slice()),
            (Pattern::Add(a1), AtomView::Add(a2)) => (a1.as_slice(), a2.to_slice()),
            (Pattern::Mul(arg) | Pattern::Add(arg), _) => {
                shortcut_done = true; // cannot match
                (arg.as_slice(), ListSlice::from_one(target))
            }
            (Pattern::Wildcard(_), AtomView::Mul(m2)) => {
                (std::slice::from_ref(pattern), m2.to_slice())
            }
            (Pattern::Wildcard(_), AtomView::Add(a2)) => {
                (std::slice::from_ref(pattern), a2.to_slice())
            }
            (_, AtomView::Mul(m2)) => {
                if do_not_match_to_single_atom_in_list {
                    shortcut_done = true; // cannot match
                }
                (std::slice::from_ref(pattern), m2.to_slice())
            }
            (_, AtomView::Add(a2)) => {
                if do_not_match_to_single_atom_in_list {
                    shortcut_done = true; // cannot match
                }
                (std::slice::from_ref(pattern), a2.to_slice())
            }
            (_, _) => (std::slice::from_ref(pattern), ListSlice::from_one(target)),
        };

        // shortcut if the number of arguments is wrong
        let min_length: usize = pat_list
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        let mut target_len = target_list.len();
        if do_not_match_entire_slice {
            target_len -= 1;
        }

        if min_length > target_len {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern: pat_list,
            iterators: if shortcut_done {
                Vec::new()
            } else {
                Vec::with_capacity(pat_list.len())
            },
            matches: if shortcut_done {
                Vec::new()
            } else {
                Vec::with_capacity(pat_list.len())
            },
            used_flag: if shortcut_done {
                vec![]
            } else {
                vec![false; target_list.len()]
            },
            target: target_list,

            initialized: shortcut_done,
            complete: false,
            ordered_gapless: false,
            do_not_match_to_single_atom_in_list,
            do_not_match_entire_slice,
        }
    }

    /// Create a new sub-slice iterator.
    pub fn from_list(
        pattern: &'b [Pattern],
        target: ListSlice<'a>,

        match_stack: &MatchStack<'a, 'b>,
        complete: bool,
        ordered: bool,
    ) -> SubSliceIterator<'a, 'b> {
        let mut shortcut_done = false;

        // shortcut if the number of arguments is wrong
        let min_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).0,
                _ => 1,
            })
            .sum();

        if min_length > target.len() {
            shortcut_done = true;
        };

        let max_length: usize = pattern
            .iter()
            .map(|x| match x {
                Pattern::Wildcard(id) => match_stack.get_range(*id).1.unwrap_or(target.len()),
                _ => 1,
            })
            .sum();

        if complete && max_length < target.len() {
            shortcut_done = true;
        };

        SubSliceIterator {
            pattern,
            iterators: Vec::with_capacity(pattern.len()),
            matches: Vec::with_capacity(pattern.len()),
            used_flag: vec![false; target.len()],
            target,

            initialized: shortcut_done,
            complete,
            ordered_gapless: ordered,
            do_not_match_to_single_atom_in_list: false,
            do_not_match_entire_slice: false,
        }
    }

    /// Get the next matches, where the map of matches is written into `match_stack`.
    /// The function returns the length of the match stack before the last subiterator
    /// matched. This value can be ignored by the end-user. If `None` is returned,
    /// all potential matches will have been generated and the iterator will generate
    /// `None` if called again.
    pub fn next(&mut self, match_stack: &mut MatchStack<'a, 'b>) -> Option<(usize, &[bool])> {
        let mut forward_pass = !self.initialized;
        self.initialized = true;

        'next_match: loop {
            if !forward_pass && self.iterators.is_empty() {
                return None; // done as all options have been exhausted
            }

            if forward_pass && self.iterators.len() == self.pattern.len() {
                // check the proposed solution for extra conditions
                if self.complete && self.used_flag.iter().any(|x| !*x)
                    || self.do_not_match_to_single_atom_in_list // TODO: a function may have more used_flags? does that clash?
                        && self.used_flag.len() > 1
                        && self.used_flag.iter().map(|x| *x as usize).sum::<usize>() == 1
                {
                    // not done as the entire target is not used
                    forward_pass = false;
                } else {
                    // yield the current match
                    return Some((*self.matches.last().unwrap(), &self.used_flag));
                }
            }

            if forward_pass {
                // add new iterator
                let it = match &self.pattern[self.iterators.len()] {
                    Pattern::Wildcard(name) => {
                        let mut size_left = self.used_flag.iter().filter(|x| !*x).count();
                        let range = match_stack.get_range(*name);

                        if self.do_not_match_entire_slice {
                            size_left -= 1;

                            if size_left < range.0 {
                                forward_pass = false;
                                continue 'next_match;
                            }
                        }

                        let mut range = (
                            range.0,
                            range.1.map(|m| m.min(size_left)).unwrap_or(size_left),
                        );

                        // bound the wildcard length based on the bounds of upcoming patterns
                        if self.complete {
                            let mut new_min = size_left;
                            let mut new_max = size_left;
                            for p in &self.pattern[self.iterators.len() + 1..] {
                                let p_range = if let Pattern::Wildcard(name) = p {
                                    match_stack.get_range(*name)
                                } else {
                                    (1, Some(1))
                                };

                                if new_min > 0 {
                                    if let Some(m) = p_range.1 {
                                        new_min -= m.min(new_min);
                                    } else {
                                        new_min = 0;
                                    }
                                }

                                if new_max < p_range.0 {
                                    forward_pass = false;
                                    continue 'next_match;
                                }

                                new_max -= p_range.0;
                            }

                            range.0 = range.0.max(new_min);
                            range.1 = range.1.min(new_max);

                            if range.0 > range.1 {
                                forward_pass = false;
                                continue 'next_match;
                            }
                        }

                        let greedy = !match_stack.settings.non_greedy_wildcards.contains(name);

                        PatternIter::Wildcard(WildcardIter {
                            initialized: false,
                            name: *name,
                            indices: Vec::new(),
                            size_target: if greedy {
                                range.1 as u32
                            } else {
                                range.0 as u32
                            },
                            min_size: range.0 as u32,
                            max_size: range.1 as u32,
                            greedy,
                        })
                    }
                    Pattern::Fn(name, args) => PatternIter::Fn(None, *name, args, Box::new(None)),
                    Pattern::Pow(base_exp) => PatternIter::Sequence(
                        None,
                        SliceType::Pow,
                        base_exp.as_slice(),
                        Box::new(None),
                    ),
                    Pattern::Mul(pat) => {
                        PatternIter::Sequence(None, SliceType::Mul, pat, Box::new(None))
                    }
                    Pattern::Add(pat) => {
                        PatternIter::Sequence(None, SliceType::Add, pat, Box::new(None))
                    }
                    Pattern::Literal(atom) => PatternIter::Literal(None, atom.as_view()),
                    Pattern::Transformer(_) => panic!("Transformer is not allowed on lhs"),
                };

                self.iterators.push(it);
            } else {
                // update an existing iterator, so pop the latest matches (this implies every iter pushes to the match)
                match_stack.truncate(self.matches.pop().unwrap());
            }

            // assume we are in forward pass mode
            // if the iterator does not match this variable is set to false
            forward_pass = true;

            match self.iterators.last_mut().unwrap() {
                PatternIter::Wildcard(w) => {
                    let mut wildcard_forward_pass = !w.initialized;
                    w.initialized = true;

                    'next_wildcard_match: loop {
                        // a wildcard collects indices in increasing order
                        // find the starting point where the last index can be moved to
                        let start_index = w.indices.last().map(|x| *x as usize + 1).unwrap_or(0);

                        if !wildcard_forward_pass {
                            let last_iterator_empty = w.indices.is_empty();
                            if let Some(last_index) = w.indices.pop() {
                                self.used_flag[last_index as usize] = false;
                            }

                            if last_iterator_empty {
                                // the wildcard iterator is exhausted for this target size
                                if w.greedy {
                                    if w.size_target > w.min_size {
                                        w.size_target -= 1;
                                    } else {
                                        break;
                                    }
                                } else if w.size_target < w.max_size {
                                    w.size_target += 1;
                                } else {
                                    break;
                                }
                            } else if self.ordered_gapless {
                                // drain the entire constructed range and start from scratch
                                continue 'next_wildcard_match;
                            }
                        }

                        // check for an empty slice match
                        if w.size_target == 0 && w.indices.is_empty() {
                            if let Some(new_stack_len) = match_stack
                                .insert(w.name, Match::Multiple(SliceType::Empty, Vec::new()))
                            {
                                self.matches.push(new_stack_len);
                                continue 'next_match;
                            } else {
                                wildcard_forward_pass = false;
                                continue 'next_wildcard_match;
                            }
                        }

                        let mut tried_first_option = false;
                        for k in start_index..self.target.len() {
                            if self.ordered_gapless && tried_first_option {
                                break;
                            }

                            if self.used_flag[k] {
                                continue;
                            }

                            self.used_flag[k] = true;
                            w.indices.push(k as u32);

                            if w.indices.len() == w.size_target as usize {
                                tried_first_option = true;

                                // simplify case of 1 argument, this is important for matching to work, since mul(x) = add(x) = arg(x) for any x
                                let matched = if w.indices.len() == 1 {
                                    match self.target.get(w.indices[0] as usize) {
                                        AtomView::Mul(m) => Match::Multiple(SliceType::Mul, {
                                            let mut v = Vec::new();
                                            for x in m.iter() {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        AtomView::Add(a) => Match::Multiple(SliceType::Add, {
                                            let mut v = Vec::new();
                                            for x in a.iter() {
                                                v.push(x);
                                            }
                                            v
                                        }),
                                        x => Match::Single(x),
                                    }
                                } else {
                                    let mut atoms = Vec::with_capacity(w.indices.len());
                                    for i in &w.indices {
                                        atoms.push(self.target.get(*i as usize));
                                    }

                                    Match::Multiple(self.target.get_type(), atoms)
                                };

                                // add the match to the stack if it is compatible
                                if let Some(new_stack_len) = match_stack.insert(w.name, matched) {
                                    self.matches.push(new_stack_len);
                                    continue 'next_match;
                                } else {
                                    // no match
                                    w.indices.pop();
                                    self.used_flag[k] = false;
                                }
                            }
                        }

                        // no match found, try to increase the index of the current last element
                        wildcard_forward_pass = false;
                    }
                }
                PatternIter::Fn(index, name, args, s) => {
                    let mut tried_first_option = false;

                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some((x, _)) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                if name.get_wildcard_level() > 0 {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }

                                self.used_flag[*jj] = false;
                                tried_first_option = true;
                                **s = None;
                                *jj + 1
                            }
                        }
                        None => 0,
                    };

                    // find a new match and create a new iterator
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        if let AtomView::Fun(f) = self.target.get(ii) {
                            let target_name = f.get_symbol();
                            let name_match = if name.get_wildcard_level() > 0 {
                                if let Some(new_stack_len) =
                                    match_stack.insert(*name, Match::FunctionName(target_name))
                                {
                                    self.matches.push(new_stack_len);
                                    true
                                } else {
                                    ii += 1;
                                    continue;
                                }
                            } else {
                                f.get_symbol() == *name
                            };

                            let ordered = !name.is_antisymmetric() && !name.is_symmetric();

                            if name_match {
                                let mut it = SubSliceIterator::from_list(
                                    args,
                                    f.to_slice(),
                                    match_stack,
                                    true,
                                    ordered,
                                );

                                if let Some((x, _)) = it.next(match_stack) {
                                    *index = Some(ii);
                                    **s = Some(it);
                                    self.matches.push(x);
                                    self.used_flag[ii] = true;

                                    continue 'next_match;
                                }

                                if name.get_wildcard_level() > 0 {
                                    // pop the matched name and truncate the stack
                                    // we cannot wait until the truncation at the start of 'next_match
                                    // as we will try to match this iterator to a new index
                                    match_stack.truncate(self.matches.pop().unwrap());
                                }
                            }
                        }

                        ii += 1;
                    }
                }
                PatternIter::Literal(index, atom) => {
                    let mut tried_first_option = false;
                    let mut ii = match index {
                        Some(jj) => {
                            self.used_flag[*jj] = false;
                            tried_first_option = true;
                            *jj + 1
                        }
                        None => 0,
                    };

                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        if self.target.get(ii) == *atom {
                            *index = Some(ii);
                            self.matches.push(match_stack.len());
                            self.used_flag[ii] = true;
                            continue 'next_match;
                        }
                        ii += 1;
                    }
                }
                PatternIter::Sequence(index, slice_type, pattern, s) => {
                    let mut tried_first_option = false;

                    // query an existing iterator
                    let mut ii = match index {
                        Some(jj) => {
                            // get the next iteration of the function
                            if let Some((x, _)) = s.as_mut().as_mut().unwrap().next(match_stack) {
                                self.matches.push(x);
                                continue 'next_match;
                            } else {
                                self.used_flag[*jj] = false;
                                tried_first_option = true;
                                *jj + 1
                            }
                        }
                        None => 0,
                    };

                    // find a new match and create a new iterator
                    while ii < self.target.len() {
                        if self.used_flag[ii] {
                            ii += 1;
                            continue;
                        }

                        if self.ordered_gapless && tried_first_option {
                            break;
                        }

                        tried_first_option = true;

                        let slice = match (self.target.get(ii), &slice_type) {
                            (AtomView::Mul(m), SliceType::Mul) => m.to_slice(),
                            (AtomView::Add(a), SliceType::Add) => a.to_slice(),
                            (AtomView::Pow(a), SliceType::Pow) => a.to_slice(),
                            _ => {
                                ii += 1;
                                continue;
                            }
                        };

                        let ordered = match slice_type {
                            SliceType::Add | SliceType::Mul => false,
                            SliceType::Pow => true, // make sure pattern (base,exp) is not exchanged
                            _ => unreachable!(),
                        };

                        let mut it =
                            SubSliceIterator::from_list(pattern, slice, match_stack, true, ordered);

                        if let Some((x, _)) = it.next(match_stack) {
                            *index = Some(ii);
                            **s = Some(it);
                            self.matches.push(x);
                            self.used_flag[ii] = true;

                            continue 'next_match;
                        }

                        ii += 1;
                    }
                }
            }

            // no match, so fall back one level
            forward_pass = false;
            self.iterators.pop();
        }
    }
}

/// Iterator over the atoms of an expression tree.
pub struct AtomTreeIterator<'a> {
    stack: Vec<(Option<usize>, usize, AtomView<'a>)>,
    settings: MatchSettings,
}

impl<'a> AtomTreeIterator<'a> {
    pub fn new(target: AtomView<'a>, settings: MatchSettings) -> AtomTreeIterator<'a> {
        AtomTreeIterator {
            stack: vec![(None, 0, target)],
            settings,
        }
    }
}

impl<'a> Iterator for AtomTreeIterator<'a> {
    type Item = (Vec<usize>, AtomView<'a>);

    /// Return the next position and atom in the tree.
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((ind, level, atom)) = self.stack.pop() {
            if let Some(max_level) = self.settings.level_range.1 {
                if level > max_level {
                    continue;
                }
            }

            if let Some(ind) = ind {
                let slice = match atom {
                    AtomView::Fun(f) => f.to_slice(),
                    AtomView::Pow(p) => p.to_slice(),
                    AtomView::Mul(m) => m.to_slice(),
                    AtomView::Add(a) => a.to_slice(),
                    _ => {
                        continue; // not iterable
                    }
                };

                if ind < slice.len() {
                    let new_atom = slice.get(ind);

                    self.stack.push((Some(ind + 1), level, atom));
                    self.stack.push((None, level, new_atom)); // push the new element on the stack
                }
            } else {
                // return full match and set the position to the first sub element
                let location = self
                    .stack
                    .iter()
                    .map(|(ind, _, _)| ind.unwrap() - 1)
                    .collect::<Vec<_>>();

                let new_level = if let AtomView::Fun(_) = atom {
                    level + 1
                } else if self.settings.level_is_tree_depth {
                    level + 1
                } else {
                    level
                };

                self.stack.push((Some(0), new_level, atom));

                if level >= self.settings.level_range.0 {
                    return Some((location, atom));
                }
            }
        }

        None
    }
}

/// Match a pattern to any subexpression of a target expression.
pub struct PatternAtomTreeIterator<'a, 'b> {
    pattern: &'b Pattern,
    atom_tree_iterator: AtomTreeIterator<'a>,
    current_target: Option<AtomView<'a>>,
    pattern_iter: Option<AtomMatchIterator<'a, 'b>>,
    match_stack: MatchStack<'a, 'b>,
    tree_pos: Vec<usize>,
    first_match: bool,
}

impl<'a: 'b, 'b> PatternAtomTreeIterator<'a, 'b> {
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        conditions: &'a Condition<WildcardAndRestriction>,
        settings: &'a MatchSettings,
    ) -> PatternAtomTreeIterator<'a, 'b> {
        PatternAtomTreeIterator {
            pattern,
            atom_tree_iterator: AtomTreeIterator::new(target, settings.clone()),
            current_target: None,
            pattern_iter: None,
            match_stack: MatchStack::new(conditions, settings),
            tree_pos: Vec::new(),
            first_match: false,
        }
    }

    pub fn next(&mut self) -> Option<(&[usize], Vec<bool>, AtomView<'a>, &MatchStack<'a, 'b>)> {
        loop {
            if let Some(ct) = self.current_target {
                if let Some(it) = self.pattern_iter.as_mut() {
                    if let Some((_, used_flags)) = it.next(&mut self.match_stack) {
                        let a = used_flags.to_vec();

                        self.first_match = true;
                        return Some((&self.tree_pos, a, ct, &self.match_stack));
                    } else {
                        // no match: bail
                        self.current_target = None;
                        self.pattern_iter = None;
                        continue;
                    }
                } else {
                    // prevent duplicate matches by not matching to single atoms in a list as they will
                    // be tested at a later stage in the atom tree iterator, as we want to store the position
                    self.pattern_iter = Some(AtomMatchIterator::new(self.pattern, ct));
                }
            } else {
                let res = self.atom_tree_iterator.next();

                if let Some((tree_pos, cur_target)) = res {
                    self.tree_pos = tree_pos;
                    self.current_target = Some(cur_target);
                } else {
                    return None;
                }
            }
        }
    }
}

/// Replace a pattern in the target once. Every  call to `next`,
/// will return a new match and replacement until the options are exhausted.
pub struct ReplaceIterator<'a, 'b> {
    rhs: &'b Pattern,
    pattern_tree_iterator: PatternAtomTreeIterator<'a, 'b>,
    target: AtomView<'a>,
}

impl<'a: 'b, 'b> ReplaceIterator<'a, 'b> {
    pub fn new(
        pattern: &'b Pattern,
        target: AtomView<'a>,
        rhs: &'b Pattern,
        conditions: &'a Condition<WildcardAndRestriction>,
        settings: &'a MatchSettings,
    ) -> ReplaceIterator<'a, 'b> {
        ReplaceIterator {
            pattern_tree_iterator: PatternAtomTreeIterator::new(
                pattern, target, conditions, settings,
            ),
            rhs,
            target,
        }
    }

    fn copy_and_replace(
        out: &mut Atom,
        position: &[usize],
        used_flags: &[bool],
        target: AtomView<'a>,
        rhs: AtomView<'_>,
        workspace: &Workspace,
    ) {
        if let Some((first, rest)) = position.split_first() {
            match target {
                AtomView::Fun(f) => {
                    let slice = f.to_slice();

                    let out = out.to_fun(f.get_symbol());

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);
                            out.add_arg(oa.as_view());
                        } else {
                            out.add_arg(arg);
                        }
                    }
                }
                AtomView::Pow(p) => {
                    let slice = p.to_slice();

                    if *first == 0 {
                        let mut oa = workspace.new_atom();
                        Self::copy_and_replace(
                            &mut oa,
                            rest,
                            used_flags,
                            slice.get(0),
                            rhs,
                            workspace,
                        );
                        out.to_pow(oa.as_view(), slice.get(1));
                    } else {
                        let mut oa = workspace.new_atom();
                        Self::copy_and_replace(
                            &mut oa,
                            rest,
                            used_flags,
                            slice.get(1),
                            rhs,
                            workspace,
                        );
                        out.to_pow(slice.get(0), oa.as_view());
                    }
                }
                AtomView::Mul(m) => {
                    let slice = m.to_slice();

                    let out = out.to_mul();

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);

                            // TODO: do type check or just extend? could be that we get x*y*z -> x*(w*u)*z
                            out.extend(oa.as_view());
                        } else {
                            out.extend(arg);
                        }
                    }
                }
                AtomView::Add(a) => {
                    let slice = a.to_slice();

                    let out = out.to_add();

                    for (index, arg) in slice.iter().enumerate() {
                        if index == *first {
                            let mut oa = workspace.new_atom();
                            Self::copy_and_replace(&mut oa, rest, used_flags, arg, rhs, workspace);

                            out.extend(oa.as_view());
                        } else {
                            out.extend(arg);
                        }
                    }
                }
                _ => unreachable!("Atom does not have children"),
            }
        } else {
            match target {
                AtomView::Mul(m) => {
                    let out = out.to_mul();

                    for (child, used) in m.iter().zip(used_flags) {
                        if !used {
                            out.extend(child);
                        }
                    }

                    out.extend(rhs);
                }
                AtomView::Add(a) => {
                    let out = out.to_add();

                    for (child, used) in a.iter().zip(used_flags) {
                        if !used {
                            out.extend(child);
                        }
                    }

                    out.extend(rhs);
                }
                _ => {
                    out.set_from_view(&rhs);
                }
            }
        }
    }

    /// Return the next replacement.
    pub fn next(&mut self, out: &mut Atom) -> Option<()> {
        if let Some((position, used_flags, _target, match_stack)) =
            self.pattern_tree_iterator.next()
        {
            Workspace::get_local().with(|ws| {
                let mut new_rhs = ws.new_atom();

                self.rhs
                    .substitute_wildcards(ws, &mut new_rhs, match_stack)
                    .unwrap(); // TODO: escalate?

                ReplaceIterator::copy_and_replace(
                    out,
                    position,
                    &used_flags,
                    self.target,
                    new_rhs.as_view(),
                    ws,
                );
            });

            Some(())
        } else {
            None
        }
    }
}
