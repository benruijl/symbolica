use ahash::HashMap;
use symbolica::{
    id::{Match, PatternRestriction},
    parser::parse,
    printer::AtomPrinter,
    representations::{default::DefaultRepresentation, AtomView, Num, OwnedAtom},
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::new();

    let pattern = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let mut restrictions = HashMap::default();
    restrictions.insert(
        state.get_or_insert_var("x_"),
        vec![PatternRestriction::Filter(Box::new(
            |v: &Match<DefaultRepresentation>| match v {
                Match::Single(v) => {
                    if let AtomView::Num(n) = v {
                        !n.is_one() && !n.is_zero()
                    } else {
                        false
                    }
                }
                _ => false,
            },
        ))],
    );

    let rhs = parse("f(x_ -1) + f(x_ - 2)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    // replace f(0) and f(1) by 1
    let lhs_zero_pat = parse("f(0)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let lhs_one_pat = parse("f(1)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let rhs_one = parse("1")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expand_pat = parse("(x_+y_)*z_")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expand_rhs = parse("x_*z_+y_*z_")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .into_pattern(&state);

    let expr: OwnedAtom<DefaultRepresentation> = parse("f(10)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let mut target = workspace.new_atom();
    target.get_mut().from_view(&expr.to_view());

    println!(
        "> Repeated calls of f(x_) = f(x_ - 1) + f(x_ - 2) on {}:",
        AtomPrinter::new(
            target.get().to_view(),
            symbolica::printer::PrintMode::default(),
            &state
        ),
    );

    for _ in 0..9 {
        let mut out = workspace.new_atom();
        pattern.replace_all(
            target.get().to_view(),
            &rhs,
            &state,
            &workspace,
            &restrictions,
            out.get_mut(),
        );

        // expand (f(1)+f(2))*4
        let mut out2 = workspace.new_atom();
        expand_pat.replace_all(
            out.get().to_view(),
            &expand_rhs,
            &state,
            &workspace,
            &HashMap::default(),
            out2.get_mut(),
        );

        // sort the expression
        let mut out_renom = workspace.new_atom();
        out2.get()
            .to_view()
            .normalize(&workspace, &state, out_renom.get_mut());
        out2 = out_renom;

        let mut out_renom2 = workspace.new_atom();
        lhs_zero_pat.replace_all(
            out2.get().to_view(),
            &rhs_one,
            &state,
            &workspace,
            &HashMap::default(),
            out_renom2.get_mut(),
        );

        let mut out3 = workspace.new_atom();
        lhs_one_pat.replace_all(
            out_renom2.get().to_view(),
            &rhs_one,
            &state,
            &workspace,
            &HashMap::default(),
            out3.get_mut(),
        );

        // sort expression
        let mut out_renom = workspace.new_atom();
        out3.get()
            .to_view()
            .normalize(&workspace, &state, out_renom.get_mut());

        println!(
            "\t{}",
            AtomPrinter::new(
                out_renom.get().to_view(),
                symbolica::printer::PrintMode::default(),
                &state
            ),
        );

        target = out_renom;
    }
}
