use ahash::HashMap;
use symbolica::{
    id::{Condition, Match, Pattern, PatternRestriction},
    representations::{Atom, AtomView, Num},
    state::{State, Workspace},
};

fn main() {
    let mut state = State::new();
    let workspace = Workspace::default();

    // prepare all patterns
    let pattern = Pattern::parse("f(x_)", &mut state, &workspace).unwrap();
    let rhs = Pattern::parse("f(x_ - 1) + f(x_ - 2)", &mut state, &workspace).unwrap();
    let lhs_zero_pat = Pattern::parse("f(0)", &mut state, &workspace).unwrap();
    let lhs_one_pat = Pattern::parse("f(1)", &mut state, &workspace).unwrap();
    let rhs_one = Atom::new_num(1).into_pattern(&state);

    // prepare the pattern restriction `x_ > 1`
    let mut restrictions = (
        state.get_or_insert_var("x_"),
        PatternRestriction::Filter(Box::new(|v: &Match| match v {
            Match::Single(AtomView::Num(n)) => !n.is_one() && !n.is_zero(),
            _ => false,
        })),
    )
        .into();

    let input = Atom::parse("f(10)", &mut state, &workspace).unwrap();
    let mut target = workspace.new_atom();
    target.set_from_view(&input.as_view());

    println!(
        "> Repeated calls of f(x_) = f(x_ - 1) + f(x_ - 2) on {}:",
        target.printer(&state),
    );

    for _ in 0..9 {
        let mut out = workspace.new_atom();
        pattern.replace_all(
            target.as_view(),
            &rhs,
            &state,
            &workspace,
            Some(&restrictions),
            None,
            &mut out,
        );

        let mut out2 = workspace.new_atom();
        out.as_view().expand(&workspace, &state, &mut out2);

        lhs_zero_pat.replace_all(
            out2.as_view(),
            &rhs_one,
            &state,
            &workspace,
            None,
            None,
            &mut out,
        );

        lhs_one_pat.replace_all(
            out.as_view(),
            &rhs_one,
            &state,
            &workspace,
            None,
            None,
            &mut out2,
        );

        println!("\t{}", out2.printer(&state),);

        target = out2;
    }
}
