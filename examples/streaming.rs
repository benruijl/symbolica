use ahash::HashMap;

use symbolica::{
    parser::parse,
    printer::{AtomPrinter, PrintMode},
    representations::{default::DefaultRepresentation, OwnedAtom},
    state::{ResettableBuffer, State, Workspace},
    streaming::TermStreamer,
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace<DefaultRepresentation> = Workspace::new();

    let input = parse("x+ f(x) + 2*f(y) + 7*f(z)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap();

    let pattern = parse("f(x_)")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .to_view()
        .into_pattern(&state);

    let rhs = parse("f(x) + x")
        .unwrap()
        .to_atom(&mut state, &workspace)
        .unwrap()
        .to_view()
        .into_pattern(&state);

    let mut stream = TermStreamer::new_from(input);

    // map every term in the expression
    stream = stream.map(|workspace, x| {
        let mut out1 = workspace.new_atom();
        pattern.replace_all(
            x.to_view(),
            &rhs,
            &state,
            workspace,
            &<_>::default(),
            &mut out1,
        );

        let mut out2 = workspace.new_atom();
        out1.to_view().normalize(&workspace, &state, &mut out2);

        let mut out3 = OwnedAtom::new();
        out2.to_view().expand(&workspace, &state, &mut out3);

        out3
    });

    let res = stream.to_expression(&workspace, &state);
    println!(
        "\t+ {}",
        AtomPrinter::new(res.to_view(), <_>::default(), &state)
    );
}
