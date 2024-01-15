use symbolica::{
    id::Pattern,
    representations::Atom,
    state::{ResettableBuffer, State, Workspace},
    streaming::TermStreamer,
};

fn main() {
    let mut state = State::new();
    let workspace: Workspace = Workspace::default();

    let input = Atom::parse("x+ f(x) + 2*f(y) + 7*f(z)", &mut state, &workspace).unwrap();
    let pattern = Pattern::parse("f(x_)", &mut state, &workspace).unwrap();
    let rhs = Pattern::parse("f(x) + x", &mut state, &workspace).unwrap();

    let mut stream = TermStreamer::new_from(input);

    // map every term in the expression
    stream = stream.map(|workspace, x| {
        let mut out1 = workspace.new_atom();
        pattern.replace_all(x.as_view(), &rhs, &state, workspace, None, &mut out1);

        let mut out2 = workspace.new_atom();
        out1.as_view().normalize(workspace, &state, &mut out2);

        let mut out3 = Atom::new();
        out2.as_view().expand(workspace, &state, &mut out3);

        out3
    });

    let res = stream.to_expression(&workspace, &state);
    println!("\t+ {}", res.printer(&state));
}
