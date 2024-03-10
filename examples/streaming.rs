use symbolica::{id::Pattern, representations::Atom, streaming::TermStreamer};

fn main() {
    let input = Atom::parse("x+ f(x) + 2*f(y) + 7*f(z)").unwrap();
    let pattern = Pattern::parse("f(x_)").unwrap();
    let rhs = Pattern::parse("f(x) + x").unwrap();

    let mut stream = TermStreamer::new_from(input);

    // map every term in the expression
    stream = stream.map(|workspace, x| {
        let mut out1 = workspace.new_atom();
        pattern.replace_all_into(x.as_view(), &rhs, None, None, &mut out1);

        out1.expand()
    });

    let res = stream.to_expression();
    println!("\t+ {}", res);
}
