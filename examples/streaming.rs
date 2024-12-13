use brotli::CompressorWriter;
use symbolica::{
    atom::{Atom, AtomCore},
    id::Pattern,
    streaming::{TermStreamer, TermStreamerConfig},
};

fn main() {
    let input = Atom::parse("x+ f(x) + 2*f(y) + 7*f(z)").unwrap();
    let pattern = Pattern::parse("f(x_)").unwrap();
    let rhs = Pattern::parse("f(x) + x").unwrap();

    let mut stream = TermStreamer::<CompressorWriter<_>>::new(TermStreamerConfig {
        n_cores: 4,
        path: ".".to_owned(),
        max_mem_bytes: 40,
    });
    stream.push(input);

    // map every term in the expression
    stream = stream.map(|x| x.replace_all(&pattern, &rhs, None, None).expand());

    let res = stream.to_expression();
    println!("\t+ {}", res);
}
