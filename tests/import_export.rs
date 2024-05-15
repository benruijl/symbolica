use std::io::Cursor;

use smartstring::SmartString;
use symbolica::{
    atom::Atom,
    state::{FunctionAttribute, State},
};

fn conflict() {
    State::get_symbol("x");
    State::get_symbol("y");
    State::get_symbol_with_attributes("f", &[FunctionAttribute::Symmetric]).unwrap();

    let a = Atom::parse("f(x, y)*x^2").unwrap();

    let mut a_export = vec![];
    a.as_view().write(&mut a_export).unwrap();

    let mut state_export = vec![];
    State::export(&mut state_export).unwrap();

    // reset the state and create a conflict
    unsafe { State::reset() };

    State::get_symbol("y");
    State::get_symbol("x");
    State::get_symbol("f");

    let state_map = State::import(
        Cursor::new(&state_export),
        Some(Box::new(|old_name| SmartString::from(old_name) + "1")),
    )
    .unwrap();

    let a_rec = Atom::import(Cursor::new(&a_export), &state_map).unwrap();

    let r = Atom::parse("x^2*f1(y, x)").unwrap();
    assert_eq!(a_rec, r);
}

#[test]
fn rational_rename() {
    State::get_symbol("x");

    let a = Atom::parse("x^2*coeff(x)").unwrap();

    let mut a_export = vec![];
    a.as_view().write(&mut a_export).unwrap();

    let mut state_export = vec![];
    State::export(&mut state_export).unwrap();

    // reset the state and create a conflict
    unsafe { State::reset() };

    State::get_symbol("y");

    let state_map = State::import(Cursor::new(&state_export), None).unwrap();

    let a_rec = Atom::import(Cursor::new(&a_export), &state_map).unwrap();

    let r = Atom::parse("x^2*coeff(x)").unwrap();
    assert_eq!(a_rec, r);

    unsafe { State::reset() };
    conflict();
}
