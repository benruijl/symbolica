use std::ffi::{c_char, CStr};

use crate::{
    parser::parse,
    printer::{PrintMode, RationalPolynomialPrinter},
    representations::default::DefaultRepresentation,
    rings::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};

pub struct Symbolica {
    state: State,
    workspace: Workspace<DefaultRepresentation>,
}

/// Create a new Symbolica handle.
#[no_mangle]
pub extern "C" fn init() -> *mut Symbolica {
    let s = Symbolica {
        state: State::new(),
        workspace: Workspace::new(),
    };
    let p = Box::into_raw(Box::new(s));
    p
}

/// Simplify a rational polynomial.
#[no_mangle]
pub extern "C" fn simplify(symbolica: *mut Symbolica, input: *const c_char, out: *mut c_char) {
    let c = unsafe { CStr::from_ptr(input) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    let atom = parse(cstr)
        .unwrap()
        .to_atom(&mut symbolica.state, &symbolica.workspace)
        .unwrap();
    let r: RationalPolynomial<u8> = atom
        .to_view()
        .to_rational_polynomial(&symbolica.workspace, &symbolica.state, None)
        .unwrap();

    let out_str = format!(
        "{}",
        RationalPolynomialPrinter {
            poly: &r,
            state: &symbolica.state,
            print_mode: PrintMode::default()
        }
    );

    unsafe {
        std::ptr::copy(out_str.as_bytes().as_ptr().cast(), out, out_str.len());
        std::ptr::write(out.offset(out_str.len() as isize) as *mut u8, 0u8);
    }
}

/// Free the Symbolica handle.
#[no_mangle]
pub unsafe extern "C" fn drop(symbolica: *mut Symbolica) {
    let _ = Box::from_raw(symbolica);
}
