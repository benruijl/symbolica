use std::ffi::{c_char, CStr};
use std::fmt::Write;
use std::os::raw::c_ulonglong;

use crate::printer::SymbolicaPrintOptions;
use crate::rings::finite_field::FiniteField;
use crate::rings::integer::IntegerRing;
use crate::rings::rational::RationalField;
use crate::{
    parser::parse,
    printer::{PrintMode, RationalPolynomialPrinter},
    representations::default::DefaultRepresentation,
    rings::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};

pub struct LocalState {
    buffer: String,
}

pub struct Symbolica {
    state: State,
    workspace: Workspace<DefaultRepresentation>,
    local_state: LocalState,
}

/// Create a new Symbolica handle.
#[no_mangle]
pub extern "C" fn init() -> *mut Symbolica {
    let s = Symbolica {
        state: State::new(),
        workspace: Workspace::new(),
        local_state: LocalState {
            buffer: String::with_capacity(2048),
        },
    };
    let p = Box::into_raw(Box::new(s));
    p
}

/// Simplify a rational polynomial. The return value is only valid until the next call to
/// `simplify`.
#[no_mangle]
pub extern "C" fn simplify(
    symbolica: *mut Symbolica,
    input: *const c_char,
    prime: c_ulonglong,
) -> *const c_char {
    let c = unsafe { CStr::from_ptr(input) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    let atom = parse(cstr)
        .unwrap()
        .to_atom(&mut symbolica.state, &symbolica.workspace)
        .unwrap();

    if prime == 0 {
        let r: RationalPolynomial<IntegerRing, u8> = atom
            .to_view()
            .to_rational_polynomial::<RationalField, IntegerRing, u8>(
                &symbolica.workspace,
                &symbolica.state,
                RationalField::new(),
                IntegerRing::new(),
                None,
            )
            .unwrap();

        symbolica.local_state.buffer.clear();
        write!(
            &mut symbolica.local_state.buffer,
            "{}\0", // add the NUL character
            RationalPolynomialPrinter {
                poly: &r,
                state: &symbolica.state,
                print_mode: PrintMode::default()
            }
        )
        .unwrap();
    } else {
        if prime < u32::MAX as c_ulonglong {
            let field = FiniteField::<u32>::new(prime as u32);
            let rf: RationalPolynomial<FiniteField<u32>, u8> = atom
                .to_view()
                .to_rational_polynomial(&symbolica.workspace, &symbolica.state, field, field, None)
                .unwrap();

            symbolica.local_state.buffer.clear();
            write!(
                &mut symbolica.local_state.buffer,
                "{}\0", // add the NUL character
                RationalPolynomialPrinter {
                    poly: &rf,
                    state: &symbolica.state,
                    print_mode: PrintMode::Symbolica(SymbolicaPrintOptions {
                        terms_on_new_line: false,
                        color_top_level_sum: false,
                        print_finite_field: false
                    })
                }
            )
            .unwrap();
        } else {
            panic!("Prime is too large");
        }
    }

    unsafe { CStr::from_bytes_with_nul_unchecked(symbolica.local_state.buffer.as_bytes()) }.as_ptr()
}

/// Free the Symbolica handle.
#[no_mangle]
pub unsafe extern "C" fn drop(symbolica: *mut Symbolica) {
    let _ = Box::from_raw(symbolica);
}
