use std::ffi::{c_char, CStr};
use std::fmt::Write;
use std::os::raw::c_ulonglong;
use std::sync::Arc;

use smartstring::{LazyCompact, SmartString};

use crate::parser::Token;
use crate::poly::Variable;
use crate::domains::finite_field::{FiniteField, FiniteFieldCore, Mersenne64};
use crate::domains::integer::IntegerRing;
use crate::domains::rational::RationalField;
use crate::LicenseManager;
use crate::{
    printer::{FactorizedRationalPolynomialPrinter, PrintOptions, RationalPolynomialPrinter},
    representations::default::Linear,
    domains::factorized_rational_polynomial::FactorizedRationalPolynomial,
    domains::rational_polynomial::RationalPolynomial,
    state::{State, Workspace},
};

struct LocalState {
    buffer: String,
    var_map: Arc<Vec<Variable>>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

struct Symbolica {
    state: State,
    workspace: Workspace<Linear>,
    local_state: LocalState,
}

/// Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions.
#[no_mangle]
unsafe extern "C" fn set_license_key(key: *const c_char) -> bool {
    let key = unsafe { CStr::from_ptr(key) }.to_str().unwrap();
    LicenseManager::set_license_key(key)
        .map_err(|e| eprintln!("{}", e))
        .is_ok()
}

/// Check if the current Symbolica instance has a valid license key set.
#[no_mangle]
unsafe extern "C" fn is_licensed() -> bool {
    LicenseManager::is_licensed()
}

/// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
/// `email`.
#[no_mangle]
unsafe extern "C" fn request_hobbyist_license(name: *const c_char, email: *const c_char) -> bool {
    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    let email = unsafe { CStr::from_ptr(email) }.to_str().unwrap();

    LicenseManager::request_hobbyist_license(name, email)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(|e| eprintln!("{}", e))
        .is_ok()
}

/// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
/// `email`.
#[no_mangle]
unsafe extern "C" fn request_trial_license(
    name: *const c_char,
    email: *const c_char,
    company: *const c_char,
) -> bool {
    let name = unsafe { CStr::from_ptr(name) }.to_str().unwrap();
    let email = unsafe { CStr::from_ptr(email) }.to_str().unwrap();
    let company = unsafe { CStr::from_ptr(company) }.to_str().unwrap();

    LicenseManager::request_trial_license(name, email, company)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(|e| eprintln!("{}", e))
        .is_ok()
}

/// Create a new Symbolica handle.
#[no_mangle]
unsafe extern "C" fn init() -> *mut Symbolica {
    let s = Symbolica {
        state: State::new(),
        workspace: Workspace::new(),
        local_state: LocalState {
            buffer: String::with_capacity(2048),
            var_map: Arc::new(vec![]),
            var_name_map: vec![],
            input_has_rational_numbers: false,
            exp_fits_in_u8: true,
        },
    };

    Box::into_raw(Box::new(s))
}

#[no_mangle]
unsafe extern "C" fn set_options(
    symbolica: *mut Symbolica,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
) {
    let symbolica = unsafe { &mut *symbolica };

    symbolica.local_state.input_has_rational_numbers = input_has_rational_numbers;
    symbolica.local_state.exp_fits_in_u8 = exp_fits_in_u8;
}

#[no_mangle]
unsafe extern "C" fn set_vars(symbolica: *mut Symbolica, vars: *const c_char) {
    let c = unsafe { CStr::from_ptr(vars) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    symbolica.local_state.var_name_map.clear();

    let mut var_map = vec![];

    for var in cstr.split(',') {
        var_map
            .push(Variable::Identifier(symbolica.state.get_or_insert_var(var)));
        symbolica.local_state.var_name_map.push(var.into());
    }

    symbolica.local_state.var_map = Arc::new(var_map);
}

/// Simplify a rational polynomial. The return value is only valid until the next call to
/// `simplify`.
#[no_mangle]
unsafe extern "C" fn simplify(
    symbolica: *mut Symbolica,
    input: *const c_char,
    prime: c_ulonglong,
    explicit_rational_polynomial: bool,
) -> *const c_char {
    let c = unsafe { CStr::from_ptr(input) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    let token = Token::parse(cstr).unwrap();

    let opts = PrintOptions {
        terms_on_new_line: false,
        color_top_level_sum: false,
        color_builtin_functions: false,
        print_finite_field: false,
        explicit_rational_polynomial,
        number_thousands_separator: None,
        multiplication_operator: '*',
        square_brackets_for_function: false,
        num_exp_as_superscript: false,
        latex: false,
    };

    macro_rules! to_rational {
        ($in_field: ty, $exp_size: ty) => {
            if prime == 0 {
                let r: RationalPolynomial<IntegerRing, $exp_size> = token
                    .to_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &<$in_field>::new(),
                        &IntegerRing::new(),
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    RationalPolynomialPrinter {
                        poly: &r,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else if prime <= u32::MAX as c_ulonglong {
                let field = FiniteField::<u32>::new(prime as u32);
                let rf: RationalPolynomial<FiniteField<u32>, $exp_size> = token
                    .to_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    RationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else if prime == Mersenne64::PRIME {
                let field = FiniteField::<Mersenne64>::new(Mersenne64::new());
                let rf: RationalPolynomial<FiniteField<Mersenne64>, $exp_size> = token
                    .to_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    RationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else {
                let field = FiniteField::<u64>::new(prime as u64);
                let rf: RationalPolynomial<FiniteField<u64>, $exp_size> = token
                    .to_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    RationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            }
        };
    }

    match (
        symbolica.local_state.input_has_rational_numbers,
        symbolica.local_state.exp_fits_in_u8,
    ) {
        (false, true) => to_rational!(IntegerRing, u8),
        (true, true) => to_rational!(RationalField, u8),
        (false, false) => to_rational!(IntegerRing, u16),
        (true, false) => to_rational!(RationalField, u16),
    }

    unsafe { CStr::from_bytes_with_nul_unchecked(symbolica.local_state.buffer.as_bytes()) }.as_ptr()
}

/// Simplify a rational polynomial, factorizing the denominator. The return value is only valid until the next call to
/// `simplify`.
#[no_mangle]
unsafe extern "C" fn simplify_factorized(
    symbolica: *mut Symbolica,
    input: *const c_char,
    prime: c_ulonglong,
    explicit_rational_polynomial: bool,
) -> *const c_char {
    let c = unsafe { CStr::from_ptr(input) };
    let cstr = c.to_str().unwrap();

    let symbolica = unsafe { &mut *symbolica };

    let token = Token::parse(cstr).unwrap();

    let opts = PrintOptions {
        terms_on_new_line: false,
        color_top_level_sum: false,
        color_builtin_functions: false,
        print_finite_field: false,
        explicit_rational_polynomial,
        number_thousands_separator: None,
        multiplication_operator: '*',
        square_brackets_for_function: false,
        num_exp_as_superscript: false,
        latex: false,
    };

    macro_rules! to_rational {
        ($in_field: ty, $exp_size: ty) => {
            if prime == 0 {
                let r: FactorizedRationalPolynomial<IntegerRing, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &<$in_field>::new(),
                        &IntegerRing::new(),
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    FactorizedRationalPolynomialPrinter {
                        poly: &r,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else if prime <= u32::MAX as c_ulonglong {
                let field = FiniteField::<u32>::new(prime as u32);
                let rf: FactorizedRationalPolynomial<FiniteField<u32>, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    FactorizedRationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else if prime == Mersenne64::PRIME {
                let field = FiniteField::<Mersenne64>::new(Mersenne64::new());
                let rf: FactorizedRationalPolynomial<FiniteField<Mersenne64>, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    FactorizedRationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            } else {
                let field = FiniteField::<u64>::new(prime as u64);
                let rf: FactorizedRationalPolynomial<FiniteField<u64>, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &symbolica.workspace,
                        &mut symbolica.state,
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                symbolica.local_state.buffer.clear();
                write!(
                    &mut symbolica.local_state.buffer,
                    "{}\0", // add the NUL character
                    FactorizedRationalPolynomialPrinter {
                        poly: &rf,
                        state: &symbolica.state,
                        opts,
                        add_parentheses: false,
                    }
                )
                .unwrap();
            }
        };
    }

    match (
        symbolica.local_state.input_has_rational_numbers,
        symbolica.local_state.exp_fits_in_u8,
    ) {
        (false, true) => to_rational!(IntegerRing, u8),
        (true, true) => to_rational!(RationalField, u8),
        (false, false) => to_rational!(IntegerRing, u16),
        (true, false) => to_rational!(RationalField, u16),
    }

    unsafe { CStr::from_bytes_with_nul_unchecked(symbolica.local_state.buffer.as_bytes()) }.as_ptr()
}

/// Free the Symbolica handle.
#[no_mangle]
unsafe extern "C" fn drop(symbolica: *mut Symbolica) {
    let _ = Box::from_raw(symbolica);
}
