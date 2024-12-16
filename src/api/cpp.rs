use std::ffi::{c_char, CStr};
use std::fmt::Write;
use std::os::raw::c_ulonglong;
use std::sync::Arc;

use smartstring::{LazyCompact, SmartString};

use crate::atom::Symbol;
use crate::domains::finite_field::{FiniteField, FiniteFieldCore, Mersenne64, Zp, Zp64};
use crate::domains::integer::{IntegerRing, Z};
use crate::domains::rational::Q;
use crate::domains::SelfRing;
use crate::parser::Token;
use crate::poly::Variable;
use crate::LicenseManager;
use crate::{
    domains::factorized_rational_polynomial::FactorizedRationalPolynomial,
    domains::rational_polynomial::RationalPolynomial, printer::PrintOptions, printer::PrintState,
};

struct LocalState {
    buffer: String,
    var_map: Arc<Vec<Variable>>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

struct Symbolica {
    local_state: LocalState,
    is_licensed: bool,
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

/// Get a license key for offline use, generated from a licensed Symbolica session. The key will remain valid for 24 hours.
/// The key is written into `key`, which must be a buffer of at least 100 bytes.
#[no_mangle]
unsafe extern "C" fn get_license_key(email: *const c_char) -> bool {
    let email = unsafe { CStr::from_ptr(email) }.to_str().unwrap();

    match LicenseManager::get_license_key(email) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("{}", e);
            false
        }
    }
}

/// Create a new Symbolica handle.
#[no_mangle]
unsafe extern "C" fn init() -> *mut Symbolica {
    LicenseManager::check();

    let s = Symbolica {
        local_state: LocalState {
            buffer: String::with_capacity(2048),
            var_map: Arc::new(vec![]),
            var_name_map: vec![],
            input_has_rational_numbers: false,
            exp_fits_in_u8: false,
        },
        is_licensed: LicenseManager::is_licensed(),
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
        var_map.push(Variable::Symbol(Symbol::new(var)));
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

    if !symbolica.is_licensed {
        LicenseManager::check();
    }

    let token = Token::parse(cstr).unwrap();

    let opts = PrintOptions {
        explicit_rational_polynomial,
        ..PrintOptions::file()
    };

    macro_rules! to_rational {
        ($in_field: expr, $exp_size: ty) => {
            symbolica.local_state.buffer.clear();
            if prime == 0 {
                let r: RationalPolynomial<IntegerRing, $exp_size> = token
                    .to_rational_polynomial(
                        &$in_field,
                        &Z,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                r.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else if prime <= u32::MAX as c_ulonglong {
                let field = Zp::new(prime as u32);
                let rf: RationalPolynomial<Zp, $exp_size> = token
                    .to_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else if prime == Mersenne64::PRIME {
                let field = FiniteField::<Mersenne64>::new(Mersenne64::new());
                let rf: RationalPolynomial<FiniteField<Mersenne64>, $exp_size> = token
                    .to_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else {
                let field = Zp64::new(prime as u64);
                let rf: RationalPolynomial<Zp64, $exp_size> = token
                    .to_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            }

            write!(
                &mut symbolica.local_state.buffer,
                "\0", // add the NUL character
            )
            .unwrap()
        };
    }

    match (
        symbolica.local_state.input_has_rational_numbers,
        symbolica.local_state.exp_fits_in_u8,
    ) {
        (false, true) => {
            to_rational!(Z, u8);
        }
        (true, true) => {
            to_rational!(Q, u8);
        }
        (false, false) => {
            to_rational!(Z, u16);
        }
        (true, false) => {
            to_rational!(Q, u16);
        }
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
        explicit_rational_polynomial,
        ..PrintOptions::file()
    };

    macro_rules! to_rational {
        ($in_field: expr, $exp_size: ty) => {
            symbolica.local_state.buffer.clear();
            if prime == 0 {
                let r: FactorizedRationalPolynomial<IntegerRing, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &$in_field,
                        &Z,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                r.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else if prime <= u32::MAX as c_ulonglong {
                let field = Zp::new(prime as u32);
                let rf: FactorizedRationalPolynomial<Zp, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else if prime == Mersenne64::PRIME {
                let field = FiniteField::<Mersenne64>::new(Mersenne64::new());
                let rf: FactorizedRationalPolynomial<FiniteField<Mersenne64>, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            } else {
                let field = Zp64::new(prime as u64);
                let rf: FactorizedRationalPolynomial<Zp64, $exp_size> = token
                    .to_factorized_rational_polynomial(
                        &field,
                        &field,
                        &symbolica.local_state.var_map,
                        &symbolica.local_state.var_name_map,
                    )
                    .unwrap();

                rf.format(&opts, PrintState::new(), &mut symbolica.local_state.buffer)
                    .unwrap();
            }

            write!(
                &mut symbolica.local_state.buffer,
                "\0", // add the NUL character
            )
            .unwrap()
        };
    }

    match (
        symbolica.local_state.input_has_rational_numbers,
        symbolica.local_state.exp_fits_in_u8,
    ) {
        (false, true) => {
            to_rational!(Z, u8);
        }
        (true, true) => {
            to_rational!(Q, u8);
        }
        (false, false) => {
            to_rational!(Z, u16);
        }
        (true, false) => {
            to_rational!(Q, u16);
        }
    }

    unsafe { CStr::from_bytes_with_nul_unchecked(symbolica.local_state.buffer.as_bytes()) }.as_ptr()
}

/// Free the Symbolica handle.
#[no_mangle]
unsafe extern "C" fn drop(symbolica: *mut Symbolica) {
    let _ = Box::from_raw(symbolica);
}

#[cfg(test)]
mod test {
    use std::ffi::{c_char, CStr};

    use crate::domains::finite_field::Mersenne64;

    use super::{drop, init, set_options};

    #[test]
    fn simplify() {
        let symbolica = unsafe { init() };

        unsafe { set_options(symbolica, true, false) };

        unsafe { super::set_vars(symbolica, b"d,y\0".as_ptr() as *const c_char) };

        let input = "-(4096-4096*y^2)/(-3072+1024*d)*(1536-512*d)-(-8192+8192*y^2)/(2)*((-6+d)/2)-(-8192+8192*y^2)/(-2)*((-13+3*d)/2)-(-8192+8192*y^2)/(-4)*(-8+2*d)\0";
        let result = unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, 0, true) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "[32768-32768*y^2-8192*d+8192*d*y^2]");

        unsafe { set_options(symbolica, true, true) };

        let result = unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, 0, false) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "32768-32768*y^2-8192*d+8192*d*y^2");

        unsafe { set_options(symbolica, false, false) };

        let result =
            unsafe { super::simplify_factorized(symbolica, input.as_ptr() as *const i8, 0, true) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "[8192]*[4-4*y^2-d+d*y^2]");

        let result =
            unsafe { super::simplify_factorized(symbolica, input.as_ptr() as *const i8, 0, false) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        unsafe { drop(symbolica) };
        assert_eq!(result, "8192*(4-4*y^2-d+d*y^2)");
    }

    #[test]
    fn simplify_ff() {
        let symbolica = unsafe { init() };

        unsafe { super::set_vars(symbolica, b"d,y\0".as_ptr() as *const c_char) };

        let prime = 4293491017;

        let input = "-(4096-4096*y^2)/(-3072+1024*d)*(1536-512*d)-(-8192+8192*y^2)/(2)*((-6+d)/2)-(-8192+8192*y^2)/(-2)*((-13+3*d)/2)-(-8192+8192*y^2)/(-4)*(-8+2*d)\0";
        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, true) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "[32768+4293458249*y^2+4293482825*d+8192*d*y^2]");

        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, false) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "32768+4293458249*y^2+4293482825*d+8192*d*y^2");

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, true)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(result, "[32768+4293458249*y^2+4293482825*d+8192*d*y^2]");

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, false)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        unsafe { drop(symbolica) };
        assert_eq!(result, "32768+4293458249*y^2+4293482825*d+8192*d*y^2");
    }

    #[test]
    fn simplify_mersenne() {
        let symbolica = unsafe { init() };

        unsafe { super::set_vars(symbolica, b"d,y\0".as_ptr() as *const c_char) };

        let prime = Mersenne64::PRIME;

        let input = "-(4096-4096*y^2)/(-3072+1024*d)*(1536-512*d)-(-8192+8192*y^2)/(2)*((-6+d)/2)-(-8192+8192*y^2)/(-2)*((-13+3*d)/2)-(-8192+8192*y^2)/(-4)*(-8+2*d)\0";
        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, true) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "[32768+2305843009213661183*y^2+2305843009213685759*d+8192*d*y^2]"
        );

        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, false) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "32768+2305843009213661183*y^2+2305843009213685759*d+8192*d*y^2"
        );

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, true)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "[32768+2305843009213661183*y^2+2305843009213685759*d+8192*d*y^2]"
        );

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, false)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        unsafe { drop(symbolica) };
        assert_eq!(
            result,
            "32768+2305843009213661183*y^2+2305843009213685759*d+8192*d*y^2"
        );
    }

    #[test]
    fn simplify_u64_prime() {
        let symbolica = unsafe { init() };

        unsafe { super::set_vars(symbolica, b"d,y\0".as_ptr() as *const c_char) };

        let prime = 18446744073709551163;

        let input = "-(4096-4096*y^2)/(-3072+1024*d)*(1536-512*d)-(-8192+8192*y^2)/(2)*((-6+d)/2)-(-8192+8192*y^2)/(-2)*((-13+3*d)/2)-(-8192+8192*y^2)/(-4)*(-8+2*d)\0";
        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, true) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "[32768+18446744073709518395*y^2+18446744073709542971*d+8192*d*y^2]"
        );

        let result =
            unsafe { super::simplify(symbolica, input.as_ptr() as *const i8, prime, false) };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "32768+18446744073709518395*y^2+18446744073709542971*d+8192*d*y^2"
        );

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, true)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        assert_eq!(
            result,
            "[32768+18446744073709518395*y^2+18446744073709542971*d+8192*d*y^2]"
        );

        let result = unsafe {
            super::simplify_factorized(symbolica, input.as_ptr() as *const i8, prime, false)
        };
        let result = unsafe { CStr::from_ptr(result).to_str().unwrap() }.to_owned();

        unsafe { drop(symbolica) };
        assert_eq!(
            result,
            "32768+18446744073709518395*y^2+18446744073709542971*d+8192*d*y^2"
        );
    }
}
