//! Symbolica is a blazing fast computer algebra system.
//!
//! It can be used to perform mathematical operations,
//! such as symbolic differentiation, integration, simplification,
//! pattern matching and solving equations.
//!
//! For example:
//!
//! ```
//! use symbolica::{atom::Atom, state::State};
//!
//! fn main() {
//!     let input = Atom::parse("x^2*log(2*x + y) + exp(3*x)").unwrap();
//!     let a = input.derivative(State::get_symbol("x"));
//!     println!("d({})/dx = {}:", input, a);
//! }
//! ```
//!
//! Check out the [guide](https://symbolica.io/docs/get_started.html) for more information, examples,
//! and additional documentation.

use std::{
    collections::HashMap,
    env,
    io::{Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    process::abort,
    sync::atomic::{AtomicBool, Ordering::Relaxed},
    thread::ThreadId,
    time::{Duration, SystemTime},
};

use colored::Colorize;
use once_cell::sync::OnceCell;
use tinyjson::JsonValue;

#[cfg(feature = "python_no_module")]
pub mod api;
#[cfg(not(feature = "python_no_module"))]
mod api;
pub mod atom;
pub mod coefficient;
mod collect;
pub mod combinatorics;
mod derivative;
pub mod domains;
pub mod evaluate;
mod expand;
pub mod graph;
pub mod id;
mod normalize;
pub mod numerical_integration;
pub mod parser;
pub mod poly;
pub mod printer;
mod solve;
pub mod state;
pub mod streaming;
pub mod tensors;
pub mod transformer;
pub mod utils;

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

static LICENSE_KEY: OnceCell<String> = OnceCell::new();
static LICENSE_MANAGER: OnceCell<LicenseManager> = OnceCell::new();
static LICENSED: AtomicBool = LicenseManager::init();

#[allow(dead_code)]
pub struct LicenseManager {
    lock: Option<TcpListener>,
    core_limit: Option<usize>,
    pid: u32,
    thread_id: ThreadId,
    has_license: bool,
}

const MULTIPLE_INSTANCE_WARNING: &str = "┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Cannot start new unlicensed Symbolica instance since there is already another one running on the machine. │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────┘"
;

const RESOLVE_ERROR: &str = "
┌───────────────────────────────────────────────────────────┐
│ Could not resolve the IP of the Symbolica license server. │
│                                                           │
│ Please check your DNS configuration.                      │
└───────────────────────────────────────────────────────────┘";

const CONNECTION_ERROR: &str = "
┌────────────────────────────────────────────────┐
│ Could not connect to Symbolica license server. │
│                                                │
│ Some networks block traffic to uncommon ports. │
│ Consider switching networks or using a VPN.    │
└────────────────────────────────────────────────┘";

const NETWORK_ERROR: &str = "
┌───────────────────────────────────────────────────┐
│ Connection to Symbolica license server timed out. │
│                                                   │
│ Please check your network configuration.          │
└───────────────────────────────────────────────────┘";

const ACTIVATION_ERROR: &str = "
┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘";

const MISSING_LICENSE_ERROR: &str = "
┌───────────────────────────────┐
│ Symbolica license key missing │
└───────────────────────────────┘";

impl Default for LicenseManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LicenseManager {
    pub fn new() -> LicenseManager {
        let pid = std::process::id();
        let thread_id = std::thread::current().id();

        match Self::check_license_key() {
            Ok(()) => {
                return LicenseManager {
                    lock: None,
                    core_limit: None,
                    pid,
                    thread_id,
                    has_license: true,
                };
            }
            Err(e) => {
                if !e.contains("missing") {
                    eprintln!("{}", e);
                }
            }
        }

        if env::var("SYMBOLICA_HIDE_BANNER").is_err() {
            println!(
                "┌────────────────────────────────────────────────────────┐
│ You are running a restricted Symbolica instance.       │
│                                                        │
│ This mode is only permitted for non-commercial use and │
│ is limited to one instance and core per machine.       │
│                                                        │
│ {} can easily acquire a {} license key        │
│ that unlocks all cores and removes this banner:        │
│                                                        │
│   from symbolica import *                              │
│   request_hobbyist_license('YOUR_NAME', 'YOUR_EMAIL')  │
│                                                        │
│ All other users can obtain a free 30-day trial key:    │
│                                                        │
│   from symbolica import *                              │
│   request_trial_license('NAME', 'EMAIL', 'EMPLOYER')   │
│                                                        │
│ See https://symbolica.io/docs/get_started.html#license │
└────────────────────────────────────────────────────────┘",
                "Hobbyists".bold(),
                "free".bold(),
            );
        }

        let port = env::var("SYMBOLICA_PORT").unwrap_or_else(|_| "12011".to_owned());

        match TcpListener::bind(format!("127.0.0.1:{}", port)) {
            Ok(o) => {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build_global()
                    .unwrap();

                drop(o);

                std::thread::spawn(move || loop {
                    let new_port =
                        env::var("SYMBOLICA_PORT").unwrap_or_else(|_| "12011".to_owned());

                    if port != new_port {
                        println!("{}", MULTIPLE_INSTANCE_WARNING);
                        abort();
                    }

                    match TcpListener::bind(&format!("127.0.0.1:{}", port)) {
                        Ok(_) => {
                            std::thread::sleep(Duration::from_secs(1));
                        }
                        Err(_) => {
                            println!("{}", MULTIPLE_INSTANCE_WARNING);
                            abort();
                        }
                    }
                });

                LicenseManager {
                    lock: None,
                    core_limit: Some(1),
                    pid,
                    thread_id,
                    has_license: false,
                }
            }
            Err(_) => {
                println!("{}", MULTIPLE_INSTANCE_WARNING);
                abort();
            }
        }
    }

    const fn init() -> AtomicBool {
        AtomicBool::new(false)
    }

    fn check_license_key() -> Result<(), String> {
        let key = LICENSE_KEY
            .get()
            .cloned()
            .or(env::var("SYMBOLICA_LICENSE").ok());

        let Some(mut key) = key else {
            std::thread::spawn(|| {
                let mut m: HashMap<String, JsonValue> = HashMap::default();
                m.insert(
                    "version".to_owned(),
                    env!("CARGO_PKG_VERSION").to_owned().into(),
                );
                let mut v = JsonValue::from(m).stringify().unwrap();
                v.push('\n');

                if let Ok(mut stream) = Self::connect() {
                    let _ = stream.write_all(v.as_bytes());
                };
            });

            return Err(MISSING_LICENSE_ERROR.to_owned());
        };

        if key.contains('#') {
            let mut a = key.split('#');
            let f1 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;
            let f2 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;
            let f3 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;

            let mut h: u32 = 5381;
            for b in f2.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in f3.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }

            let h = format!("{:x}", h);
            if f1 != h {
                Err(ACTIVATION_ERROR.to_owned())?;
            }

            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let t2 = u64::from_str_radix(f2, 16)
                .map_err(|_| ACTIVATION_ERROR.to_owned())
                .unwrap();

            if t > t2 {
                Err("┌───────────────────────────────────┐
│ The Symbolica license has expired │
└───────────────────────────────────┘"
                    .to_owned())?;
            }

            key = f3.to_owned();
            std::thread::spawn(|| {
                if let Err(e) = Self::check_registration(key) {
                    if e.contains("expired") {
                        println!("{}", e);
                        abort();
                    }
                }
            });
        } else {
            Self::check_registration(key)?;
        }

        LICENSED.store(true, Relaxed);
        Ok(())
    }

    fn connect() -> Result<TcpStream, String> {
        let mut ip = ("symbolica.io", 12012)
            .to_socket_addrs()
            .map_err(|e| format!("{}\nError: {}", RESOLVE_ERROR, e))?;
        let Some(n) = ip.next() else {
            return Err(RESOLVE_ERROR.to_owned());
        };

        let stream = match TcpStream::connect_timeout(&n, Duration::from_secs(5)) {
            Ok(stream) => stream,
            Err(_) => {
                return Err(CONNECTION_ERROR.to_owned());
            }
        };

        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;

        Ok(stream)
    }

    fn check_registration(key: String) -> Result<(), String> {
        let mut stream = Self::connect()?;

        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert(
            "version".to_owned(),
            env!("CARGO_PKG_VERSION").to_owned().into(),
        );
        m.insert("license".to_owned(), key.into());
        let mut v = JsonValue::from(m).stringify().unwrap();
        v.push('\n');

        stream
            .write_all(v.as_bytes())
            .map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;
        let read_str =
            std::str::from_utf8(&buf).map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;

        if read_str == "{\"status\":\"ok\"}\n" {
            Ok(())
        } else if read_str.is_empty() {
            Err("┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘"
                .to_owned())
        } else {
            let message: JsonValue = read_str[..read_str.len() - 1]
                .parse()
                .map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;
            let message_parsed: &HashMap<_, _> = message
                .get()
                .ok_or_else(|| format!("{}\nError: Empty response", NETWORK_ERROR))?;
            let status: &String = message_parsed
                .get("status")
                .unwrap()
                .get()
                .ok_or_else(|| format!("{}\nError: missing status", NETWORK_ERROR))?;
            Err(format!(
                "┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘
Error: {}",
                status,
            ))
        }
    }

    #[inline(always)]
    fn check() {
        if LICENSED.load(Relaxed) {
            return;
        }

        Self::check_impl();
    }

    fn check_impl() {
        let manager = LICENSE_MANAGER.get_or_init(LicenseManager::new);

        if manager.has_license {
            return;
        }

        let pid = std::process::id();
        let thread_id = std::thread::current().id();

        if manager.pid != pid || manager.thread_id != thread_id {
            println!("{}", MULTIPLE_INSTANCE_WARNING);
            abort();
        }
    }

    /// Set the license key. Can only be called before calling any other Symbolica functions.
    pub fn set_license_key(key: &str) -> Result<(), String> {
        if LICENSE_KEY.get_or_init(|| key.to_owned()) != key {
            Err("Different license key cannot be set in same session")?;
        }

        Self::check_license_key()
    }

    /// Returns `true` iff this instance has a valid license key set.
    pub fn is_licensed() -> bool {
        LICENSED.load(Relaxed) || Self::check_license_key().is_ok()
    }

    /// Get the current Symbolica version.
    pub fn get_version() -> &'static str {
        env!("SYMBOLICA_VERSION")
    }

    fn request_license_email(data: HashMap<String, JsonValue>) -> Result<(), String> {
        let mut stream = Self::connect()?;
        let mut v = JsonValue::from(data).stringify().unwrap();
        v.push('\n');

        stream
            .write_all(v.as_bytes())
            .map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .map_err(|e| format!("{}\nError: {}", NETWORK_ERROR, e))?;
        let read_str = std::str::from_utf8(&buf).map_err(|_| "Bad server response".to_string())?;

        if read_str == "{\"status\":\"email sent\"}\n" {
            Ok(())
        } else if read_str.is_empty() {
            Err("Empty response".to_owned())
        } else {
            let message: JsonValue = read_str[..read_str.len() - 1]
                .parse()
                .map_err(|_| "Bad server response".to_string())?;
            let message_parsed: &HashMap<_, _> = message
                .get()
                .ok_or_else(|| "Bad server response".to_string())?;
            let status: &String = message_parsed
                .get("status")
                .unwrap()
                .get()
                .ok_or_else(|| "Bad server response".to_string())?;
            Err(status.clone())
        }
    }

    /// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_hobbyist_license(name: &str, email: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("type".to_owned(), "hobbyist".to_owned().into());
        Self::request_license_email(m)
    }

    /// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_trial_license(name: &str, email: &str, company: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("company".to_owned(), company.to_owned().into());
        m.insert("type".to_owned(), "trial".to_owned().into());
        Self::request_license_email(m)
    }

    /// Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    /// The key will be sent to the e-mail address `email`.
    pub fn request_sublicense(
        name: &str,
        email: &str,
        company: &str,
        super_license: &str,
    ) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("company".to_owned(), company.to_owned().into());
        m.insert("type".to_owned(), "sublicense".to_owned().into());
        m.insert("super_license".to_owned(), super_license.to_owned().into());
        Self::request_license_email(m)
    }

    /// Get the license key for the account registered with the provided email address.
    pub fn get_license_key(email: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("email".to_owned(), email.to_owned().into());
        Self::request_license_email(m)
    }
}
