use std::{
    collections::HashMap,
    env,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::abort,
    thread::ThreadId,
    time::{Duration, SystemTime},
};

use colored::Colorize;
use once_cell::sync::OnceCell;
use tinyjson::JsonValue;

pub mod api;
pub mod coefficient;
pub mod collect;
pub mod combinatorics;
pub mod derivative;
pub mod domains;
pub mod evaluate;
pub mod expand;
pub mod id;
pub mod normalize;
pub mod numerical_integration;
pub mod parser;
pub mod poly;
pub mod printer;
pub mod representations;
pub mod solve;
pub mod state;
pub mod streaming;
pub mod transformer;
pub mod utils;

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

static LICENSE_KEY: OnceCell<String> = OnceCell::new();
static LICENSE_MANAGER: OnceCell<LicenseManager> = OnceCell::new();

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

const NETWORK_ERROR: &str = "┌────────────────────────────────────────────────┐
│ Could not connect to Symbolica license server. │
│                                                │
│ Please check your network configuration.       │
└────────────────────────────────────────────────┘";

const ACTIVATION_ERROR: &str = "┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘";

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
                eprintln!("{}", e);
            }
        }

        println!(
            "┌────────────────────────────────────────────────────────┐
│ You are running an unlicensed Symbolica instance.      │
│                                                        │
│ This mode is only allowed for non-professional use and │
│ is limited to one instance and core.                   │
│                                                        │
│ {} can easily acquire a free license to unlock  │
│ all cores and to remove this banner and the prompt:    │
│                                                        │
│   from symbolica import request_hobbyist_license       │
│   request_hobbyist_license('YOUR_NAME', 'YOUR_EMAIL')  │
│                                                        │
│ {} users must obtain an appropriate license, │
│ or can get a free 14-day trial license:                │
│                                                        │
│   from symbolica import request_trial_license          │
│   request_trial_license('NAME', 'EMAIL', 'EMPLOYER')   │
│                                                        │
│ See https://symbolica.io/docs/get_started.html#license │
└────────────────────────────────────────────────────────┘",
            "Hobbyists".bold(),
            "Professional".bold(),
        );
        print!("Confirm that you are a non-professional user of Symbolica (y/N): ");
        std::io::stdout().flush().unwrap();
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer).unwrap();
        if !["Y\n", "y\n", "yes\n"].contains(&buffer.as_str()) {
            abort();
        }

        match TcpListener::bind("127.0.0.1:12011") {
            Ok(o) => {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build_global()
                    .unwrap();

                drop(o);

                std::thread::spawn(|| loop {
                    match TcpListener::bind("127.0.0.1:12011") {
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

    fn check_license_key() -> Result<(), String> {
        let key = LICENSE_KEY
            .get()
            .cloned()
            .or(env::var("SYMBOLICA_LICENSE").ok());

        let Some(key) = key else {
            return Err(ACTIVATION_ERROR.to_owned());
        };

        if key.contains('@') {
            let mut a = key.split('@');
            let f1 = a.next().unwrap();
            let f2 = a.next().unwrap();
            let f3 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;

            let mut h: u32 = 5381;
            for b in env!("SYMBOLICA_VERSION").as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in f2.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in f3.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }

            if f1 != h.to_string() {
                Err(ACTIVATION_ERROR.to_owned())?;
            }

            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let t2 = f2.parse::<u64>().map_err(|_| ACTIVATION_ERROR.to_owned())?;

            if t < t2 || t - t2 > 24 * 60 * 60 {
                Err("┌───────────────────────────────────────────┐
│ The offline Symbolica license has expired │
└───────────────────────────────────────────┘"
                    .to_owned())?;
            }

            return Ok(());
        }

        let Ok(mut stream) = TcpStream::connect("symbolica.io:12012") else {
            return Err(NETWORK_ERROR.to_owned());
        };

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

    fn check(&self) {
        if self.has_license {
            return;
        }

        let pid = std::process::id();
        let thread_id = std::thread::current().id();

        if self.pid != pid || self.thread_id != thread_id {
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
        Self::check_license_key().is_ok()
    }

    /// Get the current Symbolica version.
    pub fn get_version() -> &'static str {
        env!("SYMBOLICA_VERSION")
    }

    /// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_hobbyist_license(name: &str, email: &str) -> Result<(), String> {
        if let Ok(mut stream) = TcpStream::connect("symbolica.io:12012") {
            let mut m: HashMap<String, JsonValue> = HashMap::default();
            m.insert("name".to_owned(), name.to_owned().into());
            m.insert("email".to_owned(), email.to_owned().into());
            m.insert("type".to_owned(), "hobbyist".to_owned().into());
            let mut v = JsonValue::from(m).stringify().unwrap();
            v.push('\n');

            stream.write_all(v.as_bytes()).unwrap();

            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).unwrap();
            let read_str = std::str::from_utf8(&buf).unwrap();

            if read_str == "{\"status\":\"email sent\"}\n" {
                Ok(())
            } else if read_str.is_empty() {
                Err("Empty response".to_owned())
            } else {
                let message: JsonValue = read_str[..read_str.len() - 1].parse().unwrap();
                let message_parsed: &HashMap<_, _> = message.get().unwrap();
                let status: &String = message_parsed.get("status").unwrap().get().unwrap();
                Err(status.clone())
            }
        } else {
            Err("Could not connect to the license server".to_owned())
        }
    }

    /// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_trial_license(name: &str, email: &str, company: &str) -> Result<(), String> {
        if let Ok(mut stream) = TcpStream::connect("symbolica.io:12012") {
            let mut m: HashMap<String, JsonValue> = HashMap::default();
            m.insert("name".to_owned(), name.to_owned().into());
            m.insert("email".to_owned(), email.to_owned().into());
            m.insert("company".to_owned(), company.to_owned().into());
            m.insert("type".to_owned(), "trial".to_owned().into());
            let mut v = JsonValue::from(m).stringify().unwrap();
            v.push('\n');

            stream.write_all(v.as_bytes()).unwrap();

            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).unwrap();
            let read_str = std::str::from_utf8(&buf).unwrap();

            if read_str == "{\"status\":\"email sent\"}\n" {
                Ok(())
            } else if read_str.is_empty() {
                Err("Empty response".to_owned())
            } else {
                let message: JsonValue = read_str[..read_str.len() - 1].parse().unwrap();
                let message_parsed: &HashMap<_, _> = message.get().unwrap();
                let status: &String = message_parsed.get("status").unwrap().get().unwrap();
                Err(status.clone())
            }
        } else {
            Err("Could not connect to the license server".to_owned())
        }
    }

    /// Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    /// The key will be sent to the e-mail address `email`.
    pub fn request_sublicense(
        name: &str,
        email: &str,
        company: &str,
        super_license: &str,
    ) -> Result<(), String> {
        if let Ok(mut stream) = TcpStream::connect("symbolica.io:12012") {
            let mut m: HashMap<String, JsonValue> = HashMap::default();
            m.insert("name".to_owned(), name.to_owned().into());
            m.insert("email".to_owned(), email.to_owned().into());
            m.insert("company".to_owned(), company.to_owned().into());
            m.insert("type".to_owned(), "sublicense".to_owned().into());
            m.insert("super_license".to_owned(), super_license.to_owned().into());
            let mut v = JsonValue::from(m).stringify().unwrap();
            v.push('\n');

            stream.write_all(v.as_bytes()).unwrap();

            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).unwrap();
            let read_str = std::str::from_utf8(&buf).unwrap();

            if read_str == "{\"status\":\"email sent\"}\n" {
                Ok(())
            } else if read_str.is_empty() {
                Err("Empty response".to_owned())
            } else {
                let message: JsonValue = read_str[..read_str.len() - 1].parse().unwrap();
                let message_parsed: &HashMap<_, _> = message.get().unwrap();
                let status: &String = message_parsed.get("status").unwrap().get().unwrap();
                Err(status.clone())
            }
        } else {
            Err("Could not connect to the license server".to_owned())
        }
    }

    /// Get a license key for offline use, generated from a licensed Symbolica session. The key will remain valid for 24 hours.
    pub fn get_offline_license_key() -> Result<String, String> {
        if !Self::check_license_key().is_ok() {
            Err("Cannot request offline license from an unlicensed session".to_owned())?;
        }
        let key = LICENSE_KEY
            .get()
            .cloned()
            .or(env::var("SYMBOLICA_LICENSE").ok());

        let Some(key) = key else {
            return Err(ACTIVATION_ERROR.to_owned());
        };

        if !key.contains('@') {
            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string();

            let mut h: u32 = 5381;
            for b in env!("SYMBOLICA_VERSION").as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in t.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in key.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }

            Ok(format!("{}@{}@{}", h, t, key))
        } else {
            Err("Cannot request offline license key from an offline session".to_owned())
        }
    }
}
