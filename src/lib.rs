pub mod api;
pub mod coefficient;
pub mod derivative;
pub mod expand;
pub mod id;
pub mod normalize;
pub mod numerical_integration;
pub mod parser;
pub mod poly;
pub mod printer;
pub mod representations;
pub mod rings;
pub mod state;
pub mod streaming;
pub mod transformer;
pub mod utils;

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
