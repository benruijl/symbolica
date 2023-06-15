pub mod api;
pub mod coefficient;
pub mod expand;
pub mod id;
pub mod normalize;
pub mod parser;
pub mod poly;
pub mod printer;
pub mod representations;
pub mod rings;
pub mod state;
pub mod utils;

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
