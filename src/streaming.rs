use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    ops::{Add, AddAssign},
    sync::{Arc, Mutex},
};

use brotli::{CompressorWriter, Decompressor};
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::{
    atom::{Atom, AtomView},
    state::RecycledAtom,
};

pub trait ReadableNamedStream: Read + Send {
    fn open(name: &str) -> Self;
}

impl ReadableNamedStream for BufReader<File> {
    fn open(name: &str) -> Self {
        BufReader::new(File::open(name).unwrap())
    }
}

impl ReadableNamedStream for Decompressor<BufReader<File>> {
    fn open(name: &str) -> Self {
        brotli::Decompressor::new(BufReader::new(File::open(name).unwrap()), 4096)
    }
}

pub trait WriteableNamedStream: Write + Send {
    type Reader: ReadableNamedStream;

    fn create(name: &str) -> Self;
}

impl WriteableNamedStream for BufWriter<File> {
    type Reader = BufReader<File>;

    fn create(name: &str) -> Self {
        BufWriter::new(File::create(name).unwrap())
    }
}

impl WriteableNamedStream for CompressorWriter<BufWriter<File>> {
    type Reader = Decompressor<BufReader<File>>;

    fn create(name: &str) -> Self {
        CompressorWriter::new(BufWriter::new(File::create(name).unwrap()), 4096, 6, 22)
    }
}

#[derive(Clone)]
pub struct TermStreamerConfig {
    pub n_cores: usize,
    pub path: String,
    pub max_mem_bytes: usize,
}

impl Default for TermStreamerConfig {
    fn default() -> Self {
        Self {
            n_cores: 4,
            path: ".".to_owned(),
            max_mem_bytes: 1073741824, // 1 GB
        }
    }
}

struct TermInputStream<'a, R: ReadableNamedStream> {
    mem_buf: &'a [Atom],
    file_buf: Vec<R>,
    pos: usize,
    mem_pos: usize,
}

impl<'a, R: ReadableNamedStream> Iterator for TermInputStream<'a, R> {
    type Item = Atom;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == 0 {
            if self.mem_pos < self.mem_buf.len() {
                self.mem_pos += 1;
                return Some(self.mem_buf[self.mem_pos - 1].clone());
            }

            self.pos += 1;
        }

        while self.pos <= self.file_buf.len() {
            let mut a = Atom::new();
            if let Ok(()) = a.read(&mut self.file_buf[self.pos - 1]) {
                return Some(a);
            }

            self.pos += 1;
        }

        None
    }
}

/// A term streamer that has terms partly in memory and partly on another storage device.
pub struct TermStreamer<W: WriteableNamedStream> {
    mem_buf: Vec<Atom>,
    mem_size: usize,
    num_terms: usize,
    total_size: usize,
    file_buf: Vec<W>,
    config: TermStreamerConfig,
    filename: String,
    thread_pool: Arc<rayon::ThreadPool>,
    generation: usize,
}

impl<W: WriteableNamedStream> Drop for TermStreamer<W> {
    fn drop(&mut self) {
        for x in &mut (0..self.file_buf.len()) {
            std::fs::remove_file(&format!("{}_{}_{}", self.filename, self.generation, x)).unwrap();
        }
    }
}

impl<W: WriteableNamedStream> Default for TermStreamer<W>
where
    for<'a> AtomView<'a>: Send,
    Atom: Send,
{
    fn default() -> Self {
        Self::new(TermStreamerConfig::default())
    }
}

impl<W: WriteableNamedStream> Add<&mut TermStreamer<W>> for &mut TermStreamer<W> {
    type Output = TermStreamer<W>;

    fn add(self, rhs: &mut TermStreamer<W>) -> Self::Output {
        let mut n = self.next_generation();
        let r1 = self.reader();
        let r2 = rhs.reader();

        for a in r1.chain(r2) {
            n.push(a);
        }

        n
    }
}

impl<W: WriteableNamedStream> AddAssign<&mut TermStreamer<W>> for TermStreamer<W> {
    fn add_assign(&mut self, rhs: &mut TermStreamer<W>) {
        for a in rhs.reader() {
            self.push(a);
        }
    }
}

impl<W: WriteableNamedStream> Add<Atom> for TermStreamer<W> {
    type Output = TermStreamer<W>;

    fn add(mut self, rhs: Atom) -> Self::Output {
        self.push(rhs);
        self
    }
}

impl<W: WriteableNamedStream> TermStreamer<W> {
    /// Create a new term streamer.
    pub fn new(config: TermStreamerConfig) -> Self {
        let filename = loop {
            let name = format!("{}/{:x}", config.path, thread_rng().gen::<u64>());
            if !std::path::Path::new(&name).exists() {
                break name;
            }
        };

        Self {
            mem_buf: vec![],
            mem_size: 0,
            num_terms: 0,
            total_size: 0,
            file_buf: vec![],
            filename,
            thread_pool: Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.n_cores)
                    .build()
                    .unwrap(),
            ),
            config,
            generation: 0,
        }
    }

    fn next_generation(&self) -> Self {
        Self {
            mem_buf: vec![],
            mem_size: 0,
            num_terms: 0,
            total_size: 0,
            file_buf: vec![],
            filename: self.filename.clone(),
            config: self.config.clone(),
            thread_pool: self.thread_pool.clone(),
            generation: self.generation + 1,
        }
    }

    /// Returns true iff the stream fits in memory.
    pub fn fits_in_memory(&self) -> bool {
        self.file_buf.is_empty()
    }

    /// Get the number of terms in the stream.
    pub fn get_num_terms(&self) -> usize {
        self.num_terms
    }

    /// Add terms to the buffer.
    pub fn push(&mut self, a: Atom) {
        if let AtomView::Add(aa) = a.as_view() {
            for arg in aa.iter() {
                self.push_sorted_impl(arg.to_owned());
            }
        } else {
            self.push_sorted_impl(a);
        }
    }

    fn push_sorted_impl(&mut self, a: Atom) {
        let size = a.as_view().get_byte_size();
        self.mem_buf.push(a);
        self.mem_size += size;
        self.num_terms += 1;
        self.total_size += size;

        if self.mem_size >= self.config.max_mem_bytes {
            self.sort();

            if self.mem_size * 2 > self.config.max_mem_bytes {
                self.file_buf.push(W::create(&format!(
                    "{}_{}_{}",
                    self.filename,
                    self.generation,
                    self.file_buf.len()
                )));

                let f = self.file_buf.last_mut().unwrap();
                for x in self.mem_buf.drain(..) {
                    x.as_view().write(&mut *f).unwrap();
                }
                self.mem_size = 0;
            }
        }
    }

    /// Sort all the terms in the memory buffer.
    fn sort(&mut self) {
        self.mem_buf
            .par_sort_by(|a, b| a.as_view().cmp_terms(&b.as_view()));

        let mut out = Vec::with_capacity(self.mem_buf.len());
        let old_size = self.mem_buf.len();
        let mut new_size = 0;

        if !self.mem_buf.is_empty() {
            let mut last_buf = self.mem_buf.remove(0);

            let mut handle: RecycledAtom = Atom::new().into();

            for cur_buf in self.mem_buf.drain(..) {
                if !last_buf.merge_terms(cur_buf.as_view(), &mut handle) {
                    // we are done merging
                    {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if !n.is_zero() {
                                new_size += v.get_byte_size();
                                out.push(last_buf);
                            }
                        } else {
                            new_size += v.get_byte_size();
                            out.push(last_buf);
                        }
                    }
                    last_buf = cur_buf;
                }
            }

            if let AtomView::Num(n) = last_buf.as_view() {
                if !n.is_zero() {
                    new_size += last_buf.as_view().get_byte_size();
                    out.push(last_buf);
                }
            } else {
                new_size += last_buf.as_view().get_byte_size();
                out.push(last_buf);
            }
        }

        self.mem_buf = out;
        self.num_terms += self.mem_buf.len();
        self.num_terms -= old_size;
        self.total_size += new_size;
        self.total_size -= self.mem_size;
        self.mem_size = new_size;
    }

    /// Fuse sorted streams into one sorted stream.
    pub fn normalize(&mut self) {
        self.sort();

        if self.file_buf.is_empty() {
            return;
        }

        self.mem_buf.reverse();

        let mut head = vec![self.mem_buf.pop()];

        let n_files = self.file_buf.len();

        for b in &mut self.file_buf {
            b.flush().unwrap();
        }

        let mut files: Vec<_> = (0..n_files)
            .map(|i| W::Reader::open(&format!("{}_{}_{}", self.filename, self.generation, i)))
            .collect();

        for ff in &mut files {
            let mut a = Atom::new();
            if let Ok(()) = a.read(ff) {
                head.push(Some(a));
            } else {
                head.push(None);
            }
        }

        let mut new_stream = self.next_generation();

        let mut last = Atom::new();

        let mut smallest = (0..head.len()).collect::<Vec<_>>();
        let mut helper = Atom::new();
        loop {
            // find minimal element
            smallest.sort_unstable_by(|a, b| {
                if let Some(aa) = &head[*a] {
                    if let Some(bb) = &head[*b] {
                        aa.as_view().cmp_terms(&bb.as_view())
                    } else {
                        std::cmp::Ordering::Less
                    }
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            let Some(c) = head[smallest[0]].take() else {
                // all None
                break;
            };

            // load the next element
            if smallest[0] == 0 {
                head[0] = self.mem_buf.pop();
            } else {
                let mut a = Atom::new();
                if let Ok(()) = a.read(&mut files[smallest[0] - 1]) {
                    head[smallest[0]] = Some(a);
                }
            }

            if !last.merge_terms(c.as_view(), &mut helper) {
                if let AtomView::Num(n) = last.as_view() {
                    if !n.is_zero() {
                        new_stream.push_sorted_impl(last.clone());
                    }
                } else {
                    new_stream.push_sorted_impl(last.clone());
                }

                last.set_from_view(&c.as_view());
            }
        }

        if let AtomView::Num(n) = last.as_view() {
            if !n.is_zero() {
                new_stream.push_sorted_impl(last.clone());
            }
        } else {
            new_stream.push_sorted_impl(last.clone());
        }

        *self = new_stream;
    }

    /// Convert the term stream into an expression. This may exceed the available memory.
    pub fn to_expression(&mut self) -> Atom {
        self.normalize();

        let mut a = Atom::new();
        let add = a.to_add();

        for x in self.reader() {
            add.extend(x.as_view());
        }

        if add.get_nargs() == 1 {
            let mut b = Atom::new();
            b.set_from_view(&add.to_add_view().iter().next().unwrap());
            return b;
        }

        add.set_normalized(true);
        a
    }

    fn reader(&mut self) -> TermInputStream<W::Reader> {
        let num_files = self.file_buf.len();

        for x in &mut self.file_buf {
            x.flush().unwrap();
        }

        TermInputStream {
            mem_buf: &self.mem_buf,
            file_buf: (0..num_files)
                .map(|i| W::Reader::open(&format!("{}_{}_{}", self.filename, self.generation, i)))
                .collect(),
            pos: 0,
            mem_pos: 0,
        }
    }

    /// Map every term in the stream using the function `f`. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map(&mut self, f: impl Fn(Atom) -> Atom + Send + Sync) -> Self {
        let t = self.thread_pool.clone();

        let new_out = self.next_generation();

        let reader = self.reader();

        let out_wrap = Mutex::new(new_out);

        t.install(
            #[inline(always)]
            || {
                reader.par_bridge().for_each(|x| {
                    out_wrap.lock().unwrap().push(f(x));
                });
            },
        );

        out_wrap.into_inner().unwrap()
    }

    /// Map every term in the stream using the function `f` using a single thread. The resulting terms
    /// are a stream as well, which is returned by this function.
    pub fn map_single_thread(&mut self, f: impl Fn(Atom) -> Atom) -> Self {
        let mut new_out = self.next_generation();

        let reader = self.reader();

        for x in reader {
            new_out.push(f(x));
        }

        new_out
    }

    pub fn eq(&mut self, other: &mut Self) -> bool {
        self.normalize();
        other.normalize();

        self.reader().eq(other.reader())
    }

    pub fn get_byte_size(&self) -> usize {
        self.total_size
    }
}

#[cfg(test)]
mod test {
    use std::{fs::File, io::BufWriter};

    use brotli::CompressorWriter;

    use crate::{
        atom::{Atom, AtomType},
        id::{Pattern, PatternRestriction},
        state::State,
        streaming::{TermStreamer, TermStreamerConfig},
    };

    #[test]
    fn file_stream() {
        let mut streamer =
            TermStreamer::<CompressorWriter<BufWriter<File>>>::new(TermStreamerConfig {
                n_cores: 4,
                path: ".".to_owned(),
                max_mem_bytes: 20,
            });

        let input = Atom::parse("v1 + f1(v1) + 2*f1(v2) + 7*f1(v3) + v2 + v3 + v4").unwrap();
        streamer.push(input);

        let _ = streamer.reader();

        streamer = streamer + Atom::parse("f1(v1)").unwrap();

        streamer = streamer.map(|f| f);

        let pattern = Pattern::parse("f1(x_)").unwrap();
        let rhs = Pattern::parse("f1(v1) + v1").unwrap();

        streamer = streamer.map(|x| pattern.replace_all(x.as_view(), &rhs, None, None).expand());

        streamer.normalize();

        let r = streamer.to_expression();

        let res = Atom::parse("12*v1+v2+v3+v4+11*f1(v1)").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn file_stream_with_rationals() {
        let mut streamer =
            TermStreamer::<CompressorWriter<BufWriter<File>>>::new(TermStreamerConfig {
                n_cores: 4,
                path: ".".to_owned(),
                max_mem_bytes: 20,
            });

        let input = Atom::parse("v1*coeff(v2/v3+1)+v2*coeff(v3+1)+v3*coeff(1/v2)").unwrap();
        streamer.push(input);

        let pattern = Pattern::parse("v1_").unwrap();
        let rhs = Pattern::parse("v1").unwrap();

        streamer = streamer.map(|x| {
            pattern
                .replace_all(
                    x.as_view(),
                    &rhs,
                    Some(
                        &(
                            State::get_symbol("v1_"),
                            PatternRestriction::IsAtomType(AtomType::Var),
                        )
                            .into(),
                    ),
                    None,
                )
                .expand()
        });

        streamer.normalize();

        let r = streamer.to_expression();

        let res = Atom::parse("coeff((v3+2*v2*v3+v2*v3^2+v2^2)/(v2*v3))*v1").unwrap();
        assert_eq!(r, res);
    }

    #[test]
    fn memory_stream() {
        let input = Atom::parse("v1 + f1(v1) + 2*f1(v2) + 7*f1(v3)").unwrap();
        let pattern = Pattern::parse("f1(x_)").unwrap();
        let rhs = Pattern::parse("f1(v1) + v1").unwrap();

        let mut stream = TermStreamer::<BufWriter<File>>::new(TermStreamerConfig::default());
        stream.push(input);

        // map every term in the expression
        stream = stream.map(|x| pattern.replace_all(x.as_view(), &rhs, None, None).expand());

        let r = stream.to_expression();

        let res = Atom::parse("11*v1+10*f1(v1)").unwrap();
        assert_eq!(r, res);
    }
}
