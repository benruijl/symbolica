use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use serde::{Deserialize, Serialize};

use crate::domains::float::{NumericalFloatComparison, Real};

/// Keep track of statistical quantities, such as the average,
/// the error and the chi-squared of samples added over multiple
/// iterations.
///
/// Samples can be added using [`StatisticsAccumulator::add_samples()`]. When an iteration of
/// samples is finished, call [`StatisticsAccumulator::update_iter()`], which
/// updates the average, error and chi-squared over all iterations with the average
/// and error of the current iteration in a weighted fashion.
///
/// This accumulator can be merged with another accumulator using [`StatisticsAccumulator::merge()`] or
///  [`StatisticsAccumulator::merge_samples_no_reset()`]. This is useful when
/// samples are collected in multiple threads.
///
/// The accumulator also stores which samples yielded the highest weight thus far.
/// This can be used to study the input that impacted the average and error the most.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct StatisticsAccumulator<T: Real + NumericalFloatComparison> {
    sum: T,
    sum_sq: T,
    weight_sum: T,
    avg_sum: T,
    pub avg: T,
    pub err: T,
    guess: T,
    pub chi_sq: T,
    chi_sum: T,
    chi_sq_sum: T,
    new_samples: usize,
    new_zero_evaluations: usize,
    pub cur_iter: usize,
    pub processed_samples: usize,
    pub max_eval_positive: T,
    pub max_eval_positive_xs: Option<Sample<T>>,
    pub max_eval_negative: T,
    pub max_eval_negative_xs: Option<Sample<T>>,
    pub num_zero_evaluations: usize,
}

impl<T: Real + NumericalFloatComparison> StatisticsAccumulator<T> {
    /// Create a new [StatisticsAccumulator].
    pub fn new() -> StatisticsAccumulator<T> {
        StatisticsAccumulator {
            sum: T::zero(),
            sum_sq: T::zero(),
            weight_sum: T::zero(),
            avg_sum: T::zero(),
            avg: T::zero(),
            err: T::zero(),
            guess: T::zero(),
            chi_sq: T::zero(),
            chi_sum: T::zero(),
            chi_sq_sum: T::zero(),
            new_samples: 0,
            new_zero_evaluations: 0,
            cur_iter: 0,
            processed_samples: 0,
            max_eval_positive: T::zero(),
            max_eval_positive_xs: None,
            max_eval_negative: T::zero(),
            max_eval_negative_xs: None,
            num_zero_evaluations: 0,
        }
    }

    /// Copy the statistics accumulator, skipping the samples
    /// that evaluated to the maximum point.
    ///
    /// This function does not allocate.
    pub fn shallow_copy(&self) -> StatisticsAccumulator<T> {
        StatisticsAccumulator {
            sum: self.sum,
            sum_sq: self.sum_sq,
            weight_sum: self.weight_sum,
            avg_sum: self.avg_sum,
            avg: self.avg,
            err: self.err,
            guess: self.guess,
            chi_sq: self.chi_sq,
            chi_sum: self.chi_sum,
            chi_sq_sum: self.chi_sq_sum,
            new_samples: self.new_samples,
            new_zero_evaluations: self.new_zero_evaluations,
            cur_iter: self.cur_iter,
            processed_samples: self.processed_samples,
            max_eval_positive: self.max_eval_positive,
            max_eval_positive_xs: None,
            max_eval_negative: self.max_eval_negative,
            max_eval_negative_xs: None,
            num_zero_evaluations: self.num_zero_evaluations,
        }
    }

    /// Add a new `sample` to the accumulator with its corresponding evaluation `eval`.
    /// Note that the average and error are only updated upon calling [`Self::update_iter()`].
    pub fn add_sample(&mut self, eval: T, sample: Option<&Sample<T>>) {
        self.sum += &eval;
        self.sum_sq += eval * eval;
        self.new_samples += 1;

        if eval == T::zero() {
            self.new_zero_evaluations += 1;
        }

        if self.max_eval_positive_xs.is_none() || eval > self.max_eval_positive {
            self.max_eval_positive = eval;
            self.max_eval_positive_xs = sample.cloned();
        }

        if self.max_eval_negative_xs.is_none() || eval < self.max_eval_negative {
            self.max_eval_negative = eval;
            self.max_eval_negative_xs = sample.cloned();
        }
    }

    /// Add the non-processed samples of `other` to non-processed samples of this
    /// accumulator. The non-processed samples are removed from `other`.
    pub fn merge_samples(&mut self, other: &mut StatisticsAccumulator<T>) {
        self.merge_samples_no_reset(other);

        // reset the other
        other.sum = T::zero();
        other.sum_sq = T::zero();
        other.new_samples = 0;
        other.new_zero_evaluations = 0;
    }

    /// Add the non-processed samples of `other` to non-processed samples of this
    /// accumulator without removing the samples from `other`.
    pub fn merge_samples_no_reset(&mut self, other: &StatisticsAccumulator<T>) {
        self.sum += &other.sum;
        self.sum_sq += &other.sum_sq;
        self.new_samples += other.new_samples;
        self.new_zero_evaluations += other.new_zero_evaluations;

        if other.max_eval_positive > self.max_eval_positive {
            self.max_eval_positive = other.max_eval_positive;
            self.max_eval_positive_xs = other.max_eval_positive_xs.clone();
        }

        if other.max_eval_negative < self.max_eval_negative {
            self.max_eval_negative = other.max_eval_negative;
            self.max_eval_negative_xs = other.max_eval_negative_xs.clone();
        }
    }

    /// Process the samples added with `[`Self::add_sample()`]` and
    /// compute a new average, error, and chi-squared.
    pub fn update_iter(&mut self) -> bool {
        // TODO: we could be throwing away events that are very rare
        if self.new_samples < 2 {
            self.cur_iter += 1;
            return false;
        }

        self.processed_samples += self.new_samples;
        self.num_zero_evaluations += self.new_zero_evaluations;
        let n = T::from_usize(self.new_samples);
        self.sum /= &n;
        self.sum_sq /= n * n;
        let mut w = (self.sum_sq * n).sqrt();

        w = ((w + self.sum) * (w - self.sum)) / (n - T::one());
        if w == T::zero() {
            // all sampled points are the same
            // set the weight to a large number
            w = T::from_usize(usize::MAX);
        } else {
            w = w.inv();
        }

        self.weight_sum += w;
        self.avg_sum += w * self.sum;
        let sigma_sq = self.weight_sum.inv();
        self.avg = sigma_sq * self.avg_sum;
        self.err = sigma_sq.sqrt();
        if self.cur_iter == 0 {
            self.guess = self.sum;
        }
        w *= self.sum - self.guess;
        self.chi_sum += w;
        self.chi_sq_sum += w * self.sum;
        self.chi_sq = self.chi_sq_sum - self.avg * self.chi_sum;

        // reset
        self.sum = T::zero();
        self.sum_sq = T::zero();
        self.new_samples = 0;
        self.new_zero_evaluations = 0;
        self.cur_iter += 1;

        true
    }

    /// Format `mean ± sdev` as `mean(sdev)` in a human-readable way with the correct number of digits.
    ///
    /// Based on the Python package [gvar](https://github.com/gplepage/gvar) by Peter Lepage.
    pub fn format_uncertainty(&self) -> String {
        Self::format_uncertainty_impl(self.avg.to_f64(), self.err.to_f64())
    }

    fn format_uncertainty_impl(mean: f64, sdev: f64) -> String {
        fn ndec(x: f64, offset: usize) -> i32 {
            let mut ans = (offset as f64 - x.log10()) as i32;
            if ans > 0 && x * 10.0f64.powi(ans) >= [0.5, 9.5, 99.5][offset] {
                ans -= 1;
            }
            if ans < 0 {
                0
            } else {
                ans
            }
        }
        let v = mean;
        let dv = sdev;

        // special cases
        if v.is_nan() || dv.is_nan() {
            format!("{:e} ± {:e}", v, dv)
        } else if dv.is_infinite() {
            format!("{:e} ± inf", v)
        } else if v == 0. && !(1e-4..1e5).contains(&dv) {
            if dv == 0. {
                "0(0)".to_owned()
            } else {
                let e = format!("{:.1e}", dv);
                let mut ans = e.split('e');
                let e1 = ans.next().unwrap();
                let e2 = ans.next().unwrap();
                "0.0(".to_owned() + e1 + ")e" + e2
            }
        } else if v == 0. {
            if dv >= 9.95 {
                format!("0({:.0})", dv)
            } else if dv >= 0.995 {
                format!("0.0({:.1})", dv)
            } else {
                let ndecimal = ndec(dv, 2);
                format!(
                    "{:.*}({:.0})",
                    ndecimal as usize,
                    v,
                    dv * 10.0f64.powi(ndecimal)
                )
            }
        } else if dv == 0. {
            let e = format!("{:e}", v);
            let mut ans = e.split('e');
            let e1 = ans.next().unwrap();
            let e2 = ans.next().unwrap();
            if e2 != "0" {
                e1.to_owned() + "(0)e" + e2
            } else {
                e1.to_owned() + "(0)"
            }
        } else if dv > 1e4 * v.abs() {
            format!("{:.1e} ± {:.2e}", v, dv)
        } else if v.abs() >= 1e6 || v.abs() < 1e-5 {
            // exponential notation for large |self.mean|
            let exponent = v.abs().log10().floor();
            let fac = 10.0.powf(exponent);
            let mantissa = Self::format_uncertainty_impl(v / fac, dv / fac);
            let e = format!("{:.0e}", fac);
            let mut ee = e.split('e');
            mantissa + "e" + ee.nth(1).unwrap()
        }
        // normal cases
        else if dv >= 9.95 {
            if v.abs() >= 9.5 {
                format!("{:.0}({:.0})", v, dv)
            } else {
                let ndecimal = ndec(v.abs(), 1);
                format!("{:.*}({:.*})", ndecimal as usize, v, ndecimal as usize, dv)
            }
        } else if dv >= 0.995 {
            if v.abs() >= 0.95 {
                format!("{:.1}({:.1})", v, dv)
            } else {
                let ndecimal = ndec(v.abs(), 1);
                format!("{:.*}({:.*})", ndecimal as usize, v, ndecimal as usize, dv)
            }
        } else {
            let ndecimal = ndec(v.abs(), 1).max(ndec(dv, 2));
            format!(
                "{:.*}({:.0})",
                ndecimal as usize,
                v,
                dv * 10.0f64.powi(ndecimal)
            )
        }
    }
}

/// A sample taken from a [Grid] that approximates a function. The sample is more likely to fall in a region
/// where the function changes rapidly.
///
/// If the sample comes from a [ContinuousGrid], it is the variant [Continuous](Sample::Continuous)
/// and contains the weight and the list of sample points.
/// If the sample comes from a [DiscreteGrid], it is the variant [Discrete](Sample::Discrete) and contains
/// the weight, the bin and the subsample if the bin has a nested grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Sample<T: Real + NumericalFloatComparison> {
    Continuous(T, Vec<T>),
    Discrete(T, usize, Option<Box<Sample<T>>>),
}

impl<T: Real + NumericalFloatComparison> Default for Sample<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + NumericalFloatComparison> Sample<T> {
    /// Construct a new empty sample that can be handed over to [`Grid::sample()`].
    pub fn new() -> Sample<T> {
        Sample::Continuous(T::zero(), vec![])
    }

    /// Get the weight of the sample.
    pub fn get_weight(&self) -> T {
        match self {
            Sample::Continuous(w, _) | Sample::Discrete(w, _, _) => *w,
        }
    }

    /// Transform the sample to a discrete grid, used for recycling memory.
    fn to_discrete_grid(&mut self) -> (&mut T, &mut usize, &mut Option<Box<Sample<T>>>) {
        if let Sample::Continuous(..) = self {
            *self = Sample::Discrete(T::zero(), 0, None);
        }

        match self {
            Sample::Discrete(weight, index, sub_sample) => (weight, index, sub_sample),
            _ => unreachable!(),
        }
    }

    /// Transform the sample to a continuous, used for recycling memory.
    fn to_continuous_grid(&mut self) -> (&mut T, &mut Vec<T>) {
        if let Sample::Continuous(..) = self {
            *self = Sample::Continuous(T::zero(), vec![])
        }

        match self {
            Sample::Continuous(weight, sub_samples) => (weight, sub_samples),
            _ => unreachable!(),
        }
    }
}

/// An adapting grid that captures the enhancements of an integrand.
/// It supports discrete and continuous dimensions. The discrete dimensions
/// can have a nested grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Grid<T: Real + NumericalFloatComparison> {
    Continuous(ContinuousGrid<T>),
    Discrete(DiscreteGrid<T>),
}

impl<T: Real + NumericalFloatComparison> Grid<T> {
    /// Sample a position in the grid. The sample is more likely to land in a region
    /// where the function the grid is based on is changing rapidly.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample<T>) {
        match self {
            Grid::Continuous(g) => g.sample(rng, sample),
            Grid::Discrete(g) => g.sample(rng, sample),
        }
    }

    /// Add a sample point and its corresponding evaluation `eval` to the grid as training.
    /// Upon a call to [`Grid::update`], the grid will be adapted to better represent
    /// the function that is being evaluated.
    pub fn add_training_sample(&mut self, sample: &Sample<T>, eval: T) -> Result<(), String> {
        match self {
            Grid::Continuous(g) => g.add_training_sample(sample, eval),
            Grid::Discrete(g) => g.add_training_sample(sample, eval),
        }
    }

    /// Returns `Ok` when this grid can be merged with another grid,
    /// and `Err` when the grids have a different shape.
    pub fn is_mergeable(&self, grid: &Grid<T>) -> Result<(), String> {
        match (self, grid) {
            (Grid::Continuous(c1), Grid::Continuous(c2)) => c1.is_mergeable(c2),
            (Grid::Discrete(d1), Grid::Discrete(d2)) => d1.is_mergeable(d2),
            _ => Err("Cannot merge a discrete and continuous grid".to_owned()),
        }
    }

    /// Merge a grid with exactly the same structure.
    pub fn merge(&mut self, grid: &Grid<T>) -> Result<(), String> {
        // first do a complete check to see if the grids are mergeable
        self.is_mergeable(grid)?;
        self.merge_unchecked(grid);

        Ok(())
    }

    /// Merge a grid without checks. For internal use only.
    fn merge_unchecked(&mut self, grid: &Grid<T>) {
        match (self, grid) {
            (Grid::Continuous(c1), Grid::Continuous(c2)) => c1.merge_unchecked(c2),
            (Grid::Discrete(d1), Grid::Discrete(d2)) => d1.merge_unchecked(d2),
            _ => panic!("Cannot merge grids that have a different shape."),
        }
    }

    /// Update the grid based on the samples added through [`Grid::add_training_sample`].
    pub fn update(&mut self, learning_rate: T) {
        match self {
            Grid::Continuous(g) => g.update(learning_rate),
            Grid::Discrete(g) => g.update(learning_rate),
        }
    }

    /// Get the statistics of this grid.
    pub fn get_statistics(&mut self) -> &StatisticsAccumulator<T> {
        match self {
            Grid::Continuous(g) => &g.accumulator,
            Grid::Discrete(g) => &g.accumulator,
        }
    }
}
/// A bin of a discrete grid, which may contain a subgrid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bin<T: Real + NumericalFloatComparison> {
    pub pdf: T,
    pub accumulator: StatisticsAccumulator<T>,
    pub sub_grid: Option<Grid<T>>,
}

impl<T: Real + NumericalFloatComparison> Bin<T> {
    /// Returns `Ok` when this grid can be merged with another grid,
    /// and `Err` when the grids have a different shape.
    pub fn is_mergeable(&self, other: &Bin<T>) -> Result<(), String> {
        if self.pdf != other.pdf {
            return Err("PDF not equivalent".to_owned());
        }

        match (&self.sub_grid, &other.sub_grid) {
            (None, None) => Ok(()),
            (Some(s1), Some(s2)) => s1.is_mergeable(s2),
            (None, Some(_)) | (Some(_), None) => Err("Sub-grid not equivalent".to_owned()),
        }
    }

    /// Merge a grid without checks. For internal use only.
    fn merge(&mut self, other: &Bin<T>) {
        self.accumulator.merge_samples_no_reset(&other.accumulator);

        if let (Some(s1), Some(s2)) = (&mut self.sub_grid, &other.sub_grid) {
            s1.merge_unchecked(s2);
        }
    }
}

/// A discrete grid consisting of a given number of bins.
/// Each bin may have a nested grid.
///
/// After adding training samples and updating, the probabilities
/// of a sample from the grid landing in a bin is proportional to its
/// average value if training happens on the average, or to its
/// variance (recommended).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteGrid<T: Real + NumericalFloatComparison> {
    pub bins: Vec<Bin<T>>,
    pub accumulator: StatisticsAccumulator<T>,
    max_prob_ratio: T,
    train_on_avg: bool,
}

impl<T: Real + NumericalFloatComparison> DiscreteGrid<T> {
    /// Create a new discrete grid with `bins.len()` number of bins, where
    /// each bin may have a sub-grid.
    ///
    /// Also set the maximal probability ratio between bins, `max_prob_ratio`,
    /// that prevents one bin from getting oversampled.
    ///
    /// If you want to train on the average instead of the error, set `train_on_avg` to `true` (not recommended).
    pub fn new(
        bins: Vec<Option<Grid<T>>>,
        max_prob_ratio: T,
        train_on_avg: bool,
    ) -> DiscreteGrid<T> {
        let pdf = T::from_usize(1) / T::from_usize(bins.len());
        DiscreteGrid {
            bins: bins
                .into_iter()
                .map(|s| Bin {
                    pdf,
                    accumulator: StatisticsAccumulator::new(),
                    sub_grid: s,
                })
                .collect(),
            accumulator: StatisticsAccumulator::new(),
            max_prob_ratio,
            train_on_avg,
        }
    }

    /// Sample a bin from all bins based on the bin pdfs.
    fn sample_bin<R: Rng + ?Sized>(&self, rng: &mut R) -> (usize, T) {
        let r: T = T::sample_unit(rng);

        let mut cdf = T::zero();
        for (i, bin) in self.bins.iter().enumerate() {
            cdf += bin.pdf;
            if r <= cdf {
                // the 'volume' of the bin is 1 / pdf
                return (i, bin.pdf.inv());
            }
        }
        unreachable!(
            "Could not sample discrete dimension: {:?} at point {}",
            self, r
        );
    }

    /// Update the discrete grid probabilities of landing in a particular bin when sampling,
    /// and adapt all sub-grids based on the new training samples.
    ///
    /// If `learning_rate` is set to 0, no training happens.
    pub fn update(&mut self, learning_rate: T) {
        let mut err_sum = T::zero();
        for bin in &mut self.bins {
            if let Some(sub_grid) = &mut bin.sub_grid {
                sub_grid.update(learning_rate);
            }

            let acc = &mut bin.accumulator;
            acc.update_iter();

            if acc.processed_samples > 1 {
                err_sum += acc.err * T::from_usize(acc.processed_samples - 1).sqrt();
            }
        }

        if learning_rate.is_zero()
            || self.bins.iter().all(|x| {
                if self.train_on_avg {
                    x.accumulator.avg == T::zero()
                } else {
                    x.accumulator.err == T::zero() || x.accumulator.processed_samples < 2
                }
            })
        {
            return;
        }

        let mut max_per_bin = T::zero();
        for bin in &mut self.bins {
            let acc = &mut bin.accumulator;

            if self.train_on_avg {
                bin.pdf = acc.avg.norm()
            } else if acc.processed_samples < 2 {
                bin.pdf = T::zero();
            } else {
                let n_samples = T::from_usize(acc.processed_samples - 1);
                let var = acc.err * n_samples.sqrt();
                bin.pdf = var;
            }

            if bin.pdf > max_per_bin {
                max_per_bin = bin.pdf;
            }
        }

        let mut sum = T::zero();
        for bin in &mut self.bins {
            bin.pdf = bin.pdf.max(&(max_per_bin / self.max_prob_ratio));
            sum += bin.pdf;
        }

        for bin in &mut self.bins {
            bin.pdf /= sum;
        }

        self.accumulator.update_iter();
    }

    /// Sample a point form this grid, writing the result in `sample`.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample<T>) {
        let (weight, vs, child) = sample.to_discrete_grid();

        *weight = T::one();
        let (v, w) = self.sample_bin(rng);
        *weight *= &w;
        *vs = v;

        // get the child grid for this sample
        if let Some(sub_grid) = &mut self.bins[v].sub_grid {
            let child_sample = if let Some(sub_sample) = child {
                sub_sample
            } else {
                *child = Some(Box::new(Sample::new()));
                child.as_mut().unwrap()
            };

            sub_grid.sample(rng, child_sample);

            // multiply the weight of the subsample
            *weight *= &child_sample.get_weight();
        } else {
            *child = None;
        };
    }

    /// Add a training sample with its corresponding evaluation, i.e. `f(sample)`, to the grid.
    pub fn add_training_sample(&mut self, sample: &Sample<T>, eval: T) -> Result<(), String> {
        if !eval.is_finite() {
            return Err(format!(
                "Added training sample that is not finite: sample={:?}, fx={}",
                sample, eval
            ));
        }

        if let Sample::Discrete(weight, index, sub_sample) = sample {
            self.accumulator.add_sample(eval * weight, Some(sample));

            // undo the weight of the bin, which is 1 / pdf
            let bin_weight = *weight * self.bins[*index].pdf;
            self.bins[*index]
                .accumulator
                .add_sample(bin_weight * eval, Some(sample));

            if let Some(sg) = &mut self.bins[*index].sub_grid {
                if let Some(sub_sample) = sub_sample {
                    sg.add_training_sample(sub_sample, eval)?;
                }
            }

            Ok(())
        } else {
            Err(format!("Discrete sample expected: {:?}", sample))
        }
    }

    /// Returns `Ok` when this grid can be merged with another grid,
    /// and `Err` when the grids have a different shape.
    pub fn is_mergeable(&self, other: &DiscreteGrid<T>) -> Result<(), String> {
        if self.bins.len() != other.bins.len() {
            return Err("Discrete grid dimensions do not match".to_owned());
        }

        for (c, o) in self.bins.iter().zip(&other.bins) {
            c.is_mergeable(o)?;
        }

        Ok(())
    }

    /// Merge a grid with exactly the same structure.
    pub fn merge(&mut self, grid: &DiscreteGrid<T>) -> Result<(), String> {
        // first do a complete check to see if the grids are mergeable
        self.is_mergeable(grid)?;
        self.merge_unchecked(grid);

        Ok(())
    }

    /// Merge a grid without checks. For internal use only.
    fn merge_unchecked(&mut self, other: &DiscreteGrid<T>) {
        for (c, o) in self.bins.iter_mut().zip(&other.bins) {
            c.merge(o);
        }

        self.accumulator.merge_samples_no_reset(&other.accumulator);
    }
}

/// An adaptive continuous grid that uses factorized dimensions to approximate
/// a function. The VEGAS algorithm is used to adapt the grid
/// based on new sample points.
///
/// After adding training samples and updating, the probabilities
/// of a sample from the grid landing in a bin is proportional to its
/// average value if training happens on the average, or to its
/// variance (recommended).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousGrid<T: Real + NumericalFloatComparison> {
    pub continuous_dimensions: Vec<ContinuousDimension<T>>,
    pub accumulator: StatisticsAccumulator<T>,
}

impl<T: Real + NumericalFloatComparison> ContinuousGrid<T> {
    /// Create a new grid with `n_dims` dimensions and `n_bins` bins
    /// per dimension.
    ///
    /// With `min_samples_for_update` grid updates can be prevented if
    /// there are too few samples in a certain bin. With `bin_number_evolution`
    /// the bin numbers can be changed based on the iteration index. If the
    /// `bin_number_evolution` array is smaller than the current iteration number,
    /// the last element is taken from the list.
    ///
    /// With `train_on_avg`, the grids will be adapted based on the average value
    /// of the bin, contrary to it's variance.
    pub fn new(
        n_dims: usize,
        n_bins: usize,
        min_samples_for_update: usize,
        bin_number_evolution: Option<Vec<usize>>,
        train_on_avg: bool,
    ) -> ContinuousGrid<T> {
        ContinuousGrid {
            continuous_dimensions: vec![
                ContinuousDimension::new(
                    n_bins,
                    min_samples_for_update,
                    bin_number_evolution,
                    train_on_avg
                );
                n_dims
            ],
            accumulator: StatisticsAccumulator::new(),
        }
    }

    /// Sample a point in the grid, writing the result in `sample`.
    pub fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R, sample: &mut Sample<T>) {
        let (weight, vs) = sample.to_continuous_grid();
        *weight = T::one();
        vs.clear();
        vs.resize(self.continuous_dimensions.len(), T::zero());
        for (vs, d) in vs.iter_mut().zip(&self.continuous_dimensions) {
            let (v, w) = d.sample(rng);
            *weight *= &w;
            *vs = v;
        }
    }

    /// Add a training sample with its corresponding evaluation, i.e. `f(sample)`, to the grid.
    pub fn add_training_sample(&mut self, sample: &Sample<T>, eval: T) -> Result<(), String> {
        if !eval.is_finite() {
            return Err(format!(
                "Added training sample that is not finite: sample={:?}, fx={}",
                sample, eval
            ));
        }

        if let Sample::Continuous(weight, xs) = sample {
            self.accumulator.add_sample(eval * weight, Some(sample));

            for (d, x) in self.continuous_dimensions.iter_mut().zip(xs) {
                d.add_training_sample(*x, *weight, eval)?;
            }
            Ok(())
        } else {
            unreachable!(
                "Sample cannot be converted to continuous sample: {:?}",
                self
            );
        }
    }

    /// Update the grid based on the added training samples. This will move the partition bounds of every dimension.
    ///
    /// The `learning_rate` determines the speed of the adaptation. If it is set to `0`, no training will be performed.
    pub fn update(&mut self, learning_rate: T) {
        for d in self.continuous_dimensions.iter_mut() {
            d.update(learning_rate);
        }

        self.accumulator.update_iter();
    }

    /// Returns `Ok` when this grid can be merged with another grid,
    /// and `Err` when the grids have a different shape.
    pub fn is_mergeable(&self, grid: &ContinuousGrid<T>) -> Result<(), String> {
        if self.continuous_dimensions.len() != grid.continuous_dimensions.len() {
            return Err("Cannot merge grids that have a different shape.".to_owned());
        }

        for (c, o) in self
            .continuous_dimensions
            .iter()
            .zip(&grid.continuous_dimensions)
        {
            c.is_mergeable(o)?;
        }
        Ok(())
    }

    /// Merge a grid with exactly the same structure.
    pub fn merge(&mut self, grid: &ContinuousGrid<T>) -> Result<(), String> {
        // first do a complete check to see if the grids are mergeable
        self.is_mergeable(grid)?;
        self.merge_unchecked(grid);

        Ok(())
    }

    /// Merge a grid without checks. For internal use only.
    fn merge_unchecked(&mut self, grid: &ContinuousGrid<T>) {
        self.accumulator.merge_samples_no_reset(&grid.accumulator);

        for (c, o) in self
            .continuous_dimensions
            .iter_mut()
            .zip(&grid.continuous_dimensions)
        {
            c.merge(o);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousDimension<T: Real + NumericalFloatComparison> {
    pub partitioning: Vec<T>,
    bin_accumulator: Vec<StatisticsAccumulator<T>>,
    bin_importance: Vec<T>,
    counter: Vec<usize>,
    min_samples_for_update: usize,
    bin_number_evolution: Vec<usize>,
    update_counter: usize,
    train_on_avg: bool,
}

impl<T: Real + NumericalFloatComparison> ContinuousDimension<T> {
    /// Create a new dimension with `n_bins` bins.
    ///
    /// With `min_samples_for_update` grid updates can be prevented if
    /// there are too few samples in a certain bin. With `bin_number_evolution`
    /// the bin numbers can be changed based on the iteration index. If the
    /// `bin_number_evolution` array is smaller than the current iteration number,
    /// the last element is taken from the list.
    ///
    /// With `train_on_avg`, the grids will be adapted based on the average value
    /// of the bin, contrary to it's variance.
    fn new(
        n_bins: usize,
        min_samples_for_update: usize,
        bin_number_evolution: Option<Vec<usize>>,
        train_on_avg: bool,
    ) -> ContinuousDimension<T> {
        ContinuousDimension {
            partitioning: (0..=n_bins)
                .map(|i| T::from_usize(i) / T::from_usize(n_bins))
                .collect(),
            bin_importance: vec![T::zero(); n_bins],
            bin_accumulator: vec![StatisticsAccumulator::new(); n_bins],
            counter: vec![0; n_bins],
            min_samples_for_update,
            bin_number_evolution: bin_number_evolution.unwrap_or(vec![n_bins]),
            update_counter: 0,
            train_on_avg,
        }
    }

    /// Sample a point in this dimension, writing the result in `sample`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> (T, T) {
        let r: T = T::sample_unit(rng);

        // map the point to a bin
        let n_bins = T::from_usize(self.partitioning.len() - 1);
        let bin_index = (n_bins * r).to_usize_clamped();
        let bin_width = self.partitioning[bin_index + 1] - self.partitioning[bin_index];

        // rescale the point in the bin
        let sample =
            self.partitioning[bin_index] + (n_bins * r - T::from_usize(bin_index)) * bin_width;
        let weight = n_bins as T * bin_width; // d_sample / d_r

        (sample, weight)
    }

    /// Add a training sample with its corresponding evaluation, i.e. `f(sample)`, to the proper bin.
    fn add_training_sample(&mut self, sample: T, weight: T, eval: T) -> Result<(), String> {
        if sample < T::zero() || sample > T::one() || !eval.is_finite() || !weight.is_finite() {
            return Err(format!(
                "Malformed sample point: sample={}, weight={}, fx={}",
                sample, weight, eval
            ));
        }

        let mut index = self
            .partitioning
            .binary_search_by(|v| v.partial_cmp(&sample).unwrap())
            .unwrap_or_else(|e| e);
        index = index.saturating_sub(1);

        self.bin_accumulator[index].add_sample(weight * eval, None);
        Ok(())
    }

    /// Update the grid based on the added training samples. This will move the partition bounds of every dimension.
    ///
    /// The `learning_rate` determines the speed of the adaptation. If it is set to `0`, no training will be performed.
    fn update(&mut self, learning_rate: T) {
        for (bi, acc) in self.bin_importance.iter_mut().zip(&self.bin_accumulator) {
            if self.train_on_avg {
                *bi += &acc.sum
            } else {
                *bi += &acc.sum_sq;
            }
        }

        for (c, acc) in self.counter.iter_mut().zip(&self.bin_accumulator) {
            *c += acc.new_samples;
        }

        if self.counter.iter().sum::<usize>() < self.min_samples_for_update {
            // do not train the grid if there is a lack of samples
            return;
        }

        if learning_rate.is_zero() {
            self.bin_accumulator.clear();
            self.bin_accumulator
                .resize(self.partitioning.len() - 1, StatisticsAccumulator::new());
            self.bin_importance.clear();
            self.bin_importance
                .resize(self.partitioning.len() - 1, T::zero());
            self.counter.clear();
            self.counter.resize(self.partitioning.len() - 1, 0);
            return;
        }

        let n_bins = self.partitioning.len() - 1;

        for avg in self.bin_importance.iter_mut() {
            *avg = avg.norm();
        }

        // normalize the average
        for (avg, &c) in self.bin_importance.iter_mut().zip(&self.counter) {
            if c > 0 {
                *avg /= T::from_usize(c);
            }
        }

        // smoothen the averages between adjacent grid points
        if self.partitioning.len() > 2 {
            let mut prev = self.bin_importance[0];
            let mut cur = self.bin_importance[1];
            self.bin_importance[0] = (T::from_usize(3) * prev + cur) / T::from_usize(4);
            for bin in 1..n_bins - 1 {
                let s = prev + cur * T::from_usize(6);
                prev = cur;
                cur = self.bin_importance[bin + 1];
                self.bin_importance[bin] = (s + cur) / T::from_usize(8);
            }
            self.bin_importance[n_bins - 1] = (prev + T::from_usize(3) * cur) / T::from_usize(4);
        }

        let sum: T = self.bin_importance.iter().sum();
        let mut imp_sum = T::zero();
        for bi in self.bin_importance.iter_mut() {
            let m = if *bi == sum {
                T::one()
            } else if *bi == T::zero() {
                T::zero()
            } else {
                ((*bi / sum - T::one()) / (*bi / sum).log()).powf(learning_rate)
            };
            *bi = m;
            imp_sum += m;
        }

        let new_number_of_bins = *self
            .bin_number_evolution
            .get(self.update_counter)
            .or(self.bin_number_evolution.last())
            .unwrap_or(&self.bin_accumulator.len());
        self.update_counter += 1;
        let new_weight_per_bin = imp_sum / T::from_usize(new_number_of_bins);

        // resize the bins using their importance measure
        let mut new_partitioning = vec![T::zero(); new_number_of_bins + 1];

        // evenly distribute the bins such that each has weight_per_bin weight
        let mut acc = T::zero();
        let mut j = 0;
        let mut target = T::zero();
        for nb in &mut new_partitioning[1..].iter_mut() {
            target += new_weight_per_bin;
            // find the bin that has the accumulated weight we are looking for
            while j < self.bin_importance.len() && acc + self.bin_importance[j] < target {
                acc += &self.bin_importance[j];
                // prevent some rounding errors from going out of the bin
                if j + 1 < self.bin_importance.len() {
                    j += 1;
                } else {
                    break;
                }
            }

            // find out how deep we are in the current bin
            let bin_depth = (target - acc) / self.bin_importance[j];
            *nb = self.partitioning[j]
                + bin_depth * (self.partitioning[j + 1] - self.partitioning[j]);
        }

        // it could be that all the weights are distributed before we reach 1, for example if the first bin
        // has all the weights. we still force to have the complete input range
        new_partitioning[new_number_of_bins] = T::one();
        self.partitioning = new_partitioning;

        self.bin_importance.clear();
        self.bin_importance
            .resize(self.partitioning.len() - 1, T::zero());
        self.counter.clear();
        self.counter.resize(self.partitioning.len() - 1, 0);
        self.bin_accumulator.clear();
        self.bin_accumulator
            .resize(self.partitioning.len() - 1, StatisticsAccumulator::new());
    }

    /// Returns `Ok` when this grid can be merged with another grid,
    /// and `Err` when the grids have a different shape.
    fn is_mergeable(&self, other: &ContinuousDimension<T>) -> Result<(), String> {
        if self.partitioning != other.partitioning {
            Err("Partitions do not match".to_owned())
        } else {
            Ok(())
        }
    }

    /// Merge a grid without checks. For internal use only.
    fn merge(&mut self, other: &ContinuousDimension<T>) {
        for (bi, obi) in self.bin_accumulator.iter_mut().zip(&other.bin_accumulator) {
            bi.merge_samples_no_reset(obi);
        }
    }
}

/// A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
/// A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).
///
/// Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
/// which is an instance counter starting at 0.
pub struct MonteCarloRng {
    state: Xoshiro256StarStar,
}

impl RngCore for MonteCarloRng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.state.next_u32()
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.state.fill_bytes(dest)
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.state.try_fill_bytes(dest)
    }
}

impl MonteCarloRng {
    /// Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
    /// each thread or instance generating samples should use the same `seed` but a different `stream_id`.
    pub fn new(seed: u64, stream_id: usize) -> Self {
        let mut state = Xoshiro256StarStar::seed_from_u64(seed);
        for _ in 0..stream_id {
            state.jump();
        }

        Self { state }
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use super::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample};

    #[test]
    fn multichannel() {
        // Integrate x*pi + x^2 using multi-channeling:
        // x*pi and x^2 will have their own Vegas grid
        let fs = [|x: f64| (x * PI).sin(), |x: f64| x * x];

        let mut grid = DiscreteGrid::new(
            vec![
                Some(Grid::Continuous(ContinuousGrid::new(
                    1, 10, 1000, None, false,
                ))),
                Some(Grid::Continuous(ContinuousGrid::new(
                    1, 10, 1000, None, false,
                ))),
            ],
            0.01,
            false,
        );

        let mut rng = MonteCarloRng::new(0, 0);

        let mut sample = Sample::new();
        for _ in 1..20 {
            // sample 10_000 times per iteration
            for _ in 0..10_000 {
                grid.sample(&mut rng, &mut sample);

                if let Sample::Discrete(_weight, i, cont_sample) = &sample {
                    if let Sample::Continuous(_cont_weight, xs) =
                        cont_sample.as_ref().unwrap().as_ref()
                    {
                        grid.add_training_sample(&sample, fs[*i](xs[0])).unwrap();
                    }
                }
            }

            grid.update(1.5);
        }

        assert_eq!(grid.accumulator.avg, 0.9713543844460519);
        assert_eq!(grid.accumulator.err, 0.0009026050146732183)
    }
}
