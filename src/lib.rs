#![feature(generic_const_exprs)]

use std::ops::{Div, Mul};

use nalgebra::{SMatrix, SVector};
use rand::{prelude::Distribution, thread_rng, Rng};

#[derive(Clone)]
pub struct Exponential<Point, Stat, Prior: Distribution<Point>, const N: usize>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    multipliers: SVector<f64, N>,
    stat: Stat,
    prior: Prior,
    _marker: std::marker::PhantomData<Point>,
}

impl<Point, Stat, Prior: Distribution<Point>, const N: usize> Exponential<Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    pub fn new(multipliers: SVector<f64, N>, stat: Stat, prior: Prior) -> Self {
        Self {
            multipliers,
            stat,
            prior,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn density(&self, x: Point) -> f64 {
        (self.multipliers.dot(&(self.stat)(x))).exp()
    }

    pub fn integral<R: Rng, O: std::iter::Sum>(
        &self,
        f: impl Fn(Point) -> O,
        iterations: usize,
        rng: &mut R,
    ) -> O
    where
        O: Mul<f64, Output = O> + Div<f64, Output = O>,
    {
        self.prior
            .clone()
            .map(|x| f(x) * self.density(x))
            .sample_iter(rng)
            .take(iterations)
            .sum::<O>()
            / iterations as f64
    }

    pub fn entropy<R: Rng>(&self, iterations: usize, rng: &mut R) -> f64 {
        self.integral(|x| self.multipliers.dot(&(self.stat)(x)), iterations, rng)
    }

    pub fn entropy_gradient<R: Rng>(&self, iterations: usize, rng: &mut R) -> SVector<f64, N> {
        self.integral(
            |x| (1. + self.multipliers.dot(&(self.stat)(x))) * (self.stat)(x),
            iterations,
            rng,
        )
    }

    pub fn entropy_hessian<R: Rng>(&self, iterations: usize, rng: &mut R) -> SMatrix<f64, N, N> {
        self.integral(
            |x| {
                let stat = (self.stat)(x);
                (2. + self.multipliers.dot(&stat)) * stat * stat.transpose()
            },
            iterations,
            rng,
        )
    }
}

struct Problem<Point, Stat, Prior: Distribution<Point>, const N: usize>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    stat: Stat,
    prior: Prior,
    _t_l: SVector<f64, N>,
    _t_u: SVector<f64, N>,
    _marker: std::marker::PhantomData<Point>,
}

impl<Point, Stat, Prior: Distribution<Point>, const N: usize> Problem<Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
    [(); N + 1]:,
{
    fn stat_p1(&self) -> impl Fn(Point) -> SVector<f64, { N + 1 }> + '_ {
        |x| {
            let stat = (self.stat)(x);
            let mut new: SVector<f64, { N + 1 }> = SVector::zeros();
            new.copy_from_slice(stat.as_slice());
            new[N] = -1.;
            new
        }
    }
}

impl<Point, Stat, Prior: Distribution<Point>, const N: usize> ipopt::BasicProblem
    for Problem<Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
    [(); N + 1]:,
{
    fn num_variables(&self) -> usize {
        N + 1
    }

    fn bounds(&self, x_l: &mut [ipopt::Number], x_u: &mut [ipopt::Number]) -> bool {
        x_l[0..N].fill(-f64::INFINITY);
        x_u[0..N].fill(f64::INFINITY);
        x_l[N] = 1.;
        x_u[N] = 1.;

        true
    }

    fn initial_point(&self, _x: &mut [ipopt::Number]) -> bool {
        false
    }

    fn objective(&self, x: &[ipopt::Number], obj: &mut ipopt::Number) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), self.prior.clone());

        *obj = exp.entropy(10_000, &mut thread_rng());
        true
    }

    fn objective_grad(&self, x: &[ipopt::Number], grad_f: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), self.prior.clone());

        grad_f.copy_from_slice(exp.entropy_gradient(10_000, &mut thread_rng()).as_slice());
        true
    }
}
