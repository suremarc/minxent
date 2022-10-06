#![feature(generic_const_exprs)]

use std::ops::{Div, Mul};

use nalgebra::{SMatrix, SVector};
use rand::{prelude::Distribution, thread_rng, Rng};

#[derive(Clone)]
pub struct Exponential<Point, Stat, Prior: Distribution<Point>, const N: usize>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
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
        (&self.prior)
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
{
    stat: Stat,
    prior: Prior,
    g_l: SVector<f64, N>,
    g_u: SVector<f64, N>,
    _marker: std::marker::PhantomData<Point>,
}

impl<Point, Stat, Prior: Distribution<Point>, const N: usize> Problem<Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
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
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.prior);

        *obj = exp.entropy(10_000, &mut thread_rng());
        true
    }

    fn objective_grad(&self, x: &[ipopt::Number], grad_f: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.prior);

        grad_f.copy_from_slice(exp.entropy_gradient(10_000, &mut thread_rng()).as_slice());
        true
    }
}

impl<Point, Stat, Prior: Distribution<Point>, const N: usize> ipopt::ConstrainedProblem
    for Problem<Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    [(); N + 1]:,
{
    fn num_constraints(&self) -> usize {
        N
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        todo!()
    }

    fn constraint(&self, x: &[ipopt::Number], g: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.prior);

        g.copy_from_slice(
            exp.integral(&self.stat, 10_000, &mut thread_rng())
                .as_slice(),
        );
        true
    }

    fn constraint_bounds(&self, g_l: &mut [ipopt::Number], g_u: &mut [ipopt::Number]) -> bool {
        g_l.copy_from_slice(self.g_l.as_slice());
        g_u.copy_from_slice(self.g_u.as_slice());
        true
    }

    fn constraint_jacobian_indices(
        &self,
        _rows: &mut [ipopt::Index],
        _cols: &mut [ipopt::Index],
    ) -> bool {
        todo!()
    }

    fn constraint_jacobian_values(
        &self,
        _x: &[ipopt::Number],
        _vals: &mut [ipopt::Number],
    ) -> bool {
        todo!()
    }

    fn num_hessian_non_zeros(&self) -> usize {
        (N + 1) * (N + 1)
    }

    fn hessian_indices(&self, _rows: &mut [ipopt::Index], _cols: &mut [ipopt::Index]) -> bool {
        todo!()
    }

    fn hessian_values(
        &self,
        _x: &[ipopt::Number],
        _obj_factor: ipopt::Number,
        _lambda: &[ipopt::Number],
        _vals: &mut [ipopt::Number],
    ) -> bool {
        todo!()
    }
}
