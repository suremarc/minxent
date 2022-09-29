use std::ops::{Div, Mul};

use nalgebra::SVector;
use rand::{prelude::Distribution, rngs::ThreadRng, thread_rng, Rng};

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
}

struct Problem<'a, Point, Stat, Prior: Distribution<Point>, const N: usize>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    exp: &'a Exponential<Point, Stat, Prior, N>,
}

impl<'a, Point, Stat, Prior: Distribution<Point>, const N: usize> ipopt::BasicProblem
    for Problem<'a, Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    fn num_variables(&self) -> usize {
        N + 1
    }

    fn bounds(&self, x_l: &mut [ipopt::Number], x_u: &mut [ipopt::Number]) -> bool {
        false
    }

    fn initial_point(&self, x: &mut [ipopt::Number]) -> bool {
        false
    }

    fn objective(&self, x: &[ipopt::Number], obj: &mut ipopt::Number) -> bool {
        *obj = self.exp.entropy(10_000, &mut thread_rng());
        true
    }

    fn objective_grad(&self, x: &[ipopt::Number], grad_f: &mut [ipopt::Number]) -> bool {
        grad_f.copy_from_slice(
            self.exp
                .integral(
                    |x| (1. + self.exp.multipliers.dot(&(self.exp.stat)(x))) * (self.exp.stat)(x),
                    10_000,
                    &mut thread_rng(),
                )
                .as_slice(),
        );

        true
    }
}

impl<'a, Point, Stat, Prior: Distribution<Point>, const N: usize> ipopt::ConstrainedProblem
    for Problem<'a, Point, Stat, Prior, N>
where
    Point: Copy + Clone,
    Stat: Fn(Point) -> SVector<f64, N>,
    Prior: Clone,
{
    fn num_constraints(&self) -> usize {
        todo!()
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        todo!()
    }

    fn constraint(&self, x: &[ipopt::Number], g: &mut [ipopt::Number]) -> bool {
        todo!()
    }

    fn constraint_bounds(&self, g_l: &mut [ipopt::Number], g_u: &mut [ipopt::Number]) -> bool {
        todo!()
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        todo!()
    }

    fn constraint_jacobian_values(&self, x: &[ipopt::Number], vals: &mut [ipopt::Number]) -> bool {
        todo!()
    }

    fn num_hessian_non_zeros(&self) -> usize {
        todo!()
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
        todo!()
    }

    fn hessian_values(
        &self,
        x: &[ipopt::Number],
        obj_factor: ipopt::Number,
        lambda: &[ipopt::Number],
        vals: &mut [ipopt::Number],
    ) -> bool {
        todo!()
    }
}
