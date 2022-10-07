#![feature(generic_const_exprs)]

use std::ops::{Div, Mul};

use nalgebra::{SMatrix, SVector};
use rand::{prelude::Distribution, thread_rng, Rng};

pub trait SampleSpace: Distribution<Self::Point> {
    type Point: Copy + Clone;
}

impl<'a, S: SampleSpace> SampleSpace for &'a S {
    type Point = S::Point;
}

#[derive(Default)]
pub struct Standard<T: Copy + Clone>(std::marker::PhantomData<T>)
where
    rand::distributions::Standard: Distribution<T>;

impl<T: Copy + Clone> Distribution<T> for Standard<T>
where
    rand::distributions::Standard: Distribution<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        rand::distributions::Standard.sample(rng)
    }
}

impl<T: Copy + Clone> SampleSpace for Standard<T>
where
    rand::distributions::Standard: Distribution<T>,
{
    type Point = T;
}

#[derive(Clone)]
pub struct Exponential<Stat, Space: SampleSpace, const N: usize>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
{
    multipliers: SVector<f64, N>,
    stat: Stat,
    space: Space,
}

impl<Stat, Space: SampleSpace, const N: usize> Exponential<Stat, Space, N>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
{
    pub fn new(multipliers: SVector<f64, N>, stat: Stat, space: Space) -> Self {
        Self {
            multipliers,
            stat,
            space,
        }
    }

    pub fn density(&self, x: Space::Point) -> f64 {
        (self.multipliers.dot(&(self.stat)(x))).exp()
    }

    pub fn integral<R: Rng, O: std::iter::Sum>(
        &self,
        f: impl Fn(Space::Point) -> O,
        iterations: usize,
        rng: &mut R,
    ) -> O
    where
        O: Mul<f64, Output = O> + Div<f64, Output = O>,
    {
        (&self.space)
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

struct Problem<Stat, Space: SampleSpace, const N: usize>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
{
    stat: Stat,
    space: Space,
    g_l: SVector<f64, N>,
    g_u: SVector<f64, N>,
}

impl<Stat, Space: SampleSpace, const N: usize> Problem<Stat, Space, N>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
    [(); N + 1]:,
{
    fn stat_p1(&self) -> impl Fn(Space::Point) -> SVector<f64, { N + 1 }> + '_ {
        |x| {
            let stat = (self.stat)(x);
            let mut new: SVector<f64, { N + 1 }> = SVector::zeros();
            new.copy_from_slice(stat.as_slice());
            new[N] = -1.;
            new
        }
    }
}

impl<Stat, Space: SampleSpace, const N: usize> ipopt::BasicProblem for Problem<Stat, Space, N>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
    [(); N + 1]:,
{
    fn num_variables(&self) -> usize {
        N + 1
    }

    fn bounds(&self, _x_l: &mut [ipopt::Number], _x_u: &mut [ipopt::Number]) -> bool {
        false
    }

    fn initial_point(&self, _x: &mut [ipopt::Number]) -> bool {
        false
    }

    fn objective(&self, x: &[ipopt::Number], obj: &mut ipopt::Number) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.space);

        *obj = exp.entropy(10_000, &mut thread_rng());
        true
    }

    fn objective_grad(&self, x: &[ipopt::Number], grad_f: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.space);

        grad_f.copy_from_slice(exp.entropy_gradient(10_000, &mut thread_rng()).as_slice());
        true
    }
}

impl<Stat, Space: SampleSpace, const N: usize> ipopt::ConstrainedProblem for Problem<Stat, Space, N>
where
    Stat: Fn(Space::Point) -> SVector<f64, N>,
    [(); N + 1]:,
{
    fn num_constraints(&self) -> usize {
        N + 1
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        (N + 1) * (N + 1)
    }

    fn constraint(&self, x: &[ipopt::Number], g: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.space);

        g.copy_from_slice(
            exp.integral(self.stat_p1(), 10_000, &mut thread_rng())
                .as_slice(),
        );
        true
    }

    fn constraint_bounds(&self, g_l: &mut [ipopt::Number], g_u: &mut [ipopt::Number]) -> bool {
        g_l.copy_from_slice(self.g_l.as_slice());
        g_l[N] = 1.;
        g_u.copy_from_slice(self.g_u.as_slice());
        g_u[N] = 1.;
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        for idx in 0usize..(N + 1) * (N + 1) {
            (rows[idx], cols[idx]) = ((idx % N) as i32, (idx / N) as i32);
        }
        true
    }

    fn constraint_jacobian_values(&self, x: &[ipopt::Number], vals: &mut [ipopt::Number]) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.space);

        vals.copy_from_slice(
            exp.integral(
                |x| {
                    let stat = (exp.stat)(x);
                    stat * stat.transpose()
                },
                10_000,
                &mut thread_rng(),
            )
            .as_slice(),
        );
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        (N + 1) * (N + 1)
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
        for idx in 0usize..(N + 1) * (N + 1) {
            (rows[idx], cols[idx]) = ((idx % N) as i32, (idx / N) as i32);
        }
        true
    }

    fn hessian_values(
        &self,
        x: &[ipopt::Number],
        obj_factor: ipopt::Number,
        lambda: &[ipopt::Number],
        vals: &mut [ipopt::Number],
    ) -> bool {
        let mut multipliers: SVector<f64, { N + 1 }> = SVector::zeros();
        multipliers.copy_from_slice(x);
        let exp = Exponential::new(multipliers, self.stat_p1(), &self.space);

        let mut lambdas: SVector<f64, { N + 1 }> = SVector::zeros();
        lambdas.copy_from_slice(lambda);

        vals.copy_from_slice(
            exp.integral(
                |x| {
                    let stat = (exp.stat)(x);
                    ((lambdas - obj_factor * multipliers).dot(&stat) - 2. * obj_factor)
                        * stat
                        * stat.transpose()
                },
                10_000,
                &mut thread_rng(),
            )
            .as_slice(),
        );
        true
    }
}
