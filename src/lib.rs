use std::ops::Mul;

use nalgebra::{ClosedAdd, ClosedMul, SVector};
use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution, Rng};

#[derive(Clone)]
pub struct Exponential<
    Scalar: Float + nalgebra::Scalar + ClosedAdd + ClosedMul + std::iter::Sum,
    Point: Copy + Clone,
    Stat,
    Prior: Distribution<Point>,
    const N: usize,
> where
    Stat: Clone + Fn(Point) -> SVector<Scalar, N>,
    Prior: Clone,
{
    multipliers: SVector<Scalar, N>,
    stat: Stat,
    prior: Prior,
    _marker: std::marker::PhantomData<Point>,
}

impl<
        Scalar: Float + nalgebra::Scalar + ClosedAdd + ClosedMul + std::iter::Sum,
        Point: Copy + Clone,
        Stat,
        Prior: Distribution<Point>,
        const N: usize,
    > Exponential<Scalar, Point, Stat, Prior, N>
where
    Stat: Clone + Fn(Point) -> SVector<Scalar, N>,
    Prior: Clone,
{
    pub fn new(multipliers: SVector<Scalar, N>, stat: Stat, prior: Prior) -> Self {
        Self {
            multipliers,
            stat,
            prior,
            _marker: std::marker::PhantomData,
        }
    }
    pub fn density_unnormalized(&self, x: Point) -> Scalar {
        (self.multipliers.dot(&(self.stat)(x))).exp()
    }

    pub fn integral_unnormalized<R: Rng, O: std::iter::Sum>(
        &self,
        f: impl Fn(Point) -> O,
        rng: &mut R,
    ) -> O
    where
        Standard: Distribution<Scalar>,
        O: Mul<Scalar, Output = O>,
    {
        self.prior
            .clone()
            .map(|x| f(x) * self.density_unnormalized(x))
            .sample_iter(rng)
            .take(100000)
            .sum()
    }
}
