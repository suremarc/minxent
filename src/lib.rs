use std::ops::{Div, Mul};

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
    Standard: Distribution<Scalar>,
    Stat: Clone + Fn(Point) -> SVector<Scalar, N>,
    Prior: Clone,
{
    multipliers: SVector<Scalar, N>,
    stat: Stat,
    prior: Prior,
    denom: Scalar,
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
    Standard: Distribution<Scalar>,
    Stat: Clone + Fn(Point) -> SVector<Scalar, N>,
    Prior: Clone,
{
    pub fn new(multipliers: SVector<Scalar, N>, stat: Stat, prior: Prior) -> Self {
        let mut new = Self {
            multipliers,
            stat,
            prior,
            denom: Scalar::zero(),
            _marker: std::marker::PhantomData,
        };
        let mut rng = rand::thread_rng();
        new.denom = new.integral_unnormalized(|_| Scalar::one(), 1_000_000, &mut rng);
        new
    }
    pub fn density_unnormalized(&self, x: Point) -> Scalar {
        (self.multipliers.dot(&(self.stat)(x))).exp()
    }

    pub fn integral_unnormalized<R: Rng, O: std::iter::Sum>(
        &self,
        f: impl Fn(Point) -> O,
        iterations: usize,
        rng: &mut R,
    ) -> O
    where
        O: Mul<Scalar, Output = O>,
    {
        self.prior
            .clone()
            .map(|x| f(x) * self.density_unnormalized(x))
            .sample_iter(rng)
            .take(iterations)
            .sum()
    }

    pub fn density(&self, x: Point) -> Scalar {
        self.density_unnormalized(x) / self.denom
    }

    pub fn integral<R: Rng, O: std::iter::Sum>(
        &self,
        f: impl Fn(Point) -> O,
        iterations: usize,
        rng: &mut R,
    ) -> O
    where
        O: Mul<Scalar, Output = O> + Div<Scalar, Output = O>,
    {
        self.prior
            .clone()
            .map(|x| f(x) * self.density_unnormalized(x))
            .sample_iter(rng)
            .take(iterations)
            .sum::<O>()
            / self.denom
    }
}
