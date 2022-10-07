use minxent::{Exponential, Standard};
use nalgebra::{SMatrix, SVector};
use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    let e = Exponential::<_, Standard<f64>, 2>::solve(
        |x| SVector::from([x, x * x]),
        Default::default(),
        SMatrix::<f64, 2, 2>::new(-1., 1., 0., f64::INFINITY),
    )?;

    println!("{}", e.multipliers);

    Ok(())
}
