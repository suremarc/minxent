use std::error::Error;

#[derive(Default)]
pub struct HS071 {}

impl ipopt::BasicProblem for HS071 {
    fn num_variables(&self) -> usize {
        4
    }

    fn bounds(&self, x_l: &mut [ipopt::Number], x_u: &mut [ipopt::Number]) -> bool {
        x_l.fill(1.0);
        x_u.fill(1.0);
        true
    }

    fn initial_point(&self, x: &mut [ipopt::Number]) -> bool {
        x.copy_from_slice(&[1.0, 5.0, 5.0, 1.0]);
        true
    }

    fn objective(&self, x: &[ipopt::Number], obj: &mut ipopt::Number) -> bool {
        *obj = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
        true
    }

    fn objective_grad(&self, x: &[ipopt::Number], grad_f: &mut [ipopt::Number]) -> bool {
        grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        grad_f[1] = x[0] * x[3];
        grad_f[2] = x[0] * x[3] + 1.0;
        grad_f[3] = x[0] * (x[0] + x[1] + x[2]);
        true
    }
}

impl ipopt::ConstrainedProblem for HS071 {
    fn num_constraints(&self) -> usize {
        2
    }

    fn num_constraint_jacobian_non_zeros(&self) -> usize {
        8
    }

    fn constraint(&self, x: &[ipopt::Number], g: &mut [ipopt::Number]) -> bool {
        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        true
    }

    fn constraint_bounds(&self, g_l: &mut [ipopt::Number], g_u: &mut [ipopt::Number]) -> bool {
        g_l.copy_from_slice(&[25.0, 40.0]);
        g_u.copy_from_slice(&[2e19, 40.0]);
        true
    }

    fn constraint_jacobian_indices(
        &self,
        rows: &mut [ipopt::Index],
        cols: &mut [ipopt::Index],
    ) -> bool {
        rows.copy_from_slice(&[0, 0, 0, 0, 1, 1, 1, 1]);
        cols.copy_from_slice(&[0, 1, 2, 3, 0, 1, 2, 3]);
        true
    }

    fn constraint_jacobian_values(&self, x: &[ipopt::Number], vals: &mut [ipopt::Number]) -> bool {
        vals.copy_from_slice(&[
            x[1] * x[2] * x[3],
            x[0] * x[2] * x[3],
            x[0] * x[1] * x[3],
            x[0] * x[1] * x[2],
            2.0 * x[0],
            2.0 * x[1],
            2.0 * x[2],
            2.0 * x[3],
        ]);
        true
    }

    fn num_hessian_non_zeros(&self) -> usize {
        10
    }

    fn hessian_indices(&self, rows: &mut [ipopt::Index], cols: &mut [ipopt::Index]) -> bool {
        let mut idx = 0;
        for i in 0..4 {
            for j in 0..i + 1 {
                rows[idx] = i;
                cols[idx] = j;
                idx += 1;
            }
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
        vals.copy_from_slice(&[
            obj_factor * (2.0 * x[3]),
            obj_factor * x[3],
            0.,
            obj_factor * x[3],
            0.,
            0.,
            obj_factor * (2.0 * x[0] + x[1] + x[2]),
            obj_factor * x[0],
            obj_factor * x[0],
            0.,
        ]);

        vals[1] += lambda[0] * x[2] * x[3];
        vals[3] += lambda[0] * x[1] * x[3];
        vals[4] += lambda[0] * x[0] * x[3];
        vals[6] += lambda[0] * x[1] * x[2];
        vals[7] += lambda[0] * x[0] * x[2];
        vals[8] += lambda[0] * x[0] * x[1];

        vals[0] += lambda[1] * 2.0;
        vals[2] += lambda[1] * 2.0;
        vals[5] += lambda[1] * 2.0;
        vals[9] += lambda[1] * 2.0;

        true
    }
}

pub fn main() -> Result<(), Box<dyn Error>> {
    let mut problem = ipopt::Ipopt::new_unconstrained(HS071 {})?;
    problem.set_option("tol", 3.82e-6);
    problem.set_option("print_level", 5);
    problem.set_option("mu_strategy", "adaptive");
    problem.set_option("output_file", "ipopt.out");

    let solve_result = problem.solve();

    assert_eq!(solve_result.status, ipopt::SolveStatus::SolveSucceeded);

    Ok(())
}
