use std::pin::Pin;

use autocxx::prelude::*;
use autocxx::subclass::*;

include_cpp! {
    #include "IpIpoptApplication.hpp"
    #include "IpTypes.hpp"
    #include "IpTNLP.hpp"
    generate!("Ipopt::Number")
    generate!("Ipopt::Index")
    generate!("Ipopt::TNLP")
    subclass!("Ipopt::TNLP", HS071_NLP)
    safety!(unsafe)
}

use ffi::*;

#[is_subclass(superclass("TNLP"))]
#[derive(Default)]
pub struct HS071_NLP {}

impl Ipopt::TNLP_methods for HS071_NLP {
    fn get_nlp_info(
        &mut self,
        n: Pin<&mut autocxx::c_int>,
        m: Pin<&mut autocxx::c_int>,
        nnz_jac_g: Pin<&mut autocxx::c_int>,
        nnz_h_lag: Pin<&mut autocxx::c_int>,
        index_style: Pin<&mut Ipopt::TNLP_IndexStyleEnum>,
    ) -> bool {
        // The problem described in HS071_NLP.hpp has 4 variables, x[0] through x[3]
        *n.get_mut() = 4.into();

        // one equality constraint and one inequality constraint
        *m.get_mut() = 2.into();

        // in this example the jacobian is dense and contains 8 nonzeros
        *nnz_jac_g.get_mut() = 8.into();

        // the Hessian is also dense and has 16 total nonzeros, but we
        // only need the lower left corner (since it is symmetric)
        *nnz_h_lag.get_mut() = 10.into();

        // use the C style indexing (0-based)
        *index_style.get_mut() = Ipopt::TNLP_IndexStyleEnum::C_STYLE;

        true
    }

    unsafe fn get_bounds_info(
        &mut self,
        n: autocxx::c_int,
        x_l: *mut f64,
        x_u: *mut f64,
        m: autocxx::c_int,
        g_l: *mut f64,
        g_u: *mut f64,
    ) -> bool {
        // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
        // If desired, we could assert to make sure they are what we think they are.
        assert!(n == 4.into());
        assert!(m == 2.into());

        let x_l = std::slice::from_raw_parts_mut(x_l, 4);
        // the variables have lower bounds of 1
        for i in 0..4 {
            x_l[i] = 1.0;
        }

        let x_u = std::slice::from_raw_parts_mut(x_u, 4);
        // the variables have upper bounds of 5
        for i in 0..4 {
            x_u[i] = 5.0;
        }

        let g_l = std::slice::from_raw_parts_mut(g_l, 2);
        let g_u = std::slice::from_raw_parts_mut(g_u, 2);
        // the first constraint g1 has a lower bound of 25
        g_l[0] = 25.0;
        // the first constraint g1 has NO upper bound, here we set it to 2e19.
        // Ipopt interprets any number greater than nlp_upper_bound_inf as
        // infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
        // is 1e19 and can be changed through ipopt options.
        g_u[0] = 2e19;

        // the second constraint g2 is an equality constraint, so we set the
        // upper and lower bound to the same value
        g_l[1] = 40.0;
        g_u[1] = 40.0;

        true
    }

    unsafe fn get_starting_point(
        &mut self,
        _n: autocxx::c_int,
        init_x: bool,
        x: *mut f64,
        init_z: bool,
        _z_l: *mut f64,
        _z_u: *mut f64,
        _m: autocxx::c_int,
        init_lambda: bool,
        _lambda: *mut f64,
    ) -> bool {
        // Here, we assume we only have starting values for x, if you code
        // your own NLP, you can provide starting values for the dual variables
        // if you wish
        assert!(init_x);
        assert!(!init_z);
        assert!(!init_lambda);

        let x = std::slice::from_raw_parts_mut(x, 4);

        // initialize to the given starting point
        x[0] = 1.0;
        x[1] = 5.0;
        x[2] = 5.0;
        x[3] = 1.0;

        true
    }

    unsafe fn eval_f(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        _new_x: bool,
        obj_value: Pin<&mut f64>,
    ) -> bool {
        assert!(n == 4.into());

        let x = std::slice::from_raw_parts(x, 4);

        *obj_value.get_mut() = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];

        true
    }

    unsafe fn eval_grad_f(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        grad_f: *mut f64,
    ) -> bool {
        assert!(n == 4.into());

        let x = std::slice::from_raw_parts(x, 4);
        let grad_f = std::slice::from_raw_parts_mut(grad_f, 4);

        grad_f[0] = x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]);
        grad_f[1] = x[0] * x[3];
        grad_f[2] = x[0] * x[3] + 1.0;
        grad_f[3] = x[0] * (x[0] + x[1] + x[2]);

        true
    }

    unsafe fn eval_g(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        m: autocxx::c_int,
        g: *mut f64,
    ) -> bool {
        assert!(n == 4.into());
        assert!(m == 2.into());

        let x = std::slice::from_raw_parts(x, 4);
        let g = std::slice::from_raw_parts_mut(g, 2);

        g[0] = x[0] * x[1] * x[2] * x[3];
        g[1] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];

        return true;
    }

    unsafe fn eval_jac_g(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        m: autocxx::c_int,
        nele_jac: autocxx::c_int,
        i_row: *mut autocxx::c_int,
        j_col: *mut autocxx::c_int,
        values: *mut f64,
    ) -> bool {
        assert!(n == 4.into());
        assert!(m == 2.into());

        let x = std::slice::from_raw_parts(x, 4);
        let i_row = std::slice::from_raw_parts_mut(i_row, 8);
        let j_col = std::slice::from_raw_parts_mut(j_col, 8);

        if values.is_null() {
            // return the structure of the Jacobian

            // this particular Jacobian is dense
            i_row[0] = 0.into();
            j_col[0] = 0.into();
            i_row[1] = 0.into();
            j_col[1] = 1.into();
            i_row[2] = 0.into();
            j_col[2] = 2.into();
            i_row[3] = 0.into();
            j_col[3] = 3.into();
            i_row[4] = 1.into();
            j_col[4] = 0.into();
            i_row[5] = 1.into();
            j_col[5] = 1.into();
            i_row[6] = 1.into();
            j_col[6] = 2.into();
            i_row[7] = 1.into();
            j_col[7] = 3.into();
        } else {
            let values = std::slice::from_raw_parts_mut(values, 8);
            // return the values of the Jacobian of the constraints

            values[0] = x[1] * x[2] * x[3]; // 0,0
            values[1] = x[0] * x[2] * x[3]; // 0,1
            values[2] = x[0] * x[1] * x[3]; // 0,2
            values[3] = x[0] * x[1] * x[2]; // 0,3

            values[4] = 2.0 * x[0]; // 1,0
            values[5] = 2.0 * x[1]; // 1,1
            values[6] = 2.0 * x[2]; // 1,2
            values[7] = 2.0 * x[3]; // 1,3
        }

        true
    }

    unsafe fn eval_h(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        obj_factor: f64,
        m: autocxx::c_int,
        lambda: *const f64,
        new_lambda: bool,
        nele_hess: autocxx::c_int,
        i_row: *mut autocxx::c_int,
        j_col: *mut autocxx::c_int,
        values: *mut f64,
    ) -> bool {
        assert!(n == 4.into());
        assert!(m == 2.into());

        let x = std::slice::from_raw_parts(x, 4);
        let i_row = std::slice::from_raw_parts_mut(i_row, 8);
        let j_col = std::slice::from_raw_parts_mut(j_col, 8);

        if values.is_null() {
            // return the structure. This is a symmetric matrix, fill the lower left
            // triangle only.

            // the hessian for this problem is actually dense
            let mut idx = 0;
            for row in 0..4 {
                for col in 0..row + 1 {
                    i_row[idx as usize] = row.into();
                    j_col[idx as usize] = col.into();
                    idx += 1;
                }
            }

            assert!(idx == nele_hess.0)
        } else {
            let lambda = std::slice::from_raw_parts(lambda, 2);
            let values = std::slice::from_raw_parts_mut(values, 10);

            // return the values. This is a symmetric matrix, fill the lower left
            // triangle only

            // fill the objective portion
            values[0] = obj_factor * (2.0 * x[3]); // 0,0

            values[1] = obj_factor * (x[3]); // 1,0
            values[2] = 0.; // 1,1

            values[3] = obj_factor * (x[3]); // 2,0
            values[4] = 0.; // 2,1
            values[5] = 0.; // 2,2

            values[6] = obj_factor * (2.0 * x[0] + x[1] + x[2]); // 3,0
            values[7] = obj_factor * (x[0]); // 3,1
            values[8] = obj_factor * (x[0]); // 3,2
            values[9] = 0.; // 3,3

            // add the portion for the first constraint
            values[1] += lambda[0] * (x[2] * x[3]); // 1,0

            values[3] += lambda[0] * (x[1] * x[3]); // 2,0
            values[4] += lambda[0] * (x[0] * x[3]); // 2,1

            values[6] += lambda[0] * (x[1] * x[2]); // 3,0
            values[7] += lambda[0] * (x[0] * x[2]); // 3,1
            values[8] += lambda[0] * (x[0] * x[1]); // 3,2

            // add the portion for the second constraint
            values[0] += lambda[1] * 2.0; // 0,0

            values[2] += lambda[1] * 2.0; // 1,1

            values[5] += lambda[1] * 2.0; // 2,2

            values[9] += lambda[1] * 2.0; // 3,3
        }

        true
    }

    unsafe fn finalize_solution(
        &mut self,
        _status: Ipopt::SolverReturn,
        n: autocxx::c_int,
        x: *const f64,
        z_l: *const f64,
        z_u: *const f64,
        _m: autocxx::c_int,
        g: *const f64,
        _lambda: *const f64,
        obj_value: f64,
        _ip_data: *const Ipopt::IpoptData,
        _ip_cq: *mut Ipopt::IpoptCalculatedQuantities,
    ) {
        assert!(n == 4.into());

        let x = std::slice::from_raw_parts(x, 4);
        let z_l = std::slice::from_raw_parts(z_l, 4);
        let z_u = std::slice::from_raw_parts(z_u, 4);
        let g = std::slice::from_raw_parts(g, 2);
        // here is where we would store the solution to variables, or write to a file, etc
        // so we could use the solution.

        // For this example, we write the solution to the console
        println!("\n\nSolution of the primal variables, x");
        for (i, x_i) in x.iter().copied().enumerate() {
            println!("x[{i}] = {}", x_i);
        }

        println!("\n\nSolution of the bound multipliers, z_l and z_u");
        for (i, z_li) in z_l.iter().copied().enumerate() {
            println!("z_l[{i}] = {}", z_li);
        }
        for (i, z_ui) in z_u.iter().copied().enumerate() {
            println!("z_u[{i}] = {}", z_ui);
        }

        println!("\n\nObjective value");
        println!("f(x*) = {obj_value}");

        println!("\nFinal value of the constraints:");
        for (i, g_i) in g.iter().copied().enumerate() {
            println!("g({i}) = {}", g_i);
        }
    }
}
