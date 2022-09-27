use std::pin::Pin;

use autocxx::prelude::*;
use autocxx::subclass::*;

include_cpp! {
    #include "IpTypes.hpp"
    #include "IpTNLP.hpp"
    generate!("Ipopt::Number")
    generate!("Ipopt::Index")
    generate!("Ipopt::TNLP")
    subclass!("Ipopt::TNLP", MyNLP)
    safety!(unsafe)
}

use ffi::*;

#[is_subclass(superclass("TNLP"))]
pub struct MyNLP {}

impl Ipopt::TNLP_methods for MyNLP {
    fn get_nlp_info(
        &mut self,
        n: Pin<&mut autocxx::c_int>,
        m: Pin<&mut autocxx::c_int>,
        nnz_jac_g: Pin<&mut autocxx::c_int>,
        nnz_h_lag: Pin<&mut autocxx::c_int>,
        index_style: Pin<&mut Ipopt::TNLP_IndexStyleEnum>,
    ) -> bool {
        todo!()
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
        todo!()
    }

    unsafe fn get_starting_point(
        &mut self,
        n: autocxx::c_int,
        init_x: bool,
        x: *mut f64,
        init_z: bool,
        z_l: *mut f64,
        z_u: *mut f64,
        m: autocxx::c_int,
        init_lambda: bool,
        lambda: *mut f64,
    ) -> bool {
        todo!()
    }

    unsafe fn eval_f(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        obj_value: Pin<&mut f64>,
    ) -> bool {
        todo!()
    }

    unsafe fn eval_grad_f(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        grad_f: *mut f64,
    ) -> bool {
        todo!()
    }

    unsafe fn eval_g(
        &mut self,
        n: autocxx::c_int,
        x: *const f64,
        new_x: bool,
        m: autocxx::c_int,
        g: *mut f64,
    ) -> bool {
        todo!()
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
        todo!()
    }

    unsafe fn finalize_solution(
        &mut self,
        status: Ipopt::SolverReturn,
        n: autocxx::c_int,
        x: *const f64,
        z_l: *const f64,
        z_u: *const f64,
        m: autocxx::c_int,
        g: *const f64,
        lambda: *const f64,
        obj_value: f64,
        ip_data: *const Ipopt::IpoptData,
        ip_cq: *mut Ipopt::IpoptCalculatedQuantities,
    ) {
        todo!()
    }
}
