//! CUDA f32 fast path. Outputs are separate allocations, so Var::set retains
//! the existing variable storage and alias semantics.
use super::ParamsAdamW;
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use candle::cuda_backend::WrapErr;
use candle::{op::BackpropOp, CudaStorage, DType, Result, Storage, Tensor};
use std::sync::OnceLock;

const SOURCE: &str = r#"
extern "C" __global__ void adamw_f32(
    unsigned int n, const float* theta, const float* m, const float* v,
    const float* g, float* next_m, float* next_v, float* next_theta,
    float beta1, float one_minus_beta1, float beta2, float one_minus_beta2,
    float scale_m, float scale_v, float decay, float eps, float lr) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Explicit roundings preserve the boundaries of the old tensor operations.
    float nm = __fadd_rn(__fmul_rn(m[i], beta1), __fmul_rn(g[i], one_minus_beta1));
    float nv = __fadd_rn(__fmul_rn(v[i], beta2),
                        __fmul_rn(__fmul_rn(g[i], g[i]), one_minus_beta2));
    float mh = __fmul_rn(nm, scale_m);
    float vh = __fmul_rn(nv, scale_v);
    float adjusted = __fdiv_rn(mh, __fadd_rn(sqrtf(vh), eps));
    next_m[i] = nm;
    next_v[i] = nv;
    next_theta[i] = __fsub_rn(__fmul_rn(theta[i], decay), __fmul_rn(adjusted, lr));
}
"#;

pub(super) fn update(
    theta: &Tensor,
    m: &Tensor,
    v: &Tensor,
    g: &Tensor,
    params: &ParamsAdamW,
    scale_m: f64,
    scale_v: f64,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let inputs = [theta, m, v, g];
    let n = theta.elem_count();
    if n == 0
        || n > u32::MAX as usize
        || !theta.device().is_cuda()
        || inputs.iter().any(|t| {
            t.dtype() != DType::F32
                || !t.is_contiguous()
                || t.shape() != theta.shape()
                || !t.device().same_device(theta.device())
        })
    {
        return Ok(None);
    }
    let (ts, tl) = theta.storage_and_layout();
    let (ms, ml) = m.storage_and_layout();
    let (vs, vl) = v.storage_and_layout();
    let (gs, gl) = g.storage_and_layout();
    let (Storage::Cuda(ts), Storage::Cuda(ms), Storage::Cuda(vs), Storage::Cuda(gs)) =
        (&*ts, &*ms, &*vs, &*gs)
    else {
        return Ok(None);
    };
    let dev = ts.device();
    static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
    let ptx = PTX.get_or_init(|| {
        compile_ptx_with_opts(
            SOURCE,
            CompileOptions {
                fmad: Some(false),
                use_fast_math: Some(false),
                ..Default::default()
            },
        )
        .map(|p| p.to_src())
        .map_err(|e| e.to_string())
    });
    let ptx = ptx
        .as_ref()
        .map_err(|e| candle::Error::Msg(e.clone()).bt())?;
    let func = dev.get_or_load_custom_func("adamw_f32", "candle_nn_adamw_f32_v1", ptx)?;
    let theta = ts
        .as_cuda_slice::<f32>()?
        .slice(tl.start_offset()..tl.start_offset() + n);
    let m = ms
        .as_cuda_slice::<f32>()?
        .slice(ml.start_offset()..ml.start_offset() + n);
    let v = vs
        .as_cuda_slice::<f32>()?
        .slice(vl.start_offset()..vl.start_offset() + n);
    let g = gs
        .as_cuda_slice::<f32>()?
        .slice(gl.start_offset()..gl.start_offset() + n);
    // SAFETY: The kernel writes every element of all three outputs before use.
    let mut nm = unsafe { dev.alloc::<f32>(n)? };
    let mut nv = unsafe { dev.alloc::<f32>(n)? };
    let mut nt = unsafe { dev.alloc::<f32>(n)? };
    let mut builder = func.builder();
    let n_u32 = n as u32;
    builder
        .arg(&n_u32)
        .arg(&theta)
        .arg(&m)
        .arg(&v)
        .arg(&g)
        .arg(&mut nm)
        .arg(&mut nv)
        .arg(&mut nt);
    candle::builder_arg!(
        builder,
        params.beta1 as f32,
        (1.0 - params.beta1) as f32,
        params.beta2 as f32,
        (1.0 - params.beta2) as f32,
        scale_m as f32,
        scale_v as f32,
        (1.0 - params.lr * params.weight_decay) as f32,
        params.eps as f32,
        params.lr as f32
    );
    // SAFETY: Contiguous input ranges and output lengths are checked above;
    // cudarc tracks input reads and output writes on the device stream.
    unsafe { builder.launch(LaunchConfig::for_num_elems(n_u32)) }.w()?;
    let wrap = |slice| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            tl.shape().clone(),
            BackpropOp::none(),
            false,
        )
    };
    Ok(Some((wrap(nm), wrap(nv), wrap(nt))))
}
