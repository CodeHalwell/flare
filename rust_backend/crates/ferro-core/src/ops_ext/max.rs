//! Global `max` reduction to a scalar. Forward returns the maximum over all
//! elements with torch semantics: empty input is an error and NaN propagates
//! (the first NaN encountered wins). Backward matches torch's
//! evenly_distribute_backward: the incoming scalar grad is split evenly across
//! every element equal to the max, so ties share it; a NaN result matches no
//! element (NaN != NaN) and yields zero gradients, as in torch.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

impl Tensor {
    pub fn max(&self) -> Result<Tensor> {
        let xv = self.to_vec();
        if xv.is_empty() {
            return Err(Error::InvalidShape { op: "max", msg: "cannot reduce an empty tensor".into() });
        }
        let mut m = xv[0];
        for &v in &xv {
            if v.is_nan() {
                m = v;
                break;
            }
            if v > m {
                m = v;
            }
        }
        // Capture only the tie indices, not the materialized input: the closure
        // lives as long as the output, and xv would pin a full copy in memory.
        let ties: Vec<usize> =
            xv.iter().enumerate().filter(|(_, &v)| v == m).map(|(i, _)| i).collect();
        let len = xv.len();
        let out = Tensor::scalar(m);
        let shape = self.shape().to_vec();
        Ok(out.record_fn(vec![self.clone()], move |g| {
            let mut grad = vec![0.0; len];
            if !ties.is_empty() {
                let share = g.item() / ties.len() as f32;
                for &i in &ties {
                    grad[i] = share;
                }
            }
            vec![Tensor::from_vec(grad, &shape).unwrap()]
        }))
    }
}
