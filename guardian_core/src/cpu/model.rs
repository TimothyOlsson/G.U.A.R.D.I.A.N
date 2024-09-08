use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use anyhow::{ensure, Result};

use super::process::clip;

pub type Row = Array1<f32>;
pub type Array = Array2<f32>;
pub type Weight = Array2<f32>;
pub type Bias = Array1<f32>;

#[derive(Clone)]
pub struct Layer {
    weight: Weight,
    bias: Bias
}

#[derive(Clone)]
pub struct Model {
    settings: ModelSettings,
    input_weights: Vec<Weight>,
    input_bias: Bias,
    hidden_layers: Vec<Layer>,
    output_layers: Vec<Layer>
}

#[derive(Clone)]
pub struct ModelSettings {
    // Easier for GPU calculations
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
    input_sizes: Vec<usize>,
    hidden_sizes: Vec<usize>,
    output_sizes: Vec<usize>
}

impl ModelSettings {
    pub fn new(input_sizes: Vec<usize>, hidden_sizes: Vec<usize>, output_sizes: Vec<usize>) -> Result<Self> {
        ensure!(!input_sizes.is_empty());
        ensure!(!hidden_sizes.is_empty());
        ensure!(!output_sizes.is_empty());
        Ok(
            Self {
                n_inputs: input_sizes.len(),
                n_hidden: hidden_sizes.len(),
                n_outputs: output_sizes.len(),
                input_sizes,
                hidden_sizes,
                output_sizes
            }
        )
    }
}

impl Model {
    pub fn new(settings: ModelSettings) -> Result<Self> {
        ensure!(!settings.input_sizes.is_empty());
        ensure!(!settings.hidden_sizes.is_empty());
        ensure!(!settings.output_sizes.is_empty());

        // Handle input layers
        let next_size = settings.hidden_sizes.first().unwrap();
        let mut input_weights = vec![];
        for input_size in settings.input_sizes.iter() {
            let weight = new_weight(*input_size, *next_size);
            input_weights.push(weight);
        }
        let input_bias = new_bias(*next_size);

        // Handle hidden layers
        let mut prev_size = next_size;
        let mut hidden_layers = vec![];
        for hidden_size in settings.hidden_sizes.iter().skip(1) {
            let layer = Layer::new(*prev_size, *hidden_size);
            hidden_layers.push(layer);
            prev_size = hidden_size;
        }

        // Handle outputs
        let last_hidden_size = prev_size;
        let mut output_layers = vec![];
        for output_size in settings.output_sizes.iter() {
            let layer = Layer::new(*last_hidden_size, *output_size);
            output_layers.push(layer);
        }
        Ok(Self { settings, input_weights, input_bias, hidden_layers, output_layers })
    }

    /// Apply the full model on the input arrays
    pub fn forward_from_precalc(
        &self,
        inputs: &[(usize, ArrayView2<f32>)],
        precalculated: &[&Row],
    ) -> Vec<Array> {
        let batch_size = inputs.first().unwrap().1.shape()[0];
        let mut x: Array = Array2::zeros((batch_size, self.input_bias.len()));

        // Add precalc
        for precalc in precalculated {
            x = x + *precalc;
        }

        // Add inputs not precalculated
        for (i, input) in inputs {
            let weight = &self.input_weights[*i];
            x = x + input.dot(weight);
        }

        x = x + &self.input_bias;
        x = activation_fn(x);
        // Done with inputs, now go through all hidden
        for layer in self.hidden_layers.iter() {
            x = layer.forward_with_bias(&x);
            x = activation_fn(x);
            //x = activation_fn_output(x);
        }

        // Now, we can calculate the outputs
        let mut outputs = vec![];
        for layer in &self.output_layers {
            let mut res = layer.forward_weight(&x);  // TODO: No bias on output?
            res = activation_fn_output(res);
            outputs.push(res);
        }
        outputs
    }


    /// NOTE: Bias is NOT added here!
    pub fn precalculate(&self, input_index: usize, x: ArrayView1<f32>) -> Row {
        let weight = &self.input_weights[input_index];
        let hidden = x.dot(weight);
        hidden
    }
}


impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weight = new_weight(input_size, output_size);
        let bias = new_bias(output_size);
        Self {
            weight,
            bias
        }
    }

    // Apply the forward computation on the input array
    pub fn forward_with_bias(&self, x: &Array) -> Array {
        let mut y = x.dot(&self.weight);
        y = y + &self.bias;
        y
    }

    pub fn forward_weight(&self, x: &Array) -> Array {
        let y = x.dot(&self.weight);
        y
    }

    // Ownership to prevent copy
    pub fn apply_bias(&self, mut x: Array) -> Array {
        x = x + &self.bias;
        x
    }
}

/// Clamping
/// Does it inplace
fn activation_fn(arr: Array) -> Array {
    clip(arr, 0.0, 1.0)
}

/// Delta
/// Can be negative
fn activation_fn_output(arr: Array) -> Array {
    clip(arr, -0.1, 0.1)
}

fn new_weight(input_size: usize, output_size: usize) -> Weight {
    Array2::random((input_size, output_size), Normal::new(0.0, 0.1).unwrap())
}

fn new_bias(size: usize) -> Bias {
    Array1::random(size, Normal::new(0.0, 0.1).unwrap())
}

#[cfg(test)]
pub mod tests {
    use rand::distributions::Uniform;
    use super::*;

    #[test]
    pub fn test_model() {
        let model = Model::new(
            ModelSettings::new(vec![4, 2], vec![8, 10], vec![2, 4]).unwrap()
        ).unwrap();
        let batch_size = 4;
        let x1 = Array2::random((batch_size, 4), Uniform::new(0.0, 1.0));
        let x2 = Array2::random((batch_size, 2), Uniform::new(0.0, 1.0));
        println!("INPUT1:\n{x1:#?}");
        println!("INPUT2:\n{x2:#?}");
        let res = model.forward_from_precalc(
            &[(0, x1.view()), (1, x2.view())],
            &[]
        );
        println!("OUTPUT:\n{res:#?}");
    }
}