data {
  int<lower=1> K;  // Number of sources
  int<lower=1> M;  // Number of measurement points
  array[M] vector[1] y; // outcome y is 1D
  array[M] vector[2] x; // data is x is 2D
}

parameters {
  array[K] vector[2] theta;  // 2D locations of the K sources
}

model {
  // Priors for theta
  for (k in 1:K) {
    theta[k] ~ normal(0, 1);  // standard normal prior
  }

  // Likelihood
  for (m in 1:M) {
    vector[K] sq_two_norm;
    for (k in 1:K) {
      sq_two_norm[k] = dot_self(x[m] - theta[k]);
    }
    real mean_y = log(0.1 + sum(inv(1e-4 + sq_two_norm)));  // 0.1 and 1e-4 fixed
    target += normal_lpdf(y[m] | mean_y, 0.5);  // noise_sd is 0.5 fixed
  }
  // add prior lpdf to target
  for (k in 1:K) {
    target += normal_lpdf(theta[k] | 0, 1);
  }
  // target is the joint logprob
}
