functions {
  array[] real sir_model(real t,  array[] real y,  array[] real theta,  array[] real x_r, array[] int x_i) {
    real S = y[1];
    real I = y[2];
    real R = y[3];
    real beta = theta[1];
    real gamma = theta[2];
    real N = x_i[1];

    real dS_dt = -beta * S * I / N;
    real dI_dt = beta * S * I / N - gamma * I;
    real dR_dt = gamma * I;
    return {dS_dt, dI_dt, dR_dt};
  }
}
data {
  int<lower=1> N;          // Total population
  int<lower=1> T;          // Time horizon
  real<lower=0> I0;        // Initial number of infected
  real<lower=0> R0;        // Initial number of recovered
  array[T] int<lower=0> y;       // Observed data: number of new infections
  array[T] real ts;
  int<lower=1> total_count;// Total counts for binomial observation model
}
transformed data {
  array[0] real x_r;
  array[1] int x_i = {N};
  array[3] real y0 = {N - I0 - R0, I0, R0};
}
parameters {
  real beta_log;  // Log-transformed transmission rate
  real gamma_log; // Log-transformed recovery rate
}
transformed parameters {
  real<lower=0> beta = exp(beta_log);
  real<lower=0> gamma = exp(gamma_log);
  array[2] real theta = {beta, gamma};
  array[T,3] real solution = integrate_ode_rk45(sir_model, y0, 0, ts, theta, x_r, x_i, 1e-8, 1e-7, 1e5);
}
model {
  beta_log ~ normal(log(0.4), 0.5);
  gamma_log ~ normal(log(1.0 / 8.0), 0.2);

  for (t in 1:T) {
    real bin_I = solution[t,2] + 1e-1; //fmax(, 0.0);
    y[t] ~ binomial(total_count, bin_I / N); // Using the 2nd column (I) of the solution
  }
}
