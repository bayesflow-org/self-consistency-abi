functions {
  array[] real sir(real t, array[] real y, array[] real theta, array[] real x_r, array[] int x_i) {
    real S = y[1];
    real I = y[2];
    real R = y[3];
    real N = y[4];

    real beta = theta[1];
    real gamma = theta[2];

    real dS_dt = -beta * S * I / N;
    real dI_dt = beta * S * I / N - gamma * I;
    real dR_dt = gamma * I;

    return {dS_dt, 0.0, dI_dt, dR_dt};
  }
}

data {
  int<lower=1> T;                 // Total time points
  array[T] int infections;        // Observed data (infections at each time point)
  array[T] real ts;               // Time steps corresponding to the infections
  array[4] real y0;               // Initial state including total population
  int<lower=1> total_count;       // Total count for binomial distribution
}
transformed data {
  array[0] real x_r; // Empty array, no additional real-valued data needed
  array[0] int x_i;  // Empty array, no additional integer-valued data needed
}
parameters {
  real log_beta;  // Log of transmission rate
  real log_gamma; // Log of recovery rate
}
transformed parameters {
  real<lower=0> beta = exp(log_beta);  // Transmission rate on the original scale
  real<lower=0> gamma = exp(log_gamma); // Recovery rate on the original scale
}

model {
  // Prior distributions on the log scale; these should get transformed (exp-ed)
  log_beta ~ normal(log(0.4), 0.5);
  log_gamma ~ normal(log(1.0/8.0), 0.2);

  // SIR model integration
  array[2] real theta = {beta, gamma};
  // Solution to the ode (<-> odeint)
  // t0 = 0
  array[T, 3] real sir_sol = integrate_ode_rk45(sir, y0, 0, ts, theta, x_r, x_i, 1e-6, 1e-5, 1e5);

  // Binomial likelihood for the observed infections
  for (i in 1:T) {
    // sample Bin(1000, I / N)
    sir_sol[i, 3] = fmax(sir_sol[i, 3], 0.0);  // deal with underflow
    target += binomial_lpmf(infections[i] | total_count, sir_sol[i, 3] / y0[4]);
  }
  // add prior:
  target += normal_lpdf(log_beta | log(0.4), 0.5);
  target += normal_lpdf(log_gamma | log(1.0/8.0), 0.2);
}
