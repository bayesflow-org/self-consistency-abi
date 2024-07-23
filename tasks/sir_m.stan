functions {
  array[] real sir(real t, array[] real y, array[] real theta, array[] real  x_r, array[] int  x_i) {
      real S = y[1];
      real I = y[2];
      real R = y[3];
      real N = y[4];

      real beta = theta[1];
      real gamma = theta[2];

      real dS_dt = -beta * I * S / N;
      real dI_dt =  beta * I * S / N - gamma * I;
      real dR_dt =  gamma * I;

      return {dS_dt, 0.0, dI_dt, dR_dt};
  }
}
data {
  int<lower=1> n_days;
  array[4] real y0; //
  real t0;
  array[n_days] real ts;
  // int N;
  array[n_days] int cases;
  // int prior_predictive;
}
transformed data {
  array[0] real x_r;
  //array[1] int x_i = { N };
  array[0] int x_i;
}
parameters {
  real<lower=0> gamma;
  real<lower=0> beta;
}
transformed parameters{
  array[n_days, 4] real y;
  array[2] real theta = {beta, gamma};
  // ode, initial condition, ode params, [rel_tol, abs_tol, max_step]
  y = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i, 1e-8, 1e-7, 1e5);
}
model {
  //priors
  //target += normal_lpdf(beta | 2, 0.1) - normal_lccdf(0 | 2, 0.1);
  //target += normal_lpdf(gamma | 0.4, 0.1) - normal_lccdf(0 | 0.4, 0.1);
  target += lognormal_lpdf(beta | log(0.4), 0.5);
  target += lognormal_lpdf(gamma | log(1.0/8.0), 0.2);

  //sampling distribution
  //if (prior_predictive == 0){
  target += binomial_lpmf(cases | 1000, (col(to_matrix(y), 3) + 1e-8) / y0[4]);
  //}

}
generated quantities {
  real R0 = beta / gamma;
  real recovery_time = 1 / gamma;
  array[n_days] real pred_cases;

  //col(matrix x, int n) - The n-th column of matrix x. Here the number of infected people
  pred_cases = binomial_rng(1000, (col(to_matrix(y), 3) + 1e-8) / y0[4]);
}
