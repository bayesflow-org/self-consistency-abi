functions {
    vector derivative(real t, vector y, real p0, real h, real k1, real nu) {
        vector[3] dydt;
        dydt[1] = -0.03*y[1] + 1/(1+pow(y[3]/p0, h));
        dydt[2] = -0.03*y[2] + nu*y[1] - k1*y[2];
        dydt[3] = -0.03*y[3] + k1*y[2];

        return dydt;
    }

    real normal_lb_rng(real mu, real sigma, real lb) {
        real p_lb = normal_cdf(0 | mu, sigma);
        real u = uniform_rng(p_lb, 1);
        real y = mu + sigma * inv_Phi(u);
        return y;
    }
}

data {
    int N;
    vector[N] y;
    array[N] real x;
}

parameters {
    real<lower=0> p0;
    real<lower=0> h;
    real<lower=0> k1;
    real<lower=0> nu;
}

transformed parameters {
    vector[3] y0 = to_vector({2, 5, 3});
}

model {
    p0 ~ gamma(2, 1); //normal(2.4, 0.2);
    h ~ gamma(10, 1); //normal(7, 1);
    k1 ~ gamma(2, 50);//normal(0.1, 0.1);
    nu ~ gamma(2, 50); // normal(0, 0.01);

    array[N] vector[3] mu = ode_rk45(derivative, y0, 0.0, x, p0, h, k1, nu);

    for (t in 1:N) {
        y[t] ~ normal(mu[t, 1], 1.0);
    }
}

generated quantities {
    real p0_prior = gamma_rng(2, 1);
    real h_prior = gamma_rng(10, 1);
    real k1_prior = gamma_rng(2, 50);
    real nu_prior = gamma_rng(2, 50);
  }