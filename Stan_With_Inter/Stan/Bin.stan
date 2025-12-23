data {
  int<lower = 0> n_sam;
  int<lower = 0> y_sam;
  int<lower = 0> y_sens;
  int<lower = 0> n_sens;
  int<lower = 0> y_spec;
  int<lower = 0> n_spec;
}

parameters {
  real<lower=0,upper=1> prev; // True prevalence
  real<lower=0,upper=1> sens;
  real<lower=0,upper=1> spec;
}

model {
  real p_sam = prev * sens + (1 - prev) * (1 - spec);
  
  // Compute log likelihood contributions
  target += binomial_lpmf(y_sam | n_sam, p_sam);
  target += binomial_lpmf(y_sens | n_sens, sens);
  target += binomial_lpmf(y_spec | n_spec, spec);
}

// Optionally, you can also create a generated quantities block
generated quantities {
  real log_lik;
  log_lik = binomial_lpmf(y_sam | n_sam, prev * sens + (1 - prev) * (1 - spec))
           + binomial_lpmf(y_sens | n_sens, sens)
           + binomial_lpmf(y_spec | n_spec, spec);
}



