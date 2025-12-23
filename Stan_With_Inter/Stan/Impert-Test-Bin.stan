data {
  int<lower = 0> n_sam;   // Number of samples for the main outcome
  int<lower = 0> y_sam;   // Successes for the main outcome
  int<lower = 0> y_sens;  // Successes for the sensitivity
  int<lower = 0> n_sens;   // Number of samples for the sensitivity
  int<lower = 0> y_spec;   // Successes for the specificity
  int<lower = 0> n_spec;   // Number of samples for the specificity
}

parameters {
  real<lower=0,upper=1> prev;  // Prevalence
  real<lower=0,upper=1> sens;   // Sensitivity
  real<lower=0,upper=1> spec;   // Specificity
}

model {
  real p_sam = prev * sens + (1 - prev) * (1 - spec);
  
  // Likelihoods
  y_sam ~ binomial(n_sam, p_sam);
  y_sens ~ binomial(n_sens, sens);
  y_spec ~ binomial(n_spec, spec);
  
  target += binomial_lpmf(y_sam | n_sam, p_sam);
  target += binomial_lpmf(y_sens | n_sens, sens);
  target += binomial_lpmf(y_spec | n_spec, spec);
}

generated quantities {
  real log_lik;

  // Calculate the log likelihood for the observations
  log_lik = binomial_lpmf(y_sam | n_sam, prev * sens + (1 - prev) * (1 - spec))
           + binomial_lpmf(y_sens | n_sens, sens)
           + binomial_lpmf(y_spec | n_spec, spec);
}

