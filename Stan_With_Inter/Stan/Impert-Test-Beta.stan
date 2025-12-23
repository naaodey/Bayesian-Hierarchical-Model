data {
  int <lower = 0> n_sam;
  int <lower = 0> y_sam;
  int <lower = 0> alpha;
  int <lower = 0> beta;
  int <lower = 0> spec_alpha;
  int <lower = 0>  spec_beta;
  int <lower = 0>  sens_alpha;
  int <lower = 0>  sens_beta;
}

parameters {
  real<lower=0,upper=1> prev; 
  real<lower=0,upper=1> sens;
  real<lower=0,upper=1> spec;
}

model{
  real p_sam = prev*sens+(1-prev)*(1-spec);
  prev ~ beta(alpha, beta);
  sens ~ beta(sens_alpha,sens_beta);
  spec ~ beta(spec_alpha,spec_beta);
  
  for(j in 1:y_sam){
    target += log_sum_exp(log(sens) + log(prev),
     log(spec) + log (1-prev));
  }
  for (j in 1:n_sam-y_sam){
    target += log_sum_exp(log(1-sens) + log(prev),
    log(1-spec) +log(1-prev));
  }
}

generated quantities {
  real log_lik;
  
  // Calculate log likelihood for the observation
  log_lik = binomial_lpmf(y_sam | n_sam, prev * sens + (1 - prev) * (1 - spec));
}



