data {
  int<lower = 0> n_sam;
  int<lower = 0, upper = 1> y[n_sam];
  vector<lower = 0, upper = 1> [n_sam] male;
  int<lower = 1, upper = 4> age[n_sam];
  int<lower = 1, upper = 8> eth[n_sam];
  int<lower = 0> Region;
  int<lower = 1, upper = Region> Reg[n_sam];
  vector[Region] x_Reg;
  int<lower = 0> J_spec;
  int<lower = 0> y_spec[J_spec];
  int<lower = 0> n_spec[J_spec];
  int<lower = 0> J_sens;
  int<lower = 0> y_sens[J_sens];
  int<lower = 0> n_sens[J_sens];
  int<lower = 0> J;  // number of population cells
  vector<lower = 0>[J] Post;  // population sizes for poststratification
  real<lower = 0> coef_prior_scale;
  real<lower = 0> logit_spec_prior_scale;
  real<lower = 0> logit_sens_prior_scale;
}
parameters {
  real mu_logit_spec;
  real mu_logit_sens;
  real<lower = 0> sigma_logit_spec;
  real<lower = 0> sigma_logit_sens;
  vector<offset = mu_logit_spec, multiplier = sigma_logit_spec>[J_spec] logit_spec;
  vector<offset = mu_logit_sens, multiplier = sigma_logit_sens>[J_sens] logit_sens;
  vector[3] beta;
  real<lower = 0> sigma_age;
  real<lower = 0> sigma_eth;
  real<lower = 0> sigma_reg;
  vector<multiplier = sigma_age>[4] a_age;
  vector<multiplier = sigma_eth>[8] a_eth;
  vector<multiplier = sigma_reg>[Region] a_reg;
}
transformed parameters {
  vector[J_spec] spec = inv_logit(logit_spec);
  vector[J_sens] sens = inv_logit(logit_sens);
}
model {
  vector[n_sam] prev = inv_logit(beta[1]
                      + beta[2] * male
                      + beta[3] * x_Reg[Reg]
                      + a_age[age]
                      + a_eth[eth]
                      + a_reg[Reg]);
  
  vector[n_sam] p_sample = prev .* sens[1] + (1 - prev) .* (1 - spec[1]);

  // Log-likelihood components
  target += bernoulli_logit_lpmf(y | beta[1] + beta[2] * male + beta[3] * x_Reg[Reg] + a_age[age] + a_eth[eth] + a_reg[Reg]);

  for (j in 1:J_spec) {
    target += binomial_logit_lpmf(y_spec[j] | n_spec[j], logit_spec[j]);
  }

  for (j in 1:J_sens) {
    target += binomial_logit_lpmf(y_sens[j] | n_sens[j], logit_sens[j]);
  }

  // Priors
  logit_spec ~ normal(mu_logit_spec, sigma_logit_spec);
  logit_sens ~ normal(mu_logit_sens, sigma_logit_sens);
  sigma_logit_spec ~ normal(0, logit_spec_prior_scale);
  sigma_logit_sens ~ normal(0, logit_sens_prior_scale);
  mu_logit_spec ~ normal(4, 2);
  mu_logit_sens ~ normal(4, 2);
  a_age ~ normal(0, sigma_age);
  a_eth ~ normal(0, sigma_eth);
  a_reg ~ normal(0, sigma_reg);
  beta[1] + beta[2] * mean(male) + beta[3] * mean(x_Reg[Reg]) ~ logistic(0, 1);
  beta[2] ~ normal(0, coef_prior_scale);
  beta[3] ~ normal(0, coef_prior_scale / sd(x_Reg[Reg]));
  sigma_age ~ normal(0, coef_prior_scale);
  sigma_eth ~ normal(0, coef_prior_scale);
  sigma_reg ~ normal(0, coef_prior_scale);
}
generated quantities {
  real prev_avg;
  vector[J] p_pop;
  int count;
  real log_likelihood;  // Variable to store log likelihood

  count = 1;
  log_likelihood = 0;  // Initialize log likelihood

  for (i in 1:n_sam) {
    // Calculate the log likelihood for y
    log_likelihood += bernoulli_logit_lpmf(y[i] | beta[1] + beta[2] * male[i] + beta[3] * x_Reg[Reg[i]] + a_age[age[i]] + a_eth[eth[i]] + a_reg[Reg[i]]);
  }

  for (j in 1:J_spec) {
    // Calculate the log likelihood for y_spec
    log_likelihood += binomial_logit_lpmf(y_spec[j] | n_spec[j], logit_spec[j]);
  }

  for (j in 1:J_sens) {
    // Calculate the log likelihood for y_sens
    log_likelihood += binomial_logit_lpmf(y_sens[j] | n_sens[j], logit_sens[j]);
  }

  // Calculate p_pop as before
  count = 1;
  for (i_reg in 1:Region) {
    for (i_eth in 1:8) {
      for (i_age in 1:4) {
        for (i_male in 0:1) {
          p_pop[count] = inv_logit(beta[1]
                                + beta[2] * i_male
                                + beta[3] * x_Reg[i_reg]
                                + a_age[i_age]
                                + a_eth[i_eth]
                                + a_reg[i_reg]);
          count += 1;
        }
      }
    }
  }
  prev_avg = sum(Post .* p_pop) / sum(Post);
}
