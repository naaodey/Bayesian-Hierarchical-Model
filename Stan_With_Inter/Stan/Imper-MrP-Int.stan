data {
  int<lower = 0> n_sam; // Number of samples
  int<lower = 0, upper = 1> y[n_sam]; // Binary outcome
  vector<lower = 0, upper = 1> [n_sam]male; // Gender variable
  int<lower = 1, upper = 4> age[n_sam]; // Age group
  int<lower = 1, upper = 8> eth[n_sam]; // Ethnicity
  int<lower = 0> Region; // Number of regions
  int<lower = 1, upper = Region> Reg[n_sam]; // Region index for each sample
  vector[Region] x_Reg; // Region-specific covariates
  int<lower = 0> J_spec; // Number of specification groups
  int<lower = 0> y_spec[J_spec]; // Responses for specifications
  int<lower = 0> n_spec[J_spec]; // Number of observations for specifications
  int<lower = 0> J_sens; // Number of sensitivity groups
  int<lower = 0> y_sens[J_sens]; // Responses for sensitivities
  int<lower = 0> n_sens[J_sens]; // Number of observations for sensitivities
  int<lower = 0> J; // Number of population cells
  vector<lower = 0>[J] Post; // Population sizes for poststratification
  real<lower = 0> coef_prior_scale; // Coefficient prior scale
  real<lower = 0> logit_spec_prior_scale; // Logit specification prior scale
  real<lower = 0> logit_sens_prior_scale; // Logit sensitivity prior scale
}

parameters {
  real mu_logit_spec; // Mean logit specification
  real mu_logit_sens; // Mean logit sensitivity
  real<lower = 0> sigma_logit_spec; // SD logit specification
  real<lower = 0> sigma_logit_sens; // SD logit sensitivity
  vector<offset = mu_logit_spec, multiplier = sigma_logit_spec>[J_spec] logit_spec; // Logit specifications
  vector<offset = mu_logit_sens, multiplier = sigma_logit_sens>[J_sens] logit_sens; // Logit sensitivities
  vector[3] beta; // Coefficients for the main model
  real<lower = 0> sigma_age; // SD for age effects
  real<lower = 0> sigma_eth; // SD for ethnicity effects
  real<lower = 0> sigma_reg; // SD for region effects
  real<lower = 0> sigma_interaction; // SD for interaction terms
  vector [4] a_age; // Age effects
  vector [8] a_eth; // Ethnicity effects
  vector [Region] a_reg; // Region effects
  matrix[4, 8] gamma_age_eth; // Interaction term for age and ethnicity
}

transformed parameters {
  vector[J_spec] spec = inv_logit(logit_spec); // Inverse logit for specifications
  vector[J_sens] sens = inv_logit(logit_sens); // Inverse logit for sensitivities
}

model {
  vector[n_sam] prev;
  
  // Create each term as a vector of length N
  vector[n_sam] male_effect = beta[2] * male;
  vector[n_sam] reg_effect = beta[3] * x_Reg[Reg];
  vector[n_sam] eth_effect = a_eth[eth];
  vector[n_sam] age_effect = a_age[age];
  vector[n_sam] reg_effect_2 = a_reg[Reg];
  
  // Compute interaction effects based on age and ethnicity
  vector[n_sam] interaction_effect;
  for (n in 1:n_sam) {
    interaction_effect[n] = gamma_age_eth[age[n], eth[n]];
  }
  
  // Combine all terms
  prev = inv_logit(beta[1] + male_effect + reg_effect + eth_effect + age_effect + reg_effect_2 + interaction_effect);

  vector[n_sam] p_sample = prev * sens[1] + (1 - prev) * (1 - spec[1]);
  
  y ~ bernoulli(p_sample);
  y_spec ~ binomial(n_spec, spec);
  y_sens ~ binomial(n_sens, sens);
  
  logit_spec ~ normal(mu_logit_spec, sigma_logit_spec);
  logit_sens ~ normal(mu_logit_sens, sigma_logit_sens);
  sigma_logit_spec ~ normal(0, logit_spec_prior_scale);
  sigma_logit_sens ~ normal(0, logit_sens_prior_scale);
  mu_logit_spec ~ normal(4, 2);
  mu_logit_sens ~ normal(4, 2);
  
  a_eth ~ normal(0, sigma_eth);
  a_age ~ normal(0, sigma_age);
  a_reg ~ normal(0, sigma_reg);
  to_vector(gamma_age_eth) ~ normal(0, sigma_interaction);
  
  beta[1] + beta[2] * mean(male) + beta[3] * mean(x_Reg[Reg]) ~ logistic(0, 1);
  beta[2] ~ normal(0, coef_prior_scale);
  beta[3] ~ normal(0, coef_prior_scale / sd(x_Reg[Reg]));
  
  sigma_eth ~ normal(0, coef_prior_scale);
  sigma_age ~ normal(0, coef_prior_scale);
  sigma_reg~ normal(0, coef_prior_scale);
  sigma_interaction ~ normal(0, coef_prior_scale);
}

generated quantities {
  real prev_avg;
  vector[J] p_pop;
  int count = 1;
  real log_likelihood = 0; // Initialize log likelihood

  for (i in 1:n_sam) {
    // Calculate the log likelihood for y
    log_likelihood += bernoulli_logit_lpmf(y[i] | beta[1]
                      + beta[2] * male[i] 
                      + beta[3] * x_Reg[Reg[i]] 
                      + a_age[age[i]] 
                      + a_eth[eth[i]] 
                      + a_reg[Reg[i]] 
                      + gamma_age_eth[age[i], eth[i]]); // Use prev[i] for log-likelihood
  }

  for (j in 1:J_spec) {
    // Calculate the log likelihood for y_spec
    log_likelihood += binomial_logit_lpmf(y_spec[j] | n_spec[j], logit_spec[j]);
  }

  for (j in 1:J_sens) {
    // Calculate the log likelihood for y_sens
    log_likelihood += binomial_logit_lpmf(y_sens[j] | n_sens[j], logit_sens[j]);
  }

  // Calculate population prevalence
  count = 1; // Reset count for population calculations
  for (i_reg in 1:Region) {
    for (i_eth in 1:8) {
      for (i_age in 1:4) {
        for (i_male in 0:1) {
          p_pop[count] = inv_logit(beta[1]
                                + beta[2] * i_male
                                + beta[3] * x_Reg[i_reg]
                                + a_age[i_age]
                                + a_eth[i_eth]
                                + a_reg[i_reg]
                                + gamma_age_eth[i_age, i_eth]);  // Include interaction
          count += 1; // Increment count
        }
      }
    }
  }
  prev_avg = sum(Post .* p_pop) / sum(Post); // Average prevalence calculation
}
