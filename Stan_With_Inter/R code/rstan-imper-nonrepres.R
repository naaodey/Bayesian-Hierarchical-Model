# Importing libraries for the analysis
library(rstan)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(bayesplot)
library (shinystan)

options(mc.cores = parallel::detectCores())

# spin = shortest posterior interval
# equivalent to  highest posterior density interval for unimodal posteriors
spin = function(x, lower=NULL, upper=NULL, conf=0.95){
  x = sort(as.vector(x))
  if (!is.null(lower)) {
    if (lower > min(x)) stop("lower bound is not lower than all the data")
    else x = c(lower, x)
  }
  if (!is.null(upper)) {
    if (upper < max(x)) stop("upper bound is not higher than all the data")
    else x = c(x, upper)
  }
  n = length(x)
  gap = round(conf*n)
  width = x[(gap+1):n] - x[1:(n-gap)]
  index = min(which(width==min(width)))
  x[c(index, index + gap)]
}

################################## Binomial Distribution ##################################

uss_model_bin = stan_model("Bin.stan")

uss_data_bin = list(
  y_sam = 50, n_sam = 3330,
  y_spec = 369+30, n_spec = 371+30,
  y_sens = 25+78, n_sens = 37+85
)

fit_1 = sampling(uss_model_bin, data = uss_data_bin, chains = 4, iter = 20000, refresh = 0,control=list(adapt_delta=0.95))

print(fit_1, digits_summary=3)

# Posterior draws as data frame
draws_1 = rstan::extract(fit_1, pars=c("spec", "sens", "prev"))

hist(draws_1$prev)

# Inference for the population prevalence
y = as.vector(draws_1[["prev"]])
par(mar=c(3,3,0,1), mgp=c(2, .7, 0), tck=-.02)
hist(y, yaxt="n", yaxs="i", xlab=expression(paste("Prevalence, ", pi)), ylab="", main="")


subset = sample(40000, 1000)
x = as.vector(draws_1[["spec"]][subset])
y = as.vector(draws_1[["prev"]][subset])
z = as.vector(draws_1[["sens"]][subset])

par(mar=c(3,3,0,1), mgp=c(2, .7, 0), tck=-.02)
plot(x, y, xlim=c(min(x), 1), ylim=c(0, max(y)), xaxs="i", yaxs="i", xlab=expression(paste("Specificity, ", gamma)), ylab=expression(paste("Prevalence, ", pi)), bty="l", pch=20, cex=.3)


par(mar=c(3,3,0,1), mgp=c(2, .7, 0), tck=-.02)
plot(z, y, xlim=c(min(x), 1), ylim=c(0, max(y)), xaxs="i", yaxs="i", xlab=expression(paste("Sensitivity, ", delta)), ylab=expression(paste("Prevalence, ", pi)), bty="l", pch=20, cex=.3)


# Use the shortest posterior interval, which makes more sense than a central interval because of the skewness of the posterior and the hard boundary at 0
print(spin(draws_1[["prev"]], lower=0, upper=1, conf=0.95))




##################################Beta Binomial##################################

uss_model_beta = stan_model("Beta-Bin.stan")

uss_data_beta = list(
  y_sam = 50, n_sam = 3330,
  sens_alpha = 103+1, sens_beta = 19+1,
  spec_alpha = 2+1, spec_beta = 399+1,
  alpha = 50, beta = 3280
)


fit_1a = sampling(uss_model_beta, data = uss_data_beta, chains = 4, iter = 20000, refresh = 0,control=list(adapt_delta=0.95))

print(fit_1a, digits_summary=3)

# Posterior draws as data frame
draws_1a = rstan::extract(fit_1a, pars=c("spec", "sens", "prev"))

hist(draws_1a$prev)

# Inference for the population prevalence
y1 = as.vector(draws_1a[["prev"]])
par(mar=c(3,3,0,1), mgp=c(2, .7, 0), tck=-.02)
hist(y1, yaxt="n", yaxs="i", xlab=expression(paste("Prevalence, ", pi)), ylab="", main="")


subset = sample(40000, 1000)
x_1 = as.vector(draws_1a[["spec"]][subset])
y_1 = as.vector(draws_1a[["prev"]][subset])
z_1 = as.vector(draws_1a[["sens"]][subset])

par(mar=c(3,3,0,1), mgp=c(2, .7, 0), tck=-.02)
plot(x_1, y_1, xlim=c(min(x), 1), ylim=c(0, max(y)), xaxs="i", yaxs="i", xlab=expression(paste("Specificity, ", gamma)), ylab=expression(paste("Prevalence, ", pi)), bty="l", pch=20, cex=.3)
plot(z_1, y_1, xlim=c(min(x), 1), ylim=c(0, max(y)), xaxs="i", yaxs="i", xlab=expression(paste("Sensitivity, ", delta)), ylab=expression(paste("Prevalence, ", pi)), bty="l", pch=20, cex=.3)


print(spin(draws_1a[["prev"]], lower=0, upper=1, conf=0.95))


########################## Bin Imperfect Test ###############################################

imper_test_model1 = stan_model("Impert-Test-Bin.stan")

imper_test_data1 = list(
  y_sam = 50, n_sam = 3330,
  y_spec = 369+30, n_spec = 371+30,
  y_sens = 25+78, n_sens = 37+85
)

fit_2 = sampling(imper_test_model1, data = imper_test_data1, chains = 4, iter = 20000, refresh = 0)

print(fit_2, digits_summary=3)

draws_2 = rstan::extract(fit_2, pars = "prev")

hist(draws_2$prev)

print(spin(draws_2[["prev"]], lower=0, upper=1, conf=0.95))



########################## Beta-Bin Imperfect Test ###############################################

imper_test_model2 = stan_model("Impert-Test-Beta.stan")


imper_test_data2 = list(
  y_sam = 50, n_sam = 3330,
  sens_alpha = 103+1, sens_beta = 19+1,
  spec_alpha = 2+1, spec_beta = 399+1,
  alpha = 50, beta = 3250
)

fit_2a = sampling(imper_test_model2, data = imper_test_data2, chains = 4, iter = 20000, refresh = 0)

print(fit_2a, digits_summary=3)

draws_2a = rstan::extract(fit_2a, pars = "prev")

hist(draws_2a$prev)

print(spin(draws_2a[["prev"]], lower=0, upper=1, conf=0.95))



#################################### Non-representative Sample #########################################

hmrp_model = stan_model("Imper-MrP.stan")

n_sam = 3330
y = sample(rep(c(0, 1), c(3330 - 50, 50)))
n = rep(1,3330)

male = sample(rep(c(0,1), c(2101, 1229)))
eth = sample(rep(1:8, c(500+220, 150, 250+50, 150, 200, 40+60, 950+100, 660)))
age = sample(rep(1:4, c(71,550,2542,167)))
Region = 16
Reg = sample(1:Region, 3330, replace = TRUE)
x_Reg = round(rnorm(Region, 50, 20))


# Poststratification Table

J = 2*4*8*Region
Post = rep(NA, J)
count = 1
for (i_reg in 1:Region) {
  for (i_eth in 1:8) {
    for (i_age in 1:4) {
      for (i_male in 0:1) {
        Post[count] = 1000
        count = count + 1
      }
    }
  }
}


# Putting together the data and fit the model

hmrp_data = list(
  n_sam = n_sam, y = y, male = male, age = age, Region = Region, eth = eth,
  x_Reg = x_Reg, Reg = Reg, J = J, Post = Post, J_spec=14, y_spec=c(0, 368, 30, 70, 1102, 300, 311, 500, 198, 99, 29, 146, 105, 50), n_spec=c(0, 371, 30, 70, 1102, 300, 311, 500, 200, 99, 31, 150, 108, 52),
  J_sens=4, y_sens=c(0, 78, 27, 25), n_sens=c(0, 85, 37, 35), logit_spec_prior_scale=0.3, logit_sens_prior_scale=0.3, coef_prior_scale=0.5
)

fit_3 = sampling(hmrp_model, data = hmrp_data, iter = 20000, chains = 4, refresh = 0, control=list(adapt_delta=0.9))

print(fit_3, pars=c("prev_avg", "beta", "a_age", "a_eth", "sigma_eth", "sigma_age", 
                    "sigma_reg", "mu_logit_spec", "sigma_logit_spec",  "mu_logit_sens",
                    "sigma_logit_sens", "p_pop[1]", "p_pop[2]", "p_pop[3]","log_likelihood"), digits=3)





##################################Non-representative Sample With Interaction##########################################


hmrp_int_model = stan_model("4.stan")


hmrp_int_data = list(
  n_sam = n_sam, y = y, male = male, age = age, Region = Region, eth = eth,
  x_Reg = x_Reg, Reg = Reg, J = J, Post = Post, J_spec=14, y_spec=c(0, 368, 30, 70, 1102, 300, 311, 500, 198, 99, 29, 146, 105, 50), n_spec=c(0, 371, 30, 70, 1102, 300, 311, 500, 200, 99, 31, 150, 108, 52),
  J_sens=4, y_sens=c(0, 78, 27, 25), n_sens=c(0, 85, 37, 35), logit_spec_prior_scale=0.3, logit_sens_prior_scale=0.3, coef_prior_scale=0.5
)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
fit_4 = sampling(hmrp_int_model, data = hmrp_int_data, iter = 20000, chains = 4, refresh = 0, control=list(adapt_delta=0.9))

print(fit_4, pars=c("prev_avg", "gamma_age_eth", 
                    "sigma_reg", "mu_logit_spec", "sigma_logit_spec",  "mu_logit_sens",
                    "sigma_logit_sens", "p_pop[1]", "p_pop[2]", "p_pop[3]","log_likelihood"), digits=3)


draws_4 = rstan::extract(fit_4)
print(spin(draws_4[["prev_avg"]], lower=0, upper=1, conf=0.95))

hist(draws_4[["gamma_age_eth"]])

quantile(draws_4[["gamma_age_eth"]], c(0.05, 0.95))


################################# Model Selection #########################################

# Extract the log likelihood from the fit
log_lik_samples1 = rstan::extract(fit_1, "log_lik")$log_lik

log_lik_samples1a = rstan::extract(fit_1a, "log_lik")$log_lik

log_lik_samples2 = rstan::extract(fit_2, "log_lik")$log_lik

log_lik_samples2a = rstan::extract(fit_2a, "log_lik")$log_lik

log_lik_samples3 = rstan::extract(fit_3, "log_likelihood")$log_likelihood

log_lik_samples4 = rstan::extract(fit_4, "log_likelihood")$log_likelihood


# Summarize the log likelihood
log_lik_summary1 = apply(log_lik_samples1, 2, mean) # Mean for each observation
print(log_lik_summary1)




    