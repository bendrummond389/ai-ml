# Gaussian Distribution

## Formula

The Gaussian distribution, also known as the normal distribution, is described by the probability density function (PDF):

\[
f(x \;|\; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\]

- \( \mu \) is the mean
- \( \sigma^2 \) is the variance
- \( x \) is the point at which you're evaluating the function

## Mean

The mean (\( \mu \)) of a data set \( X \) with \( N \) elements is calculated as:

\[
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
\]

## Variance

The variance (\( \sigma^2 \)) of a data set \( X \) with \( N \) elements is calculated as:

\[
\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2
\]

Note that this formula calculates the population variance. For sample variance, you would divide by \( N - 1 \) instead of \( N \).
