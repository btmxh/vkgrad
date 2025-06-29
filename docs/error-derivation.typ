= Error bound for dual-process matrix multiplication difference

*Problem*: Let $A in RR^(M times K), B in RR^(K times N)$ be random uniform matrices ($A_(i j), B_(i j) tilde cal(U)[-a, a]$). Consider two processes that calculate $C = A B$, giving results $C^1, C^2$. Assuming the processes are computer-based (representing numbers as `float32`, subjected to floating-point errors, etc.), find a upper bound for $E = max abs(C^1 - C^2)_(i j)$.

== Folded Gaussian Lemma

*Lemma:* The absolute value of a Gaussian $X tilde cal(N)(0, sigma)$ has mean $sigma sqrt(2/pi)$.

*Proof*

Integrating:
$
  EE[abs(X)] & = integral_RR abs(x) 1/(sigma sqrt(2 pi)) exp(-x^2/(2 sigma^2)) dif x                       \
             & = 1/(sigma sqrt(2 pi)) integral_0^infinity exp(-x^2/(2 sigma^2)) dif (x^2)                  \
             & = (sqrt(2) sigma)/(sqrt(pi)) integral_0^infinity exp(-x^2/(2 sigma^2)) dif (x^2/(2sigma^2)) \
             & = sigma sqrt(2/pi).
$

== Derivation of error upper bound

Let $Delta C = C^1 - C^2$, assuming $Delta C$ are independent, then: $max |Delta C_(i j)|$ is the maximum of $M times N$ i.i.d. random variables.

We look at each $abs(Delta C_(i j))$. The `f32` error of one floating-point operation can be estimated as a uniform distribution $cal(U)[-t, t]$, for $t = epsilon abs(C_(i j))$, where $epsilon$ is the machine epsilon. The accumulation of such error can be modelled as the sum of i.i.d. random variables, which is the Gaussian $cal(N)(0, sigma)$, where $sigma$ is:

$ sigma_1 = sqrt(K t^2 / 3) = epsilon abs(C_(i j)) sqrt(K/3). $

To estimate $abs(C_(i j))$, we use the Central Limit Theorem again: it is the sum of $A_(i k) B_(k j)$, which are independent products of two uniformly distributed variables. We can calculate the mean (which is 0) and the standard deviation of each product as
$
  sigma_2 & = sqrt(VV[C_(i j)]) = sqrt(K VV[A_(i k) B_(k j)])                                          \
          & = sqrt(K (EE[A_(i k)]^2 VV[B_(k j)] + EE[B_(k j)]^2 VV[A_(i k)] + VV[A_(i k)]VV[B_(k j)])) \
          & = sqrt(K) VV[A_(i k)] = a^2/3 sqrt(K).
$

Then, $abs(C_(i j))$ has mean $sigma' sqrt(2/pi)$, using the lemma above. Substituting everything in:
$ sigma_1 = epsilon a^2/3 sqrt((2K)/pi) sqrt(K/3) = sqrt(2/(27pi)) epsilon a^2K. $

Once we have the distribution of the error of one floating-point process, we can simply subtract them to find the distribution of the error between the two processes. Assuming the two process is independent, which means the two errors aer also independent, so the difference is yet another Gaussian $cal(N)(0, sigma sqrt(2))$. Then, the absolute value of that difference is just the absolute value of a Gaussian. Denote $sigma = sigma_1 sqrt(2) = sqrt(4/(27pi)) epsilon a^2K.$

Now, onto the maximum part. We simply evaluate the CDF:

#let erf = math.op("erf")
$ F_E (x) = PP (E <= x) = PP(abs((Delta C)_(i j)) <= x, forall i, j) = erf (x/(sigma sqrt(2)))^(M N). $

To find an upper bound $U_alpha$ that works $1 - alpha$ of the time, we need:
$
  F_E (U_alpha) = 1 - alpha => erf(U_alpha/(sigma sqrt(2))) = root(M N, 1-alpha) => U_alpha = sigma sqrt(2) erf^(-1) (root(M N, 1 - alpha)).
$

For the full formula:
$ U_alpha = sqrt(8/(27pi)) epsilon a^2K erf^(-1) (root(M N, 1 - alpha)). $

The erf term can be approximated:
$
  erf^(-1) (root(M N, 1 - alpha)) & approx erf^(-1) (1 - alpha/(M N))                                     \
                                  & approx 1/sqrt(2) sqrt(
                                      log 2/(pi (a/(M N))^2)
                                      - log log 2/(pi (a/(M N))^2)
                                    )                                                                     \
                                  & = sqrt(log (M N)/alpha sqrt(2/pi) - log log (M N)/alpha sqrt(2/pi)) .
$
In conclusion,
$ U_alpha approx sqrt(8/(27pi) (log (M N)/alpha sqrt(2/pi) - log log (M N)/alpha sqrt(2/pi))) epsilon a^2K. $

