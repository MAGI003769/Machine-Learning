import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
mu = 0
gamma = [0.2, 0.4, 0.6, 0.8, 1.0]
legends = []

fig, cells = plt.subplots(1, 2, figsize=(8,4))

for i, gam in enumerate(gamma):
	power = -(x-mu) / gam
	log_cdf = 1 / (1+np.exp(power))
	cells[0].plot(x, log_cdf)
	tmp_legend = '$\gamma$='+str(gam)
	legends.append(tmp_legend)
cells[0].axhline(0.5, c='k', ls='--')
cells[0].legend(legends)
cells[0].set_ylabel('$F(x)$')
cells[0].grid()

for i, gam in enumerate(gamma):
	power = -(x-mu) / gam
	log_pdf = np.exp(power) / (gam * (1+np.exp(power))**2)
	cells[1].plot(x, log_pdf)
	tmp_legend = '$\gamma$='+str(gam)
	legends.append(tmp_legend)
cells[1].legend(legends)
cells[1].set_ylabel('$f(x)$')
cells[1].grid()

fig.tight_layout()
fig.savefig('logistic_curves.png', bbox_inches='tight')
plt.show()

