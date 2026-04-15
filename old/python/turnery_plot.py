import ternary
import numpy as np
import matplotlib.pyplot as plt

with np.load('runtimes.npz') as runs:
    runtime_ratio = runs['runtime_ratio']
    scale = runs['scale']

data = dict()
for p in range(runtime_ratio.shape[0]):
    data[(runtime_ratio[p,0], runtime_ratio[p,1], runtime_ratio[p,2])] = runtime_ratio[p,3]



figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmap(data=data, cmap=None)
tax.boundary(linewidth=2.0)
tax.set_title("Wall-clock Runtime Ratio ($\Delta t_{OCI}/\Delta t_{SI}$) in $S_2$ on GPU")
tax.set_axis_limits({'b': [0.05, 1.0], 'l': [0.0, 0.95], 'r': [0.05, 10.0]})
tick_formats = {'b': "%.2f", 'r': "%d", 'l': "%.2f"}
# get and set the custom ticks:
tax.get_ticks_from_axis_limits()
tax.set_custom_ticks(fontsize=10, offset=0.02, tick_formats=tick_formats)
tax.get_axes().axis('off')
tax.left_axis_label("scattering ratio [$\Sigma_s/\Sigma$]", offset=0.13)
tax.right_axis_label("mfp thickness [$\Sigma*\Delta x$]", offset=0.13)
tax.bottom_axis_label("$\Delta t$", offset=0.0)

tax.clear_matplotlib_ticks()
tax._redraw_labels()
plt.tight_layout()
tax.show()
