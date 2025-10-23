from scipy.integrate import trapezoid, cumulative_trapezoid

I = trapezoid(y, x)
partial = cumulative_trapezoid(y, x, initial=0)  # running integral