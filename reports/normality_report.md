# Normality Diagnostics

- Source: `MiniProyecto\coffee_db.parquet`
- Column: `Total_domestic_consumption`
- N: **55**
- Mean: **1112486638.2364**
- Std: **3834783126.9728**

## Tests
- Shapiro-Wilk: stat=0.2810, p=0.000000
- Kolmogorov-Smirnov: stat=0.3859, p=0.000000
- Anderson-Darling: stat=13.2031, critical@5%=0.7390

## Recommendation
Hay desviaciones de normalidad. Si usas inferencia parametrica, valida residuos, considera transformaciones (log/Yeo-Johnson) o metodos robustos.

## Figures
- Histogram: `reports\figures\Total_domestic_consumption_hist.png`
- QQ-plot: `reports\figures\Total_domestic_consumption_qqplot.png`