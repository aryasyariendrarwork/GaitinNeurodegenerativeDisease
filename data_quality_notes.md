# Data quality notes

## hunt20
- Right stride median: 42.91 seconds; left stride median: 0.99 seconds.
- Right stride values form long plateaus in the graph.
- Negative double support values occur in the file.
- These values match the original PhysioNet file.
- The cause has not been established.
- Raw data retained unchanged. Handling for modelling remains pending.

## als12
- Very long stride intervals strongly affect the mean and standard deviation.
- Left stride maximum: 55.15 seconds.
- Left stride median: 1.3833 seconds.
- Raw data retained unchanged.

## Dataset documentation
- PhysioNet states that the derived time series have not been filtered.