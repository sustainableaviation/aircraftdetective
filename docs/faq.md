# FAQ

<div class="video-wrapper">
  <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/NfDUkR3DOFw?si=MuoWNsIrBm_QjA5_" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

## How are DataFrames with Units handled?

The package uses the [`pint` library](https://pint.readthedocs.io/en/stable/index.html) to manage physical units in DataFrames. This allows for seamless unit conversions and calculations. Units in columns of DataFrames are handled using the [`pint-pandas` extension](https://pint-pandas.readthedocs.io/en/latest/index.html), which integrates `pint` with `pandas`.

!!! note
    Note that `pint-pandas` _must_ use the same `pint` unit registry as the rest of the package. This is ensured by following the instructions in the [pint-pandas documentation: Using a Shared Unit Registry](https://pint-pandas.readthedocs.io/en/latest/getting/projects.html).

```python
import pint
ureg = pint.get_application_registry()
```