# `wavelet.bumpsteer`

"Bump steerable wavelets.

Commonly used in phase harmonic scattering networks. Introduced in [1].
Implementation used here is based on the pyWPH implementation, see [2].

## References

[1] Mallat, Stéphane, Sixin Zhang, and Gaspar Rochette. "Phase harmonic correlations and convolutional neural networks." Information and Inference: A Journal of the IMA 9.3 (2020): 721-747.

[2] Regaldo-Saint Blancard, B., Allys, E., Boulanger, F., Levrier, F., & Jeffrey, N. (2021). A new approach for the statistical denoising of Planck interstellar dust polarization data. arXiv:2102.03160

::: calcite.wavelet.bumpsteer.filter_bank_2d
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Real Space

::: calcite.wavelet.bumpsteer.bump_steerable_kernel_2d_real
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Fourier Space

::: calcite.wavelet.bumpsteer.bump_steerable_kernel_2d_fourier
    handler: python
    options:
        show_source: false
        show_root_heading: true
