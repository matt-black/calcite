# `wavelet.cake`

Cake wavelets.

2D cake wavelets were introduced in [1], while 3D cake wavelets were introduced in [2].

## References

[1] Bekkers, Erik, et al. "A multi-orientation analysis approach to retinal vessel tracking." Journal of Mathematical Imaging and Vision 49 (2014): 583-610.

[2] Janssen, Michiel HJ, et al. "Design and processing of invertible orientation scores of 3D images." Journal of mathematical imaging and vision 60 (2018): 1427-1458.

## Filter Banks

::: calcite.wavelet.cake.filter_bank_2d
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: calcite.wavelet.cake.orientation_bank_2d
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: calcite.wavelet.cake.orientation_bank_2d_real
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: calcite.wavelet.cake.orientation_bank_2d_fourier
    handler: python
    options:
        show_source: false
        show_root_heading: true

::: calcite.wavelet.cake.orientation_bank_3d_fourier
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Single Kernels

::: calcite.wavelet.cake.cake_wavelet_3d_fourier
    handler: python
    options:
        show_source: false
        show_root_heading: true

## Utilities

::: calcite.wavelet.cake.split_cake_wavelet_fourier
    handler: python
    options:
        show_source: false
        show_root_heading: true