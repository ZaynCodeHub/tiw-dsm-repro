# Third-party code (reused, not written by our team)

`guided_diffusion/` is copied **unchanged** from the official TIW-DSM repository
(https://github.com/alsdudrla10/TIW-DSM, folder `guided_diffusion/`), which in turn
comes from OpenAI's *guided-diffusion* / *improved-diffusion* (MIT licence) as modified
by Kim et al. for Discriminator Guidance (ICML 2023).

The only edit is in `fp16_util.py`, where we commented out `from . import logger`
because the logger module is not needed. The functions that used it are never called.

We use it for exactly one purpose: building the ADM encoder (`EncoderUNetModel`) for
the **pretrained, frozen feature extractor** and the **shallow discriminator head**
with the same architecture as the paper. This lets us load the authors' released
`32x32_classifier.pt` feature-extractor weights.
