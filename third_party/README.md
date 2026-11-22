# Third-party code (reused, not written by our team)

`guided_diffusion/` is copied **unchanged** from the official TIW-DSM repository
(https://github.com/alsdudrla10/TIW-DSM, folder `guided_diffusion/`), which in turn
comes from OpenAI's *guided-diffusion* / *improved-diffusion* (MIT licence) as modified
by Kim et al. for Discriminator Guidance (ICML 2023).

There are two small edits. (1) In `nn.py`, `checkpoint()` always calls the function directly. The original
gradient-checkpointing path does not restore `torch.autocast` when it re-runs the forward pass, which crashes
under mixed precision; at 32x32 the memory saving is not needed, and the outputs are unchanged.
(2) In `fp16_util.py`, where we commented out `from . import logger`
because the logger module is not needed. The functions that used it are never called.

We use it for exactly one purpose: building the ADM encoder (`EncoderUNetModel`) for
the **pretrained, frozen feature extractor** and the **shallow discriminator head**
with the same architecture as the paper. This lets us load the authors' released
`32x32_classifier.pt` feature-extractor weights.
