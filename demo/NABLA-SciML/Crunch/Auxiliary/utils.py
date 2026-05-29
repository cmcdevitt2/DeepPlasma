"""
Minimal utilities for SSBroyden optimizer configuration.

This file contains only the static configuration dictionary needed by the
training scripts. All other utilities from the original NABLA-SciML library
have been removed to minimize dependencies.
"""

# Static configuration for SSBroyden optimizer
static_options_SSBroyden = {
    'gtol': 2.22e-16,
    'update_method': "ssbroyden2",
    'initial_scale': True,
    'ls_normal_c1': 1e-4, 'ls_normal_c2': 0.9, 'ls_normal_maxiter': 15,
    'ls_fb_c1_try1': 1e-4, 'ls_fb_c2_try1': 0.8, 'ls_fb_maxiter_try1': 10,
    'ls_fb_c1_try2': 1e-4, 'ls_fb_c2_try2': 0.5, 'ls_fb_maxiter_try2': 25
}
