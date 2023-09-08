import torch
import numpy as np
from tqdm import tqdm
from lib.models.ddsm import *

def Euler_Maruyama_sampler(
    score_model,
    sample_shape,
    init=None,
    mask=None,
    alpha=None,
    beta=None,
    max_time=4,
    min_time=0.01,
    time_dilation=1,
    time_dilation_start_time=None,
    batch_size=64,
    num_steps=100,
    device="cuda",
    random_order=False,
    speed_balanced=True,
    speed_factor=None,
    concat_input=None,
    eps=1e-5,
):
    """
    Generate samples from score-based models with the Euler-Maruyama solver
    for (multivariate) Jacobi diffusion processes with stick-breaking
    construction.

    Parameters
    ----------
    score_model : torch.nn.Module
        A PyTorch time-dependent score model.
    sample_shape : tuple
        Shape of all dimensions of sample tensor without the batch dimension.
    init: torch.Tensor, default is None
        If specified, use as initial values instead of sampling from stationary distribution.
    alpha :  torch.Tensor, default is None
        Jacobi Diffusion parameters. If None, use default choices of alpha, beta =
        (1, k-1), (1, k-2), (1, k-3), ..., (1, 1) where k is the number of categories.
    beta : torch.Tensor, default is None
        See above `for alpha`.
    max_time : float, default is 4
        Max time of reverse diffusion sampling.
    min_time : float, default is 0.01
        Min time of reverse diffusion sampling.
    time_dilation : float, default is 1
        Use `time_dilation > 1` to bias samples toward high density areas.
    time_dilation_start_time : float, default is None
        If specified, start time dilation from this timepoint to min_time.
    batch_size : int, default is 64
        Number of samples to generate
    num_steps: int, default is 100
        Total number of steps for reverse diffusion sampling.
    device: str, default is 'cuda'
        Use 'cuda' to run on GPU or 'cpu' to run on CPU
    random_order : bool, default is False
        Whether to convert x to v space with randomly ordered stick-breaking transform.
    speed_balanced : bool, default is True
        If True use speed factor `s=(a+b)/2`, otherwise use `s=1`.
    eps: float, default is 1e-5
        All state values are clamped to (eps, 1-eps) for numerical stability.


    Returns
    -------
    Samples : torch.Tensor
        Samples in x space.
    """
    sb = UnitStickBreakingTransform()
    if alpha is None:
        alpha = torch.ones(sample_shape[-1] - 1, dtype=torch.float, device=device)
    if beta is None:
        beta = torch.arange(
            sample_shape[-1] - 1, 0, -1, dtype=torch.float, device=device
        )

    if speed_balanced:
        if speed_factor is None:
            s = 2.0 / (alpha + beta)
        else:
            s = speed_factor * 2.0 / (alpha + beta)
    else:
        s = torch.ones(sample_shape[-1] - 1).to(device)

    if init is None:
        init_v = Beta(alpha, beta).sample((batch_size,) + sample_shape[:-1]).to(device)
    else:
        init_v = sb._inverse(init).to(device)

    if time_dilation_start_time is None:
        time_steps = torch.linspace(
            max_time, min_time, num_steps * time_dilation + 1, device=device
        )
    else:
        time_steps = torch.cat(
            [
                torch.linspace(
                    max_time,
                    time_dilation_start_time,
                    round(num_steps * (max_time - time_dilation_start_time) / max_time)
                    + 1,
                )[:-1],
                torch.linspace(
                    time_dilation_start_time,
                    min_time,
                    round(num_steps * (time_dilation_start_time - min_time) / max_time)
                    * time_dilation
                    + 1,
                ),
            ]
        )
    step_sizes = time_steps[:-1] - time_steps[1:]
    time_steps = time_steps[:-1]
    v = init_v.detach()

    if mask is not None:
        assert mask.shape[-1] == v.shape[-1] + 1

    if random_order:
        order = np.arange(sample_shape[-1])
    else:
        if mask is not None:
            mask_v = sb.inv(mask)

    with torch.no_grad():
        for i_step in tqdm.tqdm(range(len(time_steps))):
            time_step = time_steps[i_step]
            step_size = step_sizes[i_step]
            x = sb(v)

            if time_dilation_start_time is not None:
                if time_step < time_dilation_start_time:
                    c = time_dilation
                else:
                    c = 1
            else:
                c = time_dilation

            if not random_order:
                g = torch.sqrt(v * (1 - v))
                batch_time_step = torch.ones(batch_size, device=device) * time_step

                with torch.enable_grad():
                    if concat_input is None:
                        score = score_model(x, batch_time_step)
                    else:
                        score = score_model(
                            torch.cat([x, concat_input], -1), batch_time_step
                        )

                    mean_v = (
                        v
                        + s[(None,) * (v.ndim - 1)]
                        * (
                            (
                                0.5
                                * (
                                    alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                    - beta[(None,) * (v.ndim - 1)] * v
                                )
                            )
                            - (1 - 2 * v)
                            - (g**2) * gx_to_gv(score, x)
                        )
                        * (-step_size)
                        * c
                    )

                next_v = mean_v + torch.sqrt(step_size * c) * torch.sqrt(
                    s[(None,) * (v.ndim - 1)]
                ) * g * torch.randn_like(v)

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

                v = torch.clamp(next_v, eps, 1 - eps).detach()
            else:
                x = x[..., np.argsort(order)]
                order = np.random.permutation(np.arange(sample_shape[-1]))

                if mask is not None:
                    mask_v = sb.inv(mask[..., order])

                v = sb._inverse(x[..., order], prevent_nan=True)
                v = torch.clamp(v, eps, 1 - eps).detach()

                g = torch.sqrt(v * (1 - v))
                batch_time_step = torch.ones(batch_size, device=device) * time_step

                with torch.enable_grad():
                    if concat_input is None:
                        score = score_model(x, batch_time_step)
                    else:
                        score = score_model(
                            torch.cat([x, concat_input], -1), batch_time_step
                        )
                    mean_v = (
                        v
                        + s[(None,) * (v.ndim - 1)]
                        * (
                            (
                                0.5
                                * (
                                    alpha[(None,) * (v.ndim - 1)] * (1 - v)
                                    - beta[(None,) * (v.ndim - 1)] * v
                                )
                            )
                            - (1 - 2 * v)
                            - (g**2) * (gx_to_gv(score[..., order], x[..., order]))
                        )
                        * (-step_size)
                        * c
                    )
                next_v = mean_v + torch.sqrt(step_size * c) * torch.sqrt(
                    s[(None,) * (v.ndim - 1)]
                ) * g * torch.randn_like(v)

                if mask is not None:
                    next_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

                v = torch.clamp(next_v, eps, 1 - eps).detach()

    if mask is not None:
        mean_v[~torch.isnan(mask_v)] = mask_v[~torch.isnan(mask_v)]

    # Do not include any noise in the last sampling step.
    if not random_order:
        return sb(torch.clamp(mean_v, eps, 1 - eps))
    else:
        return sb(torch.clamp(mean_v, eps, 1 - eps))[..., np.argsort(order)]

### possibly to test these samplers:
# from: https://github.com/henriupton99/score-based-generative-models/blob/main/code/sampler.py

import torch
import numpy as np
from tqdm import tqdm
from scipy import integrate
## The number of sampling steps.
def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           num_steps, 
                           batch_size=64, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
    * marginal_prob_std(t)[:, None, None, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x


#@title Define the Predictor-Corrector sampler (double click to expand or collapse)

def pc_sampler(score_model, 
               marginal_prob_std,
               diffusion_coeff,
               snr,
               num_steps, 
               batch_size=64,                
               device='cuda',
               eps=1e-3):
  """Generate samples from score-based models with Predictor-Corrector method.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient 
      of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.    
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns: 
    Samples.
  """
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
  time_steps = np.linspace(1., eps, num_steps)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      # Corrector step (Langevin MCMC)
      grad = score_model(x, batch_time_step)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = np.sqrt(np.prod(x.shape[1:]))
      langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
      x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

      # Predictor step (Euler-Maruyama)
      g = diffusion_coeff(batch_time_step)
      x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
      x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)      
    
    # The last step does not include any noise
    return x_mean
  

#@title Define the ODE sampler (double click to expand or collapse)

## The error tolerance for the black-box ODE solver
def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                atol, 
                rtol, 
                batch_size=64, 
                device='cuda', 
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  t = torch.ones(batch_size, device=device)
  # Create the latent code
  if z is None:
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
      * marginal_prob_std(t)[:, None, None, None]
  else:
    init_x = z
    
  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
    with torch.no_grad():    
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t    
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)
  
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x