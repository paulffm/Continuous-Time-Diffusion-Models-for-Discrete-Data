import jax



def get_per_process_batch_size(batch_size):
    num_devices = jax.device_count()
    assert (batch_size // num_devices * num_devices == batch_size), (
        'Batch size %d must be divisible by num_devices %d', batch_size,
        num_devices)
    batch_size = batch_size // jax.process_count()

    return batch_size