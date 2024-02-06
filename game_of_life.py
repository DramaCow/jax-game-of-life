import pygame
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from functools import partial
import numpy as np
class Canvas(object):
    def __init__(self):
        self.screen = None
        self.clock = None
        
    def render(self, image):
        width_px, height_px, _ = image.shape
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((width_px, height_px))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        surface = pygame.surfarray.make_surface(np.asarray(image))

        assert self.screen is not None
        self.screen.blit(surface, (0, 0))
        pygame.event.pump()
        self.clock.tick(300)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
         
@jax.jit
def update(cells):
    counts = convolve(jnp.pad(cells, 1, mode='wrap'), jnp.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]), mode='valid')
    return jnp.where(cells, (counts == 2) | (counts == 3), (counts == 3))

@partial(jax.jit, static_argnums=(1,))
def cells_to_image(cells, scale_factor=1):
    cells = jnp.kron(cells, jnp.ones((scale_factor, scale_factor)))
    return jnp.transpose(cells[..., None].repeat(3, axis=-1).astype(jnp.uint8) * 255, (1, 0, 2))

if __name__=="__main__":
    canvas = Canvas()

    width = 800
    height = 800

    rng = jax.random.PRNGKey(1)
    cells = jax.random.bernoulli(rng, p=0.333, shape=(height, width))

    for _ in range(1000):
        canvas.render(cells_to_image(cells))
        cells = update(cells)
    canvas.close()