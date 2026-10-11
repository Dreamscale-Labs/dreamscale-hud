"""Native pi0.5 camera profile for HUD-hosted LIBERO."""

from hud import Environment

from dreamscale_hud.contract import PI05_PROFILE
from env import create_environment

env = create_environment(
    profile=PI05_PROFILE, environment=Environment(name="dreamscale-libero-pooled-pi05")
)
