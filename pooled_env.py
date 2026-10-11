"""Explicit raw360 pooled profile; the existing scalar environment is unchanged."""

from hud import Environment

from dreamscale_hud.contract import POOLED_PROFILE
from env import create_environment

env = create_environment(
    profile=POOLED_PROFILE, environment=Environment(name="dreamscale-libero-pooled")
)
