import numpy as np
import pytest

from conftest import drive

LAYOUTS = {
    "single_block": [("🚗 Vehicle", "Static", 15, 0)],
    "slalom": [("🚗 Vehicle", "Static", 12, 0.5), ("🛒 Pushcart", "Static", 22, -2.5), ("🐄 Cow", "Static", 31, 1.5)],
    "crossing": [("🧍 Pedestrian", "Crossing the road", 16, 4), ("🚗 Vehicle", "Static", 26, -1)],
}


def plan(sim, obstacles, choice="Path 2"):
    paths = sim["make_paths"](3)
    chosen = sim["densify"]([(0.0, 0.0)] + paths[choice])
    return sim["plan_route"](paths[choice][-1], obstacles, (0.0, 0.0), 2.5, sim["path_dev"](chosen))


def test_route_goes_around_a_parked_car(sim):
    car = sim["make_obstacle"]("🚗 Vehicle", "Static", 15, 0)
    route = plan(sim, [car])
    assert route is not None
    beside = route[np.abs(route[:, 0] - car["x"]) < car["L"] / 2 + sim["HX0"]]
    # the planner keeps a margin, so alongside the car the route must clear it with room to spare
    assert np.all(np.abs(beside[:, 1] - car["y"]) >= car["W"] / 2 + sim["HY0"])


def test_route_stays_on_the_road(sim):
    route = plan(sim, [sim["make_obstacle"](*o) for o in LAYOUTS["slalom"]])
    assert np.all(np.abs(route[:, 1]) + sim["HALF_W"] <= sim["ROAD_HALF"])
    assert np.all(np.diff(route[:, 0]) >= 0), "routes only ever move forward"


@pytest.mark.parametrize("name", LAYOUTS)
def test_fixed_layouts_reach_the_goal_cleanly(sim, name):
    state = drive(sim, [sim["make_obstacle"](*o) for o in LAYOUTS[name]])
    assert state["status"] == "done"
    assert not state["collided"]


def test_random_layouts_never_collide(sim, monkeypatch):
    real_rng = np.random.default_rng
    finished = 0
    for seed in range(30):
        monkeypatch.setattr(np.random, "default_rng", lambda *a, s=seed: real_rng(s))
        obstacles = sim["random_obstacles"](4)
        monkeypatch.setattr(np.random, "default_rng", real_rng)
        state = drive(sim, obstacles, choice=f"Path {seed % 3 + 1}")
        assert not state["collided"], f"collision with seed {seed}"
        finished += state["status"] == "done"
    # a crowded layout can leave the car waiting for a gap, but that should stay rare
    assert finished >= 27
