import logging
import pathlib
import runpy

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "vehicle_path_standalone.py"


@pytest.fixture(scope="session")
def sim():
    """The app's module globals, loaded outside `streamlit run` so the planner functions can be called directly."""
    logging.getLogger("streamlit").setLevel(logging.ERROR)  # bare mode warns about every st.* call
    return runpy.run_path(str(APP))


def drive(sim, obstacles, choice="Path 2", n_paths=3, speed_inc=2, max_ticks=600):
    """Run the 2D tab's simulation to completion and return the final sim state."""
    paths = sim["make_paths"](n_paths)
    sim["restore"](obstacles)
    state = sim["new_sim"](choice, paths, obstacles, speed_inc * sim["RES"] / sim["DT"])
    for _ in range(max_ticks):
        if not state["running"]:
            break
        sim["step_world"](state, obstacles, speed_inc)
    return state
