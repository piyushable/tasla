import pathlib

from streamlit.testing.v1 import AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "vehicle_path_standalone.py")


def test_page_renders_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_drive_tab_starts_the_car():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value("2️⃣ Drive").run()
    start = next(b for b in at.button if "Start driving" in b.label)
    start.click().run()
    assert not at.exception
    assert any("Driving Path" in s.value for s in at.success)
