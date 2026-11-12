
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

@pytest.fixture
def client(monkeypatch):
    # Подменяем тяжёлый модуль нейросети
    fake_core = ModuleType("predict_core")

    fake_core.get_model_status = lambda: (
        False,
        "Тест: модель отсутствует"
    )
    fake_core.row_dict_from_flask_form = lambda form: {}
    fake_core.predict_from_dict = lambda data: {}

    monkeypatch.setitem(sys.modules, "predict_core", fake_core)

    # Загружаем настоящий Flask app.py
    app_path = (
        Path(__file__).resolve().parents[1]
        / "project"
        / "app.py"
    )

    spec = importlib.util.spec_from_file_location(
        "agl_app_test", app_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.app.config["TESTING"] = True

    return module.app.test_client()


# Тест №1: главная страница открывается
def test_home_page(client):
    response = client.get("/")

    assert response.status_code == 404 # broken for testing jenkins, fix to 200 later
    assert b"AGL Predictor" in response.data


# Тест №2: приложение сообщает об отсутствии модели
def test_model_not_loaded(client):
    response = client.post(
        "/",
        data={"action": "predict"}
    )

    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Модель не загружена:" in html
    assert "Тест: модель отсутствует" in html


# Тест №3: несуществующая страница возвращает 404
def test_not_found(client):
    response = client.get("/nonexistent-page")

    assert response.status_code == 404
