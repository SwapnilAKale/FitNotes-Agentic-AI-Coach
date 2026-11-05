"""
The pull-to-reload badge's curling arm (frontend/vendor/curl-arm) is loaded with a dynamic
import by vendor/pull-reload.js. Like the graph viewer, a missing file in its module graph
doesn't raise anything visible: the badge just shows its ring with no arm. So the graph is
walked here, and the route that serves the folder is checked for the file types the browser
needs (a .glb served as JavaScript or JSON would not load) and for staying inside the folder.
"""

import importlib.util
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent
CURL_ARM = _ROOT / "frontend" / "vendor" / "curl-arm"
ENTRY = CURL_ARM / "curl-arm.js"

_FROM = re.compile(r"""(?:^|\n)\s*(?:import|export)\b[^;\n]*?\sfrom\s*['"]([^'"]+)['"]""")
_SIDE = re.compile(r"""(?:^|\n)\s*import\s*['"]([^'"]+)['"]""")


def _server():
    name = "fitnotes_frontend_server"
    fe = sys.modules.get(name)
    if fe is None:
        spec = importlib.util.spec_from_file_location(name, _ROOT / "frontend" / "server.py")
        fe = importlib.util.module_from_spec(spec)
        sys.modules[name] = fe
        spec.loader.exec_module(fe)
    return TestClient(fe.app)


def test_module_graph_is_closed_and_relative():
    """No bare 'three' specifiers: the pages already map 'three' to release 180 (or not at
    all), and these files are release 170, so every import must be a relative path."""
    seen, missing, queue = set(), [], [ENTRY]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            missing.append(str(path))
            continue
        body = path.read_text(encoding="utf-8", errors="replace")
        for spec in set(_FROM.findall(body)) | set(_SIDE.findall(body)):
            assert spec.startswith("./"), f"{path.name} imports {spec!r}, not a relative path"
            queue.append((path.parent / spec).resolve())
    assert not missing, "unresolved import(s): " + ", ".join(missing)
    assert (CURL_ARM / "three" / "three.module.min.js").resolve() in seen


def test_model_and_texture_are_present():
    for name in ("arm-athletic.glb", "arm-athletic-normal.webp"):
        assert (CURL_ARM / name).is_file(), f"frontend/vendor/curl-arm/{name} is missing"


def test_pull_reload_loads_the_arm_module():
    js = (_ROOT / "frontend" / "vendor" / "pull-reload.js").read_text(encoding="utf-8")
    assert "curl-arm/curl-arm.js" in js and "createCurlArm" in js


def test_route_serves_each_file_type():
    client = _server()
    for path, kind in (("curl-arm.js", "text/javascript"),
                       ("three/three.module.min.js", "text/javascript"),
                       ("arm-athletic.glb", "model/gltf-binary"),
                       ("arm-athletic-normal.webp", "image/webp")):
        r = client.get(f"/vendor/curl-arm/{path}")
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith(kind), (path, r.headers["content-type"])


def test_route_stays_inside_the_folder():
    client = _server()
    assert client.get("/vendor/curl-arm/three/LICENSE").status_code == 404      # not a served type
    assert client.get("/vendor/curl-arm/missing.js").status_code == 404
    r = client.get("/vendor/curl-arm/..%2F..%2Fserver.py")
    assert r.status_code == 404 or "FastAPI" not in r.text
