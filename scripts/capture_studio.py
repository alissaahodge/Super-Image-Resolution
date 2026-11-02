"""Capture genuine browser screenshots after the LFW model is trained."""
from pathlib import Path
from playwright.sync_api import sync_playwright

Path("docs/images").mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page.goto("http://127.0.0.1:5173", wait_until="networkidle")
    page.screenshot(path="docs/images/studio-upload.png", full_page=True)
    image = next(Path("lfw-eval").glob("*.jpg"))
    page.locator('input[type="file"]').set_input_files(str(image))
    page.get_by_role("button", name="Enhance image").click()
    page.get_by_role("link", name="Download PNG").wait_for(timeout=120000)
    page.screenshot(path="docs/images/studio-comparison.png", full_page=True)
    browser.close()
