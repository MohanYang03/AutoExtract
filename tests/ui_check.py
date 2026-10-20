"""
End-to-end check of the Batch Extract page in headless Chromium.

    .venv/Scripts/python tests/ui_check.py [base_url] [--pdf path] [--shots dir]

Needs the app running (backend serving the built frontend). Exercises:
option defaults, upload, values, drag + keyboard correction, SEM/SD switch,
n for all bars, clipboard copy, Excel download, exclude, console errors.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

import openpyxl
from playwright.sync_api import expect, sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def nums(page, row, cols):
    cells = page.locator("table tbody tr").nth(row).locator("td")
    return [cells.nth(c).inner_text() for c in cols]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:9000")
    ap.add_argument("--chart", default=None, help="chart image (default: a generated one)")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--shots", default=None)
    args = ap.parse_args()
    if args.chart is None:
        import tempfile
        import cv2
        sys.path.insert(0, HERE)
        from synth import ChartSpec, render
        args.chart = os.path.join(tempfile.mkdtemp(), "test_chart.png")
        cv2.imwrite(args.chart, render(ChartSpec([12.0, 31.5, 22.0], [1.5, 3.0, 2.2],
                                                 labels=["Control", "CUMS", "CUMS+EA"],
                                                 colors=["#ffffff", "#999999", "#d62728"])))

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000}, accept_downloads=True)
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=args.base)
        page = ctx.new_page()
        console_errors = []
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        print("page load")
        page.goto(f"{args.base}/extract")
        expect(page.get_by_role("heading", name="Batch Extract")).to_be_visible()
        check(page.locator("#opt-bar_names").is_checked(), "bar names on by default")
        check(page.locator("#opt-titles").is_checked(), "titles on by default")
        check(not page.locator("#opt-legend_colors").is_checked(), "legend colours off by default")
        check(page.locator("#opt-error").input_value() == "", "error type defaults to auto")

        print("upload chart image")
        page.set_input_files("#batch-input", args.chart)
        page.locator(".panel-card").first.wait_for(timeout=60000)
        means = [float(nums(page, i, [2])[0]) for i in range(3)]
        check(all(abs(m - t) < 0.4 for m, t in zip(means, [12.0, 31.5, 22.0])), f"means {means}")
        labels = [page.locator("table tbody tr").nth(i).locator("input").first.input_value() for i in range(3)]
        check(labels == ["Control", "CUMS", "CUMS+EA"], f"labels {labels}")
        check("3 bars" in page.locator(".results-toolbar").inner_text(), "toolbar counts bars")

        print("drag bar 1 mean line up")
        svg = page.locator(".panel-svg").first
        hit = svg.locator("line[stroke='#fd7e14']").first
        svg.scroll_into_view_if_needed()
        box = hit.bounding_box()
        before = float(nums(page, 0, [2])[0])
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 30, steps=5)
        page.mouse.up()
        after = float(nums(page, 0, [2])[0])
        check(after > before + 1, f"drag raised mean {before} -> {after}")

        print("arrow-key nudge")
        page.keyboard.press("ArrowDown")
        nudged = float(nums(page, 0, [2])[0])
        check(0 < after - nudged < 1, f"ArrowDown lowered mean slightly {after} -> {nudged}")

        print("SEM/SD switch and n for all")
        page.get_by_role("button", name="SD", exact=True).first.click()
        heads = page.locator("table thead th").all_inner_texts()
        check("sd" in heads and "mean+sd" in heads, f"headers after SD: {heads}")
        page.locator(".panel-card input[type=number]").last.fill("8")
        page.get_by_role("button", name="Apply").first.click()
        sem_text = nums(page, 1, [6])[0]
        check(sem_text not in ("", "0"), f"derived sem shown: {sem_text}")
        n_val = page.locator("table tbody tr").nth(2).locator("input[type=number]").input_value()
        check(n_val == "8", "n applied to every bar")

        print("copy rows")
        page.get_by_role("button", name="Copy rows").first.click()
        clip = page.evaluate("navigator.clipboard.readText()")
        lines = clip.split("\n")
        check(len(lines) == 3 and all(len(l.split("\t")) == 12 for l in lines), f"clipboard {len(lines)} rows x 12 cols")
        check(lines[0].split("\t")[6] == "8", "clipboard sample_size column")

        print("download Excel")
        with page.expect_download() as dl:
            page.get_by_role("button", name="Download Excel").click()
        data = open(dl.value.path(), "rb").read()
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb[wb.sheetnames[2]]
        header = [c.value for c in ws[1]][:12]
        check(header[-1] == "mean+sd", f"sheet header ends {header[-1]}")
        row = [c.value for c in ws[2]][:12]
        check(abs(row[0] - nudged) < 0.01 and row[6] == 8, f"edited values exported: mean {row[0]:.3f}, n {row[6]}")

        if args.pdf:
            print("upload a paper")
            page.set_input_files("#batch-input", args.pdf)
            page.wait_for_function("document.querySelectorAll('.panel-card').length > 5", timeout=180000)
            n_cards = page.locator(".panel-card").count()
            check(n_cards > 5, f"paper panels shown: {n_cards}")

        print("exclude a panel")
        text0 = page.locator(".results-toolbar").inner_text()
        page.get_by_role("button", name="Exclude").first.click()
        text1 = page.locator(".results-toolbar").inner_text()
        check(text0 != text1, "toolbar updates on exclude")

        if args.shots:
            os.makedirs(args.shots, exist_ok=True)
            page.get_by_role("button", name="Include").first.click()
            page.screenshot(path=os.path.join(args.shots, "dashboard.png"), full_page=False)
            page.locator(".panel-card").first.screenshot(path=os.path.join(args.shots, "panel.png"))

        check(not console_errors, f"no console errors {console_errors[:3]}")
        browser.close()
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
