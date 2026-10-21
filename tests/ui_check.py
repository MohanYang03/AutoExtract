"""
End-to-end check of the Bar Analyzer and Box Analyzer pages in headless Chromium.

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
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:2800")
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

        print("page load (the old /extract link lands on the Bar Analyzer)")
        page.goto(f"{args.base}/extract")
        expect(page.get_by_role("heading", name="Bar Analyzer")).to_be_visible()
        check(page.url.endswith("/bar_analyzer"), f"redirected to {page.url}")
        check(page.get_by_role("link", name="Batch Extract").count() == 0, "no Batch Extract link in the nav")
        bar = page.locator("#page-bar")
        check(bar.locator("#opt-bar-bar_names").is_checked(), "bar names on by default")
        check(bar.locator("#opt-bar-titles").is_checked(), "titles on by default")
        check(not bar.locator("#opt-bar-legend_colors").is_checked(), "legend colours off by default")
        check(bar.locator("#opt-error").input_value() == "", "error type defaults to auto")

        print("upload chart image")
        bar.locator("#batch-input-bar").set_input_files(args.chart)
        bar.locator(".panel-card").first.wait_for(timeout=60000)
        means = [float(nums(bar, i, [2])[0]) for i in range(3)]
        check(all(abs(m - t) < 0.4 for m, t in zip(means, [12.0, 31.5, 22.0])), f"means {means}")
        labels = [bar.locator("table tbody tr").nth(i).locator("input").first.input_value() for i in range(3)]
        check(labels == ["Control", "CUMS", "CUMS+EA"], f"labels {labels}")
        check("3 bars" in bar.locator(".results-toolbar").inner_text(), "toolbar counts bars")

        print("drag bar 1 mean line up")
        svg = bar.locator(".panel-svg").first
        hit = svg.locator("line[stroke='#fd7e14']").first
        svg.scroll_into_view_if_needed()
        box = hit.bounding_box()
        before = float(nums(bar, 0, [2])[0])
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2 - 30, steps=5)
        page.mouse.up()
        after = float(nums(bar, 0, [2])[0])
        check(after > before + 1, f"drag raised mean {before} -> {after}")

        print("arrow-key nudge")
        page.keyboard.press("ArrowDown")
        nudged = float(nums(bar, 0, [2])[0])
        check(0 < after - nudged < 1, f"ArrowDown lowered mean slightly {after} -> {nudged}")

        print("SEM/SD switch and n for all")
        bar.get_by_role("button", name="SD", exact=True).first.click()
        heads = bar.locator("table thead th").all_inner_texts()
        check("sd" in heads and "mean+sd" in heads, f"headers after SD: {heads}")
        bar.locator(".panel-card input[type=number]").last.fill("8")
        bar.get_by_role("button", name="Apply").first.click()
        sem_text = nums(bar, 1, [6])[0]
        check(sem_text not in ("", "0"), f"derived sem shown: {sem_text}")
        n_val = bar.locator("table tbody tr").nth(2).locator("input[type=number]").input_value()
        check(n_val == "8", "n applied to every bar")

        print("copy rows")
        bar.get_by_role("button", name="Copy rows").first.click()
        clip = page.evaluate("navigator.clipboard.readText()")
        lines = clip.split("\n")
        check(len(lines) == 3 and all(len(l.split("\t")) == 12 for l in lines), f"clipboard {len(lines)} rows x 12 cols")
        check(lines[0].split("\t")[6] == "8", "clipboard sample_size column")

        print("download Excel")
        with page.expect_download() as dl:
            bar.get_by_role("button", name="Download Excel").click()
        data = open(dl.value.path(), "rb").read()
        wb = openpyxl.load_workbook(io.BytesIO(data))
        ws = wb[wb.sheetnames[2]]
        header = [c.value for c in ws[1]][:12]
        check(header[-1] == "mean+sd", f"sheet header ends {header[-1]}")
        row = [c.value for c in ws[2]][:12]
        check(abs(row[0] - nudged) < 0.01 and row[6] == 8, f"edited values exported: mean {row[0]:.3f}, n {row[6]}")

        if args.pdf:
            print("upload a paper")
            bar.locator("#batch-input-bar").set_input_files(args.pdf)
            page.wait_for_function("document.querySelectorAll('.panel-card').length > 5", timeout=180000)
            n_cards = bar.locator(".panel-card").count()
            check(n_cards > 5, f"paper panels shown: {n_cards}")

        print("exclude a panel")
        text0 = bar.locator(".results-toolbar").inner_text()
        bar.get_by_role("button", name="Exclude").first.click()
        text1 = bar.locator(".results-toolbar").inner_text()
        check(text0 != text1, "toolbar updates on exclude")
        if args.shots:
            os.makedirs(args.shots, exist_ok=True)
            bar.get_by_role("button", name="Include").first.click()
            page.screenshot(path=os.path.join(args.shots, "bar_page.png"), full_page=False)

        import tempfile
        import cv2
        sys.path.insert(0, HERE)
        from synth import BoxSpec, render_box
        tmp = tempfile.mkdtemp()
        box_png = os.path.join(tmp, "box_chart.png")
        cv2.imwrite(box_png, render_box(BoxSpec([(2, 5, 7, 9, 13), (4, 8, 11, 15, 19)], ylim=(0, 20))))

        print("bar page ignores box plots")
        n_before = bar.locator(".panel-card").count()
        bar.locator("#batch-input-bar").set_input_files(box_png)
        bar.get_by_text("box_chart.png: no bar charts or dot plots found").wait_for(timeout=60000)
        check(bar.locator(".panel-card").count() == n_before, "no panel added for a box plot")

        print("box page: two files at once, only the box plot is read")
        page.get_by_role("link", name="Box Analyzer").first.click()
        expect(page.get_by_role("heading", name="Box Analyzer")).to_be_visible()
        boxp = page.locator("#page-box")
        check(boxp.locator(".panel-card").count() == 0, "box page starts empty (results kept per page)")
        check(boxp.locator("#opt-error").count() == 0, "no SEM/SD option on the box page")
        boxp.locator("#batch-input-box").set_input_files([box_png, args.chart])
        boxp.get_by_text("test_chart.png: no box plots found").wait_for(timeout=120000)
        check(boxp.locator(".panel-card").count() == 1, f"one box panel shown: {boxp.locator('.panel-card').count()}")
        card = boxp.locator(".panel-card").last
        check(card.get_by_text("box plot").count() > 0, "box panel recognised")
        cells = card.locator("table tbody tr").first.locator("td").all_inner_texts()
        five = [float(c) for c in cells[2:7]]
        check(all(abs(a - b) < 0.25 for a, b in zip(five, [2, 5, 7, 9, 13])), f"five-number summary {five}")
        check(cells[8].strip() == "", "no estimate before n")
        card.locator("input[type=number]").last.fill("12")
        card.get_by_role("button", name="Apply").click()
        cells = card.locator("table tbody tr").first.locator("td").all_inner_texts()
        est_mean, est_sd = float(cells[8]), float(cells[9])
        check(6.5 < est_mean < 7.6 and 2 < est_sd < 4.5, f"estimates with n=12: mean {est_mean}, sd {est_sd}")
        card.locator("select").first.select_option("tukey")
        cells = card.locator("table tbody tr").first.locator("td").all_inner_texts()
        check(float(cells[9]) != est_sd, f"whisker type changes SD {est_sd} -> {cells[9]}")
        print("drag a median")
        bsvg = card.locator(".panel-svg")
        med = bsvg.locator("line[stroke='#0d6efd']").first
        bsvg.scroll_into_view_if_needed()
        mb = med.bounding_box()
        page.mouse.move(mb["x"] + mb["width"] / 2, mb["y"] + mb["height"] / 2)
        page.mouse.down()
        page.mouse.move(mb["x"] + mb["width"] / 2, mb["y"] + mb["height"] / 2 - 20, steps=5)
        page.mouse.up()
        new_med = float(card.locator("table tbody tr").first.locator("td").all_inner_texts()[4])
        check(new_med > 7.3, f"drag raised median 7 -> {new_med}")
        card.get_by_role("button", name="Copy rows").click()
        bclip = page.evaluate("navigator.clipboard.readText()").split("\n")
        check(len(bclip) == 2 and bclip[0].split("\t")[0] != "" and bclip[0].split("\t")[6] == "12",
              "box clipboard rows carry estimated mean and n")
        with page.expect_download() as dl:
            boxp.get_by_role("button", name="Download Excel").click()
        check(dl.value.suggested_filename == "box_plots.xlsx", f"download name {dl.value.suggested_filename}")
        wb = openpyxl.load_workbook(io.BytesIO(open(dl.value.path(), "rb").read()))
        bws = wb[wb.sheetnames[-1]]
        bhead = [c.value for c in bws[1]]
        check("Q1" in bhead and str(bws.cell(2, 1).value).startswith("=IF("), "box sheet has formulas and quartiles")
        check("2 boxes" in boxp.locator(".results-toolbar").inner_text(), "toolbar counts boxes")

        print("violin page")
        import numpy as np
        from synth import LineSpec, ViolinSpec, render_line, render_violin
        vrng = np.random.default_rng(1)
        vspec = ViolinSpec([vrng.normal(10, 2, 14), vrng.normal(14, 3, 14)], inner="lines")
        violin_png = os.path.join(tmp, "violin_chart.png")
        cv2.imwrite(violin_png, render_violin(vspec))
        page.get_by_role("link", name="Violin Analyzer").first.click()
        expect(page.get_by_role("heading", name="Violin Analyzer")).to_be_visible()
        vp = page.locator("#page-violin")
        vp.locator("#batch-input-violin").set_input_files(violin_png)
        vp.locator(".panel-card").first.wait_for(timeout=60000)
        vcard = vp.locator(".panel-card").first
        check(vcard.get_by_text("violin plot").count() > 0, "violin panel recognised")
        vcells = vcard.locator("table tbody tr").first.locator("td").all_inner_texts()
        check(abs(float(vcells[4]) - vspec.stats[0][2]) < 0.3, f"violin median {vcells[4]} vs {vspec.stats[0][2]:.2f}")
        check(vcard.get_by_text("Violin ends").count() > 0, "violin ends option shown")
        vcard.locator("input[type=number]").last.fill("14")
        vcard.get_by_role("button", name="Apply").click()
        vcells = vcard.locator("table tbody tr").first.locator("td").all_inner_texts()
        check(vcells[8].strip() != "", f"violin estimate with n: {vcells[8]}")

        print("line page")
        lspec = LineSpec([0, 7, 14, 21], [[20, 22, 25, 27], [16, 17, 18, 19]], errs=[[1] * 4, [0.8] * 4])
        line_png = os.path.join(tmp, "line_chart.png")
        cv2.imwrite(line_png, render_line(lspec))
        page.get_by_role("link", name="Line Analyzer").first.click()
        expect(page.get_by_role("heading", name="Line Analyzer")).to_be_visible()
        lp = page.locator("#page-line")
        lp.locator("#batch-input-line").set_input_files(line_png)
        lp.locator(".panel-card").first.wait_for(timeout=60000)
        lcard = lp.locator(".panel-card").first
        check(lcard.get_by_text("line plot").count() > 0, "line panel recognised")
        lrows = lcard.locator("table tbody tr")
        check(lrows.count() == 8, f"8 point rows: {lrows.count()}")
        xs = [lrows.nth(i).locator("td").nth(1).inner_text() for i in range(4)]
        check(xs == ["0", "7.000", "14.00", "21.00"] or [round(float(x)) for x in xs] == [0, 7, 14, 21], f"x values {xs}")
        lcard.locator("table tbody tr").first.locator("input").first.fill("Vehicle")
        check("Vehicle" in lp.locator(".panel-card").first.inner_text() or
              lcard.locator("input").first.input_value() == "Vehicle", "series renamed")
        lcard.get_by_role("button", name="Copy rows").click()
        lclip = page.evaluate("navigator.clipboard.readText()").split("\n")
        check(len(lclip) == 8 and lclip[0].split("\t")[5] == "Vehicle", "line clipboard rows carry the series name")
        with page.expect_download() as dl:
            lp.get_by_role("button", name="Download Excel").click()
        check(dl.value.suggested_filename == "line_plots.xlsx", f"download name {dl.value.suggested_filename}")

        print("back to the bar page: its results are still there")
        page.get_by_role("link", name="Bar Analyzer").first.click()
        expect(page.get_by_role("heading", name="Bar Analyzer")).to_be_visible()

        check(bar.locator(".panel-card").count() >= 1 and bar.is_visible() and not boxp.is_visible(),
              "bar results kept after visiting the box page")
        check(not console_errors, f"no console errors {console_errors[:3]}")
        browser.close()
    print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
