"""Read-only compatibility for typed original Wan pilot workbooks."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ditserve import results


HEADERS = ("path_json", "type", "part", "value_json_fragment")


def historical_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def legacy_rows(value, path=(), chunk_size=17):
    kind = {dict: "dict", list: "list", str: "str", bool: "bool", int: "int", float: "float", type(None): "null"}[type(value)]
    scalar = str(len(value)) if isinstance(value, (dict, list)) else historical_json(value)
    for part, offset in enumerate(range(0, len(scalar), chunk_size)):
        yield historical_json(list(path)), kind, part, scalar[offset:offset + chunk_size]
    if isinstance(value, dict):
        for key, item in value.items():
            yield from legacy_rows(item, path + (key,), chunk_size)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from legacy_rows(item, path + (index,), chunk_size)


class WanProfileResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.value = dict(schema_version="ditserve.wan_breakdown.v1", status="completed",
            z={"last": True, "first": None, "中文🙂": "鱼游过湖面", "escape/~": "=SUM(A1:A3)"},
            a=[False, None, 123456789012345678901234567890, 1.2345678901234567],
            numeric_keys={"10": "ten", "02": "two", "": "empty"})

    def write_book(self, value=None, mutate=None):
        value = self.value if value is None else value
        digest = hashlib.sha256(historical_json(value).encode("ascii")).hexdigest()
        book = results._excel().Workbook()
        book.remove(book.active)
        for name, content in (("summary", {"payload_sha256": digest}), ("evidence", value)):
            sheet = book.create_sheet(name)
            sheet.append(HEADERS)
            for row in legacy_rows(content):
                sheet.append(row)
        if mutate:
            mutate(book)
        path = self.root / "results.xlsx"
        book.save(path)
        book.close()
        (self.root / "REPORT.md").write_text("Historical report: immutable.\n", encoding="utf-8")
        return path

    @staticmethod
    def scalar_cell(book, path, part=0):
        key = historical_json(path)
        return next(row[3] for row in book["evidence"].iter_rows(min_row=2)
                    if row[0].value == key and row[2].value == part)

    def test_original_order_boolean_null_unicode_and_exact_numbers_are_preserved(self):
        path = self.write_book()
        before = path.read_bytes()
        restored = results.load(self.root)
        self.assertEqual(historical_json(restored), historical_json(self.value))
        self.assertEqual(list(restored), list(self.value))
        self.assertEqual(list(restored["z"]), list(self.value["z"]))
        self.assertIs(restored["z"]["last"], True)
        self.assertIsNone(restored["z"]["first"])
        self.assertIs(restored["a"][0], False)
        self.assertIsNone(restored["a"][1])
        self.assertEqual(restored["a"][2], 123456789012345678901234567890)
        self.assertEqual(restored["z"]["中文🙂"], "鱼游过湖面")
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual((self.root / "REPORT.md").read_text(), "Historical report: immutable.\n")

    def test_boolean_and_null_corruption_is_rejected_without_rewriting(self):
        for path, replacement in ((["z", "last"], "1"), (["z", "first"], '"null"')):
            with self.subTest(path=path):
                target = self.write_book(mutate=lambda book: setattr(self.scalar_cell(book, path), "value", replacement))
                before = target.read_bytes()
                with self.assertRaisesRegex(ValueError, "scalar type changed"):
                    results.load(self.root)
                self.assertEqual(target.read_bytes(), before)

    def test_unicode_content_change_is_rejected_by_historical_hash(self):
        value = copy.deepcopy(self.value)
        value["z"]["中文🙂"] = "鱼"
        def corrupt(book):
            self.scalar_cell(book, ["z", "中文🙂"]).value = historical_json("鸟")
        path = self.write_book(value, mutate=corrupt)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            results.load(self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_insertion_order_is_part_of_the_original_hash(self):
        def reorder(book):
            sheet = book["evidence"]
            rows = list(sheet.values)
            first = next(i for i, row in enumerate(rows) if row[0] == '["z"]')
            second = next(i for i, row in enumerate(rows) if row[0] == '["a"]')
            rows[first], rows[second] = rows[second], rows[first]
            sheet.delete_rows(1, sheet.max_row)
            for row in rows:
                sheet.append(row)
        self.write_book(mutate=reorder)
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            results.load(self.root)

    def test_missing_fragment_and_duplicate_fragment_are_rejected(self):
        for duplicate in (False, True):
            def corrupt(book):
                sheet = book["evidence"]
                row = next(row for row in sheet.iter_rows(min_row=2)
                           if row[0].value == '["a",2]' and row[2].value == 1)
                if duplicate:
                    sheet.append([cell.value for cell in row])
                else:
                    sheet.delete_rows(row[0].row)
            self.write_book(mutate=corrupt)
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                results.load(self.root)

    def test_excel_formula_and_unknown_schema_are_rejected(self):
        self.write_book(mutate=lambda book: setattr(self.scalar_cell(book, ["z", "last"]), "value", "=TRUE()"))
        with self.assertRaisesRegex(ValueError, "formula"):
            results.load(self.root)
        value = dict(self.value, schema_version="unsupported")
        self.write_book(value)
        with self.assertRaisesRegex(ValueError, "unsupported legacy Wan schema"):
            results.load(self.root)


if __name__ == "__main__":
    unittest.main()
