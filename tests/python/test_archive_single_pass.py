"""Streaming archive reads retain exact typed values and reject tampering."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ditserve import results


class ArchiveStreamingReadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'results.xlsx'
        self.value = dict(schema=results.SCHEMA, run=dict(config={'/~': 'literal =SUM(A1:A3)'},
            metrics=dict(big=10**50, precise=1.2345678901234567, negative_zero=-0.0,
                         empty={}, rows=[None, True, False, [], '\u0000\r\n\ud800'], long='x🙂' * 24000)),
            finalization=dict(status='completed'))

    def write(self):
        with patch.object(results, 'MAX_SHEET_ROWS', 12):
            results._write_book(self.path, self.value)

    def test_exact_nested_values_and_split_sheets(self):
        self.write()
        actual = results._read_book(self.path)
        self.assertEqual(results._json(actual), results._json(self.value))

    def test_formulas_in_header_data_or_manifest_are_rejected(self):
        for location in ('header', 'data', 'manifest'):
            with self.subTest(location=location):
                self.write()
                book = results._excel().load_workbook(self.path)
                try:
                    typed = next(sheet for sheet in book if sheet.title != 'manifest')
                    if location == 'manifest':
                        book['manifest']['B2'] = '=1+1'
                    elif location == 'header':
                        typed['A1'] = '=1+1'
                    else:
                        typed.cell(typed.max_row, 4).value = '=1+1'
                    book.save(self.path)
                finally:
                    book.close()
                with self.assertRaisesRegex(ValueError, 'formulas are forbidden'):
                    results._read_book(self.path)

    def test_header_row_count_and_digest_corruption_are_rejected(self):
        for location in ('header', 'node_rows', 'metadata_sha256'):
            with self.subTest(location=location):
                self.write()
                book = results._excel().load_workbook(self.path)
                try:
                    if location == 'header':
                        next(sheet for sheet in book if sheet.title != 'manifest')['A1'] = 'wrong'
                    else:
                        for row in book['manifest']:
                            if row[0].value == location:
                                row[1].value = '0'
                    book.save(self.path)
                finally:
                    book.close()
                with self.assertRaises(ValueError):
                    results._read_book(self.path)


if __name__ == '__main__':
    unittest.main()
